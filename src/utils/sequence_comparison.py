"""CPU diagnostics of the SAME head connections and probes across sequential best checkpoints.

Curves from different hidden units/architectures are never treated as equivalent. KAN and
MLP are compared at their scalar MOS response, on the same frozen-backbone embeddings.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

SEQUENCE = ("bvcc", "singmos", "tmhint", "brspeech", "bvcc")
DATASET_IDS = {"bvcc": 1, "singmos": 2, "tmhint": 3, "brspeech": 4}
COLORS = ("#F59E0B", "#7C3AED", "#059669", "#2563EB", "#DB2777", "#64748B")
DEFAULTS = {"enabled": True, "probe_samples": 64, "grid_points": 201,
            "inputs_per_block": 2, "outputs_per_layer": 1,
            "response_points": 33, "conditional_dims": 1,
            "change_atol": 1e-6, "change_rtol": 1e-3, "wandb": True}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    """CSV for spreadsheets: semicolon delimiter and decimal comma; keep strings intact."""
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (format(value, ".12g").replace(".", ",")
                                   if isinstance(value, float) and math.isfinite(value) else
                                   "" if value is None else value) for key, value in row.items()})


class ProbeCollector:
    """Capture pooled embeddings BEFORE head LayerNorm/sigmoid, without another backbone pass."""
    def __init__(self, head, count):
        self.count = int(count)
        self.values = []
        self.size = 0
        self.handle = head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        remaining = self.count - self.size
        if remaining > 0:
            values = args[0][:remaining].detach().float().cpu().numpy().copy()
            self.values.append(values)
            self.size += len(values)

    def close(self):
        self.handle.remove()

    def save(self, path, targets, filenames, metadata):
        if not self.values or self.size > len(targets) or self.size > len(filenames):
            raise ValueError("Probe embeddings, targets and metadata rows do not align")
        values = np.concatenate(self.values)
        targets = np.asarray(targets[:self.size], dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(targets).all():
            raise ValueError("Nonfinite diagnostic probes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, embeddings=values, targets=targets,
                                filenames=np.asarray(filenames[:self.size], dtype=str),
                                metadata=np.asarray(json.dumps(metadata)))
        temporary.replace(path)


def basis_values(x, low, high, scale):
    x = np.asarray(x, dtype=np.float64)
    return (np.maximum(x[..., None] - low, 0) * np.maximum(high - x[..., None], 0) * scale) ** 2


def weighted_curve(x, low, high, coefficients, scale):
    return basis_values(x, low, high, scale) @ coefficients


def region_change(before, after, atol=1e-6, rtol=1e-3):
    threshold = float(atol + rtol * max(np.max(np.abs(before)), np.max(np.abs(after))))
    delta = np.asarray(after) - np.asarray(before)
    return delta, np.abs(delta) > threshold, threshold


def region_overlap(a_delta, a_mask, b_delta, b_mask):
    union = np.logical_or(a_mask, b_mask).sum()
    denominator = np.linalg.norm(a_delta) * np.linalg.norm(b_delta)
    return {"jaccard_regiao_alterada": float(np.logical_and(a_mask, b_mask).sum() / union)
            if union else None,
            "cosseno_delta_assinado": float(np.clip(np.dot(a_delta, b_delta) / denominator, -1, 1))
            if denominator > 1e-24 else None}


def selected_indices(size, count):
    return sorted(set(np.linspace(0, size - 1, min(size, count), dtype=int).tolist()))


def select_edges(head, inputs_per_block=2, outputs_per_layer=1):
    """Fixed positional selection, chosen ONCE; never reselect the most-changed unit per stage."""
    edges = []
    for layer_index, layer in enumerate(head.rk_layers):
        blocks = ([(name, (identity - 1) * head.input_size, head.input_size)
                   for name, identity in DATASET_IDS.items()] if layer_index == 0 and head.block_expansion else
                  [("compartilhado", 0, layer.input_size)])
        for block, start, size in blocks:
            for feature in selected_indices(size, inputs_per_block):
                for output in selected_indices(layer.output_size, outputs_per_layer):
                    edges.append((layer_index, block, start + feature, output))
    return edges


def load_checkpoint(path):
    # Only user-owned, locally produced training checkpoints are accepted by the caller.
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("head_state", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("model.mlp.") for key in state):
        state = {key[len("model.mlp."):]: value for key, value in state.items()
                 if key.startswith("model.mlp.")}
    return state


def build_head(config, state):
    options = config["model"]
    common = {"use_layer_norm": bool(options.get("use_layer_norm", True)),
              "layer_norm_eps": float(options.get("layer_norm_eps", 1e-5)),
              "layer_norm_affine": bool(options.get("layer_norm_affine", False)),
              "dataset_block_expansion": options.get("dataset_block_expansion")}
    if options["model_type"] == "dynamic_kan":
        from models.torch_relu_kan import ReLUKAN
        layers = sorted(int(key.split(".")[1]) for key in state if
                        key.startswith("rk_layers.") and key.endswith("equal_size_conv.weight"))
        width = [int(options["mlp_input_dim"])] + [
            int(state[f"rk_layers.{index}.equal_size_conv.weight"].shape[0]) for index in layers]
        head = ReLUKAN(width, int(options["relukan_grid"]), int(options["relukan_k"]), **common)
    else:
        from models.base_models import MLPBase
        head = MLPBase(input_size=int(options["mlp_input_dim"]),
                       hidden_dim=int(options["mlp_hidden_dim"]),
                       num_layers=int(options["mlp_num_layers"]),
                       output_size=int(options.get("mlp_output_size", 1)),
                       dropout=float(options.get("mlp_dropout", 0)),
                       activation_func=options.get("mlp_activation_func", "relu"), **common)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def run_head(head, embeddings, dataset, capture=False):
    inputs = {}
    handles = []
    if capture:
        modules = (list(enumerate(head.rk_layers)) if hasattr(head, "rk_layers") else
                   [(index, module) for index, module in enumerate(head.layers)
                    if isinstance(module, torch.nn.Linear)])
        for index, module in modules:
            def hook(module, args, index=index):
                inputs[index] = args[0].detach().float().cpu().numpy().copy()
            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            values = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
            ids = torch.full((len(values),), DATASET_IDS[dataset], dtype=torch.long)
            predicted = head(values, dataset_ids=ids).numpy().reshape(-1)
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite predictions from a diagnostic head checkpoint")
        return predicted, inputs
    finally:
        for handle in handles:
            handle.remove()


def tensor_drift(before, after):
    before, after = before.detach().float(), after.detach().float()
    if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError("Incompatible/nonfinite head parameters")
    delta = after - before
    norm = float(torch.linalg.vector_norm(before))
    return {"delta_l2": float(torch.linalg.vector_norm(delta)),
            "delta_relativo": float(torch.linalg.vector_norm(delta)) / max(norm, 1e-12),
            "delta_rms_por_parametro": float(delta.square().mean().sqrt()),
            "parametros": before.numel()}


def parameter_changes(previous, current, transition, trained_on):
    rows = []
    baseline = dict(previous.named_parameters())
    for name, value in current.named_parameters():
        groups = [("compartilhado", baseline[name], value)]
        first_weight = name in {"rk_layers.0.equal_size_conv.weight", "layers.0.weight"}
        first_phase = name in {"rk_layers.0.phase_low", "rk_layers.0.phase_height"}
        if current.block_expansion and (first_weight or first_phase):
            groups = []
            for dataset, identity in DATASET_IDS.items():
                selection = slice((identity - 1) * current.input_size, identity * current.input_size)
                before = baseline[name][..., selection] if first_weight else baseline[name][selection]
                after = value[..., selection] if first_weight else value[selection]
                groups.append((dataset, before, after))
        for region, before, after in groups:
            rows.append({"transicao": transition, "treinado_em": trained_on, "parametro": name,
                         "regiao": region,
                         "bloco_ativo_no_ft": region == trained_on if region != "compartilhado" else None,
                         **tensor_drift(before, after)})
    return rows


def support_rows(head, captured, stage, dataset):
    """Use learned endpoints, separating routed live coordinates from deliberate padding."""
    rows = []
    if not hasattr(head, "rk_layers"):
        return rows
    for index, layer in enumerate(head.rk_layers):
        values = captured[index]
        low = layer.phase_low.detach().numpy()
        high = layer.phase_height.detach().numpy()
        selections = [("compartilhado", np.arange(values.shape[1]))]
        if index == 0 and head.block_expansion:
            start = (DATASET_IDS[dataset] - 1) * head.input_size
            active = np.arange(start, start + head.input_size)
            padding = np.setdiff1d(np.arange(values.shape[1]), active)
            selections = [("bloco_ativo", active), ("padding", padding)]
        for region, selection in selections:
            activations = ((values[:, selection, None] > low[selection]) &
                           (values[:, selection, None] < high[selection]))
            row = {"etapa": stage, "dataset": dataset, "camada": index, "regiao": region,
                   "fracao_sem_base_ativa": float((~activations.any(-1)).mean()),
                   "bases_ativas_media": float(activations.sum(-1).mean()),
                   "fracao_intervalos_invalidos": float((high[selection] <= low[selection]).mean())}
            if region == "padding":
                basis = basis_values(np.full(len(selection), head.block_expansion.padding_value),
                                     low[selection], high[selection], layer.r)
                # Sum bases for each actual edge; include cancellations but also expose raw basis leakage.
                coefficients = layer.equal_size_conv.weight.detach().numpy()[:, 0][:, :, selection]
                contribution = np.einsum("ib,obi->oi", basis, coefficients)
                row.update(padding_base_maximo=float(basis.max()),
                           padding_contribuicao_abs_maxima=float(np.abs(contribution).max()))
            rows.append(row)
    return rows


def discover_experiments(root):
    root = Path(root).resolve()
    if (root / "round.json").is_file():
        manifest = read_json(root / "round.json")
        expected = []
        for task in manifest["tasks"]:
            output = Path(task["output"])
            expected.append((task["label"], sorted(p for p in output.glob("*")
                                                   if p.is_dir() and (p / "manifest.json").exists())))
        return [path for _, paths in expected for path in paths], [label for label, paths in expected if not paths]
    return sorted(p.parent for p in root.rglob("progress.json")
                  if (p.parent / "manifest.json").exists()), []


def sequence_records(directory, allow_partial=False):
    directory = Path(directory).resolve()
    progress = read_json(directory / "progress.json") if (directory / "progress.json").exists() else {}
    stages = []
    previous = None
    for index, dataset in enumerate(SEQUENCE, 1):
        stage_dir = directory / f"{index:02d}-{dataset}"
        trained_path = stage_dir / "trained.json"
        if not trained_path.exists():
            break
        trained = read_json(trained_path)
        if trained["initialized_from"] != previous:
            raise ValueError(f"Checkpoint lineage mismatch at {stage_dir}")
        checkpoint = Path(trained["best_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        stages.append({"index": index, "dataset": dataset, "label": f"{index:02d}-{dataset}",
                       "checkpoint": str(checkpoint), "trained": trained, "directory": stage_dir})
        previous = str(checkpoint)
    complete = progress.get("status") == "finished" and len(stages) == len(SEQUENCE)
    if not complete and not allow_partial:
        return None
    if not stages:
        return None
    return {"directory": directory, "stages": stages, "complete": complete,
            "progress": progress}


def load_config(path):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_probes(directory, count):
    probes = {}
    for dataset in DATASET_IDS:
        path = Path(directory) / "comparison" / "probes" / f"{dataset}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            values = data["embeddings"][:count].copy()
            target = data["targets"][:count].copy()
            filenames = data["filenames"][:count].tolist()
        if metadata["dataset"] != dataset or metadata.get("split") != "test":
            raise ValueError(f"Unexpected diagnostic probe identity: {path}")
        if (not np.isfinite(values).all() or not np.isfinite(target).all() or
                len(values) != len(target) or len(values) != len(filenames)):
            raise ValueError(f"Invalid probes: {path}")
        probes[dataset] = {"values": values, "targets": target, "filenames": filenames,
                           "metadata": metadata, "path": str(path)}
    return probes


def histogram(values, x):
    boundaries = np.r_[x[0] - (x[1] - x[0]) / 2, (x[:-1] + x[1:]) / 2,
                       x[-1] + (x[-1] - x[-2]) / 2]
    counts = np.histogram(values, bins=boundaries)[0].astype(float)
    return counts / max(float(counts.sum()), 1.)


def experiment_name(config):
    model = config["model"]
    backbone = model["model_name"].split("/")[-1].lower()
    return (f"{backbone}-kan-k{model['relukan_k']}-g{model['relukan_grid']}"
            if model["model_type"] == "dynamic_kan" else f"{backbone}-mlp")


def test_metrics(record, name):
    rows = []
    baseline = {}
    for stage in record["stages"]:
        for dataset in DATASET_IDS:
            path = stage["directory"] / "evaluations" / dataset / "metrics.json"
            if not path.exists():
                continue
            measurement = read_json(path)
            if str(measurement["checkpoint"]) != stage["checkpoint"]:
                raise ValueError(f"Test metrics refer to another checkpoint: {path}")
            baseline.setdefault(dataset, measurement)
            old = baseline[dataset]
            if measurement["test_metadata_sha256"] != old["test_metadata_sha256"]:
                raise ValueError(f"Test split changed across checkpoints: {path}")
            srcc = measurement["srcc"]
            forgetting = old["srcc"] - srcc if old["srcc"] is not None and srcc is not None else None
            rows.append({"experimento": name, "etapa": stage["label"], "dataset": dataset,
                         "srcc": srcc, "mse": measurement["mse"],
                         "esquecimento_srcc": forgetting,
                         "esquecimento_mse": measurement["mse"] - old["mse"],
                         "exemplos": measurement["num_examples"],
                         "referencia": "primeiro best treinado neste dataset; teste completo",
                         "metadata_sha256": measurement["test_metadata_sha256"]})
    return rows


def analyze_experiment(record, options):
    directory = record["directory"]
    stages = record["stages"]
    config = load_config(stages[0]["directory"] / "config.yaml")
    name = experiment_name(config)
    probes = load_probes(directory, int(options["probe_samples"]))
    for dataset, probe in probes.items():
        if (probe["metadata"].get("backbone") != config["model"]["model_name"] or
                probe["values"].ndim != 2 or probe["values"].shape[1] != int(config["model"]["mlp_input_dim"])):
            raise ValueError(f"Probe backbone/dimension differs from the checkpoint: {dataset}")
        for stage in stages:
            metrics_path = stage["directory"] / "evaluations" / dataset / "metrics.json"
            if metrics_path.exists() and (probe["metadata"].get("test_metadata_sha256") !=
                                          read_json(metrics_path)["test_metadata_sha256"]):
                raise ValueError(f"Probe/test metadata differs at {metrics_path}")
    x = np.linspace(0., 1., int(options["grid_points"]))
    response_points = int(options["response_points"])
    table_names = ("parameters", "coverage", "edges", "basis", "regions", "overlap", "functional", "responses",
                   "probe_scores", "activation_health")
    tables = {key: [] for key in table_names}
    edge_curves, basis_curves, delta_maps, usage_maps, response_curves = {}, {}, {}, {}, {}
    previous_head = previous_curves = previous_inputs = previous_predictions = None
    previous_label = None
    prediction_baselines = {}
    edge_selection = None
    parameter_count = None
    checkpoints = []
    stage_list = list(stages)
    initial = stages[0]["trained"].get("initial_head_checkpoint")
    initial_available = bool(initial and Path(initial).is_file())
    if initial_available:
        stage_list.insert(0, {"label": "00-inicializacao", "dataset": None, "checkpoint": initial, "index": 0})

    for stage in stage_list:
        state = load_checkpoint(stage["checkpoint"])
        head = build_head(config, state)
        del state
        parameter_count = sum(parameter.numel() for parameter in head.parameters())
        checkpoint_path = Path(stage["checkpoint"])
        checkpoints.append({"etapa": stage["label"], "arquivo": str(checkpoint_path),
                            "bytes": checkpoint_path.stat().st_size,
                            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
                            "training_run": stage.get("trained", {}).get("wandb"),
                            "best_validation_score": stage.get("trained", {}).get("score"),
                            "initialized_from": stage.get("trained", {}).get("initialized_from")})
        if edge_selection is None:
            edge_selection = (select_edges(head, int(options["inputs_per_block"]), int(options["outputs_per_layer"]))
                              if hasattr(head, "rk_layers") else [])
        label = stage["label"]
        transition = f"{previous_label} → {label}"
        if previous_head is not None:
            tables["parameters"].extend(parameter_changes(previous_head, head, transition, stage["dataset"]))

        inputs, predictions = {}, {}
        for dataset, probe in probes.items():
            predicted, captured = run_head(head, probe["values"], dataset, capture=True)
            inputs[dataset], predictions[dataset] = captured, predicted
            tables["coverage"].extend(support_rows(head, captured, label, dataset))
            from scipy.stats import spearmanr
            constant = float(np.ptp(predicted)) <= 1e-6
            rank = float(spearmanr(predicted, probe["targets"]).statistic) if not constant else None
            tables["probe_scores"].append({
                "etapa": label, "dataset": dataset, "probes": len(predicted),
                "srcc_subconjunto": rank if rank is not None and math.isfinite(rank) else None,
                "mse_subconjunto": float(np.mean((predicted - probe["targets"]) ** 2)),
                "predicao_desvio_padrao": float(predicted.std()), "predicoes_constantes": constant})
            for layer_index, activations in captured.items():
                regions = [("compartilhado", activations)]
                if layer_index == 0 and head.block_expansion:
                    start = (DATASET_IDS[dataset] - 1) * head.input_size
                    regions = [("bloco_ativo", activations[:, start:start + head.input_size])]
                for region, values in regions:
                    tables["activation_health"].append({
                        "etapa": label, "dataset": dataset, "camada": layer_index,
                        "regiao": region, "medida": "entrada real da camada aprendida",
                        "media": float(values.mean()), "desvio_padrao": float(values.std()),
                        "fracao_quase_zero": float((np.abs(values) < 1e-8).mean()),
                        "fracao_coordenadas_constantes": float((values.std(axis=0) < 1e-8).mean())})
            if stage["dataset"] == dataset and dataset not in prediction_baselines:
                prediction_baselines[dataset] = predicted.copy()
            if previous_predictions is not None:
                difference = predicted - previous_predictions[dataset]
                baseline = prediction_baselines.get(dataset)
                tables["functional"].append({
                    "etapa": label, "transicao": transition, "dataset": dataset,
                    "treinado_em": stage["dataset"], "probes": len(predicted),
                    "rmse_mudanca_predicao": float(np.sqrt(np.mean(difference ** 2))),
                    "delta_mos_medio": float(difference.mean()),
                    "rmse_desde_aprendizado": float(np.sqrt(np.mean((predicted - baseline) ** 2)))
                    if baseline is not None else None,
                    "mse_subconjunto": float(np.mean((predicted - probe["targets"]) ** 2)),
                    "delta_mse_subconjunto": float(np.mean((predicted - probe["targets"]) ** 2) -
                                                   np.mean((previous_predictions[dataset] - probe["targets"]) ** 2)),
                    "dataset_ja_aprendido": dataset in prediction_baselines,
                    "nota": "mudanca funcional em probes fixos; nao substitui o teste completo"})
            anchor = np.median(probe["values"], axis=0)
            for feature in selected_indices(head.input_size, int(options["conditional_dims"])):
                lo, hi = np.quantile(probe["values"][:, feature], [.01, .99])
                if hi - lo < 1e-8:
                    lo, hi = float(lo) - 1e-3, float(hi) + 1e-3
                axis = np.linspace(lo, hi, response_points)
                values = np.repeat(anchor[None, :], response_points, axis=0)
                values[:, feature] = axis
                response, _ = run_head(head, values, dataset)
                key = f"{dataset} | dim {feature}"
                response_curves.setdefault(key, []).append((label, axis.tolist(), response.tolist()))
                for value, prediction in zip(axis, response):
                    tables["responses"].append({"etapa": label, "dataset": dataset, "dimensao": feature,
                                                 "embedding_bruto": float(value), "mos": float(prediction)})

        curves = {}
        for layer_index, block, feature, output in edge_selection:
            layer = head.rk_layers[layer_index]
            low = layer.phase_low[feature].detach().numpy()
            high = layer.phase_height[feature].detach().numpy()
            weights = layer.equal_size_conv.weight[output, 0, :, feature].detach().numpy()
            key = f"camada {layer_index} | {block} | dim {feature} → {output}"
            basis = basis_values(x, low, high, layer.r)
            curve = basis @ weights
            curves[key] = curve
            edge_curves.setdefault(key, []).append((label, x.tolist(), curve.tolist()))
            basis_curves[f"{key} | {label}"] = [(f"base {index}", x.tolist(), component.tolist())
                                                for index, component in enumerate((basis * weights).T)]
            for index, (a, b, coefficient) in enumerate(zip(low, high, weights)):
                tables["basis"].append({"etapa": label, "camada": layer_index, "bloco": block,
                                        "dimensao": feature, "saida": output, "base": index,
                                        "limite_inferior": float(a), "limite_superior": float(b),
                                        "largura": float(b - a), "coeficiente": float(coefficient)})
            for value, contribution in zip(x, curve):
                tables["edges"].append({"etapa": label, "conexao": key, "x": float(value),
                                        "contribuicao_sem_vies": float(contribution)})
            for dataset, captured in inputs.items():
                density = histogram(captured[layer_index][:, feature], x)
                usage_maps.setdefault(f"{key} | {label}", []).append((dataset, density.tolist()))
            if previous_curves is not None:
                delta, mask, threshold = region_change(previous_curves[key], curve,
                                                      float(options["change_atol"]), float(options["change_rtol"]))
                delta_maps.setdefault(key, []).append((transition, delta.tolist(), mask.tolist()))
                for dataset, captured in previous_inputs.items():
                    values = captured[layer_index][:, feature]
                    in_domain = (values >= 0.) & (values <= 1.)
                    density = histogram(values, x)
                    active_inputs = previous_inputs.get(stage["dataset"])
                    current_density = (histogram(active_inputs[layer_index][:, feature], x)
                                       if active_inputs is not None else None)
                    tables["regions"].append({
                        "transicao": transition, "treinado_em": stage["dataset"], "dataset_observado": dataset,
                        "conexao": key, "camada": layer_index, "bloco": block,
                        "delta_abs_medio_em_uso": float(np.mean(np.abs(np.interp(values[in_domain], x, delta))))
                        if in_domain.any() else None,
                        "fracao_regiao_alterada_em_uso": float(np.sum(density * mask)) if in_domain.any() else None,
                        "sobreposicao_ocupacao_com_dataset_treinado":
                        float(np.minimum(density, current_density).sum()) if current_density is not None else None,
                        "fracao_valores_no_dominio": float(in_domain.mean()),
                        "limiar_mudanca": threshold,
                        "referencia_ocupacao": "ativacoes do checkpoint ANTES do fine-tuning"})
        previous_head, previous_curves = head, curves
        previous_inputs, previous_predictions, previous_label = inputs, predictions, label
        gc.collect()

    for key, changes in delta_maps.items():
        for a, b in itertools.combinations(changes, 2):
            tables["overlap"].append({"conexao": key, "transicao_a": a[0], "transicao_b": b[0],
                                      **region_overlap(np.asarray(a[1]), a[2], np.asarray(b[1]), b[2])})
    for rows in tables.values():
        for row in rows:
            row["experimento"] = name
    model = config["model"]
    info = {"name": name, "directory": str(directory), "complete": record["complete"],
            "backbone": model["model_name"], "head": "kan" if model["model_type"] == "dynamic_kan" else "mlp",
            "k": model.get("relukan_k") if model["model_type"] == "dynamic_kan" else None,
            "g": model.get("relukan_grid") if model["model_type"] == "dynamic_kan" else None,
            "initial_basis_width": (int(model["relukan_k"]) + 1) / int(model["relukan_grid"])
            if model["model_type"] == "dynamic_kan" else None,
            "initial_grid_step": 1 / int(model["relukan_grid"]) if model["model_type"] == "dynamic_kan" else None,
            "parameters": parameter_count, "initial_checkpoint_available": initial_available,
            "effective_dropout": [float(module.p) for module in head.modules()
                                  if isinstance(module, torch.nn.Dropout)],
            "widths": (list(head.width) if hasattr(head, "width") else
                       [(module.in_features, module.out_features) for module in head.layers
                        if isinstance(module, torch.nn.Linear)]),
            "checkpoints": checkpoints, "probe_datasets": sorted(probes), "config": config,
            "selection": [list(edge) for edge in edge_selection],
            "warnings": [f"Sem probes fixos para {dataset}: ocupacao/resposta indisponivel"
                         for dataset in DATASET_IDS if dataset not in probes] +
                        ([] if initial_available else ["Inicializacao ausente; curvas comecam no primeiro best salvo"])}
    if any(row.get("padding_base_maximo", 0) > 1e-8 for row in tables["coverage"]):
        info["warnings"].append("Bases aprendidas passaram a responder ao padding; consultar coverage.csv")
    if any(row["fracao_intervalos_invalidos"] > 0 for row in tables["coverage"]):
        info["warnings"].append("Ha intervalos aprendidos com limite superior <= inferior; consultar coverage.csv")
    tables["tests"] = test_metrics(record, name)
    return {"info": info, "tables": tables, "edge_curves": edge_curves, "basis_curves": basis_curves,
            "delta_maps": delta_maps, "usage_maps": usage_maps, "response_curves": response_curves,
            "probes": probes}


def styled(figure, title, x_title, y_title):
    figure.update_layout(template="plotly_white", separators=",.", title={"text": title, "x": .02},
                         font={"family": "Arial, sans-serif", "color": "#64748B"},
                         paper_bgcolor="white", plot_bgcolor="white", height=560,
                         margin={"t": 130, "r": 30, "b": 110, "l": 85},
                         legend={"orientation": "h", "y": -.23},
                         xaxis_title=x_title, yaxis_title=y_title)
    figure.update_xaxes(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1")
    figure.update_yaxes(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1")
    return figure


def selector_plot(groups, title, x_title, y_title, heatmap=False):
    import plotly.graph_objects as go
    figure = go.Figure()
    buttons, offset = [], 0
    for group_index, (key, entries) in enumerate(groups.items()):
        start = len(figure.data)
        if heatmap:
            x, labels, z = entries
            figure.add_trace(go.Heatmap(x=x, y=labels, z=z, colorscale="Viridis",
                                       visible=group_index == 0, colorbar={"title": y_title}, name=key))
        else:
            for series_index, (label, x, values) in enumerate(entries):
                figure.add_trace(go.Scatter(x=x, y=values, mode="lines", name=label,
                                           line={"color": COLORS[series_index % len(COLORS)]},
                                           visible=group_index == 0))
        buttons.append((key, start, len(figure.data)))
    if not buttons:
        return None
    menu = [{"label": key, "method": "update", "args": [
        {"visible": [start <= index < end for index in range(len(figure.data))]},
        {"title.text": f"{title}<br><sup>{key}</sup>"}]} for key, start, end in buttons]
    figure.update_layout(updatemenus=[{"buttons": menu, "x": 0., "y": 1.15,
                                       "direction": "down", "xanchor": "left", "yanchor": "top"}])
    return styled(figure, f"{title}<br><sup>{buttons[0][0]}</sup>", x_title, y_title)


def experiment_plots(result):
    figures = {}
    name = result["info"]["name"]
    if result["edge_curves"]:
        figures["curvas_mesmas_conexoes"] = selector_plot(
            result["edge_curves"], f"{name}: mesmas curvas em cada best checkpoint",
            "Entrada da camada KAN (apos sigmoid)", "Contribuicao da conexao, sem vies")
        # Basis decomposition is deliberately limited to ONE fixed edge per layer in the panel;
        # all selected connections/bases remain available in the CSV artifact.
        first_keys = {}
        for edge, series in result["edge_curves"].items():
            layer = edge.split(" | ")[0]
            first_keys.setdefault(layer, edge)
        bases = {key: entries for key, entries in result["basis_curves"].items()
                 if any(key.startswith(edge + " | ") for edge in first_keys.values())}
        figures["bases_individuais"] = selector_plot(bases, f"{name}: componentes das bases aprendidas",
                                                     "Entrada da camada KAN", "Base × coeficiente")
        changes = {key: (result["edge_curves"][key][0][1], [item[0] for item in entries],
                         np.abs(np.asarray([item[1] for item in entries])).tolist())
                   for key, entries in result["delta_maps"].items()}
        figures["onde_as_curvas_mudaram"] = selector_plot(changes, f"{name}: regiao alterada por fine-tuning",
                                                         "Entrada da camada KAN", "|Delta contribuicao|", True)
        overlaps = {}
        for key, changes_for_edge in result["delta_maps"].items():
            labels = [entry[0] for entry in changes_for_edge]
            z = [[region_overlap(np.asarray(a[1]), a[2], np.asarray(b[1]), b[2])["jaccard_regiao_alterada"]
                  for b in changes_for_edge] for a in changes_for_edge]
            overlaps[key] = (labels, labels, z)
        figures["mesmas_regioes_entre_datasets"] = selector_plot(
            overlaps, f"{name}: sobreposicao das regioes alteradas (Jaccard)", "Transicao", "Jaccard", True)
        usage = {key: (result["edge_curves"][key.rsplit(" | ", 1)[0]][0][1],
                        [entry[0] for entry in entries], [entry[1] for entry in entries])
                 for key, entries in result["usage_maps"].items()}
        figures["ocupacao_por_dataset"] = selector_plot(usage, f"{name}: onde os datasets usam cada conexao",
                                                       "Entrada (padding fora de 0–1 fica na tabela de cobertura)",
                                                       "Massa relativa", True)
    if result["response_curves"]:
        figures["resposta_mos_condicional"] = selector_plot(
            result["response_curves"], f"{name}: resposta MOS condicional da head completa",
            "Uma coordenada do embedding bruto; demais fixadas na mediana", "MOS predito")
    return {key: value for key, value in figures.items() if value is not None}


def config_value(config, dotted):
    current = config
    for key in dotted.split("."):
        current = current.get(key) if isinstance(current, dict) else None
    return current


def paired_rows(results):
    rows = []
    fields = ("seed", "model.model_name", "model.mlp_input_dim", "model.last_layer", "model.pooling_strategy",
              "model.freeze_backbone", "model.layer_norm_affine", "model.dataset_block_expansion.dataset_to_id",
              "model.dataset_block_expansion.output_multiplier", "optimizer.name", "optimizer.params",
              "model.mlp_hidden_dim", "model.mlp_num_layers", "model.mlp_dropout",
              "train.batch_size", "trainer.accumulate_grad_batches", "trainer.max_epochs", "early_stopping")
    for kan in (item for item in results if item["info"]["head"] == "kan"):
        candidates = [item for item in results if item["info"]["head"] == "mlp" and
                      item["info"]["backbone"] == kan["info"]["backbone"] and
                      item["info"]["config"].get("seed") == kan["info"]["config"].get("seed")]
        if len(candidates) != 1:
            rows.append({"kan": kan["info"]["name"], "mlp": None,
                         "protocolo_compartilhado_pareado": False, "diferencas": "MLP pareada ausente ou ambigua"})
            continue
        mlp = candidates[0]
        differences = [field for field in fields if
                       config_value(kan["info"]["config"], field) != config_value(mlp["info"]["config"], field)]
        for dataset in DATASET_IDS:
            a, b = kan["probes"].get(dataset), mlp["probes"].get(dataset)
            if a is None or b is None:
                differences.append(f"probes.{dataset}.ausentes")
            elif (a["filenames"] != b["filenames"] or a["values"].shape != b["values"].shape or
                  not np.allclose(a["values"], b["values"], atol=1e-6, rtol=1e-5) or
                  not np.array_equal(a["targets"], b["targets"])):
                differences.append(f"probes.{dataset}.diferentes")
        mlp_tests = {(row["etapa"], row["dataset"]): row for row in mlp["tables"]["tests"]}
        for row in kan["tables"]["tests"]:
            partner = mlp_tests.get((row["etapa"], row["dataset"]))
            if partner is None:
                continue
            split_match = row["metadata_sha256"] == partner["metadata_sha256"]
            rows.append({"kan": kan["info"]["name"], "mlp": mlp["info"]["name"],
                         "backbone": kan["info"]["backbone"], "k": kan["info"]["k"], "g": kan["info"]["g"],
                         "etapa": row["etapa"], "dataset": row["dataset"],
                         "protocolo_compartilhado_pareado": not differences and split_match,
                         "diferencas": ", ".join(differences + ([] if split_match else ["test_split"])),
                         "parametros_kan": kan["info"]["parameters"], "parametros_mlp": mlp["info"]["parameters"],
                         **{f"{key}_kan": row[key] for key in ("mse", "srcc", "esquecimento_mse", "esquecimento_srcc")},
                         **{f"{key}_mlp": partner[key] for key in ("mse", "srcc", "esquecimento_mse", "esquecimento_srcc")},
                         "transformacoes": "KAN: LN+sigmoid, padding abaixo do suporte inicial; MLP: LN, padding zero",
                         "dropout_efetivo_kan": str(kan["info"]["effective_dropout"]),
                         "dropout_efetivo_mlp": str(mlp["info"]["effective_dropout"]),
                         "controle_capacidade": False})
    return rows


def metric_plot(results, table_name, metrics, title):
    groups = {}
    for dataset in DATASET_IDS:
        for metric in metrics:
            series = []
            for result in results:
                rows = [row for row in result["tables"][table_name]
                        if row.get("dataset") == dataset and row.get(metric) is not None]
                if rows:
                    series.append((result["info"]["name"], [row["etapa"] for row in rows],
                                   [row[metric] for row in rows]))
            if series:
                groups[f"{dataset} | {metric}"] = series
    return selector_plot(groups, title, "Best checkpoint apos cada etapa", "Valor da metrica")


def coverage_plot(results):
    groups = {}
    for dataset in DATASET_IDS:
        layers = sorted({row["camada"] for result in results for row in result["tables"]["coverage"]})
        for layer in layers:
            series = []
            for result in results:
                rows = [row for row in result["tables"]["coverage"] if row["dataset"] == dataset and
                        row["camada"] == layer and row["regiao"] != "padding"]
                if rows:
                    series.append((result["info"]["name"], [row["etapa"] for row in rows],
                                   [row["fracao_sem_base_ativa"] for row in rows]))
            if series:
                groups[f"{dataset} | camada {layer}"] = series
    return selector_plot(groups, "Cobertura APRENDIDA: entradas reais sem nenhuma base ativa",
                         "Best checkpoint", "Fracao sem base ativa (padding excluido)")


def parameter_plot(results):
    groups = {}
    for result in results:
        rows = result["tables"]["parameters"]
        transitions = list(dict.fromkeys(row["transicao"] for row in rows))
        regions = list(dict.fromkeys(row["regiao"] for row in rows))
        if not transitions:
            continue
        matrix = []
        for region in regions:
            values = []
            for transition in transitions:
                selected = [row for row in rows if row["regiao"] == region and row["transicao"] == transition]
                count = sum(row["parametros"] for row in selected)
                values.append(math.sqrt(sum(row["delta_rms_por_parametro"] ** 2 * row["parametros"]
                                            for row in selected) / count) if count else None)
            matrix.append(values)
        groups[result["info"]["name"]] = (transitions, regions, matrix)
    return selector_plot(groups, "Alteracao dos parametros: blocos por dataset e partes compartilhadas",
                         "Transicao", "RMS delta por parametro", True)


def conditional_pair_plot(results):
    groups = {}
    for kan in (item for item in results if item["info"]["head"] == "kan"):
        partners = [item for item in results if item["info"]["head"] == "mlp" and
                    item["info"]["backbone"] == kan["info"]["backbone"] and
                    item["info"]["config"].get("seed") == kan["info"]["config"].get("seed")]
        if len(partners) != 1:
            continue
        mlp = partners[0]
        for key, curves in kan["response_curves"].items():
            dataset = key.split(" | ")[0]
            a, b = kan["probes"][dataset], mlp["probes"].get(dataset)
            if b is None or a["filenames"] != b["filenames"] or a["values"].shape != b["values"].shape:
                continue
            if not np.allclose(a["values"], b["values"], atol=1e-6, rtol=1e-5):
                continue
            partner_curves = {entry[0]: entry for entry in mlp["response_curves"].get(key, [])}
            for stage, x, values in curves:
                if stage in partner_curves:
                    partner = partner_curves[stage]
                    groups[f"{kan['info']['name']} | {key} | {stage}"] = [
                        ("KAN", x, values), ("MLP", partner[1], partner[2])]
    return selector_plot(groups, "KAN × MLP: mesma ancora e coordenada bruta (ver pareamento de configs)",
                         "Coordenada do embedding bruto", "MOS predito pela head completa")


CAVEATS = [
    "Curvas KAN sao somas ponderadas das bases ReLU locais; nao sao B-splines polinomiais tradicionais.",
    "Conexoes sao escolhidas uma unica vez por posicao; graficos amostram conexoes, tabelas de pesos cobrem todos os parametros.",
    "Mudanca de pesos/regioes ou ocupacao sobreposta nao prova interferencia prejudicial; confrontar com predicoes e teste completo.",
    "AdamW pode alterar pesos de blocos inativos por weight decay mesmo com gradiente zero.",
    "Fora do bloco ativo, cobertura zero e intencional. Padding e medido separadamente usando os limites APRENDIDOS.",
    "Limites das bases treinaveis podem sair do suporte inicial [-K/G,1+K/G].",
    "Probes sao os primeiros N exemplos fixos do teste, reutilizados sem novo ajuste; servem ao diagnostico, nao a escolha de modelo.",
    "Respostas condicionais variam uma coordenada bruta com as demais na mediana; nao representam efeito causal nem uma spline isolada.",
    "As transformacoes KAN/MLP diferem (sigmoid/padding). G tambem altera a capacidade. Diferencas de protocolo ficam na tabela de pareamento.",
    "Se todos os valores G testados forem maiores que K, falta controle G<=K para testar especificamente esse limiar.",
    "Uma unica semente nao estabelece superioridade estatistica; a analise e descritiva.",
]


def generate_report(experiments, output_dir, options=None, wandb_options=None, allow_partial=False):
    options = {**DEFAULTS, **(options or {})}
    for field in ("probe_samples", "grid_points", "inputs_per_block", "outputs_per_layer", "response_points", "conditional_dims"):
        if int(options[field]) < (2 if field in {"grid_points", "response_points"} else 1):
            raise ValueError(f"Invalid comparison option {field}={options[field]}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, pending = [], []
    for directory in experiments:
        record = sequence_records(directory, allow_partial=allow_partial)
        if record is None:
            pending.append(str(directory))
        else:
            records.append(record)
    if (pending and not allow_partial) or not records:
        status = {"status": "waiting_for_complete_sequences", "pending": pending,
                  "expected": [str(path) for path in experiments], "wandb_run_created": False}
        write_json(output_dir / "status.json", status)
        return status
    lock = output_dir / ".analysis.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {"status": "analysis_already_running"}
    os.close(descriptor)
    try:
        # This process reconstructs heads on CPU; it never loads a backbone or uses CUDA.
        torch.set_num_threads(4)
        code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps({"version": code_sha256, "options": options, "partial": allow_partial,
            "sources": [[(stage["checkpoint"], Path(stage["checkpoint"]).stat().st_mtime_ns,
                           Path(stage["checkpoint"]).stat().st_size,
                           (stage["directory"] / "config.yaml").stat().st_mtime_ns,
                           [(str(path), path.stat().st_mtime_ns) for path in sorted(
                               (stage["directory"] / "evaluations").glob("*/metrics.json"))]) for stage in record["stages"]]
                        for record in records],
            "initials": [(str(path), path.stat().st_mtime_ns, path.stat().st_size) if path and path.is_file() else None
                         for path in [Path(record["stages"][0]["trained"]["initial_head_checkpoint"])
                                      if record["stages"][0]["trained"].get("initial_head_checkpoint") else None
                                      for record in records]],
            "probes": [(str(path), path.stat().st_mtime_ns, path.stat().st_size) for record in records
                       for path in sorted((record["directory"] / "comparison/probes").glob("*.npz"))]},
            sort_keys=True).encode()).hexdigest()
        status_path = output_dir / "status.json"
        old = read_json(status_path) if status_path.exists() else {}
        publish = bool(wandb_options and wandb_options.get("enabled") and options.get("wandb"))
        tracking_matches = old.get("wandb_run") and (
            bool(old["wandb_run"].get("offline")) == bool((wandb_options or {}).get("offline")))
        if old.get("fingerprint") == fingerprint and old.get("status") == "finished" and (
                not publish or tracking_matches):
            return old
        write_json(status_path, {"status": "analyzing", "fingerprint": fingerprint})
        results = [analyze_experiment(record, options) for record in records]
        names = [result["info"]["name"] for result in results]
        if len(set(names)) != len(names):
            raise ValueError("Ambiguous duplicate backbone/head/K/G names in comparison; select one chain per configuration")
        tables = {key: [row for result in results for row in result["tables"][key]]
                  for key in results[0]["tables"]}
        tables["paired"] = paired_rows(results)
        figures = {}
        for result in results:
            for key, figure in experiment_plots(result).items():
                figures[f"{result['info']['name']}/{key}"] = figure
        figures["comparacao/teste_completo"] = metric_plot(
            results, "tests", ("mse", "srcc", "esquecimento_mse", "esquecimento_srcc"),
            "KAN e MLP: teste completo e esquecimento desde o primeiro aprendizado")
        figures["comparacao/mudanca_funcional_probes"] = metric_plot(
            results, "functional", ("rmse_mudanca_predicao", "rmse_desde_aprendizado", "delta_mse_subconjunto"),
            "KAN e MLP: resposta aos mesmos probes fixos")
        figures["comparacao/cobertura_g_k"] = coverage_plot(results)
        figures["comparacao/pesos_por_bloco"] = parameter_plot(results)
        figures["comparacao/resposta_kan_mlp"] = conditional_pair_plot(results)
        figures = {key: value for key, value in figures.items() if value is not None}
        for key, rows in tables.items():
            write_rows(output_dir / f"{key}.csv", rows)
        manifest = {"complete": all(record["complete"] for record in records) and not pending and not allow_partial,
                    "skipped_incomplete": pending, "fingerprint": fingerprint, "analysis_code_sha256": code_sha256,
                    "options": options, "experiments": [result["info"] for result in results],
                    "caveats": CAVEATS, "references": ["https://arxiv.org/abs/2406.02075"],
                    "source": "best checkpoint de cada etapa; inicializacao quando disponivel",
                    "g_greater_than_k": [result["info"]["g"] > result["info"]["k"] for result in results
                                          if result["info"]["head"] == "kan"],
                    "row_counts": {key: len(rows) for key, rows in tables.items()}}
        write_json(output_dir / "manifest.json", manifest)
        save_dashboard(output_dir, figures, manifest)
        save_overview(output_dir, results, manifest)
        tracking = publish_report(output_dir, figures, tables, manifest, wandb_options) if publish else None
        status = {"status": "finished", "complete": manifest["complete"], "fingerprint": fingerprint,
                  "experiments": names, "wandb_run": tracking, "dashboard": str(output_dir / "dashboard.html")}
        write_json(status_path, status)
        return status
    except Exception as exc:
        write_json(output_dir / "status.json", {"status": "analysis_failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        lock.unlink()


def save_dashboard(output_dir, figures, manifest):
    import html
    from plotly.offline import get_plotlyjs
    prefix = "Comparacao sequencial completa" if manifest["complete"] else "DIAGNOSTICO PARCIAL — sequencia incompleta"
    if manifest["options"].get("synthetic_validation"):
        prefix = "VALIDACAO SINTETICA — estes nao sao resultados dos experimentos"
    parts = ["<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Comparacao das heads</title>",
             "<style>body{font:15px Arial;color:#111827;background:white;margin:32px}summary{padding:14px;cursor:pointer}details{border:1px solid #E5E7EB;margin:14px 0}li{margin:7px 0}</style>",
             f"<script>{get_plotlyjs()}</script><h1>{prefix}</h1><p>Mesmas conexoes e probes em todos os best checkpoints. CPU; sem novo treino.</p><ul>",
             *[f"<li>{html.escape(text)}</li>" for text in manifest["caveats"]], "</ul>"]
    for index, (key, figure) in enumerate(figures.items()):
        parts.append(f"<details{' open' if index == 0 else ''}><summary>{html.escape(key)}</summary>")
        parts.append(figure.to_html(full_html=False, include_plotlyjs=False))
        parts.append("</details>")
    parts.append("<script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{if(d.open)d.querySelectorAll('.plotly-graph-div').forEach(g=>Plotly.Plots.resize(g));}));</script></html>")
    (output_dir / "dashboard.html").write_text("\n".join(parts), encoding="utf-8")


def save_overview(output_dir, results, manifest):
    """Static fallback using the installed Matplotlib runtime; Plotly panels remain interactive."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(15, 10), facecolor="white", constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    representative = next((result for result in results if result["edge_curves"]), None)
    if representative is not None:
        key = next(iter(representative["edge_curves"]))
        for index, (stage, x, y) in enumerate(representative["edge_curves"][key]):
            axes[0, 0].plot(x, y, label=stage, color=COLORS[index % len(COLORS)])
        axes[0, 0].set(title="Mesma conexao KAN ao longo da sequencia", xlabel="Entrada apos sigmoid",
                        ylabel="Contribuicao sem vies")
        axes[0, 0].legend(fontsize=8)
        changes = representative["delta_maps"].get(key, [])
        if changes:
            image = axes[0, 1].imshow(np.abs([entry[1] for entry in changes]), aspect="auto", cmap="viridis",
                                      extent=(0, 1, len(changes) - .5, -.5))
            axes[0, 1].set_yticks(range(len(changes)), [entry[0] for entry in changes], fontsize=7)
            figure.colorbar(image, ax=axes[0, 1], label="|Delta contribuicao|")
        axes[0, 1].set(title="Regioes alteradas da mesma curva", xlabel="Entrada apos sigmoid")
    else:
        for axis in axes[0]:
            axis.text(.5, .5, "Splines: KAN nao disponivel neste grupo", ha="center", transform=axis.transAxes)
    for index, result in enumerate(results):
        rows = [row for row in result["tables"]["tests"] if row["dataset"] == "bvcc"]
        if rows:
            axes[1, 0].plot([row["etapa"] for row in rows], [row["esquecimento_mse"] for row in rows],
                            marker="o", label=result["info"]["name"], color=COLORS[index % len(COLORS)])
        rows = [row for row in result["tables"]["functional"] if row["dataset"] == "bvcc"]
        if rows:
            axes[1, 1].plot([row["etapa"] for row in rows], [row["rmse_mudanca_predicao"] for row in rows],
                            marker="o", label=result["info"]["name"], color=COLORS[index % len(COLORS)])
    axes[1, 0].set(title="BVCC: esquecimento no teste completo", ylabel="MSE depois − MSE primeiro best")
    axes[1, 1].set(title="BVCC: mudanca de predicoes nos probes fixos", ylabel="RMSE de mudanca por transicao")
    if not any(result["tables"]["functional"] for result in results):
        axes[1, 1].text(.5, .5, "Probes ausentes: medida indisponivel", ha="center", transform=axes[1, 1].transAxes)
    for axis in axes.flat:
        axis.grid(alpha=.25, color="#E5E7EB")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.tick_params(axis="x", labelrotation=25)
        if axis.lines:
            axis.legend(fontsize=7)
    title = "Comparacao sequencial — melhores checkpoints" if manifest["complete"] else "Comparacao PARCIAL"
    if manifest["options"].get("synthetic_validation"):
        title = "VALIDACAO SINTETICA — somente formato dos graficos"
    figure.suptitle(title + "\nConexoes amostradas; detalhes completos no dashboard interativo", fontsize=15)
    figure.savefig(output_dir / "overview.png", dpi=150)
    figure.clear()


def publish_report(output_dir, figures, tables, manifest, options):
    import wandb
    if manifest["options"].get("synthetic_validation") and not options.get("offline"):
        raise ValueError("Synthetic validation may only be logged in W&B OFFLINE mode")
    backbones = sorted({item["backbone"].split("/")[-1].lower() for item in manifest["experiments"]})
    grids = sorted({item["g"] for item in manifest["experiments"] if item["g"] is not None})
    k_values = sorted({item["k"] for item in manifest["experiments"] if item["k"] is not None})
    name = "comparacao-" + "-".join(backbones) + "-" + "-".join(f"k{k}" for k in k_values)
    name += "-" + "-".join(f"g{grid}" for grid in grids) + "-kan-mlp-sequencial-4x"
    if manifest["options"].get("synthetic_validation"):
        name = "OFFLINE-TEST-" + name
    if not manifest["complete"]:
        name = "PARCIAL-" + name
    with wandb.init(entity=options.get("entity"), project=options.get("project", "MOS-Prediction"),
                    name=name, group=options.get("group"), job_type="sequence-comparison",
                    tags=["comparacao-splines", "kan-mlp", "blocos4",
                          "sequencia-completa" if manifest["complete"] else "parcial"],
                    dir=str(output_dir), mode="offline" if options.get("offline") else "online",
                    config=manifest) as run:
        run.log({"comparacao/visao_geral": wandb.Image(str(output_dir / "overview.png"))})
        for key, figure in figures.items():
            run.log({key: wandb.Plotly(figure)})
        for key, rows in tables.items():
            if not rows or key in {"edges", "responses"}:
                continue  # Full raw point tables are in the artifact, not silently capped in the UI.
            columns = list(dict.fromkeys(field for row in rows for field in row))
            run.log({f"tabelas/{key}_primeiras_10000": wandb.Table(
                columns=columns, data=[[row.get(field) for field in columns] for row in rows[:10000]])})
        artifact = wandb.Artifact("comparacao-heads-" + manifest["fingerprint"][:12], type="sequence-comparison")
        for path in output_dir.iterdir():
            if path.is_file() and (path.suffix in {".csv", ".html", ".png"} or path.name == "manifest.json"):
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.summary.update({"sequencia_completa": manifest["complete"],
                            "g_maior_que_k_controle_ausente": bool(manifest["g_greater_than_k"]) and all(manifest["g_greater_than_k"]),
                            **{f"total_linhas/{key}": len(rows) for key, rows in tables.items()}})
        tracking = {"id": run.id, "url": run.url, "offline": bool(options.get("offline"))}
    write_json(output_dir / "wandb_run.json", tracking)
    return tracking


def schedule_comparison(output_root, expected_experiments, protocol):
    """After training, launch ONE CPU report for the whole queue; release the GPU worker promptly."""
    import subprocess
    import sys

    options = {**DEFAULTS, **protocol.get("comparison", {})}
    if not options["enabled"]:
        return {"status": "disabled"}
    output_root = Path(output_root).resolve()
    report_root = output_root
    expected = [Path(path).resolve() for path in expected_experiments]
    missing_labels = []
    for ancestor in (output_root, *output_root.parents):
        if (ancestor / "round.json").is_file():
            round_manifest = read_json(ancestor / "round.json")
            if "tasks" in round_manifest and any(output_root == Path(task["output"]).resolve()
                                                 for task in round_manifest["tasks"]):
                expected, missing_labels = discover_experiments(ancestor)
                report_root = ancestor
                break
    report_dir = report_root / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    pending = missing_labels + [str(path) for path in expected if sequence_records(path) is None]
    if pending or not expected:
        status = {"status": "waiting_for_complete_sequences", "pending": pending,
                  "expected": [str(path) for path in expected], "wandb_run_created": False}
        write_json(report_dir / "status.json", status)
        return status
    scheduled = report_dir / ".scheduled.lock"
    try:
        descriptor = os.open(scheduled, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {"status": "analysis_already_scheduled"}
    os.close(descriptor)
    request_path = report_dir / "request.json"
    write_json(request_path, {"experiments": [str(path) for path in expected], "output": str(report_dir),
                              "options": options, "wandb": protocol.get("wandb", {})})
    status_path = report_dir / "status.json"
    # Preserve the completed fingerprint so a resume does not create another identical W&B run.
    if not status_path.exists() or read_json(status_path).get("status") != "finished":
        write_json(status_path, {"status": "scheduled", "request": str(request_path)})
    repo = Path(__file__).resolve().parents[2]
    try:
        with (report_dir / "analysis.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(repo / "src/compare_sequence.py"),
                                        "--request", str(request_path)], cwd=repo,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
        write_json(report_dir / "launcher.json", {"pid": process.pid, "request": str(request_path), "device": "cpu"})
        print(f"[Comparison] CPU report scheduled: {report_dir}; pid={process.pid}", flush=True)
        return {"status": "scheduled", "pid": process.pid}
    except Exception:
        scheduled.unlink(missing_ok=True)
        raise
