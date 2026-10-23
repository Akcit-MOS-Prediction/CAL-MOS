"""Small, local CSV outputs for sequential head training (no background services)."""
import csv
import json
import math
import os
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

from utils.dataset_identity import kan_block_padding


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def cell(value):
    if isinstance(value, float):
        return format(value, ".17g").replace(".", ",") if math.isfinite(value) else ""
    return value


def csv_write(path, rows, columns, append=False):
    """Semicolon delimiter + decimal comma; strings/paths are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not append or not path.exists() or path.stat().st_size == 0
    destination = path if append else path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with destination.open("a" if append else "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";", extrasaction="raise")
        if header:
            writer.writeheader()
        writer.writerows({key: cell(value) for key, value in row.items()} for row in rows)
    if not append:
        os.replace(destination, path)


def metric(value):
    if value is None:
        return None
    value = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    return value if math.isfinite(value) else None


DATASET_LABELS = {
    "bvcc": "BVCC",
    "singmos": "SingMOS",
    "tmhint": "TMHINT",
    "brspeech": "BRSpeech",
}
PARAMETER_DRIFT_COLUMNS = [
    "epoca", "parametro", "baseline_l2", "atual_l2", "delta_l2",
    "delta_relativo", "delta_epoca_l2", "delta_epoca_relativo",
    "similaridade_cosseno", "fracao_alterada", "delta_maximo_abs", "fracao_nao_finita",
]
BLOCK_DRIFT_COLUMNS = [
    "epoca", "dataset_bloco", "dataset_treinado", "ativo_no_estagio",
    "baseline_l2", "atual_l2", "delta_l2", "delta_relativo",
    "delta_epoca_l2", "delta_epoca_relativo", "similaridade_cosseno",
    "fracao_alterada", "delta_maximo_abs", "fracao_nao_finita", "gradiente_primeiro_update_l2",
    "learning_rate", "weight_decay",
]
SPLINE_DRIFT_COLUMNS = [
    "epoca", "camada", "dataset_bloco", "low_delta_l2", "low_delta_relativo",
    "height_delta_l2", "height_delta_relativo", "largura_media_baseline",
    "largura_media_atual", "delta_largura_media", "low_fracao_nao_finita",
    "height_fracao_nao_finita",
]
ACTIVATION_HEALTH_COLUMNS = [
    "epoca", "camada", "lotes_amostrados", "valores", "media", "desvio_padrao",
    "minimo", "maximo", "fracao_quase_zero", "fracao_unidades_silenciosas",
    "fracao_nao_finita",
]


def _tensor_drift(reference, current, previous=None):
    """Summarize parameter movement against the stage-start and prior-epoch tensors."""
    reference = reference.detach().float().cpu().reshape(-1)
    current = current.detach().float().cpu().reshape(-1)
    current_all = current
    finite = torch.isfinite(reference) & torch.isfinite(current)
    nonfinite_fraction = float((~finite).float().mean()) if finite.numel() else 0.0
    if not finite.any():
        return {
            "baseline_l2": None, "atual_l2": None, "delta_l2": None,
            "delta_relativo": None, "similaridade_cosseno": None,
            "fracao_alterada": None, "delta_maximo_abs": None,
            "delta_epoca_l2": None, "delta_epoca_relativo": None,
            "fracao_nao_finita": nonfinite_fraction,
        }
    reference = reference[finite]
    current = current[finite]
    delta = current - reference
    reference_norm = float(torch.linalg.vector_norm(reference))
    current_norm = float(torch.linalg.vector_norm(current))
    delta_norm = float(torch.linalg.vector_norm(delta))
    denominator = max(reference_norm, 1e-12)
    cosine_denominator = reference_norm * current_norm
    cosine = (float(torch.dot(reference, current) / cosine_denominator)
              if cosine_denominator > 1e-24 else None)
    summary = {
        "baseline_l2": reference_norm,
        "atual_l2": current_norm,
        "delta_l2": delta_norm,
        "delta_relativo": delta_norm / denominator,
        "similaridade_cosseno": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "fracao_alterada": float((delta.abs() > 1e-12).float().mean()),
        "delta_maximo_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "fracao_nao_finita": nonfinite_fraction,
    }
    if previous is None:
        summary["delta_epoca_l2"] = None
        summary["delta_epoca_relativo"] = None
    else:
        previous = previous.detach().float().cpu().reshape(-1)
        epoch_finite = torch.isfinite(current_all) & torch.isfinite(previous)
        if epoch_finite.any():
            epoch_current, epoch_previous = current_all[epoch_finite], previous[epoch_finite]
            epoch_delta_norm = float(torch.linalg.vector_norm(epoch_current - epoch_previous))
            summary["delta_epoca_l2"] = epoch_delta_norm
            summary["delta_epoca_relativo"] = epoch_delta_norm / max(
                float(torch.linalg.vector_norm(epoch_previous)), 1e-12
            )
        else:
            summary["delta_epoca_l2"] = summary["delta_epoca_relativo"] = None
    return summary


class SequenceOutputs(Callback):
    """Capture the ACTUAL first learned layer input, once per training epoch."""
    def __init__(self, output_dir, save_head_inputs=True, active_dataset=None):
        self.output_dir = Path(output_dir)
        self.save_head_inputs = save_head_inputs
        self.active_dataset = str(active_dataset or "").lower()
        self.handles = []
        self.rows = []
        self.epoch = None
        self.route_input = None
        self.gradient_norm = None
        self.val_predictions = []
        self.val_targets = []
        self.snapshot = {}
        self.kan_epochs = {}
        self.kan_support = {}
        self.parameter_gradients = {}
        self.grad_blocks = []
        self.head_input = None
        self.kan_support_accumulators = {}
        self.kan_support_batch_counts = {}
        self.activation_accumulators = {}
        self.activation_batch_counts = {}
        self.activation_call_counts = {}
        self.activation_health_rows = []
        self.initial_parameters = {}
        self.previous_parameters = {}
        self.current_parameters = {}
        self.parameter_drift_rows = []
        self.block_drift_rows = []
        self.spline_drift_rows = []
        self.dataset_names = []
        self.dataset_names_by_id = {}
        self.head_kind = "mlp"
        self.input_width = 0

    def _wandb_run(self, trainer):
        from lightning.pytorch.loggers import WandbLogger
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger.experiment
        return None

    def on_fit_start(self, trainer, pl_module):
        head = pl_module.model.mlp
        router = head.block_expansion
        if router is None or router.output_size != 4 * head.input_size:
            raise ValueError("Expected a non-affine LayerNorm followed by exactly four blocks")
        is_kan = hasattr(head, "rk_layers")
        self.head_kind = "kan" if is_kan else "mlp"
        self.input_width = int(head.input_size)
        dataset_to_id = {str(name).lower(): int(index)
                         for name, index in dict(router.dataset_to_id).items()}
        self.dataset_names = [name for name, _ in sorted(dataset_to_id.items(), key=lambda item: item[1])]
        self.dataset_names_by_id = {index: name for name, index in dataset_to_id.items()}
        self.initial_parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in head.named_parameters()
        }
        self.previous_parameters = self.initial_parameters
        self.padding_value = float(router.padding_value)
        if tuple(router.start_positions) != tuple(i * head.input_size for i in range(4)):
            raise ValueError("Expected four disjoint dataset blocks")
        if is_kan and not router.padding_value < head.grid_min:
            raise ValueError("KAN dataset blocks must be padded below the ReLU-KAN grid")
        if not is_kan and router.padding_value != 0:
            raise ValueError("MLP dataset blocks must be padded with zero")
        first = head.rk_layers[0] if hasattr(head, "rk_layers") else head.layers[0]
        run = self._wandb_run(trainer)
        if run is not None:
            run.define_metric("debug/epoca")
            run.define_metric("debug/*", step_metric="debug/epoca")
            protocol = {"dim_original": head.input_size, "dim_head": router.output_size,
                        "dataset_to_id": dict(router.dataset_to_id), "fill": router.padding_value,
                        "normalization": ("LayerNorm non-affine + sigmoid BEFORE expansion; LayerNorm + sigmoid "
                                          "between KAN layers" if is_kan else
                                          "LayerNorm non-affine BEFORE expansion"),
                        "capture": "first sample of first training batch per epoch; full vector, no truncation",
                        "weight_drift_reference": "head parameters loaded at the start of this dataset stage",
                        "block_weight_slices": "first learned layer input columns, four equal contiguous dataset blocks",
                        "inactive_block_note": "AdamW weight decay can change zero-gradient weights; compare gradients and deltas together",
                        "spline_profiles": ("weighted per-input-dimension contribution, stage-start vs current"
                                            if is_kan else None),
                        "activation_health": ("KAN layer outputs" if is_kan else
                                              "outputs of the configured MLP activation modules"),
                        "activation_sampling_batches_per_epoch": 16,
                        "near_zero_threshold": 1e-8,
                        "kan_input_transform_version": getattr(head, "INPUT_TRANSFORM", {}).get("version") if is_kan else None,
                        "kan_spline_endpoints_trainable": bool(is_kan),
                        "kan_support_sample_batches_per_epoch": 16 if is_kan else 0}
            run.config.update({"debug_protocol": protocol}, allow_val_change=True)
            json_write(self.output_dir / "debug_protocol.json", protocol)
            run.save(str(self.output_dir / "debug_protocol.json"), base_path=str(self.output_dir), policy="now")

        def capture_norm(module, args, output):
            if pl_module.training and self.epoch != trainer.current_epoch:
                self.snapshot = {"pooled": args[0][0].detach().float().cpu(),
                                 "normalized": output[0].detach().float().cpu()}

        def capture_router(module, args, output):
            if pl_module.training and self.epoch != trainer.current_epoch:
                self.route_input = (args[0].detach(), args[1].detach())

        def capture_head(module, args):
            if not pl_module.training or self.epoch == trainer.current_epoch:
                return
            actual = args[0].detach()
            normalized, ids = self.route_input
            width = head.input_size
            expected = torch.full_like(actual, router.padding_value)
            indices = (ids[:, None] - 1) * width + torch.arange(width, device=actual.device)
            expected.scatter_(1, indices, normalized)
            if actual.shape[1] != 4 * width or not torch.isfinite(actual).all():
                raise ValueError("Invalid/nonfinite vector entering the head")
            if not torch.equal(actual, expected):
                raise ValueError("Head input differs from the expected normalized dataset block")
            self.head_input = actual[0].float().cpu()
            self.snapshot["head_input"] = self.head_input
            self.snapshot["dataset_id"] = int(ids[0])
            if self.save_head_inputs:
                values = actual[0].float().cpu().tolist()
                csv_write(self.output_dir / "head_inputs.csv", (
                    {"epoca": trainer.current_epoch + 1, "dataset_id": int(ids[0]),
                     "indice": index, "bloco": index // width + 1, "valor": value}
                    for index, value in enumerate(values)
                ), ["epoca", "dataset_id", "indice", "bloco", "valor"], append=True)
            active_run = self._wandb_run(trainer)
            if active_run is not None:
                import wandb
                table = wandb.Table(columns=["indice", "valor", "bloco"], data=[
                    [index, float(value), index // width + 1]
                    for index, value in enumerate(self.head_input.tolist())])
                active_run.log({"debug/epoca": trainer.current_epoch + 1,
                                "debug/dim_original": width, "debug/dim_head": 4 * width,
                                "debug/dataset_id": self.snapshot["dataset_id"],
                                "debug/tabela_vetor_completo": table,
                                "debug/vetor_completo": wandb.plot.line(table, "indice", "valor",
                                    title="Entrada COMPLETA da head — primeira amostra")})
                if self.save_head_inputs:
                    active_run.save(str(self.output_dir / "head_inputs.csv"), base_path=str(self.output_dir), policy="now")
            print(f"[Sequence] epoch={trainer.current_epoch + 1} normalized={tuple(normalized.shape)} "
                  f"head_input={tuple(actual.shape)} dataset_ids={ids.tolist()} "
                  f"routing_error=0 padding={router.padding_value:g}", flush=True)
            self.epoch = trainer.current_epoch
            self.route_input = None

        def capture_validation(module, args, output):
            if trainer.validating and not trainer.sanity_checking:
                self.val_predictions.extend(output.detach().float().cpu().reshape(-1).tolist())

        self.handles = [router.register_forward_hook(capture_router),
                        head.input_layer_norm.register_forward_hook(capture_norm),
                        first.register_forward_pre_hook(capture_head),
                        pl_module.model.register_forward_hook(capture_validation)]
        if hasattr(head, "rk_layers"):
            for index, layer in enumerate(head.rk_layers):
                def capture_support(module, args, layer_index=index):
                    if not pl_module.training:
                        return
                    values = args[0].detach()
                    if self.kan_support_batch_counts.get(layer_index, 0) >= 16:
                        return
                    active = ((values.unsqueeze(-1) > module.phase_low) &
                              (values.unsqueeze(-1) < module.phase_height))
                    # Layer 0: padded positions sit below the grid on purpose; measure real inputs only.
                    real = (values != self.padding_value) if layer_index == 0 else torch.ones_like(values, dtype=torch.bool)
                    valid_values = values[real]
                    outside = (~active.any(-1) & real).sum()
                    active_count = (active & real.unsqueeze(-1)).sum()
                    accumulator = self.kan_support_accumulators.get(layer_index)
                    if accumulator is None:
                        accumulator = {
                            "valid_count": torch.zeros((), dtype=torch.long, device=values.device),
                            "outside_count": torch.zeros((), dtype=torch.long, device=values.device),
                            "active_basis_count": torch.zeros((), dtype=torch.long, device=values.device),
                            "input_min": valid_values.min().detach(),
                            "input_max": valid_values.max().detach(),
                            "support_min": module.phase_low.min().detach(),
                            "support_max": module.phase_height.max().detach(),
                        }
                        self.kan_support_accumulators[layer_index] = accumulator
                    accumulator["valid_count"] += real.sum()
                    accumulator["outside_count"] += outside
                    accumulator["active_basis_count"] += active_count
                    accumulator["input_min"] = torch.minimum(accumulator["input_min"], valid_values.min())
                    accumulator["input_max"] = torch.maximum(accumulator["input_max"], valid_values.max())
                    accumulator["support_min"] = torch.minimum(accumulator["support_min"], module.phase_low.min())
                    accumulator["support_max"] = torch.maximum(accumulator["support_max"], module.phase_height.max())
                    self.kan_support_batch_counts[layer_index] = self.kan_support_batch_counts.get(layer_index, 0) + 1
                    if self.kan_epochs.get(layer_index) != trainer.current_epoch:
                        self.snapshot[f"kan_layer_{layer_index}_input"] = values[0].float().cpu()
                        self.kan_epochs[layer_index] = trainer.current_epoch
                self.handles.append(layer.register_forward_pre_hook(capture_support))

        if is_kan:
            for index, layer in enumerate(head.rk_layers):
                def capture_kan_output(module, args, output, layer_index=index):
                    if pl_module.training:
                        self._accumulate_activation(f"kan.rk_layers.{layer_index}", output)
                self.handles.append(layer.register_forward_hook(capture_kan_output))
        else:
            import torch.nn as nn
            activation_modules = {}
            for index, module in enumerate(head.layers):
                if isinstance(module, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.SiLU, nn.Tanh)):
                    activation_modules.setdefault(id(module), {"module": module, "indices": []})["indices"].append(index)
            for module_id, entry in activation_modules.items():
                indices = tuple(entry["indices"])
                def capture_mlp_activation(module, args, output, layer_indices=indices, identity=module_id):
                    if not pl_module.training:
                        return
                    call_index = self.activation_call_counts.get(identity, 0)
                    self.activation_call_counts[identity] = call_index + 1
                    if call_index < len(layer_indices):
                        layer_index = layer_indices[call_index]
                        self._accumulate_activation(f"mlp.layers.{layer_index}.{module.__class__.__name__}", output)
                self.handles.append(entry["module"].register_forward_hook(capture_mlp_activation))

    def on_train_epoch_start(self, trainer, pl_module):
        self.gradient_norm = None
        self.parameter_gradients = {}
        self.kan_support = {}
        self.kan_support_accumulators = {}
        self.kan_support_batch_counts = {}
        self.kan_epochs = {}
        self.activation_accumulators = {}
        self.activation_batch_counts = {}
        self.activation_call_counts = {}
        self.activation_health_rows = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.activation_call_counts = {}

    def _accumulate_activation(self, name, output):
        if self.activation_batch_counts.get(name, 0) >= 16 or not isinstance(output, torch.Tensor):
            return
        values = output.detach().float().reshape(-1, output.shape[-1])
        finite = torch.isfinite(values)
        absolute = values.abs()
        accumulator = self.activation_accumulators.get(name)
        if accumulator is None:
            units = values.shape[-1]
            accumulator = {
                "total_count": torch.zeros((), dtype=torch.long, device=values.device),
                "finite_count": torch.zeros((), dtype=torch.long, device=values.device),
                "nonfinite_count": torch.zeros((), dtype=torch.long, device=values.device),
                "near_zero_count": torch.zeros((), dtype=torch.long, device=values.device),
                "sum": torch.zeros((), dtype=torch.float64, device=values.device),
                "sum_squares": torch.zeros((), dtype=torch.float64, device=values.device),
                "minimum": torch.full((), float("inf"), device=values.device),
                "maximum": torch.full((), float("-inf"), device=values.device),
                "active_per_unit": torch.zeros(units, dtype=torch.long, device=values.device),
                "finite_per_unit": torch.zeros(units, dtype=torch.long, device=values.device),
            }
            self.activation_accumulators[name] = accumulator
        finite_values = torch.where(finite, values, torch.zeros_like(values)).double()
        accumulator["total_count"] += values.numel()
        accumulator["finite_count"] += finite.sum()
        accumulator["nonfinite_count"] += (~finite).sum()
        accumulator["near_zero_count"] += (finite & (absolute <= 1e-8)).sum()
        accumulator["sum"] += finite_values.sum()
        accumulator["sum_squares"] += finite_values.square().sum()
        accumulator["minimum"] = torch.minimum(
            accumulator["minimum"], torch.where(finite, values, torch.full_like(values, float("inf"))).min()
        )
        accumulator["maximum"] = torch.maximum(
            accumulator["maximum"], torch.where(finite, values, torch.full_like(values, float("-inf"))).max()
        )
        accumulator["active_per_unit"] += (finite & (absolute > 1e-8)).sum(dim=0)
        accumulator["finite_per_unit"] += finite.sum(dim=0)
        self.activation_batch_counts[name] = self.activation_batch_counts.get(name, 0) + 1

    def _finalize_activation_stats(self, epoch):
        rows = []
        for name, accumulator in self.activation_accumulators.items():
            total = int(accumulator["total_count"].detach().cpu())
            finite_count = int(accumulator["finite_count"].detach().cpu())
            nonfinite_count = int(accumulator["nonfinite_count"].detach().cpu())
            if finite_count:
                mean = float((accumulator["sum"] / finite_count).detach().cpu())
                variance = max(float((accumulator["sum_squares"] / finite_count).detach().cpu()) - mean ** 2, 0.0)
                minimum = float(accumulator["minimum"].detach().cpu())
                maximum = float(accumulator["maximum"].detach().cpu())
            else:
                mean = variance = minimum = maximum = None
            finite_units = accumulator["finite_per_unit"] > 0
            if finite_units.any():
                silent = float((accumulator["active_per_unit"][finite_units] == 0).float().mean().detach().cpu())
            else:
                silent = None
            near_zero = (int(accumulator["near_zero_count"].detach().cpu()) / finite_count
                         if finite_count else None)
            rows.append({
                "epoca": epoch,
                "camada": name,
                "lotes_amostrados": self.activation_batch_counts.get(name, 0),
                "valores": total,
                "media": mean,
                "desvio_padrao": math.sqrt(variance) if variance is not None else None,
                "minimo": minimum,
                "maximo": maximum,
                "fracao_quase_zero": near_zero,
                "fracao_unidades_silenciosas": silent,
                "fracao_nao_finita": nonfinite_count / max(total, 1),
            })
        return rows

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.gradient_norm is None:
            gradients = [p.grad.detach().float().square().sum()
                         for p in pl_module.model.mlp.parameters() if p.grad is not None]
            self.gradient_norm = metric(torch.stack(gradients).sum().sqrt()) if gradients else 0.0
            self.parameter_gradients = {name: float(parameter.grad.detach().float().norm())
                                       for name, parameter in pl_module.model.mlp.named_parameters()
                                       if parameter.grad is not None}
            head = pl_module.model.mlp
            weight = head.rk_layers[0].equal_size_conv.weight if hasattr(head, "rk_layers") else head.layers[0].weight
            if weight.grad is not None:
                grad = weight.grad.detach().float()
                per_feature = grad.square().sum(tuple(range(grad.ndim - 1)))
                self.grad_blocks = per_feature.reshape(4, head.input_size).sum(1).sqrt().cpu().tolist()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_predictions = []
        self.val_targets = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self.val_targets.extend(batch[1].detach().float().cpu().reshape(-1).tolist())

    def _snapshot_parameters(self, pl_module):
        return {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in pl_module.model.mlp.named_parameters()
        }

    @staticmethod
    def _parameter_name(name):
        return name.replace(".", "/")

    def _parameter_rows(self, current, epoch, previous=None):
        rows = []
        for name, reference in self.initial_parameters.items():
            stats = _tensor_drift(reference, current[name],
                                  previous[name] if previous is not None and name in previous else None)
            rows.append({
                "epoca": epoch,
                "parametro": name,
                "baseline_l2": stats["baseline_l2"],
                "atual_l2": stats["atual_l2"],
                "delta_l2": stats["delta_l2"],
                "delta_relativo": stats["delta_relativo"],
                "delta_epoca_l2": stats["delta_epoca_l2"],
                "delta_epoca_relativo": stats["delta_epoca_relativo"],
                "similaridade_cosseno": stats["similaridade_cosseno"],
                "fracao_alterada": stats["fracao_alterada"],
                "delta_maximo_abs": stats["delta_maximo_abs"],
                "fracao_nao_finita": stats["fracao_nao_finita"],
            })
        return rows

    def _first_weight_name(self, parameters):
        candidates = (
            "rk_layers.0.equal_size_conv.weight",
            "layers.0.weight",
        )
        for name in candidates:
            if name in parameters:
                return name
        return None

    def _block_weight_rows(self, current, epoch, previous=None, trainer=None):
        name = self._first_weight_name(current)
        if name is None or current[name].shape[-1] != 4 * self.input_width:
            return []
        reference = self.initial_parameters[name]
        previous_weight = previous.get(name) if previous is not None else None
        optimizer = trainer.optimizers[0] if trainer is not None and trainer.optimizers else None
        learning_rate = float(optimizer.param_groups[0]["lr"]) if optimizer else None
        weight_decay = float(optimizer.param_groups[0].get("weight_decay", 0.0)) if optimizer else None
        rows = []
        for index, dataset in enumerate(self.dataset_names):
            start = index * self.input_width
            end = start + self.input_width
            baseline_block = reference[..., start:end]
            current_block = current[name][..., start:end]
            previous_block = previous_weight[..., start:end] if previous_weight is not None else None
            stats = _tensor_drift(baseline_block, current_block, previous_block)
            rows.append({
                "epoca": epoch,
                "dataset_bloco": DATASET_LABELS.get(dataset, dataset),
                "dataset_treinado": DATASET_LABELS.get(self.active_dataset, self.active_dataset),
                "ativo_no_estagio": int(dataset == self.active_dataset),
                "baseline_l2": stats["baseline_l2"],
                "atual_l2": stats["atual_l2"],
                "delta_l2": stats["delta_l2"],
                "delta_relativo": stats["delta_relativo"],
                "delta_epoca_l2": stats["delta_epoca_l2"],
                "delta_epoca_relativo": stats["delta_epoca_relativo"],
                "similaridade_cosseno": stats["similaridade_cosseno"],
                "fracao_alterada": stats["fracao_alterada"],
                "delta_maximo_abs": stats["delta_maximo_abs"],
                "fracao_nao_finita": stats["fracao_nao_finita"],
                "gradiente_primeiro_update_l2": (self.grad_blocks[index]
                                                  if index < len(self.grad_blocks) else None),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            })
        return rows

    def _spline_rows(self, current, epoch, head):
        if not hasattr(head, "rk_layers"):
            return []
        rows = []
        for index, _ in enumerate(head.rk_layers):
            low_name = f"rk_layers.{index}.phase_low"
            height_name = f"rk_layers.{index}.phase_height"
            low_before, low_after = self.initial_parameters[low_name], current[low_name]
            height_before, height_after = self.initial_parameters[height_name], current[height_name]

            def make_row(dataset, selection):
                low = _tensor_drift(low_before[selection], low_after[selection])
                height = _tensor_drift(height_before[selection], height_after[selection])
                width_before_values = height_before[selection] - low_before[selection]
                width_after_values = height_after[selection] - low_after[selection]
                finite_before = torch.isfinite(width_before_values)
                finite_after = torch.isfinite(width_after_values)
                width_before = (float(width_before_values[finite_before].mean())
                                if finite_before.any() else None)
                width_after = (float(width_after_values[finite_after].mean())
                               if finite_after.any() else None)
                rows.append({
                    "epoca": epoch,
                    "camada": index,
                    "dataset_bloco": dataset,
                    "low_delta_l2": low["delta_l2"],
                    "low_delta_relativo": low["delta_relativo"],
                    "height_delta_l2": height["delta_l2"],
                    "height_delta_relativo": height["delta_relativo"],
                    "largura_media_baseline": width_before,
                    "largura_media_atual": width_after,
                    "delta_largura_media": (width_after - width_before
                                             if width_before is not None and width_after is not None else None),
                    "low_fracao_nao_finita": low["fracao_nao_finita"],
                    "height_fracao_nao_finita": height["fracao_nao_finita"],
                })

            if index == 0 and low_before.shape[0] == 4 * self.input_width:
                make_row("TODOS", slice(None))
                for block_index, dataset in enumerate(self.dataset_names):
                    start = block_index * self.input_width
                    make_row(DATASET_LABELS.get(dataset, dataset),
                             slice(start, start + self.input_width))
            else:
                make_row("TODOS", slice(None))
        return rows

    def _spline_profile_figure(self, pl_module, layer_index, current, epoch, best=False):
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        head = pl_module.model.mlp
        layer = head.rk_layers[layer_index]
        weight_name = f"rk_layers.{layer_index}.equal_size_conv.weight"
        weight_before = self.initial_parameters[weight_name].squeeze(1)
        weight_after = current[weight_name].squeeze(1)
        weight_delta = weight_after - weight_before
        input_size = int(weight_after.shape[-1])
        if layer_index == 0 and input_size == 4 * self.input_width:
            groups = [(DATASET_LABELS.get(name, name), i * self.input_width,
                       (i + 1) * self.input_width)
                      for i, name in enumerate(self.dataset_names)]
        else:
            edges = np.linspace(0, input_size, 5, dtype=int)
            groups = [(f"faixa {i + 1}", int(edges[i]), int(edges[i + 1]))
                      for i in range(4)]

        samples = []
        low_before = self.initial_parameters[f"rk_layers.{layer_index}.phase_low"]
        low_after = current[f"rk_layers.{layer_index}.phase_low"]
        height_before = self.initial_parameters[f"rk_layers.{layer_index}.phase_height"]
        height_after = current[f"rk_layers.{layer_index}.phase_height"]
        for label, start, end in groups:
            if start >= end:
                continue
            base_norm = torch.linalg.vector_norm(weight_before[..., start:end], dim=(0, 1))
            delta_norm = torch.linalg.vector_norm(weight_delta[..., start:end], dim=(0, 1))
            phase_delta = torch.linalg.vector_norm(
                low_after[start:end] - low_before[start:end], dim=-1
            ) + torch.linalg.vector_norm(
                height_after[start:end] - height_before[start:end], dim=-1
            )
            score = delta_norm / (base_norm + 1e-12) + phase_delta
            feature = start + int(torch.argmax(score))
            output_scores = torch.linalg.vector_norm(weight_delta[:, :, feature], dim=-1)
            output_unit = int(torch.argmax(output_scores))
            samples.append((label, feature, output_unit))

        figure = Figure(figsize=(12, 8), facecolor="white", constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(samples), 2, squeeze=False)
        x = np.linspace(0.0, 1.0, 201, dtype=np.float32)
        phase_scale = float(layer.r)

        def contribution(low, height, coefficients):
            basis = (np.maximum(x[:, None] - low[None, :], 0.0) *
                     np.maximum(height[None, :] - x[:, None], 0.0) * phase_scale) ** 2
            return basis @ coefficients

        for row_index, (label, feature, output_unit) in enumerate(samples):
            base_low = low_before[feature].numpy()
            now_low = low_after[feature].numpy()
            base_high = height_before[feature].numpy()
            now_high = height_after[feature].numpy()
            before_curve = contribution(base_low, base_high,
                                        weight_before[output_unit, :, feature].numpy())
            after_curve = contribution(now_low, now_high,
                                       weight_after[output_unit, :, feature].numpy())
            ax_curve, ax_delta = axes[row_index]
            ax_curve.plot(x, before_curve, color="#F59E0B", label="Checkpoint de origem")
            ax_curve.plot(x, after_curve, color="#7C3AED", label="Atual")
            ax_curve.set(title=f"{label}: dimensão {feature} → neurônio {output_unit}",
                         ylabel="Contribuição sem viés")
            ax_delta.plot(x, after_curve - before_curve, color="#059669")
            ax_delta.axhline(0.0, color="#64748B", linewidth=.7)
            ax_delta.set(title="Atual − origem", ylabel="Δ contribuição")
            for boundary in (base_low[0], base_high[-1], now_low[0], now_high[-1]):
                if 0.0 < boundary < 1.0:
                    ax_curve.axvline(boundary, color="#CBD5E1", linewidth=.5, alpha=.65)
            if row_index == 0:
                ax_curve.legend(loc="best", fontsize=8)
            for axis in (ax_curve, ax_delta):
                axis.set_xlabel("Entrada da spline (0–1)")
                axis.grid(alpha=.25, color="#E5E7EB")
                axis.spines[["top", "right"]].set_visible(False)
        title = "Checkpoint selecionado" if best else f"Época {epoch}"
        figure.suptitle(f"ReLU-KAN — perfis de contribuição das splines | camada {layer_index} | {title}",
                        fontsize=14)
        suffix = "best" if best else f"epoch_{epoch:03d}"
        path = self.output_dir / f"spline_profiles_{suffix}_layer_{layer_index}.png"
        figure.savefig(path, dpi=150)
        figure.clear()
        return path

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        predictions = self.val_predictions
        finite_predictions = [value for value in predictions if math.isfinite(value)]
        finite_targets = [value for value in self.val_targets if math.isfinite(value)]
        paired = [(target, prediction) for target, prediction in zip(self.val_targets, predictions)
                  if math.isfinite(target) and math.isfinite(prediction)]
        constant = (len(finite_predictions) > 1 and
                    max(finite_predictions) - min(finite_predictions) <= 1e-6)
        row = {"epoca": trainer.current_epoch + 1, "passo": trainer.global_step,
               "lr": trainer.optimizers[0].param_groups[0]["lr"],
               "grad_norm_primeiro_update": self.gradient_norm,
               "val_mse": metric(trainer.callback_metrics.get("val/mse")),
               "val_srcc": metric(trainer.callback_metrics.get("val/spearman")),
               "pred_min": min(finite_predictions) if finite_predictions else None,
               "pred_max": max(finite_predictions) if finite_predictions else None,
               "pred_nao_finita_fracao": 1.0 - len(finite_predictions) / max(len(predictions), 1),
               "target_nao_finita_fracao": 1.0 - len(finite_targets) / max(len(self.val_targets), 1),
               "pares_finitos_validacao": len(paired),
               "pred_constante": int(constant)}
        for index, accumulator in self.kan_support_accumulators.items():
            valid_count = int(accumulator["valid_count"].detach().cpu())
            outside_count = int(accumulator["outside_count"].detach().cpu())
            active_basis_count = int(accumulator["active_basis_count"].detach().cpu())
            self.kan_support[index] = {
                "fora_de_todas_bases": outside_count / max(valid_count, 1),
                "bases_ativas_media": active_basis_count / max(valid_count, 1),
                "entradas_validas": valid_count,
                "lotes_amostrados": self.kan_support_batch_counts.get(index, 0),
                "entrada_min": metric(accumulator["input_min"]),
                "entrada_max": metric(accumulator["input_max"]),
                "support_min": metric(accumulator["support_min"]),
                "support_max": metric(accumulator["support_max"]),
            }
        self.activation_health_rows = self._finalize_activation_stats(trainer.current_epoch + 1)
        self.rows.append(row)
        csv_write(self.output_dir / "epochs.csv", self.rows, list(row))
        if self.activation_health_rows:
            csv_write(self.output_dir / "activation_health.csv", self.activation_health_rows,
                      ACTIVATION_HEALTH_COLUMNS, append=True)
        current = self._snapshot_parameters(pl_module)
        epoch = trainer.current_epoch + 1
        self.parameter_drift_rows = self._parameter_rows(current, epoch, self.previous_parameters)
        self.block_drift_rows = self._block_weight_rows(
            current, epoch, self.previous_parameters, trainer
        )
        self.spline_drift_rows = self._spline_rows(current, epoch, pl_module.model.mlp)
        if self.parameter_drift_rows:
            csv_write(self.output_dir / "parameter_drift.csv", self.parameter_drift_rows,
                      PARAMETER_DRIFT_COLUMNS, append=True)
        if self.block_drift_rows:
            csv_write(self.output_dir / "dataset_block_weights.csv", self.block_drift_rows,
                      BLOCK_DRIFT_COLUMNS, append=True)
        if self.spline_drift_rows:
            csv_write(self.output_dir / "kan_spline_drift.csv", self.spline_drift_rows,
                      SPLINE_DRIFT_COLUMNS, append=True)
        self.current_parameters = current
        run = self._wandb_run(trainer)
        if run is not None:
            self._log_debug(run, trainer, pl_module, row)
        self.previous_parameters = current
        if not finite_predictions or len(finite_predictions) != len(predictions):
            print("[WARNING] Validation predictions contain nonfinite values; debug outputs record the fraction.",
                  flush=True)
        if constant:
            print("[WARNING] Validation predictions are constant; inspect KAN support/gradients. "
                  "No architectural fix is applied automatically.", flush=True)

    def _log_debug(self, run, trainer, pl_module, row):
        import numpy as np
        import wandb
        values = self.head_input.numpy()
        width = len(values) // 4
        names = [DATASET_LABELS.get(name, name) for name in self.dataset_names]
        blocks = values.reshape(4, width)
        finite_pairs = [(float(target), float(prediction))
                        for target, prediction in zip(self.val_targets, self.val_predictions)
                        if math.isfinite(target) and math.isfinite(prediction)]
        vector_table = wandb.Table(columns=["indice", "valor", "bloco"],
                                   data=[[i, float(value), names[i // width]] for i, value in enumerate(values)])
        padding = getattr(self, "padding_value", 0.0)
        block_table = wandb.Table(columns=["dataset", "norma_L2", "fracao_nao_padding", "grad_norm"],
                                  data=[[name, float(np.linalg.norm(block)), float(np.mean(block != padding)),
                                         self.grad_blocks[i] if self.grad_blocks else 0.0]
                                        for i, (name, block) in enumerate(zip(names, blocks))])
        prediction_table = wandb.Table(columns=["MOS_real", "MOS_predito"],
                                       data=[[target, prediction] for target, prediction in finite_pairs])
        gradient_table = wandb.Table(columns=["parametro", "grad_norm"],
                                     data=[[name, value] for name, value in self.parameter_gradients.items()])
        parameter_table = wandb.Table(
            columns=["parametro", "delta_relativo", "delta_epoca_relativo", "delta_l2"],
            data=[[item["parametro"], item["delta_relativo"], item["delta_epoca_relativo"],
                   item["delta_l2"]] for item in self.parameter_drift_rows],
        )
        dataset_weight_table = wandb.Table(
            columns=["bloco_dataset", "treinado_em", "delta_relativo", "delta_epoca_relativo",
                     "gradiente_primeiro_update_l2"],
            data=[[item["dataset_bloco"], item["dataset_treinado"], item["delta_relativo"],
                   item["delta_epoca_relativo"], item["gradiente_primeiro_update_l2"]]
                  for item in self.block_drift_rows],
        )
        activation_table = wandb.Table(
            columns=["camada", "fracao_quase_zero", "fracao_unidades_silenciosas",
                     "desvio_padrao", "fracao_nao_finita"],
            data=[[item["camada"], item["fracao_quase_zero"],
                   item["fracao_unidades_silenciosas"], item["desvio_padrao"],
                   item["fracao_nao_finita"]] for item in self.activation_health_rows],
        )
        payload = {"debug/epoca": trainer.current_epoch + 1,
                   **{f"debug/{key}": value for key, value in row.items() if value is not None},
                   "debug/dim_original": width, "debug/dim_head": len(values),
                   "debug/dataset_id": self.snapshot["dataset_id"],
                   "debug/erro_roteamento": 0, "debug/erro_fill": 0,
                   "debug/vetor_completo": wandb.plot.line(vector_table, "indice", "valor", title="Entrada COMPLETA da head — primeira amostra"),
                   "debug/tabela_vetor_completo": vector_table,
                   "debug/normas_blocos": wandb.plot.bar(block_table, "dataset", "norma_L2", title="Norma L2 por bloco"),
                   "debug/gradientes_blocos": wandb.plot.bar(block_table, "dataset", "grad_norm", title="Gradiente da primeira camada por bloco"),
                   "debug/tabela_blocos": block_table,
                   "debug/pesos/tabela_delta_parametros": parameter_table,
                   "debug/pesos/tabela_blocos_dataset": dataset_weight_table,
                   "debug/ativacoes/tabela_saude_por_camada": activation_table,
                   "debug/pred_nao_finita_fracao": row["pred_nao_finita_fracao"],
                   "debug/target_nao_finita_fracao": row["target_nao_finita_fracao"],
                   "debug/hist_head_input": wandb.Histogram(values),
                   "debug/hist_normalizado": wandb.Histogram(self.snapshot["normalized"].numpy()),
                   "debug/hist_pooled": wandb.Histogram(self.snapshot["pooled"].numpy())}
        if self.parameter_gradients:
            payload["debug/gradientes_parametros"] = wandb.plot.bar(
                gradient_table, "parametro", "grad_norm",
                title="Gradientes — primeiro update da época")
        if finite_pairs:
            payload.update({
                "debug/predicao_vs_real": wandb.plot.scatter(
                    prediction_table, "MOS_real", "MOS_predito",
                    title="Validação — pares finitos MOS real × predito"),
                "debug/pred_std": float(np.std([prediction for _, prediction in finite_pairs])),
                "debug/pred_range": float(np.ptp([prediction for _, prediction in finite_pairs])),
                "debug/target_std": float(np.std([target for target, _ in finite_pairs])),
            })
        zero_rows = [item for item in self.activation_health_rows if item["fracao_quase_zero"] is not None]
        if zero_rows:
            zero_table = wandb.Table(columns=["camada", "fracao_quase_zero"],
                                     data=[[item["camada"], item["fracao_quase_zero"]]
                                           for item in zero_rows])
            payload["debug/ativacoes/quase_zero_por_camada"] = wandb.plot.bar(
                zero_table, "camada", "fracao_quase_zero",
                title="Fração de ativações zero ou quase zero")
        silent_rows = [item for item in self.activation_health_rows
                       if item["fracao_unidades_silenciosas"] is not None]
        if silent_rows:
            silent_table = wandb.Table(columns=["camada", "fracao_unidades_silenciosas"],
                                       data=[[item["camada"], item["fracao_unidades_silenciosas"]]
                                             for item in silent_rows])
            payload["debug/ativacoes/unidades_silenciosas_por_camada"] = wandb.plot.bar(
                silent_table, "camada", "fracao_unidades_silenciosas",
                title="Unidades sem ativação na amostra de treino")
        for item in self.parameter_drift_rows:
            key = self._parameter_name(item["parametro"])
            for metric_name in ("atual_l2", "delta_l2", "delta_relativo", "delta_epoca_l2",
                                "delta_epoca_relativo", "similaridade_cosseno", "fracao_alterada",
                                "delta_maximo_abs", "fracao_nao_finita"):
                value = item.get(metric_name)
                if value is not None:
                    payload[f"debug/pesos/parametro/{key}/{metric_name}"] = value
        for item in self.block_drift_rows:
            key = item["dataset_bloco"].lower()
            for metric_name in ("delta_l2", "delta_relativo", "delta_epoca_relativo",
                                "similaridade_cosseno", "fracao_alterada", "fracao_nao_finita",
                                "gradiente_primeiro_update_l2"):
                value = item.get(metric_name)
                if value is not None:
                    payload[f"debug/pesos/bloco_dataset/{key}/{metric_name}"] = value
            payload[f"debug/pesos/bloco_dataset/{key}/ativo_no_estagio"] = item["ativo_no_estagio"]
        first_weight_name = self._first_weight_name(self.current_parameters)
        if first_weight_name is not None:
            base_weight = self.initial_parameters[first_weight_name]
            current_weight = self.current_parameters[first_weight_name]
            for index, name in enumerate(self.dataset_names):
                start, end = index * self.input_width, (index + 1) * self.input_width
                delta_values = (current_weight[..., start:end] - base_weight[..., start:end]).numpy().reshape(-1)
                delta_values = delta_values[np.isfinite(delta_values)]
                if len(delta_values):
                    stride = max(1, len(delta_values) // 10000)
                    dataset_key = DATASET_LABELS.get(name, name).lower()
                    payload[f"debug/pesos/bloco_dataset/{dataset_key}/delta_distribution"] = wandb.Histogram(
                        delta_values[::stride]
                    )
        valid_parameter_rows = [item for item in self.parameter_drift_rows
                                if item["delta_relativo"] is not None]
        if valid_parameter_rows:
            parameter_plot_table = wandb.Table(
                columns=["parametro", "delta_relativo"],
                data=[[item["parametro"], item["delta_relativo"]] for item in valid_parameter_rows],
            )
            payload["debug/pesos/delta_por_parametro"] = wandb.plot.bar(
                parameter_plot_table, "parametro", "delta_relativo",
                title="Alteração relativa dos parâmetros desde o checkpoint de origem")
        valid_block_rows = [item for item in self.block_drift_rows if item["delta_relativo"] is not None]
        if valid_block_rows:
            block_plot_table = wandb.Table(
                columns=["bloco_dataset", "delta_relativo"],
                data=[[item["dataset_bloco"], item["delta_relativo"]] for item in valid_block_rows],
            )
            payload["debug/pesos/delta_por_bloco_dataset"] = wandb.plot.bar(
                block_plot_table, "bloco_dataset", "delta_relativo",
                title="Alteração dos pesos de entrada por bloco de dataset")
        for item in self.activation_health_rows:
            key = self._parameter_name(item["camada"])
            for metric_name in ("media", "desvio_padrao", "minimo", "maximo", "fracao_quase_zero",
                                "fracao_unidades_silenciosas", "fracao_nao_finita"):
                value = item.get(metric_name)
                if value is not None:
                    payload[f"debug/ativacoes/{key}/{metric_name}"] = value
        for index, support in self.kan_support.items():
            payload.update({f"debug/kan_camada_{index}/{key}": value for key, value in support.items()})
        if hasattr(pl_module.model.mlp, "rk_layers"):
            for item in self.spline_drift_rows:
                key = f"layer_{item['camada']}/{item['dataset_bloco'].lower()}"
                for metric_name in ("low_delta_l2", "low_delta_relativo", "height_delta_l2",
                                    "height_delta_relativo", "largura_media_baseline",
                                    "largura_media_atual", "delta_largura_media"):
                    if item[metric_name] is not None:
                        payload[f"debug/splines/{key}/{metric_name}"] = item[metric_name]
            for index, layer in enumerate(pl_module.model.mlp.rk_layers):
                low_name = f"rk_layers.{index}.phase_low"
                height_name = f"rk_layers.{index}.phase_height"
                low_delta = (self.current_parameters[low_name] - self.initial_parameters[low_name]).numpy().reshape(-1)
                height_delta = (self.current_parameters[height_name] - self.initial_parameters[height_name]).numpy().reshape(-1)
                for label, values_delta in (("phase_low", low_delta), ("phase_height", height_delta)):
                    values_delta = values_delta[np.isfinite(values_delta)]
                    stride = max(1, len(values_delta) // 20000)
                    if len(values_delta):
                        payload[f"debug/splines/layer_{index}/{label}_shift_distribution"] = wandb.Histogram(
                            values_delta[::stride]
                        )
                spline_path = self._spline_profile_figure(
                    pl_module, index, self.current_parameters, trainer.current_epoch + 1
                )
                payload[f"debug/splines/layer_{index}/profiles"] = wandb.Image(str(spline_path))
            spline_table = wandb.Table(
                columns=["camada", "bloco_dataset", "low_delta_l2", "height_delta_l2",
                         "delta_largura_media"],
                data=[[item["camada"], item["dataset_bloco"], item["low_delta_l2"],
                       item["height_delta_l2"], item["delta_largura_media"]]
                      for item in self.spline_drift_rows],
            )
            payload["debug/splines/tabela_mudanca_splines"] = spline_table
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_dir / f"debug_epoch_{trainer.current_epoch + 1:03d}.npz",
                            **{key: value.numpy() if isinstance(value, torch.Tensor) else value
                               for key, value in self.snapshot.items()},
                            validation_targets=np.asarray(self.val_targets), validation_predictions=np.asarray(self.val_predictions))
        json_write(self.output_dir / f"debug_epoch_{trainer.current_epoch + 1:03d}.json",
                   {"epoch": trainer.current_epoch + 1, "metrics": row,
                    "kan_support": self.kan_support, "gradient_norms": self.parameter_gradients,
                    "parameter_drift": self.parameter_drift_rows,
                    "dataset_block_weight_drift": self.block_drift_rows,
                    "kan_spline_drift": self.spline_drift_rows,
                    "activation_health": self.activation_health_rows})
        # Dependency-free static overview alongside W&B-native interactive tables/charts.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(13, 8), facecolor="white", constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        axes[0, 0].plot(np.arange(len(values)), values, color="#F59E0B", linewidth=.6)
        for boundary in (width, 2 * width, 3 * width):
            axes[0, 0].axvline(boundary, color="#CBD5E1", linewidth=.8)
        axes[0, 0].set(title=f"Entrada completa da head: {len(values)} dimensões", xlabel="Índice", ylabel="Valor")
        axes[0, 1].barh(names, np.linalg.norm(blocks, axis=1), color="#7C3AED")
        axes[0, 1].invert_yaxis()
        axes[0, 1].set(title="Atividade dos quatro blocos", xlabel="Norma L2")
        if finite_pairs:
            finite_targets = [target for target, _ in finite_pairs]
            finite_predictions = [prediction for _, prediction in finite_pairs]
            axes[1, 0].scatter(finite_targets, finite_predictions, s=8, alpha=.4, color="#7C3AED")
            lower = min(finite_targets + finite_predictions)
            upper = max(finite_targets + finite_predictions)
            axes[1, 0].plot([lower, upper], [lower, upper], color="#059669", linestyle="--")
        axes[1, 0].set(title="Validação completa", xlabel="MOS real", ylabel="MOS predito")
        axes[1, 1].barh(names, self.grad_blocks or [0] * 4, color="#F59E0B")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set(title="Primeiro update da época", xlabel="Norma do gradiente por bloco")
        for axis in axes.flat:
            axis.grid(alpha=.25, color="#E5E7EB")
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(f"Debug — época {trainer.current_epoch + 1} | vetor: primeira amostra; MOS: validação completa", fontsize=15)
        preview = self.output_dir / f"debug_epoch_{trainer.current_epoch + 1:03d}.png"
        figure.savefig(preview, dpi=160)
        figure.clear()
        payload["debug/visao_geral"] = wandb.Image(str(preview))
        run.log(payload)
        run.save(str(self.output_dir / "*.csv"), base_path=str(self.output_dir), policy="now")
        run.save(str(self.output_dir / "debug_*"), base_path=str(self.output_dir), policy="now")

    def log_best_checkpoint(self, pl_module, checkpoint_path, run, best_epoch):
        """Record parameter and spline movement for the checkpoint that the next stage loads."""
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only was added.
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        current = {}
        for name in self.initial_parameters:
            key = f"model.mlp.{name}"
            if key not in state:
                raise KeyError(f"Best checkpoint lacks head parameter {key}")
            current[name] = state[key].detach().float().cpu().clone()

        if best_epoch is None:
            best_epoch = checkpoint.get("epoch", 0)
        epoch = int(best_epoch) + 1
        parameter_rows = self._parameter_rows(current, epoch)
        block_rows = self._block_weight_rows(current, epoch)
        for item in block_rows:
            item["gradiente_primeiro_update_l2"] = None
            item["learning_rate"] = None
            item["weight_decay"] = None
        spline_rows = self._spline_rows(current, epoch, pl_module.model.mlp)
        if parameter_rows:
            csv_write(self.output_dir / "best_checkpoint_parameter_drift.csv", parameter_rows,
                      PARAMETER_DRIFT_COLUMNS)
        if block_rows:
            csv_write(self.output_dir / "best_checkpoint_dataset_block_weights.csv", block_rows,
                      BLOCK_DRIFT_COLUMNS)
        if spline_rows:
            csv_write(self.output_dir / "best_checkpoint_kan_spline_drift.csv", spline_rows,
                      SPLINE_DRIFT_COLUMNS)

        summary = {"debug/best_checkpoint/epoch": epoch}
        for item in parameter_rows:
            key = self._parameter_name(item["parametro"])
            for metric_name in ("atual_l2", "delta_l2", "delta_relativo", "similaridade_cosseno",
                                "fracao_alterada", "delta_maximo_abs", "fracao_nao_finita"):
                value = item.get(metric_name)
                if value is not None:
                    summary[f"debug/best_checkpoint/parametro/{key}/{metric_name}"] = value
        for item in block_rows:
            key = item["dataset_bloco"].lower()
            if item["delta_l2"] is not None:
                summary[f"debug/best_checkpoint/bloco_dataset/{key}/delta_l2"] = item["delta_l2"]
            if item["delta_relativo"] is not None:
                summary[f"debug/best_checkpoint/bloco_dataset/{key}/delta_relativo"] = item["delta_relativo"]
        json_write(self.output_dir / "best_checkpoint_diagnostics.json", {
            "epoch": epoch,
            "reference": "parameters loaded at the start of this dataset stage",
            "parameter_drift": parameter_rows,
            "dataset_block_weight_drift": block_rows,
            "kan_spline_drift": spline_rows,
        })
        if run is None:
            return summary

        run.summary.update(summary)
        import wandb
        payload = {"debug/epoca": epoch}
        if parameter_rows:
            table = wandb.Table(
                columns=["parametro", "delta_relativo", "delta_l2", "similaridade_cosseno"],
                data=[[item["parametro"], item["delta_relativo"], item["delta_l2"],
                       item["similaridade_cosseno"]] for item in parameter_rows],
            )
            payload["debug/best_checkpoint/pesos_por_parametro"] = table
            valid_rows = [item for item in parameter_rows if item["delta_relativo"] is not None]
            if valid_rows:
                plot_table = wandb.Table(
                    columns=["parametro", "delta_relativo"],
                    data=[[item["parametro"], item["delta_relativo"]] for item in valid_rows],
                )
                payload["debug/best_checkpoint/delta_por_parametro"] = wandb.plot.bar(
                    plot_table, "parametro", "delta_relativo",
                    title="Checkpoint selecionado — mudança relativa desde o checkpoint de origem",
                )
        if block_rows:
            table = wandb.Table(
                columns=["bloco_dataset", "delta_relativo", "delta_l2"],
                data=[[item["dataset_bloco"], item["delta_relativo"], item["delta_l2"]]
                      for item in block_rows],
            )
            payload["debug/best_checkpoint/pesos_por_bloco_dataset"] = table
            valid_rows = [item for item in block_rows if item["delta_relativo"] is not None]
            if valid_rows:
                plot_table = wandb.Table(
                    columns=["bloco_dataset", "delta_relativo"],
                    data=[[item["dataset_bloco"], item["delta_relativo"]] for item in valid_rows],
                )
                payload["debug/best_checkpoint/delta_por_bloco_dataset"] = wandb.plot.bar(
                    plot_table, "bloco_dataset", "delta_relativo",
                    title="Checkpoint selecionado — alteração dos pesos por bloco",
                )
        if hasattr(pl_module.model.mlp, "rk_layers"):
            for index, _ in enumerate(pl_module.model.mlp.rk_layers):
                path = self._spline_profile_figure(pl_module, index, current, epoch, best=True)
                payload[f"debug/best_checkpoint/splines/layer_{index}"] = wandb.Image(str(path))
            spline_table = wandb.Table(
                columns=["camada", "bloco_dataset", "low_delta_l2", "height_delta_l2",
                         "delta_largura_media"],
                data=[[item["camada"], item["dataset_bloco"], item["low_delta_l2"],
                       item["height_delta_l2"], item["delta_largura_media"]]
                      for item in spline_rows],
            )
            payload["debug/best_checkpoint/splines/tabela_mudanca"] = spline_table
        run.log(payload)
        run.save(str(self.output_dir / "*.csv"), base_path=str(self.output_dir), policy="now")
        run.save(str(self.output_dir / "best_checkpoint_diagnostics.json"),
                 base_path=str(self.output_dir), policy="now")
        return summary

    def teardown(self, trainer, pl_module, stage):
        for handle in self.handles:
            handle.remove()
        self.handles = []


RESULT_COLUMNS = ["etapa", "tipo", "treinado_em", "dataset_teste", "backbone", "head", "k", "g",
                  "dim_original", "dim_head", "multiplicacao", "fill", "checkpoint",
                  "exemplos", "srcc", "mse", "srcc_indefinido", "pred_constante",
                  "etapa_referencia", "srcc_antes", "mse_antes", "esquecimento_srcc",
                  "esquecimento_mse", "variacao_mse", "delta_srcc_etapa_anterior",
                  "delta_mse_etapa_anterior"]


def block_fill(model):
    """Padding actually used by the head: configured value, else the head's default."""
    configured = (model.get("dataset_block_expansion") or {}).get("padding_value")
    if configured is not None:
        return float(configured)
    if model.model_type == "dynamic_kan":
        return kan_block_padding(model.get("relukan_grid", 3), model.get("relukan_k", 3))
    return 0.0


def result_rows(stages, config):
    """First own-dataset best checkpoint is the baseline, never a future checkpoint."""
    baselines, previous, rows = {}, {}, []
    for stage in stages:
        for dataset, measurement in stage.get("evaluations", {}).items():
            if dataset not in baselines and dataset == stage["dataset"]:
                baselines[dataset] = (stage["index"], measurement)
            reference_index, reference = baselines.get(dataset, (None, {}))
            before = reference.get("srcc")
            mse_before = reference.get("mse")
            srcc, mse = measurement["srcc"], measurement["mse"]
            last = previous.get(dataset, {})
            model = config.model
            delta_mse = mse - mse_before if mse_before is not None else None
            rows.append(dict(zip(RESULT_COLUMNS, [
                stage["index"], "base" if stage["index"] == 1 else "fine-tuning",
                stage["dataset"], dataset, model.model_name,
                "kan" if model.model_type == "dynamic_kan" else "mlp",
                model.get("relukan_k") if model.model_type == "dynamic_kan" else None,
                model.get("relukan_grid") if model.model_type == "dynamic_kan" else None,
                int(model.mlp_input_dim), 4 * int(model.mlp_input_dim), 4,
                block_fill(model),
                stage["best_checkpoint"], measurement["num_examples"], srcc, mse,
                measurement["srcc_undefined"], measurement["constant_predictions"],
                reference_index, before, mse_before,
                before - srcc if before is not None and srcc is not None else None,
                delta_mse, delta_mse,
                last["srcc"] - srcc if last.get("srcc") is not None and srcc is not None else None,
                mse - last["mse"] if last.get("mse") is not None else None,
            ])))
            previous[dataset] = measurement
    return rows
