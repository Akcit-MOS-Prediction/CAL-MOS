"""Small, local CSV outputs for sequential head training (no background services)."""
import csv
import json
import math
import os
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


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


class SequenceOutputs(Callback):
    """Capture the ACTUAL first learned layer input, once per training epoch."""
    def __init__(self, output_dir, save_head_inputs=True):
        self.output_dir = Path(output_dir)
        self.save_head_inputs = save_head_inputs
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
        self.padding_value = float(router.padding_value)
        if tuple(router.start_positions) != tuple(i * head.input_size for i in range(4)):
            raise ValueError("Expected four disjoint dataset blocks")
        if router.padding_value != 0:
            raise ValueError("Dataset blocks must be padded with zero")
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
                        "kan_architecture_changed": False}
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
                    if not pl_module.training or self.kan_epochs.get(layer_index) == trainer.current_epoch:
                        return
                    values = args[0].detach()
                    active = ((values.unsqueeze(-1) > module.phase_low) &
                              (values.unsqueeze(-1) < module.phase_height))
                    # Layer 0: measure the real (sigmoid) inputs only, not the zero-padded positions.
                    real = (values != self.padding_value) if layer_index == 0 else torch.ones_like(values, dtype=torch.bool)
                    self.kan_support[layer_index] = {
                        "fora_de_todas_bases": float((~active.any(-1))[real].float().mean()),
                        "entrada_min": float(values[real].min()), "entrada_max": float(values[real].max()),
                        "support_min": float(module.phase_low.min()),
                        "support_max": float(module.phase_height.max()),
                    }
                    self.snapshot[f"kan_layer_{layer_index}_input"] = values[0].float().cpu()
                    self.kan_epochs[layer_index] = trainer.current_epoch
                self.handles.append(layer.register_forward_pre_hook(capture_support))

    def on_train_epoch_start(self, trainer, pl_module):
        self.gradient_norm = None
        self.parameter_gradients = {}
        self.kan_support = {}

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

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        predictions = self.val_predictions
        constant = len(predictions) > 1 and max(predictions) - min(predictions) <= 1e-6
        row = {"epoca": trainer.current_epoch + 1, "passo": trainer.global_step,
               "lr": trainer.optimizers[0].param_groups[0]["lr"],
               "grad_norm_primeiro_update": self.gradient_norm,
               "val_mse": metric(trainer.callback_metrics.get("val/mse")),
               "val_srcc": metric(trainer.callback_metrics.get("val/spearman")),
               "pred_min": min(predictions) if predictions else None,
               "pred_max": max(predictions) if predictions else None,
               "pred_constante": int(constant)}
        self.rows.append(row)
        csv_write(self.output_dir / "epochs.csv", self.rows, list(row))
        run = self._wandb_run(trainer)
        if run is not None:
            self._log_debug(run, trainer, pl_module, row)
        if constant:
            print("[WARNING] Validation predictions are constant; inspect KAN support/gradients. "
                  "No architectural fix is applied automatically.", flush=True)

    def _log_debug(self, run, trainer, pl_module, row):
        import numpy as np
        import wandb
        values = self.head_input.numpy()
        width = len(values) // 4
        names = ["BVCC", "SingMOS", "TMHINT", "BRSpeech"]
        blocks = values.reshape(4, width)
        vector_table = wandb.Table(columns=["indice", "valor", "bloco"],
                                   data=[[i, float(value), names[i // width]] for i, value in enumerate(values)])
        padding = getattr(self, "padding_value", 0.0)
        block_table = wandb.Table(columns=["dataset", "norma_L2", "fracao_nao_padding", "grad_norm"],
                                  data=[[name, float(np.linalg.norm(block)), float(np.mean(block != padding)),
                                         self.grad_blocks[i] if self.grad_blocks else 0.0]
                                        for i, (name, block) in enumerate(zip(names, blocks))])
        prediction_table = wandb.Table(columns=["MOS_real", "MOS_predito"],
                                       data=[[float(y), float(p)] for y, p in zip(self.val_targets, self.val_predictions)])
        gradient_table = wandb.Table(columns=["parametro", "grad_norm"],
                                     data=[[name, value] for name, value in self.parameter_gradients.items()])
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
                   "debug/predicao_vs_real": wandb.plot.scatter(prediction_table, "MOS_real", "MOS_predito", title="Validação completa — MOS real × predito"),
                   "debug/gradientes_parametros": wandb.plot.bar(gradient_table, "parametro", "grad_norm", title="Gradientes — primeiro update da época"),
                   "debug/pred_std": float(np.std(self.val_predictions)),
                   "debug/pred_range": float(np.ptp(self.val_predictions)),
                   "debug/target_std": float(np.std(self.val_targets)),
                   "debug/hist_head_input": wandb.Histogram(values),
                   "debug/hist_normalizado": wandb.Histogram(self.snapshot["normalized"].numpy()),
                   "debug/hist_pooled": wandb.Histogram(self.snapshot["pooled"].numpy())}
        for index, support in self.kan_support.items():
            payload.update({f"debug/kan_camada_{index}/{key}": value for key, value in support.items()})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.output_dir / f"debug_epoch_{trainer.current_epoch + 1:03d}.npz",
                            **{key: value.numpy() if isinstance(value, torch.Tensor) else value
                               for key, value in self.snapshot.items()},
                            validation_targets=np.asarray(self.val_targets), validation_predictions=np.asarray(self.val_predictions))
        json_write(self.output_dir / f"debug_epoch_{trainer.current_epoch + 1:03d}.json",
                   {"epoch": trainer.current_epoch + 1, "metrics": row,
                    "kan_support": self.kan_support, "gradient_norms": self.parameter_gradients})
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
        axes[1, 0].scatter(self.val_targets, self.val_predictions, s=8, alpha=.4, color="#7C3AED")
        lower = min(self.val_targets + self.val_predictions)
        upper = max(self.val_targets + self.val_predictions)
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
    return float(configured) if configured is not None else 0.0


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
