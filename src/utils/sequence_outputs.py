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

    def on_fit_start(self, trainer, pl_module):
        head = pl_module.model.mlp
        router = head.block_expansion
        if router is None or router.output_size != 4 * head.input_size:
            raise ValueError("Expected a non-affine LayerNorm followed by exactly four blocks")
        if router.padding_value != 0 or tuple(router.start_positions) != tuple(
                i * head.input_size for i in range(4)):
            raise ValueError("Expected disjoint dataset blocks padded with zero")
        first = head.rk_layers[0] if hasattr(head, "rk_layers") else head.layers[0]

        def capture_router(module, args, output):
            if pl_module.training and self.epoch != trainer.current_epoch:
                self.route_input = (args[0].detach(), args[1].detach())

        def capture_head(module, args):
            if not pl_module.training or self.epoch == trainer.current_epoch:
                return
            actual = args[0].detach()
            normalized, ids = self.route_input
            width = head.input_size
            expected = torch.zeros_like(actual)
            indices = (ids[:, None] - 1) * width + torch.arange(width, device=actual.device)
            expected.scatter_(1, indices, normalized)
            if actual.shape[1] != 4 * width or not torch.isfinite(actual).all():
                raise ValueError("Invalid/nonfinite vector entering the head")
            if not torch.equal(actual, expected):
                raise ValueError("Head input differs from the expected normalized dataset block")
            if self.save_head_inputs:
                values = actual[0].float().cpu().tolist()
                csv_write(self.output_dir / "head_inputs.csv", (
                    {"epoca": trainer.current_epoch + 1, "dataset_id": int(ids[0]),
                     "indice": index, "bloco": index // width + 1, "valor": value}
                    for index, value in enumerate(values)
                ), ["epoca", "dataset_id", "indice", "bloco", "valor"], append=True)
            print(f"[Sequence] epoch={trainer.current_epoch + 1} normalized={tuple(normalized.shape)} "
                  f"head_input={tuple(actual.shape)} dataset_ids={ids.tolist()} "
                  "routing_error=0 padding=0", flush=True)
            self.epoch = trainer.current_epoch
            self.route_input = None

        def capture_validation(module, args, output):
            if trainer.validating and not trainer.sanity_checking:
                self.val_predictions.extend(output.detach().float().cpu().reshape(-1).tolist())

        self.handles = [router.register_forward_hook(capture_router),
                        first.register_forward_pre_hook(capture_head),
                        pl_module.model.register_forward_hook(capture_validation)]

    def on_train_epoch_start(self, trainer, pl_module):
        self.gradient_norm = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.gradient_norm is None:
            gradients = [p.grad.detach().float().square().sum()
                         for p in pl_module.model.mlp.parameters() if p.grad is not None]
            self.gradient_norm = metric(torch.stack(gradients).sum().sqrt()) if gradients else 0.0

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
        if constant:
            print("[WARNING] Validation predictions are constant; inspect KAN support/gradients. "
                  "No architectural fix is applied automatically.", flush=True)

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
                int(model.mlp_input_dim), 4 * int(model.mlp_input_dim), 4, 0,
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
