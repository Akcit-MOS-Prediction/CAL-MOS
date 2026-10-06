"""Per-epoch validation and training-cost metrics for forgetting runs."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() != 1:
            return None
        value = value.item()
    number = float(value)
    return number if math.isfinite(number) else None


class ForgettingMetrics(Callback):
    def __init__(self, output_dir: str, val_loss_target: float, budget_seconds: float):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.val_loss_target = float(val_loss_target)
        self.budget_seconds = float(budget_seconds)
        self.rows: list[dict] = []
        self.started = None
        self.epoch_started = None
        self.train_ended = None
        self.epoch_samples = 0
        self.total_samples = 0
        self.total_train_seconds = 0.0
        self.target_time = None
        self.target_epoch = None
        self.best_in_budget = None
        self.total_parameters = None
        self.trainable_parameters = None
        self.device = None

    def on_train_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.total_parameters = sum(parameter.numel() for parameter in pl_module.model.parameters())
        self.trainable_parameters = sum(parameter.numel() for parameter in pl_module.model.parameters()
                                        if parameter.requires_grad)
        self.device = pl_module.device
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._save("running")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started = time.perf_counter()
        self.train_ended = None
        self.epoch_samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        targets = batch[1]
        count = int(targets.shape[0])
        self.epoch_samples += count
        self.total_samples += count

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking and self.epoch_started is not None and self.train_ended is None:
            self.train_ended = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.started is None or self.epoch_started is None:
            return
        current = time.perf_counter()
        elapsed = current - self.started
        train_seconds = max(0.0, (self.train_ended or current) - self.epoch_started)
        self.total_train_seconds += train_seconds
        metrics = trainer.callback_metrics
        val_loss = scalar(metrics.get("val/loss"))
        val_srcc = scalar(metrics.get("val/spearman"))
        if val_loss is not None:
            if self.target_time is None and val_loss <= self.val_loss_target:
                self.target_time = elapsed
                self.target_epoch = trainer.current_epoch + 1
            if elapsed <= self.budget_seconds:
                self.best_in_budget = (val_loss if self.best_in_budget is None
                                       else min(self.best_in_budget, val_loss))
        peak_mb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                   if self.device.type == "cuda" else None)
        self.rows.append({
            "validation_index": len(self.rows) + 1,
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "val_loss": val_loss,
            "val_srcc": val_srcc,
            "elapsed_seconds": elapsed,
            "epoch_seconds": current - self.epoch_started,
            "train_seconds": train_seconds,
            "train_samples": self.epoch_samples,
            "train_samples_per_second": (
                self.epoch_samples / train_seconds if train_seconds > 0 else None
            ),
            "gpu_peak_allocated_mb": peak_mb,
        })
        self._save("running")
        self.epoch_started = None
        self.train_ended = None

    def on_fit_end(self, trainer, pl_module):
        self._save("finished")

    def _save(self, status: str):
        loss_rows = [row for row in self.rows if row["val_loss"] is not None]
        srcc_rows = [row for row in self.rows if row["val_srcc"] is not None]
        epoch_seconds = [row["epoch_seconds"] for row in self.rows]
        epochs_trained = max((row["epoch"] for row in self.rows), default=0)
        summary = {
            "status": status,
            "val_loss_target": self.val_loss_target,
            "budget_seconds": self.budget_seconds,
            "target_status": "reached" if self.target_time is not None else "not_reached",
            "time_to_target_seconds": self.target_time,
            "epoch_to_target": self.target_epoch,
            "best_val_loss_within_budget": self.best_in_budget,
            "budget_status": ("observed" if self.best_in_budget is not None
                              else "no_validation_within_budget"),
            "final_val_loss": self.rows[-1]["val_loss"] if self.rows else None,
            "final_val_srcc": self.rows[-1]["val_srcc"] if self.rows else None,
            "best_val_loss": min((row["val_loss"] for row in loss_rows), default=None),
            "best_val_srcc": max((row["val_srcc"] for row in srcc_rows), default=None),
            "best_val_loss_epoch": min(loss_rows, key=lambda row: row["val_loss"])["epoch"]
            if loss_rows else None,
            "best_val_srcc_epoch": max(srcc_rows, key=lambda row: row["val_srcc"])["epoch"]
            if srcc_rows else None,
            "epochs_trained": epochs_trained,
            "epochs_to_converge": self.target_epoch,
            "convergence_status": "reached" if self.target_epoch is not None else "not_reached",
            "validation_count": len(self.rows),
            "fit_elapsed_seconds": (time.perf_counter() - self.started
                                    if self.started is not None else None),
            "mean_epoch_seconds": (sum(epoch_seconds) / len(epoch_seconds)
                                   if epoch_seconds else None),
            "mean_train_epoch_seconds": (self.total_train_seconds / epochs_trained
                                         if epochs_trained else None),
            "total_train_samples": self.total_samples,
            "total_train_seconds": self.total_train_seconds,
            "train_samples_per_second": (
                self.total_samples / self.total_train_seconds
                if self.total_train_seconds > 0 else None
            ),
            "gpu_peak_allocated_mb": (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.device is not None and self.device.type == "cuda" else None
            ),
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
        }
        atomic_json(self.output_dir / "training_metrics.json", summary)
