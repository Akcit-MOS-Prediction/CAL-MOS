#!/usr/bin/env python3
"""Configurable BVCC -> SingMOS -> TMHINT -> BRSpeech -> BVCC head training."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

from omegaconf import OmegaConf

from utils.sequence_outputs import RESULT_COLUMNS, SequenceOutputs, csv_write, json_write, result_rows
from utils.dataset_identity import DATASET_IDS, kan_block_padding

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = ("bvcc", "singmos", "tmhint", "brspeech", "bvcc")


def relative_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def kan_padding_value(model):
    """Block padding strictly below the ReLU-KAN grid (defaults match ReLuKANBaseModel: g=3, k=3)."""
    return kan_block_padding(model.get("relukan_grid", 3), model.get("relukan_k", 3))


def load_plan(protocol_path, config_paths=None):
    protocol_path = Path(protocol_path).resolve(strict=True)
    protocol = OmegaConf.load(protocol_path)
    if tuple(protocol.get("sequence", SEQUENCE)) != SEQUENCE:
        raise ValueError(f"This protocol requires sequence={SEQUENCE}")
    paths = ([Path(p).resolve(strict=True) for p in config_paths] if config_paths else
             [relative_path(p, protocol_path.parent) for p in protocol.configs])
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide distinct model config paths")
    datasets = {name: OmegaConf.load(relative_path(protocol.datasets[name], protocol_path.parent))
                for name in DATASET_IDS}
    plans = []
    for index, path in enumerate(paths, 1):
        base = OmegaConf.load(path)
        stage_configs = []
        for name in SEQUENCE:
            # Dataset defines paths/default data options; model settings win over its batch defaults.
            config = OmegaConf.merge(datasets[name], base, protocol.get("overrides", {}))
            config.datasets = datasets[name].datasets
            config.only_mlp = True
            config.debug = False
            config.seed = int(config.get("seed", protocol.get("seed", 42)))
            model = config.model
            if model.model_type not in {"dynamic", "dynamic_kan"}:
                raise ValueError(f"{path}: only dynamic MLP/KAN configs are supported")
            model.freeze_backbone = True
            model.use_peft = False
            model.use_sr_embeddings = False
            model.use_layer_norm = True
            model.layer_norm_affine = False
            # Any old compact expansion, fill=0.1 or block_width is intentionally replaced.
            # MLP: zero padding. KAN: padding below the ReLU-KAN grid (inputs are sigmoid(LN) in
            # (0, 1)), so padded positions have zero basis, contribution and gradient.
            model.dataset_block_expansion = {
                "enabled": True, "num_blocks": 4, "output_multiplier": 4.0,
                "padding_value": kan_padding_value(model) if model.model_type == "dynamic_kan" else 0.0,
                "dataset_to_id": DATASET_IDS,
            }
            if model.get("input_power_transform"):
                raise ValueError("Remove input_power_transform: this protocol uses non-affine LayerNorm only")
            if int(model.mlp_input_dim) < 1 or int(config.data.num_classes) != 1:
                raise ValueError("Expected a positive backbone dimension and scalar MOS output")
            if model.get("mlp_output_size", 1) != 1:
                raise ValueError("The head must have one MOS output")
            for split in ("train", "val", "test"):
                if len(config.datasets[split]) != 1:
                    raise ValueError("Exactly one split metadata file is required")
                config.datasets[split][0].dataset_name = name
            config.trainer.val_check_interval = 1.0
            config.trainer.check_val_every_n_epoch = 1
            if config.trainer.get("limit_val_batches", 1.0) == 0:
                raise ValueError("Validation cannot be disabled: best checkpoint depends on it")
            if config.trainer.get("fast_dev_run", False):
                raise ValueError("fast_dev_run disables best checkpoint saving; use small epoch/batch limits instead")
            if config.get("scheduler") and str(config.scheduler.name).lower() not in {
                    "cosinewarmuplr", "reducelronplateau", "linearlr"}:
                raise ValueError("Unsupported scheduler; use CosineWarmupLR, ReduceLROnPlateau, LinearLR or null")
            if "scheduler" not in config:
                config.scheduler = None
            if config.optimizer.name.lower() not in {"adam", "adamw", "lion"}:
                raise ValueError("Unsupported optimizer")
            lr = float(config.optimizer.params.learning_rate)
            minimum = float(config.optimizer.params.get("min_learning_rate", min(lr, 1e-6)))
            if not 0 <= minimum <= lr or not math.isfinite(lr) or lr <= 0:
                raise ValueError("Expected 0 <= min_learning_rate <= learning_rate, learning_rate > 0")
            config.optimizer.params.min_learning_rate = minimum
            config.title = f"sequence-{path.stem}-{name}"
            OmegaConf.resolve(config)
            stage_configs.append(config)
        plans.append((f"{index:02d}-{path.stem}", path, stage_configs))
    return protocol, plans


def preflight(plans):
    """Fail before training if a later dataset is unavailable/malformed."""
    import pandas as pd
    checked = set()
    for _, _, configs in plans:
        for config in configs:
            for entries in config.datasets.values():
                for split in entries:
                    path = Path(split.metadata_path)
                    key = (str(path), split.filename_column, split.target_column, split.base_dir)
                    if key in checked:
                        continue
                    checked.add(key)
                    if not Path(split.base_dir).is_dir():
                        raise FileNotFoundError(split.base_dir)
                    data = pd.read_csv(path)
                    if len(data) < 2 or not {split.filename_column, split.target_column} <= set(data.columns):
                        raise ValueError(f"Invalid/empty MOS metadata: {path}")
                    if not pd.to_numeric(data[split.target_column], errors="coerce").map(math.isfinite).all():
                        raise ValueError(f"Nonfinite/non-numeric MOS labels: {path}")
                    if not (Path(split.base_dir) / str(data.iloc[0][split.filename_column])).is_file():
                        raise FileNotFoundError(f"First audio is missing: {path}")


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".sequence.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_stage(config, stage_dir, previous_checkpoint, gpu, protocol):
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from models.calmos_wrapper import CALMOSWrapper

    seed_everything(int(config.seed), workers=True)
    attempt = stage_dir / "attempts" / uuid.uuid4().hex[:12]
    attempt.mkdir(parents=True)
    OmegaConf.save(config, attempt / "config.yaml")
    monitor = config.model_checkpoint.get("monitor", "val/spearman")
    if monitor not in {"val/loss", "val/mse", "val/spearman", "val/pearson"}:
        raise ValueError(f"Unknown best-checkpoint monitor: {monitor}")
    mode = config.model_checkpoint.get("mode", "max" if monitor in {"val/spearman", "val/pearson"} else "min")
    checkpoint = ModelCheckpoint(dirpath=str(attempt / "checkpoints"), monitor=monitor, mode=mode,
                                 save_top_k=1, save_last=False, save_weights_only=True,
                                 filename="best-{epoch:03d}-{step}", auto_insert_metric_name=False)
    stop = OmegaConf.to_container(config.get("early_stopping", {}), resolve=True)
    stop.update(monitor=monitor, mode=mode, check_on_train_epoch_end=False)
    stop.setdefault("patience", 25)
    stage_dataset = stage_dir.name.split("-", 1)[-1].lower()
    outputs = SequenceOutputs(attempt, save_head_inputs=protocol.get("save_head_inputs", True),
                              active_dataset=stage_dataset)
    callbacks = [checkpoint, EarlyStopping(**stop), LearningRateMonitor("step"), outputs]
    loggers = [CSVLogger(str(attempt), name="lightning")]
    wandb_options = protocol.get("wandb", {})
    wandb_logger = None
    if wandb_options.get("enabled", False):
        head_label = (f"kan-k{config.model.relukan_k}-g{config.model.relukan_grid}"
                      if config.model.model_type == "dynamic_kan" else "mlp")
        backbone_label = config.model.model_name.split("/")[-1].lower()
        wandb_logger = WandbLogger(
            project=wandb_options.get("project", "MOS-Prediction"),
            entity=wandb_options.get("entity", config.get("wandb_entity")),
            name=f"{backbone_label}-{head_label}-{stage_dir.name}",
            group=wandb_options.get("group", stage_dir.parent.name),
            tags=["sequencial", "blocos4", "layernorm-nao-afim", stage_dir.name,
                  config.model.model_name, config.model.model_type],
            save_dir=str(attempt), offline=wandb_options.get("offline", False),
        )
        wandb_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
        run = wandb_logger.experiment
        parent_run = {}
        if previous_checkpoint:
            parent_run_path = Path(previous_checkpoint).parent.parent / "wandb_run.json"
            if parent_run_path.is_file():
                parent_run = load_json(parent_run_path)
        run.config.update({"debug_stage_dataset": stage_dataset,
                           "initialized_from_checkpoint": previous_checkpoint,
                           "initialized_from_wandb_run_id": parent_run.get("id"),
                           "initialized_from_wandb_run_url": parent_run.get("url"),
                           "parameter_drift_reference": "loaded head weights at the beginning of this stage"},
                          allow_val_change=True)
        json_write(attempt / "wandb_run.json", {"id": run.id, "url": run.url,
                                                 "entity": run.entity, "project": run.project})
        run.save(str(attempt / "config.yaml"), base_path=str(attempt), policy="now")
        loggers.append(wandb_logger)
    model = (CALMOSWrapper.load_from_checkpoint(str(previous_checkpoint), config=config,
                                               map_location="cpu", strict=True)
             if previous_checkpoint else CALMOSWrapper(config))
    initial_head = None
    if previous_checkpoint is None and protocol.get("comparison", {}).get("enabled", True):
        initial_head = attempt / "initial_head.pt"
        torch.save({"head_state": {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                                   for key, value in model.model.mlp.state_dict().items()}}, initial_head)
    trainer_options = OmegaConf.to_container(config.trainer, resolve=True)
    for reserved in ("devices", "logger", "callbacks", "default_root_dir", "strategy", "num_nodes"):
        trainer_options.pop(reserved, None)
    trainer_options["devices"] = 1 if gpu == "cpu" else [int(gpu)]
    trainer_options["accelerator"] = "cpu" if gpu == "cpu" else "gpu"
    trainer = Trainer(**trainer_options, callbacks=callbacks, logger=loggers,
                      default_root_dir=str(attempt))
    try:
        # Fresh optimizer/scheduler every stage; only weights come from the previous BEST.
        trainer.fit(model)
        best = Path(checkpoint.best_model_path)
        if not checkpoint.best_model_path or not best.is_file():
            raise RuntimeError("Training produced no valid best checkpoint")
        if checkpoint.best_model_score is None or not math.isfinite(float(checkpoint.best_model_score)):
            raise RuntimeError("Best validation metric is not finite")
        result = {"best_checkpoint": str(best.resolve()), "initialized_from": previous_checkpoint,
                  "monitor": monitor, "mode": mode, "score": float(checkpoint.best_model_score),
                  "attempt": str(attempt), "seed": int(config.seed)}
        if initial_head is not None:
            result["initial_head_checkpoint"] = str(initial_head.resolve())
        outputs.log_best_checkpoint(
            model, best, wandb_logger.experiment if wandb_logger is not None else None,
            getattr(checkpoint, "best_epoch", None),
        )
        debug_files = {
            "parameter_drift": str((attempt / "best_checkpoint_parameter_drift.csv").resolve()),
            "dataset_block_weights": str((attempt / "best_checkpoint_dataset_block_weights.csv").resolve()),
            "summary": str((attempt / "best_checkpoint_diagnostics.json").resolve()),
        }
        spline_diagnostics = attempt / "best_checkpoint_kan_spline_drift.csv"
        if spline_diagnostics.is_file():
            debug_files["kan_spline_drift"] = str(spline_diagnostics.resolve())
        result["debug_diagnostics"] = debug_files
        if wandb_logger is not None:
            result["wandb"] = load_json(attempt / "wandb_run.json")
            wandb_logger.experiment.summary.update({"best_checkpoint": str(best.resolve()),
                                                    "best_validation_score": result["score"],
                                                    "initialized_from": previous_checkpoint})
        json_write(stage_dir / "trained.json", result)
        return result
    finally:
        if wandb_logger is not None:
            import wandb
            wandb.finish()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(config, checkpoint, output_dir, gpu, batch_size, probe_path=None, probe_samples=64):
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from transformers import AutoFeatureExtractor
    from models.calmos_wrapper import CALMOSWrapper
    from utils.dataloader import DynamicDataset, DynamicCollate
    from eval.forgetting import move_to_device

    split = config.datasets.test[0]
    data = pd.read_csv(split.metadata_path)
    dataset = DynamicDataset(data, base_dir=Path(split.base_dir),
                             filename_column=split.filename_column, target_column=split.target_column,
                             sr_column=split.get("sr_column"), sr_dictionary=config.data.get("sr_dictionary"),
                             data_type="test", class_num=1, target_sr=config.data.target_sr,
                             dataset_id=DATASET_IDS[split.dataset_name])
    collate = DynamicCollate(target_sr=config.data.target_sr,
                             processor=AutoFeatureExtractor.from_pretrained(config.model.model_name))
    worker_count = int(config.train.num_workers)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        multiprocessing_context="spawn" if worker_count > 0 else None,
        collate_fn=collate,
    )
    device = torch.device("cpu" if gpu == "cpu" else f"cuda:{gpu}")
    model = CALMOSWrapper.load_from_checkpoint(str(checkpoint), config=config,
                                               map_location="cpu", strict=True).to(device).eval()
    collector = None
    if probe_path is not None and not Path(probe_path).exists():
        from utils.sequence_comparison import ProbeCollector
        collector = ProbeCollector(model.model.mlp, probe_samples)
    predicted, observed = [], []
    try:
        with torch.inference_mode():
            for features, targets in loader:
                predicted.extend(model(move_to_device(features, device)).float().cpu().reshape(-1).tolist())
                observed.extend(targets.float().cpu().reshape(-1).tolist())
        if len(predicted) != len(data) or len(predicted) < 2:
            raise ValueError("Missing test predictions")
        if not all(math.isfinite(v) for v in predicted + observed):
            raise ValueError("Nonfinite test predictions/targets")
        constant = max(predicted) - min(predicted) <= 1e-6
        srcc = float(spearmanr(observed, predicted).statistic)
        undefined = not math.isfinite(srcc)
        mse = math.fsum((p - y) ** 2 for p, y in zip(predicted, observed)) / len(observed)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_write(output_dir / "predictions.csv", (
            {"indice_metadata": index, "arquivo": str(data.iloc[index][split.filename_column]),
             "dataset": split.dataset_name, "dataset_id": DATASET_IDS[split.dataset_name],
             "mos_real": target, "mos_predito": prediction, "erro_quadratico": (prediction - target) ** 2}
            for index, (prediction, target) in enumerate(zip(predicted, observed))
        ), ["indice_metadata", "arquivo", "dataset", "dataset_id", "mos_real", "mos_predito", "erro_quadratico"])
        result = {"srcc": None if undefined else srcc, "mse": mse,
                  "num_examples": len(observed), "constant_predictions": constant,
                  "srcc_undefined": undefined, "checkpoint": str(checkpoint),
                  "test_metadata_path": str(split.metadata_path),
                  "test_metadata_sha256": hashlib.sha256(Path(split.metadata_path).read_bytes()).hexdigest()}
        json_write(output_dir / "metrics.json", result)
        if collector is not None:
            collector.save(probe_path, observed, data[split.filename_column].astype(str).tolist(), {
                "dataset": split.dataset_name, "split": "test",
                "test_metadata_sha256": result["test_metadata_sha256"],
                "backbone": config.model.model_name, "input_dim": int(config.model.mlp_input_dim),
                "selection": "first N metadata rows in deterministic test order",
                "used_for_model_selection": False,
            })
        return result
    finally:
        if collector is not None:
            collector.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_experiment(directory, configs, protocol, gpu, resume=False):
    if directory.exists() and not resume:
        raise FileExistsError(f"{directory} exists. Use --resume or a different --output.")
    directory.mkdir(parents=True, exist_ok=True)
    resolved = [OmegaConf.to_container(config, resolve=True) for config in configs]
    signature_protocol = OmegaConf.to_container(protocol, resolve=True)
    signature_protocol.pop("comparison", None)  # Operational reporting options do not change training lineage.
    signature = hashlib.sha256(json.dumps({"configs": resolved, "protocol":
        signature_protocol}, sort_keys=True).encode()).hexdigest()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path)["signature"] != signature:
        raise ValueError("Resume config/protocol changed; use a new output directory")
    json_write(manifest_path, {"signature": signature, "sequence": SEQUENCE,
                             "dataset_ids": DATASET_IDS, "csv_separator": ";", "decimal": ","})
    previous = None
    stages, seen = [], []
    evaluations = {}
    for index, (name, config) in enumerate(zip(SEQUENCE, configs), 1):
        evaluations.setdefault(name, config)
        if name not in seen:
            seen.append(name)
        stage_dir = directory / f"{index:02d}-{name}"
        stage_dir.mkdir(exist_ok=True)
        OmegaConf.save(config, stage_dir / "config.yaml")
        trained_path = stage_dir / "trained.json"
        trained = (load_json(trained_path) if resume and trained_path.exists() else
                   train_stage(config, stage_dir, previous, gpu, protocol))
        if trained["initialized_from"] != previous:
            raise ValueError("Saved checkpoint lineage does not match previous best")
        if not Path(trained["best_checkpoint"]).is_file():
            raise FileNotFoundError(trained["best_checkpoint"])
        previous = trained["best_checkpoint"]
        stage = {"index": index, "dataset": name, **trained, "evaluations": {}}
        stages.append(stage)
        for tested in seen:
            output_dir = stage_dir / "evaluations" / tested
            metrics_path = output_dir / "metrics.json"
            comparison = protocol.get("comparison", {})
            probe_options = ({"probe_path": directory / "comparison/probes" / f"{tested}.npz",
                              "probe_samples": int(comparison.get("probe_samples", 64))}
                             if comparison.get("enabled", True) else {})
            measurement = (load_json(metrics_path) if resume and metrics_path.exists() else
                           evaluate(evaluations[tested], previous, output_dir, gpu,
                                    int(protocol.get("inference_batch_size", config.train.batch_size)), **probe_options))
            split = evaluations[tested].datasets.test[0]
            if measurement["checkpoint"] != previous or measurement["test_metadata_sha256"] != hashlib.sha256(
                    Path(split.metadata_path).read_bytes()).hexdigest():
                raise ValueError("Saved evaluation checkpoint/test split changed")
            stage["evaluations"][tested] = measurement
            json_write(directory / "progress.json", {"status": "running", "stages": stages})
            csv_write(directory / "results.csv", result_rows(stages, configs[0]), RESULT_COLUMNS)
        print(f"[Sequence] completed stage {index}/5: {name}; best={previous}", flush=True)
        if protocol.get("wandb", {}).get("enabled", False) and trained.get("wandb"):
            import wandb
            tracking = trained["wandb"]
            with wandb.init(entity=tracking["entity"], project=tracking["project"],
                            id=tracking["id"], resume="must") as run:
                rows = result_rows(stages, configs[0])
                run.log({"tests/etapa": index, "tests/resultados": wandb.Table(
                    columns=RESULT_COLUMNS, data=[[row[key] for key in RESULT_COLUMNS] for row in rows]),
                    **{f"tests/{dataset}/{key}": measurement[key]
                       for dataset, measurement in stage["evaluations"].items()
                       for key in ("mse", "srcc", "constant_predictions") if measurement[key] is not None}})
                run.save(str(directory / "results.csv"), base_path=str(directory), policy="now")
    json_write(directory / "progress.json", {"status": "finished", "stages": stages})
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(ROOT / "config/sequential.yaml"))
    parser.add_argument("--config", action="append", help="Model/full YAML; repeat for multiple experiments")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default="0", help="Logical GPU index visible inside the container, or cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate all configs/paths without loading a backbone")
    args = parser.parse_args()
    if args.gpu != "cpu" and (not args.gpu.isdigit() or int(args.gpu) < 0):
        parser.error("--gpu must be a non-negative logical GPU index, or cpu")
    protocol, plans = load_plan(args.protocol, args.config)
    preflight(plans)
    for label, path, configs in plans:
        print(f"{label}: {path}; {SEQUENCE}; D={configs[0].model.mlp_input_dim} "
              f"-> {4 * int(configs[0].model.mlp_input_dim)}; head={configs[0].model.model_type}")
    if args.dry_run:
        print("Dry run OK: no model loaded, no training started, no output directory created.")
        return
    with output_lock(args.output.resolve()):
        failures = []
        for label, _, configs in plans:
            try:
                run_experiment(args.output.resolve() / label, configs, protocol, args.gpu, args.resume)
            except Exception as exc:
                failures.append({"experiment": label, "error": f"{type(exc).__name__}: {exc}"})
                json_write(args.output / "failures.json", failures)
                print(f"[ERROR] {label}: {exc}", file=sys.stderr, flush=True)
                if not protocol.get("continue_on_error", True):
                    raise
            finally:
                combined = []
                for saved_label, _, saved_configs in plans:
                    progress = args.output / saved_label / "progress.json"
                    if progress.exists():
                        combined.extend({"experimento": saved_label, **row} for row in result_rows(
                            load_json(progress)["stages"], saved_configs[0]))
                csv_write(args.output / "results.csv", combined, ["experimento"] + RESULT_COLUMNS)
        try:
            from utils.sequence_comparison import schedule_comparison
            schedule_comparison(args.output.resolve(), [args.output.resolve() / label for label, _, _ in plans],
                                OmegaConf.to_container(protocol, resolve=True))
        except Exception as exc:
            # Diagnostic failure must not turn a successfully trained sequence into a failed training task.
            json_write(args.output / "comparison/status.json", {"status": "schedule_failed", "error": str(exc)})
            print(f"[Comparison] Could not schedule diagnostics: {exc}", file=sys.stderr, flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
