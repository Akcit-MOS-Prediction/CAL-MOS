"""Treina os seis modelos MLP/KAN usando os quatro datasets concatenados.

Os modelos mHuBERT-147, wav2vec2 XLS-R 1B e wav2vec2 XLS-R 300M sao carregados
diretamente dos YAMLs MLP/KAN em ``config/models``. Os CSVs de train/val/test
de BRSPEECH, BVCC, SingMOS e TMHINTQI sao concatenados em arquivos de trabalho
com caminhos de audio absolutos. Os modelos sao distribuidos entre os workers
das GPUs informadas e o melhor checkpoint e escolhido pelo ``val/spearman``
agregado.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import multiprocessing as mp
import os
import queue
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\._internal\._generate_schema",
)
warnings.filterwarnings(
    "ignore",
    message=r".*UnsupportedFieldAttributeWarning.*",
    category=UserWarning,
)

import lightning as L
import pandas as pd
import yaml
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(__file__))

from eval.inference import run_inference
from main import run_train


DATASET_CONFIGS = {
    "brspeech": "config/datasets/brspeech.yaml",
    "bvcc": "config/datasets/bvcc.yaml",
    "singmos": "config/datasets/singmos.yaml",
    "tmhint": "config/datasets/tmhint.yaml",
}
DATASET_ORDER = ["brspeech", "bvcc", "singmos", "tmhint"]
MODEL_CONFIGS = {
    "mhubert_mlp": "config/models/mHuBERT-147.yaml",
    "mhubert_kan": "config/models/mHuBERT-147_kan.yaml",
    "wav2vec2_1b_mlp": "config/models/wav2vec2_1b.yaml",
    "wav2vec2_1b_kan": "config/models/wav2vec2_1b_kan.yaml",
    "wav2vec2_300m_mlp": "config/models/wav2vec2_300m.yaml",
    "wav2vec2_300m_kan": "config/models/wav2vec2_300m_kan.yaml",
}
MODEL_RUN_NAMES = {
    "mhubert_mlp": "mHuBERT-147-mlp",
    "mhubert_kan": "mHuBERT-147-kan",
    "wav2vec2_1b_mlp": "wav2vec2-xls-r-1b-mlp",
    "wav2vec2_1b_kan": "wav2vec2-xls-r-1b-kan",
    "wav2vec2_300m_mlp": "wav2vec2-xls-r-300m-mlp",
    "wav2vec2_300m_kan": "wav2vec2-xls-r-300m-kan",
}
DEFAULT_MODELS = list(MODEL_CONFIGS)
MODEL_SEED_OFFSETS = {
    model_name: offset for offset, model_name in enumerate(DEFAULT_MODELS)
}

CSV_COLUMNS = [
    "experiment_id", "model", "status", "evaluation_dataset",
    "model_config", "train_datasets", "checkpoint_path", "wandb_run_id",
    "val_mse", "val_pearson", "val_spearman",
    "test_mse", "test_pearson", "test_spearman", "test_kendall",
    "n_samples", "started_at", "ended_at", "error",
]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric(metrics: Any, key: str) -> float | None:
    try:
        return to_float(metrics.get(key))
    except AttributeError:
        return None


def load_state(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Estado invalido em {path}.")
    return value


def save_state(path: str, state: dict[str, dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        yaml.safe_dump(state, handle, allow_unicode=True, sort_keys=False)
    os.replace(temporary, path)


def update_state(
    path: str,
    experiment_id: str,
    values: dict[str, Any],
    lock: mp.Lock,
) -> None:
    with lock:
        state = load_state(path)
        state.setdefault(experiment_id, {})
        state[experiment_id].update(values)
        save_state(path, state)


def append_csv(path: str, row: dict[str, Any], lock: mp.Lock) -> None:
    with lock:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=CSV_COLUMNS,
                extrasaction="ignore",
            )
            if needs_header:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def make_combined_csvs(
    dataset_names: list[str],
    output_dir: str,
) -> dict[str, dict[str, str]]:
    """Concatena cada split e devolve specs compativeis com o config."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined_specs: dict[str, dict[str, str]] = {}

    for split in ("train", "val", "test"):
        frames: list[pd.DataFrame] = []
        for dataset_name in dataset_names:
            dataset_config = OmegaConf.load(DATASET_CONFIGS[dataset_name])
            dataset_spec = dataset_config.datasets[split][0]
            metadata_path = str(dataset_spec.metadata_path)
            frame = pd.read_csv(metadata_path).copy()
            filename_column = str(dataset_spec.filename_column)
            target_column = str(dataset_spec.target_column)
            if filename_column not in frame.columns:
                raise KeyError(
                    f"{metadata_path} nao possui a coluna '{filename_column}'."
                )
            if target_column not in frame.columns:
                raise KeyError(
                    f"{metadata_path} nao possui a coluna '{target_column}'."
                )

            # Os quatro YAMLs usam filepath/mos. A conversao para caminho
            # absoluto evita misturar os base_dir dos datasets no CSV final.
            frame["filepath"] = [
                str((Path(str(dataset_spec.base_dir)) / str(path)).resolve())
                for path in frame[filename_column].tolist()
            ]
            frame["mos"] = frame[target_column].astype(float)
            if "sample_id" not in frame.columns:
                frame["sample_id"] = [
                    f"{dataset_name}:{split}:{index}"
                    for index in range(len(frame))
                ]
            else:
                frame["sample_id"] = [
                    f"{dataset_name}:{value}"
                    for value in frame["sample_id"].tolist()
                ]
            frame["dataset"] = dataset_name
            frames.append(frame[["filepath", "mos", "sample_id", "dataset"]])

        combined = pd.concat(frames, ignore_index=True)
        combined_path = output / f"all_datasets_{split}.csv"
        combined.to_csv(combined_path, index=False)
        combined_specs[split] = {
            "name": f"Combined-{split.title()}",
            "metadata_path": str(combined_path),
            "base_dir": "/",
            "filename_column": "filepath",
            "target_column": "mos",
        }
        print(
            f"[Combined] {split}: {len(combined)} amostras -> {combined_path}"
        )
    return combined_specs


def build_config(
    model_name: str,
    combined_specs: dict[str, dict[str, str]],
    checkpoint_root: str,
    run_index: int,
):
    config = OmegaConf.merge(
        OmegaConf.load(DATASET_CONFIGS["brspeech"]),
        OmegaConf.load(MODEL_CONFIGS[model_name]),
    )
    config.datasets.train = [combined_specs["train"]]
    config.datasets.val = [combined_specs["val"]]
    config.datasets.test = [combined_specs["test"]]
    config.dataset_name = "all_datasets"
    config.title = (
        f"combined-all-{MODEL_RUN_NAMES[model_name]}"
        f"--(run-{run_index:02d})"
    )
    config.study = {
        "model": model_name,
        "datasets": DATASET_ORDER,
        "combined": True,
        "checkpoint_root": checkpoint_root,
    }
    return config


def train_args() -> Any:
    class Args:
        pass

    return Args()


def evaluate_dataset(config, dataset_name: str, checkpoint_path: str):
    evaluation_config = copy.deepcopy(config)
    evaluation_config.datasets.test = [
        OmegaConf.load(DATASET_CONFIGS[dataset_name]).datasets.test[0]
    ]
    return run_inference(evaluation_config, checkpoint_path, 0)


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_model(
    task: dict[str, Any],
    gpu: int,
    combined_specs: dict[str, dict[str, str]],
    checkpoint_root: str,
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    seed: int,
) -> None:
    experiment_id = task["experiment_id"]
    model_name = task["model"]
    started = timestamp()
    update_state(
        state_path,
        experiment_id,
        {"status": "running", "gpu": gpu, "started_at": started},
        lock,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        L.seed_everything(seed + MODEL_SEED_OFFSETS[model_name])
        config = build_config(model_name, combined_specs, checkpoint_root, 1)
        args = train_args()
        args.gpu = 0
        args.checkpoint_dir = checkpoint_root
        metrics, best_checkpoint, wandb_run_id = run_train(config, args)
        if (
            not best_checkpoint
            or best_checkpoint == "N/A"
            or not os.path.isfile(best_checkpoint)
        ):
            raise FileNotFoundError(
                f"Checkpoint invalido retornado por run_train: {best_checkpoint}"
            )

        # Avaliacao agregada e uma avaliacao independente por dataset.
        evaluation_datasets = ["all_datasets", *DATASET_ORDER]
        evaluations: dict[str, Any] = {
            "all_datasets": run_inference(config, best_checkpoint, 0)
        }
        for dataset_name in DATASET_ORDER:
            evaluations[dataset_name] = evaluate_dataset(
                config, dataset_name, best_checkpoint
            )

        ended = timestamp()
        common = {
            "model": model_name,
            "model_config": MODEL_CONFIGS[model_name],
            "train_datasets": "+".join(DATASET_ORDER),
            "checkpoint_path": best_checkpoint,
            "wandb_run_id": wandb_run_id or "N/A",
            "val_mse": metric(metrics, "val/mse"),
            "val_pearson": metric(metrics, "val/pearson"),
            "val_spearman": metric(metrics, "val/spearman"),
            "started_at": started,
            "ended_at": ended,
        }
        for evaluation_dataset in evaluation_datasets:
            evaluated = evaluations[evaluation_dataset]
            row = {
                "experiment_id": experiment_id,
                "status": "done",
                "evaluation_dataset": evaluation_dataset,
                **common,
                "test_mse": metric(evaluated, "MSE"),
                "test_pearson": metric(evaluated, "LCC"),
                "test_spearman": metric(evaluated, "SRCC"),
                "test_kendall": metric(evaluated, "KTAU"),
                "n_samples": evaluated.get("N_SAMPLES"),
            }
            append_csv(csv_path, row, lock)

        update_state(
            state_path,
            experiment_id,
            {
                **common,
                "status": "done",
                "aggregate_test_spearman": metric(
                    evaluations["all_datasets"], "SRCC"
                ),
                "per_dataset_test_spearman": {
                    name: metric(evaluations[name], "SRCC")
                    for name in DATASET_ORDER
                },
            },
            lock,
        )
        print(
            f"[{experiment_id}] concluido | checkpoint={best_checkpoint} | "
            f"agregado_SRCC={metric(evaluations['all_datasets'], 'SRCC')}"
        )
    except Exception as exc:
        ended = timestamp()
        error = f"{type(exc).__name__}: {exc}"
        append_csv(
            csv_path,
            {
                "experiment_id": experiment_id,
                "model": model_name,
                "status": "error",
                "evaluation_dataset": "all_datasets",
                "model_config": MODEL_CONFIGS[model_name],
                "train_datasets": "+".join(DATASET_ORDER),
                "started_at": started,
                "ended_at": ended,
                "error": error,
            },
            lock,
        )
        update_state(
            state_path,
            experiment_id,
            {"status": "error", "error": error, "ended_at": ended},
            lock,
        )
        print(f"[{experiment_id}] ERRO: {error}")
        traceback.print_exc()
        raise
    finally:
        cleanup_cuda()


def worker(
    gpu: int,
    task_queue: mp.Queue,
    combined_specs: dict[str, dict[str, str]],
    checkpoint_root: str,
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    seed: int,
) -> None:
    print(f"[Worker GPU {gpu}] iniciado")
    while True:
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            break
        print(f"[GPU {gpu}] iniciando {task['experiment_id']}")
        try:
            run_model(
                task,
                gpu,
                combined_specs,
                checkpoint_root,
                state_path,
                csv_path,
                lock,
                seed,
            )
        except Exception:
            # O erro ja foi salvo; os demais modelos continuam.
            continue
    print(f"[Worker GPU {gpu}] encerrado")


def show_status(path: str) -> None:
    state = load_state(path)
    for experiment_id, entry in state.items():
        print(f"{experiment_id}: {entry.get('status', 'pending')}")
        if entry.get("error"):
            print(f"  erro: {entry['error']}")


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help="Modelos do estudo; por padrao executa os seis YAMLs MLP/KAN.",
    )
    parser.add_argument("--gpu", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument("--combined-dir", default="/workspace/combined_datasets")
    parser.add_argument("--state-file", default="combined_all_6models_state.yaml")
    parser.add_argument("--csv-file", default="combined_all_6models.csv")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    args.models = list(dict.fromkeys(args.models))

    if args.status:
        show_status(args.state_file)
        return

    print(
        f"Experimento combinado | modelos={args.models} | "
        f"datasets={DATASET_ORDER} | GPUs={args.gpu}"
    )
    combined_specs = make_combined_csvs(DATASET_ORDER, args.combined_dir)
    tasks = [
        {
            "experiment_id": f"combined_all__{model_name}__run_01",
            "model": model_name,
        }
        for model_name in args.models
    ]
    print(f"Plano: {len(tasks)} modelo(s), cada um com treino e 5 avaliacoes.")
    if args.dry_run:
        return

    state = load_state(args.state_file)
    pending = []
    for task in tasks:
        experiment_id = task["experiment_id"]
        entry = state.get(experiment_id, {})
        status = entry.get("status", "pending")
        checkpoint = entry.get("checkpoint_path")
        if status in {"done", "success"} and checkpoint and os.path.isfile(checkpoint):
            print(f"Pulando {experiment_id} (concluido)")
            continue
        if status == "error" and not args.retry_errors:
            print(f"Pulando {experiment_id} (erro anterior; use --retry-errors)")
            continue
        state.setdefault(experiment_id, {})["status"] = "pending"
        pending.append(task)
    save_state(args.state_file, state)
    if not pending:
        print("Nenhum modelo pendente.")
        return

    task_queue: mp.Queue = mp.Queue()
    for task in pending:
        task_queue.put(task)
    lock = mp.Lock()
    processes = [
        mp.Process(
            target=worker,
            args=(
                gpu,
                task_queue,
                combined_specs,
                args.checkpoint_root,
                args.state_file,
                args.csv_file,
                lock,
                args.seed,
            ),
        )
        for gpu in args.gpu[: len(pending)]
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Ctrl+C recebido; encerrando workers...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()
        raise

    print(f"Resultados: {args.csv_file}")
    print(f"Estado:     {args.state_file}")


if __name__ == "__main__":
    main()
