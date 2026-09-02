"""Estudo KAN com g=12 e cross-dataset.

Este arquivo e independente de ``grid_search4.py``. Ele executa:

1. treino scratch de mHuBERT-147 KAN em cada dataset, para k=1 e k=2;
2. teste do melhor checkpoint no proprio dataset;
3. para cada checkpoint, fine-tuning em cada outro dataset;
4. teste pos-fine-tuning no dataset-alvo e novamente no dataset-fonte.

O campo ``source_spearman_forgetting_before_minus_after`` mede o esquecimento
catastrofico: valores positivos significam queda de SRCC no dataset-fonte.
O estado e salvo em YAML para permitir retomar a execucao.
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
from typing import Any

# O Pydantic usado pelo Lightning emite estes avisos de compatibilidade durante
# a importação. Eles não alteram o treinamento e são filtrados antes de carregar
# Lightning/OmegaConf, que é quando os avisos normalmente aparecem.
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
warnings.filterwarnings("ignore", category=FutureWarning)

import lightning as L
import yaml
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(__file__))

from main import run_train
from eval.inference import run_inference


RELUKAN_GRID = 12
KAN_MODEL_CONFIG = "config/models/mHuBERT-147_kan.yaml"
DATASET_CONFIG_BY_NAME = {
    "brspeech": "config/datasets/brspeech.yaml",
    "bvcc": "config/datasets/bvcc.yaml",
    "singmos": "config/datasets/singmos.yaml",
    "tmhint": "config/datasets/tmhint.yaml",
}
DEFAULT_DATASETS = ["brspeech", "bvcc", "singmos", "tmhint"]

# O titulo contem g/k para que os checkpoints deste estudo nunca sejam
# confundidos com checkpoints KAN antigos (por exemplo, g=5,k=3).
CHECKPOINT_TITLE_PREFIX = "KAN-utter-project/mHuBERT-147-g12"

CSV_COLUMNS = [
    "experiment_id", "phase", "status", "k", "grid",
    "source_dataset", "target_dataset", "source_run", "finetune_run",
    "model_config", "init_checkpoint_path", "checkpoint_path",
    "val_mse", "val_pearson", "val_spearman",
    "test_mse", "test_pearson", "test_spearman",
    "initial_source_test_mse", "initial_source_test_pearson",
    "initial_source_test_spearman", "source_test_mse",
    "source_test_pearson", "source_test_spearman",
    "initial_target_test_mse", "initial_target_test_pearson",
    "initial_target_test_spearman", "target_test_mse",
    "target_test_pearson", "target_test_spearman",
    "source_spearman_delta_after_minus_before",
    "source_spearman_forgetting_before_minus_after",
    "target_spearman_delta_after_minus_before",
    "started_at", "ended_at", "error",
]


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric(metrics: Any, key: str) -> float | None:
    if metrics is None:
        return None
    try:
        return as_float(metrics.get(key))
    except AttributeError:
        return None


def delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def load_state(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Estado invalido em {path}: esperado um mapa YAML.")
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


def build_config(
    dataset_name: str,
    k: int,
    title: str,
    checkpoint_root: str,
    init_checkpoint: str | None = None,
    source_dataset: str | None = None,
    target_dataset: str | None = None,
):
    config = OmegaConf.merge(
        OmegaConf.load(DATASET_CONFIG_BY_NAME[dataset_name]),
        OmegaConf.load(KAN_MODEL_CONFIG),
    )
    OmegaConf.update(config, "model.relukan_grid", RELUKAN_GRID, merge=True)
    OmegaConf.update(config, "model.relukan_k", k, merge=True)
    config.dataset_name = dataset_name
    config.title = title
    if init_checkpoint:
        config.finetune = {
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "init_checkpoint": init_checkpoint,
        }
    # run_train recebe o diretorio por args; manter este campo somente como
    # metadata torna o config salvo autocontido.
    config.study = {
        "relukan_grid": RELUKAN_GRID,
        "relukan_k": k,
        "checkpoint_root": checkpoint_root,
    }
    return config


def local_train_args(gpu: int, checkpoint_root: str):
    class Args:
        pass

    args = Args()
    # CUDA_VISIBLE_DEVICES restringe o processo a uma GPU; para o Lightning
    # essa GPU passa a ser o dispositivo local 0.
    args.gpu = 0
    args.checkpoint_dir = checkpoint_root
    return args


def evaluate_dataset(config, dataset_name: str, checkpoint_path: str, gpu: int):
    evaluation_config = copy.deepcopy(config)
    evaluation_config.datasets.test = OmegaConf.load(
        DATASET_CONFIG_BY_NAME[dataset_name]
    ).datasets.test
    return run_inference(evaluation_config, checkpoint_path, gpu)


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def training_task(
    task: dict[str, Any],
    gpu: int,
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    checkpoint_root: str,
    seed: int,
) -> None:
    experiment_id = task["experiment_id"]
    started = now_string()
    update_state(
        state_path,
        experiment_id,
        {"status": "running", "gpu": gpu, "started_at": started},
        lock,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dataset = task["dataset"]
    k = int(task["k"])
    run_idx = int(task["run"])
    title = (
        f"{CHECKPOINT_TITLE_PREFIX}-k{k}-{dataset}"
        f"--(run-{run_idx:02d})"
    )
    try:
        L.seed_everything(seed + 1000 * k + run_idx)
        config = build_config(dataset, k, title, checkpoint_root)
        metrics, best_checkpoint, wandb_run_id = run_train(
            config,
            local_train_args(gpu, checkpoint_root),
        )
        if not best_checkpoint or best_checkpoint == "N/A" or not os.path.isfile(best_checkpoint):
            raise FileNotFoundError(
                f"run_train nao retornou um checkpoint valido: {best_checkpoint}"
            )

        test_metrics = run_inference(config, best_checkpoint, 0)
        ended = now_string()
        row = {
            "experiment_id": experiment_id,
            "phase": "train",
            "status": "done",
            "k": k,
            "grid": RELUKAN_GRID,
            "source_dataset": dataset,
            "target_dataset": dataset,
            "source_run": run_idx,
            "finetune_run": "",
            "model_config": KAN_MODEL_CONFIG,
            "init_checkpoint_path": "",
            "checkpoint_path": best_checkpoint,
            "val_mse": metric(metrics, "val/mse"),
            "val_pearson": metric(metrics, "val/pearson"),
            "val_spearman": metric(metrics, "val/spearman"),
            "test_mse": metric(test_metrics, "MSE"),
            "test_pearson": metric(test_metrics, "LCC"),
            "test_spearman": metric(test_metrics, "SRCC"),
            "started_at": started,
            "ended_at": ended,
        }
        append_csv(csv_path, row, lock)
        update_state(
            state_path,
            experiment_id,
            {
                **row,
                "wandb_run_id": wandb_run_id or "N/A",
            },
            lock,
        )
        print(
            f"[{experiment_id}] concluido | k={k} | {dataset} | "
            f"val_SRCC={row['val_spearman']} | test_SRCC={row['test_spearman']}"
        )
    except Exception as exc:
        ended = now_string()
        error = f"{type(exc).__name__}: {exc}"
        append_csv(
            csv_path,
            {
                "experiment_id": experiment_id,
                "phase": "train",
                "status": "error",
                "k": k,
                "grid": RELUKAN_GRID,
                "source_dataset": dataset,
                "target_dataset": dataset,
                "source_run": run_idx,
                "model_config": KAN_MODEL_CONFIG,
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


def cross_task(
    task: dict[str, Any],
    gpu: int,
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    checkpoint_root: str,
    seed: int,
) -> None:
    experiment_id = task["experiment_id"]
    started = now_string()
    update_state(
        state_path,
        experiment_id,
        {"status": "running", "gpu": gpu, "started_at": started},
        lock,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    source = task["source_dataset"]
    target = task["target_dataset"]
    k = int(task["k"])
    source_run = int(task["source_run"])
    finetune_run = int(task["finetune_run"])
    init_checkpoint = task["init_checkpoint"]
    title = (
        f"{CHECKPOINT_TITLE_PREFIX}-k{k}-{source}-to-{target}"
        f"--(source-run-{source_run:02d})--(run-{finetune_run:02d})"
    )
    try:
        if not os.path.isfile(init_checkpoint):
            raise FileNotFoundError(f"Checkpoint fonte nao encontrado: {init_checkpoint}")
        L.seed_everything(seed + 10000 + 1000 * k + finetune_run)
        config = build_config(
            target,
            k,
            title,
            checkpoint_root,
            init_checkpoint=init_checkpoint,
            source_dataset=source,
            target_dataset=target,
        )

        # Baselines: o checkpoint treinado em X e medido em X e em Y antes do
        # fine-tuning. Depois do treino, repetimos exatamente as duas medidas.
        initial_source = evaluate_dataset(config, source, init_checkpoint, 0)
        initial_target = run_inference(config, init_checkpoint, 0)

        metrics, best_checkpoint, wandb_run_id = run_train(
            config,
            local_train_args(gpu, checkpoint_root),
        )
        if not best_checkpoint or best_checkpoint == "N/A" or not os.path.isfile(best_checkpoint):
            raise FileNotFoundError(
                f"run_train nao retornou checkpoint de fine-tuning valido: {best_checkpoint}"
            )

        after_target = run_inference(config, best_checkpoint, 0)
        after_source = evaluate_dataset(config, source, best_checkpoint, 0)
        initial_source_srcc = metric(initial_source, "SRCC")
        after_source_srcc = metric(after_source, "SRCC")
        initial_target_srcc = metric(initial_target, "SRCC")
        after_target_srcc = metric(after_target, "SRCC")
        ended = now_string()
        row = {
            "experiment_id": experiment_id,
            "phase": "cross_finetune",
            "status": "done",
            "k": k,
            "grid": RELUKAN_GRID,
            "source_dataset": source,
            "target_dataset": target,
            "source_run": source_run,
            "finetune_run": finetune_run,
            "model_config": KAN_MODEL_CONFIG,
            "init_checkpoint_path": init_checkpoint,
            "checkpoint_path": best_checkpoint,
            "val_mse": metric(metrics, "val/mse"),
            "val_pearson": metric(metrics, "val/pearson"),
            "val_spearman": metric(metrics, "val/spearman"),
            "initial_source_test_mse": metric(initial_source, "MSE"),
            "initial_source_test_pearson": metric(initial_source, "LCC"),
            "initial_source_test_spearman": initial_source_srcc,
            "source_test_mse": metric(after_source, "MSE"),
            "source_test_pearson": metric(after_source, "LCC"),
            "source_test_spearman": after_source_srcc,
            "initial_target_test_mse": metric(initial_target, "MSE"),
            "initial_target_test_pearson": metric(initial_target, "LCC"),
            "initial_target_test_spearman": initial_target_srcc,
            "target_test_mse": metric(after_target, "MSE"),
            "target_test_pearson": metric(after_target, "LCC"),
            "target_test_spearman": after_target_srcc,
            "source_spearman_delta_after_minus_before": delta(
                after_source_srcc, initial_source_srcc
            ),
            "source_spearman_forgetting_before_minus_after": delta(
                initial_source_srcc, after_source_srcc
            ),
            "target_spearman_delta_after_minus_before": delta(
                after_target_srcc, initial_target_srcc
            ),
            "started_at": started,
            "ended_at": ended,
        }
        append_csv(csv_path, row, lock)
        update_state(
            state_path,
            experiment_id,
            {**row, "wandb_run_id": wandb_run_id or "N/A"},
            lock,
        )
        print(
            f"[{experiment_id}] concluido | {source}->{target} | k={k} | "
            f"target antes/depois={initial_target_srcc}/{after_target_srcc} | "
            f"source antes/depois={initial_source_srcc}/{after_source_srcc}"
        )
    except Exception as exc:
        ended = now_string()
        error = f"{type(exc).__name__}: {exc}"
        append_csv(
            csv_path,
            {
                "experiment_id": experiment_id,
                "phase": "cross_finetune",
                "status": "error",
                "k": k,
                "grid": RELUKAN_GRID,
                "source_dataset": source,
                "target_dataset": target,
                "source_run": source_run,
                "finetune_run": finetune_run,
                "model_config": KAN_MODEL_CONFIG,
                "init_checkpoint_path": init_checkpoint,
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
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    checkpoint_root: str,
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
            if task["phase"] == "train":
                training_task(task, gpu, state_path, csv_path, lock, checkpoint_root, seed)
            else:
                cross_task(task, gpu, state_path, csv_path, lock, checkpoint_root, seed)
        except Exception:
            # O erro ja foi persistido pela funcao da tarefa; manter o worker
            # vivo para processar os demais experimentos.
            continue
    print(f"[Worker GPU {gpu}] encerrado")


def run_batch(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    state_path: str,
    csv_path: str,
) -> None:
    state = load_state(state_path)
    pending: list[dict[str, Any]] = []
    for task in tasks:
        experiment_id = task["experiment_id"]
        status = state.get(experiment_id, {}).get("status", "pending")
        checkpoint = state.get(experiment_id, {}).get("checkpoint_path")
        if (
            status in {"done", "success"}
            and checkpoint
            and os.path.isfile(checkpoint)
        ):
            print(f"Pulando {experiment_id} (concluido)")
            continue
        if status == "error" and not args.retry_errors:
            print(f"Pulando {experiment_id} (erro anterior; use --retry-errors)")
            continue
        state.setdefault(experiment_id, {})["status"] = "pending"
        pending.append(task)
    save_state(state_path, state)
    if not pending:
        print("Nenhum experimento pendente neste lote.")
        return

    print(f"Enfileirando {len(pending)} experimento(s) em {len(args.gpu)} GPU(s).")
    task_queue: mp.Queue = mp.Queue()
    for task in pending:
        task_queue.put(task)
    lock = mp.Lock()
    processes = [
        mp.Process(
            target=worker,
            args=(gpu, task_queue, state_path, csv_path, lock, args.checkpoint_root, args.seed),
        )
        for gpu in args.gpu
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


def train_tasks(datasets: list[str], k_values: list[int], runs: int) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": f"train__kan_g12_k{k}__{dataset}__run_{run:02d}",
            "phase": "train",
            "dataset": dataset,
            "k": k,
            "run": run,
        }
        for k in k_values
        for dataset in datasets
        for run in range(1, runs + 1)
    ]


def cross_tasks(
    datasets: list[str],
    k: int,
    runs: int,
    state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = []
    for source in datasets:
        for source_run in range(1, runs + 1):
            train_id = f"train__kan_g12_k{k}__{source}__run_{source_run:02d}"
            source_entry = state.get(train_id, {})
            checkpoint = source_entry.get("checkpoint_path")
            if source_entry.get("status") not in {"done", "success"} or not checkpoint:
                print(
                    f"Aviso: sem checkpoint concluido para k={k}, "
                    f"source={source}, run={source_run}; cross sera ignorado."
                )
                continue
            for target in datasets:
                if target == source:
                    continue
                experiment_id = (
                    f"cross__kan_g12_k{k}__{source}_to_{target}"
                    f"__source_run_{source_run:02d}__run_{source_run:02d}"
                )
                tasks.append({
                    "experiment_id": experiment_id,
                    "phase": "cross",
                    "source_dataset": source,
                    "target_dataset": target,
                    "source_run": source_run,
                    "finetune_run": source_run,
                    "k": k,
                    "init_checkpoint": checkpoint,
                })
    return tasks


def show_status(path: str) -> None:
    state = load_state(path)
    counts: dict[str, int] = {}
    for entry in state.values():
        status = entry.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    print(f"Estado: {path}")
    print(", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for experiment_id, entry in state.items():
        if entry.get("status") == "error":
            print(f"ERROR {experiment_id}: {entry.get('error', 'sem mensagem')}")


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["train", "cross", "all"],
        default="all",
        help="train=scratch; cross=apenas transferencias; all=ambas as fases.",
    )
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG_BY_NAME),
        default=DEFAULT_DATASETS,
    )
    parser.add_argument(
        "--k",
        dest="k_values",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Valores de k; g permanece fixo em 12.",
    )
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument(
        "--state-file",
        default="grid_state__kan_g12_k1_k2_cross.yaml",
    )
    parser.add_argument("--csv-file", default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs deve ser maior que zero")
    args.k_values = list(dict.fromkeys(args.k_values))
    args.datasets = list(args.datasets)
    state_path = args.state_file
    csv_path = args.csv_file or state_path.replace("grid_state__", "experiments__").replace(
        ".yaml", ".csv"
    )

    if args.status:
        show_status(state_path)
        return

    print(
        f"KAN mHuBERT-147 | g={RELUKAN_GRID} | k={args.k_values} | "
        f"datasets={','.join(args.datasets)} | GPUs={args.gpu}"
    )
    expected_train = len(args.datasets) * len(args.k_values) * args.runs
    expected_cross = expected_train * (len(args.datasets) - 1)
    print(f"Plano: {expected_train} treinos scratch + {expected_cross} fine-tunings cross.")
    if args.dry_run:
        return

    if args.phase in {"train", "all"}:
        # O lote de k=1 termina antes do lote de k=2, como solicitado.
        for k in args.k_values:
            tasks = [task for task in train_tasks(args.datasets, [k], args.runs)]
            print(f"\n=== SCRATCH: g={RELUKAN_GRID}, k={k} ===")
            run_batch(tasks, args, state_path, csv_path)

    if args.phase in {"cross", "all"}:
        # Fazemos tambem o cross em lotes separados por k para manter a
        # comparacao de k=1 e k=2 claramente identificavel.
        for k in args.k_values:
            state = load_state(state_path)
            tasks = cross_tasks(args.datasets, k, args.runs, state)
            print(f"\n=== CROSS-DATASET: g={RELUKAN_GRID}, k={k} ===")
            run_batch(tasks, args, state_path, csv_path)

    print(f"\nResultados CSV: {csv_path}")
    print(f"Estado YAML:    {state_path}")


if __name__ == "__main__":
    main()
