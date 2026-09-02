"""Workers independentes para o estudo KAN g=10/g=15.

Cada processo/container executa somente uma GPU logica, mas todos consomem a
mesma fila persistente no YAML. Um flock interprocesso torna a reivindicacao
de tarefas e as escritas no YAML/CSV atomicas. Leases com heartbeat permitem
recuperar uma tarefa quando um worker e interrompido ou morto.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import socket
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any

import grid_search4_kan_g10_g15_cross as plan


TERMINAL_SUCCESS = {"done", "success"}
SKIP_OBSERVATION = (
    "SKIPPED: modelo pausado por solicitacao do usuario; "
    "metricas nao devem ser interpretadas como sucesso."
)


class WorkerShutdown(BaseException):
    """Interrupcao intencional do worker via SIGTERM/SIGINT."""


class InterProcessFileLock:
    """Lock compativel com ``with lock`` usado pelo motor existente."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._thread_lock = threading.RLock()
        self._local = threading.local()

    def __enter__(self) -> "InterProcessFileLock":
        self._thread_lock.acquire()
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            parent = os.path.dirname(self.path) or "."
            os.makedirs(parent, exist_ok=True)
            handle = open(self.path, "a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._local.handle = handle
        self._local.depth = depth + 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        depth = self._local.depth - 1
        self._local.depth = depth
        if depth == 0:
            handle = self._local.handle
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            del self._local.handle
        self._thread_lock.release()


def now_epoch() -> float:
    return time.time()


def checkpoint_is_valid(entry: dict[str, Any]) -> bool:
    checkpoint = entry.get("checkpoint_path")
    return bool(checkpoint and os.path.isfile(str(checkpoint)))


def successful(entry: dict[str, Any]) -> bool:
    return entry.get("status") in TERMINAL_SUCCESS and checkpoint_is_valid(entry)


def task_details(task: dict[str, Any]) -> dict[str, Any]:
    values = {**plan.task_metadata(task), "phase": task["phase"]}
    for key in (
        "dataset",
        "run",
        "source_dataset",
        "target_dataset",
        "source_run",
        "finetune_run",
    ):
        if key in task:
            values[key] = task[key]
    return values


def cross_blueprints(
    args: argparse.Namespace,
    state: dict[str, dict[str, Any]],
    *,
    include_skipped_models: bool = False,
) -> list[dict[str, Any]]:
    """Cria todos os pares X->Y, mesmo antes do checkpoint scratch existir."""
    tasks: list[dict[str, Any]] = []
    for grid in args.grid_values:
        for model_key in args.models:
            if not include_skipped_models and model_key in args.skip_models:
                continue
            k = int(plan.MODEL_SPECS[model_key]["k"])
            stem = plan.experiment_stem(model_key, grid)
            for source in args.datasets:
                for source_run in range(1, args.runs + 1):
                    train_id = (
                        f"train__{stem}__{source}__run_{source_run:02d}"
                    )
                    checkpoint = state.get(train_id, {}).get(
                        "checkpoint_path", ""
                    )
                    for target in args.datasets:
                        if target == source:
                            continue
                        experiment_id = (
                            f"cross__{stem}__{source}_to_{target}"
                            f"__source_run_{source_run:02d}"
                            f"__run_{source_run:02d}"
                        )
                        tasks.append(
                            {
                                "experiment_id": experiment_id,
                                "phase": "cross",
                                "model_key": model_key,
                                "grid": grid,
                                "k": k,
                                "source_dataset": source,
                                "target_dataset": target,
                                "source_run": source_run,
                                "finetune_run": source_run,
                                "init_checkpoint": checkpoint,
                                "source_train_id": train_id,
                            }
                        )
    return tasks


def all_model_train_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_args = SimpleNamespace(**vars(args))
    manifest_args.skip_models = []
    return plan.all_train_tasks(manifest_args)


def active_train_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    return plan.all_train_tasks(args)


def synchronize_manifest(
    args: argparse.Namespace,
    state_path: str,
    lock: InterProcessFileLock,
) -> None:
    """Garante que as 96 entradas existam e aplica/retira pausas de modelo."""
    with lock:
        state = plan.engine.load_state(state_path)
        tasks = all_model_train_tasks(args)
        tasks += cross_blueprints(
            args,
            state,
            include_skipped_models=True,
        )
        for task in tasks:
            experiment_id = task["experiment_id"]
            entry = state.setdefault(experiment_id, {})
            for key, value in task_details(task).items():
                entry.setdefault(key, value)
            model_key = task["model_key"]
            if model_key in args.skip_models:
                if entry.get("status") not in TERMINAL_SUCCESS:
                    entry["status"] = "skipped"
                    entry["observation"] = SKIP_OBSERVATION
            elif (
                entry.get("status") == "skipped"
                and str(entry.get("observation", "")).startswith("SKIPPED:")
            ):
                entry["status"] = "pending"
                entry["observation"] = "Retomado apos pausa do modelo."
            else:
                entry.setdefault("status", "pending")
        plan.engine.save_state(state_path, state)


def lease_is_active(entry: dict[str, Any], timestamp: float) -> bool:
    if entry.get("status") != "running":
        return False
    try:
        return float(entry.get("lease_expires_epoch", 0)) > timestamp
    except (TypeError, ValueError):
        return False


def runnable(
    entry: dict[str, Any],
    retry_errors: bool,
    timestamp: float,
) -> bool:
    status = entry.get("status", "pending")
    if status == "skipped":
        return False
    if successful(entry):
        return False
    if status == "error" and not retry_errors:
        return False
    if lease_is_active(entry, timestamp):
        return False
    return True


def claim_from(
    tasks: list[dict[str, Any]],
    state: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    timestamp: float,
) -> tuple[dict[str, Any], str] | None:
    for task in tasks:
        if task["phase"] == "cross" and not os.path.isfile(
            str(task.get("init_checkpoint", ""))
        ):
            continue
        experiment_id = task["experiment_id"]
        entry = state.setdefault(experiment_id, {})
        if not runnable(entry, args.retry_errors, timestamp):
            continue
        previous_status = entry.get("status", "pending")
        previous_worker = entry.get("worker_id")
        token = uuid.uuid4().hex
        entry.update(task_details(task))
        entry.update(
            {
                "status": "running",
                "worker_id": args.worker_id,
                "worker_gpu": args.physical_gpu,
                "claim_token": token,
                "claimed_at_epoch": timestamp,
                "lease_expires_epoch": timestamp + args.lease_seconds,
                "attempt": int(entry.get("attempt", 0)) + 1,
            }
        )
        if previous_status == "running":
            entry["observation"] = (
                "Tarefa recuperada apos lease expirar; "
                f"worker anterior={previous_worker or 'desconhecido'}."
            )
        return task, token
    return None


def phase_summary(
    tasks: list[dict[str, Any]],
    state: dict[str, dict[str, Any]],
    retry_errors: bool,
    timestamp: float,
) -> str:
    entries = [state.get(task["experiment_id"], {}) for task in tasks]
    if any(lease_is_active(entry, timestamp) for entry in entries):
        return "waiting"
    if any(
        entry.get("status") == "error" and not retry_errors
        for entry in entries
    ):
        return "blocked"
    if all(successful(entry) or entry.get("status") == "skipped" for entry in entries):
        return "complete"
    return "available"


def claim_next_task(
    args: argparse.Namespace,
    state_path: str,
    lock: InterProcessFileLock,
) -> tuple[dict[str, Any] | None, str | None, str]:
    with lock:
        state = plan.engine.load_state(state_path)
        timestamp = now_epoch()
        train_tasks = active_train_tasks(args)

        if args.phase in {"train", "all"}:
            claimed = claim_from(train_tasks, state, args, timestamp)
            if claimed:
                plan.engine.save_state(state_path, state)
                return claimed[0], claimed[1], "claimed"

        train_state = phase_summary(
            train_tasks,
            state,
            args.retry_errors,
            timestamp,
        )
        if args.phase == "train":
            plan.engine.save_state(state_path, state)
            return None, None, train_state
        if train_state == "blocked":
            plan.engine.save_state(state_path, state)
            return None, None, train_state

        cross_tasks = cross_blueprints(args, state)
        if train_state != "complete":
            if not args.allow_cross_while_training:
                plan.engine.save_state(state_path, state)
                return None, None, train_state
            # Cross-dataset so pode iniciar quando o checkpoint scratch da
            # origem ja existe. Os demais continuam aguardando o scratch.
            cross_tasks = [
                task
                for task in cross_tasks
                if os.path.isfile(str(task.get("init_checkpoint", "")))
            ]
            if not cross_tasks:
                plan.engine.save_state(state_path, state)
                return None, None, train_state
        claimed = claim_from(cross_tasks, state, args, timestamp)
        if claimed:
            plan.engine.save_state(state_path, state)
            return claimed[0], claimed[1], "claimed"

        cross_state = phase_summary(
            cross_tasks,
            state,
            args.retry_errors,
            timestamp,
        )
        plan.engine.save_state(state_path, state)
        return None, None, cross_state


def heartbeat_loop(
    stop_event: threading.Event,
    experiment_id: str,
    claim_token: str,
    args: argparse.Namespace,
    state_path: str,
    lock: InterProcessFileLock,
) -> None:
    while not stop_event.wait(args.heartbeat_seconds):
        with lock:
            state = plan.engine.load_state(state_path)
            entry = state.get(experiment_id, {})
            if (
                entry.get("status") != "running"
                or entry.get("claim_token") != claim_token
            ):
                return
            entry["lease_expires_epoch"] = now_epoch() + args.lease_seconds
            entry["heartbeat_at_epoch"] = now_epoch()
            plan.engine.save_state(state_path, state)


def finish_claim(
    experiment_id: str,
    claim_token: str,
    state_path: str,
    lock: InterProcessFileLock,
) -> None:
    with lock:
        state = plan.engine.load_state(state_path)
        entry = state.get(experiment_id, {})
        if entry.get("claim_token") == claim_token:
            entry.pop("claim_token", None)
            entry.pop("lease_expires_epoch", None)
            entry.pop("heartbeat_at_epoch", None)
            plan.engine.save_state(state_path, state)


def return_claim_to_queue(
    experiment_id: str,
    claim_token: str,
    args: argparse.Namespace,
    state_path: str,
    lock: InterProcessFileLock,
) -> None:
    with lock:
        state = plan.engine.load_state(state_path)
        entry = state.get(experiment_id, {})
        if (
            entry.get("status") == "running"
            and entry.get("claim_token") == claim_token
        ):
            entry.update(
                {
                    "status": "pending",
                    "observation": (
                        "Worker interrompido; tarefa devolvida a fila. "
                        f"worker={args.worker_id}."
                    ),
                }
            )
            entry.pop("claim_token", None)
            entry.pop("lease_expires_epoch", None)
            entry.pop("heartbeat_at_epoch", None)
            plan.engine.save_state(state_path, state)


def execute_task(
    task: dict[str, Any],
    claim_token: str,
    args: argparse.Namespace,
    state_path: str,
    csv_path: str,
    lock: InterProcessFileLock,
) -> None:
    experiment_id = task["experiment_id"]
    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(
            stop_event,
            experiment_id,
            claim_token,
            args,
            state_path,
            lock,
        ),
        daemon=True,
    )
    heartbeat.start()
    try:
        plan.configure_engine(task)
        if task["phase"] == "train":
            plan.engine.training_task(
                task,
                args.gpu,
                state_path,
                csv_path,
                lock,
                args.checkpoint_root,
                args.seed,
            )
        else:
            plan.engine.cross_task(
                task,
                args.gpu,
                state_path,
                csv_path,
                lock,
                args.checkpoint_root,
                args.seed,
            )
    except WorkerShutdown:
        return_claim_to_queue(
            experiment_id,
            claim_token,
            args,
            state_path,
            lock,
        )
        raise
    except Exception as exc:
        print(f"[{args.worker_id}] tarefa terminou com erro: {exc}", flush=True)
    finally:
        stop_event.set()
        heartbeat.join(timeout=5)
        finish_claim(experiment_id, claim_token, state_path, lock)


def install_signal_handlers() -> None:
    received = {"count": 0}

    def stop_worker(signum, frame) -> None:
        received["count"] += 1
        if received["count"] > 1:
            os._exit(128 + signum)
        raise WorkerShutdown()

    signal.signal(signal.SIGTERM, stop_worker)
    signal.signal(signal.SIGINT, stop_worker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train", "cross", "all"], default="all")
    parser.add_argument("--gpu", type=int, default=0, help="GPU logica dentro do container.")
    parser.add_argument("--physical-gpu", required=True, help="Rotulo da GPU fisica.")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(plan.MODEL_SPECS),
        default=plan.DEFAULT_MODELS,
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        choices=sorted(plan.MODEL_SPECS),
        default=[],
    )
    parser.add_argument(
        "--grid-values",
        nargs="+",
        type=int,
        choices=plan.DEFAULT_GRIDS,
        default=plan.DEFAULT_GRIDS,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(plan.engine.DATASET_CONFIG_BY_NAME),
        default=plan.DEFAULT_DATASETS,
    )
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument(
        "--state-file",
        default="grid_state__kan_best_k_g10_g15_cross.yaml",
    )
    parser.add_argument("--csv-file", default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--allow-cross-while-training",
        action="store_true",
        help=(
            "Libera cross-dataset com checkpoint de origem pronto mesmo "
            "enquanto outro scratch ainda esta treinando."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--lease-seconds", type=float, default=300)
    args = parser.parse_args()

    args.models = list(dict.fromkeys(args.models))
    args.skip_models = list(dict.fromkeys(args.skip_models))
    args.grid_values = list(dict.fromkeys(args.grid_values))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.worker_id = args.worker_id or (
        f"{socket.gethostname()}-gpu{args.physical_gpu}-{os.getpid()}"
    )
    if args.runs < 1:
        parser.error("--runs deve ser maior que zero")
    if args.heartbeat_seconds <= 0 or args.lease_seconds <= 0:
        parser.error("heartbeat e lease devem ser positivos")
    if args.lease_seconds < args.heartbeat_seconds * 3:
        parser.error("--lease-seconds deve ser pelo menos 3x o heartbeat")
    for grid in args.grid_values:
        for model_key in args.models:
            if grid <= int(plan.MODEL_SPECS[model_key]["k"]):
                parser.error(f"g deve ser maior que k: {model_key}, g={grid}")
    return args


def main() -> int:
    args = parse_args()
    install_signal_handlers()
    state_path = os.path.abspath(args.state_file)
    csv_path = os.path.abspath(
        args.csv_file or plan.default_csv_path(args.state_file)
    )
    lock = InterProcessFileLock(f"{state_path}.lock")
    synchronize_manifest(args, state_path, lock)
    print(
        f"[{args.worker_id}] iniciado | GPU fisica={args.physical_gpu} "
        f"| GPU logica={args.gpu} | fila={state_path}",
        flush=True,
    )

    try:
        while True:
            task, token, queue_state = claim_next_task(args, state_path, lock)
            if task is not None and token is not None:
                print(
                    f"[{args.worker_id}] reivindicou {task['experiment_id']}",
                    flush=True,
                )
                execute_task(task, token, args, state_path, csv_path, lock)
                continue
            if queue_state == "complete":
                print(f"[{args.worker_id}] fila concluida; encerrando.", flush=True)
                return 0
            if queue_state == "blocked":
                print(
                    f"[{args.worker_id}] fila bloqueada por erro; "
                    "reinicie com --retry-errors.",
                    flush=True,
                )
                return 2
            time.sleep(args.poll_seconds)
    except WorkerShutdown:
        print(f"[{args.worker_id}] encerrado por solicitacao.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
