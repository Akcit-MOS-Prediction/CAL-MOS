"""Estudo ReLUKAN com g > k para todos os modelos e datasets.

O estudo usa o melhor ``k`` encontrado para cada backbone KAN e compara
``g=10`` com ``g=15``:

* mHuBERT-147 KAN: k=3;
* wav2vec2 XLS-R 1B KAN: k=6;
* wav2vec2 XLS-R 300M KAN: k=2.

Para cada combinacao (modelo, g), o programa:

1. treina um modelo-base em cada dataset e testa no proprio dataset;
2. usa exatamente esse checkpoint-base para cada par ordenado X -> Y;
3. mede X e Y antes do fine-tuning;
4. faz fine-tuning em Y;
5. mede novamente Y e X para quantificar adaptacao e esquecimento.

Com os valores padrao sao 24 treinos-base e 72 fine-tunings cross-dataset,
totalizando 96 experimentos. Todos os treinos-base formam uma fila global e,
depois que ela termina, todos os cross-dataset formam outra fila global. Assim,
qualquer quantidade de GPUs passada em ``--gpu`` consome dinamicamente a mesma
fila, sem reservar uma GPU para um modelo especifico. Checkpoints, IDs, YAML e
CSV incluem modelo, g e k para impedir que configuracoes diferentes sejam
misturadas. O arquivo reutiliza somente as funcoes de treino/inferencia e
persistencia da variante g=12; o plano e os checkpoints sao independentes.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
from typing import Any

import grid_search4_kan_g12_cross as engine


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "mhubert_kan": {
        "label": "mHuBERT-147 KAN",
        "config": "config/models/mHuBERT-147_kan.yaml",
        "k": 3,
        "checkpoint_title_prefix": "KAN-utter-project/mHuBERT-147",
    },
    "wav2vec2_1b_kan": {
        "label": "wav2vec2 XLS-R 1B KAN",
        "config": "config/models/wav2vec2_1b_kan.yaml",
        "k": 6,
        "checkpoint_title_prefix": "KAN-facebook/wav2vec2-xls-r-1b",
    },
    "wav2vec2_300m_kan": {
        "label": "wav2vec2 XLS-R 300M KAN",
        "config": "config/models/wav2vec2_300m_kan.yaml",
        "k": 2,
        "checkpoint_title_prefix": "KAN-facebook/wav2vec2-xls-r-300m",
    },
}

DEFAULT_MODELS = list(MODEL_SPECS)
DEFAULT_GRIDS = [10, 15]
DEFAULT_DATASETS = list(engine.DEFAULT_DATASETS)


def experiment_stem(model_key: str, grid: int) -> str:
    """Prefixo unico usado nos IDs do YAML e CSV."""
    k = int(MODEL_SPECS[model_key]["k"])
    return f"{model_key}__g{grid}_k{k}"


def configure_engine(task: dict[str, Any]) -> None:
    """Configura o motor dentro do processo worker para a tarefa atual."""
    spec = MODEL_SPECS[task["model_key"]]
    grid = int(task["grid"])
    engine.RELUKAN_GRID = grid
    engine.KAN_MODEL_CONFIG = str(spec["config"])
    engine.CHECKPOINT_TITLE_PREFIX = (
        f"{spec['checkpoint_title_prefix']}-g{grid}"
    )


def task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    spec = MODEL_SPECS[task["model_key"]]
    return {
        "model_key": task["model_key"],
        "model_label": spec["label"],
        "model_config": spec["config"],
        "grid": int(task["grid"]),
        "k": int(task["k"]),
    }


def worker(
    gpu: int,
    task_queue: mp.Queue,
    state_path: str,
    csv_path: str,
    lock: mp.Lock,
    checkpoint_root: str,
    seed: int,
) -> None:
    """Executa tarefas sequencialmente em uma GPU fisica."""
    print(f"[Worker GPU {gpu}] iniciado")
    while True:
        try:
            task = task_queue.get(timeout=3)
        except queue.Empty:
            break

        experiment_id = task["experiment_id"]
        configure_engine(task)
        engine.update_state(
            state_path,
            experiment_id,
            task_metadata(task),
            lock,
        )
        print(f"[GPU {gpu}] iniciando {experiment_id}")
        try:
            if task["phase"] == "train":
                engine.training_task(
                    task,
                    gpu,
                    state_path,
                    csv_path,
                    lock,
                    checkpoint_root,
                    seed,
                )
            else:
                engine.cross_task(
                    task,
                    gpu,
                    state_path,
                    csv_path,
                    lock,
                    checkpoint_root,
                    seed,
                )
        except Exception:
            # A funcao da tarefa ja gravou traceback, status e mensagem.
            # O worker continua para que um erro nao cancele todo o estudo.
            continue
    print(f"[Worker GPU {gpu}] encerrado")


def run_batch(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    state_path: str,
    csv_path: str,
) -> None:
    """Remove tarefas concluidas, enfileira as demais e inicia os workers."""
    state = engine.load_state(state_path)
    pending: list[dict[str, Any]] = []

    for task in tasks:
        experiment_id = task["experiment_id"]
        entry = state.get(experiment_id, {})
        status = entry.get("status", "pending")
        checkpoint = entry.get("checkpoint_path")
        # Tarefas marcadas explicitamente como ignoradas nao devem voltar
        # para a fila por nao possuirem checkpoint.
        observation = str(entry.get("observation", ""))
        if status == "skipped" or observation.startswith("SKIPPED:"):
            print(f"Pulando {experiment_id} (marcado como ignorado)")
            continue

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

        state.setdefault(experiment_id, {}).update(task_metadata(task))
        state[experiment_id]["status"] = "pending"
        pending.append(task)

    engine.save_state(state_path, state)
    if not pending:
        print("Nenhum experimento pendente neste lote.")
        return

    print(
        f"Enfileirando {len(pending)} experimento(s) "
        f"em {len(args.gpu)} GPU(s)."
    )
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
                state_path,
                csv_path,
                lock,
                args.checkpoint_root,
                args.seed,
            ),
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


def train_tasks(
    model_key: str,
    grid: int,
    datasets: list[str],
    runs: int,
) -> list[dict[str, Any]]:
    k = int(MODEL_SPECS[model_key]["k"])
    stem = experiment_stem(model_key, grid)
    return [
        {
            "experiment_id": (
                f"train__{stem}__{dataset}__run_{run:02d}"
            ),
            "phase": "train",
            "model_key": model_key,
            "grid": grid,
            "k": k,
            "dataset": dataset,
            "run": run,
        }
        for dataset in datasets
        for run in range(1, runs + 1)
    ]


def cross_tasks(
    model_key: str,
    grid: int,
    datasets: list[str],
    runs: int,
    state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cria todos os pares X -> Y usando apenas o checkpoint scratch exato."""
    k = int(MODEL_SPECS[model_key]["k"])
    stem = experiment_stem(model_key, grid)
    tasks: list[dict[str, Any]] = []

    for source in datasets:
        for source_run in range(1, runs + 1):
            train_id = f"train__{stem}__{source}__run_{source_run:02d}"
            source_entry = state.get(train_id, {})
            checkpoint = source_entry.get("checkpoint_path")
            if (
                source_entry.get("status") not in {"done", "success"}
                or not checkpoint
                or not os.path.isfile(checkpoint)
            ):
                print(
                    "Aviso: sem checkpoint scratch valido para "
                    f"modelo={model_key}, g={grid}, k={k}, "
                    f"source={source}, run={source_run}; cross ignorado."
                )
                continue

            for target in datasets:
                if target == source:
                    continue
                finetune_run = source_run
                experiment_id = (
                    f"cross__{stem}__{source}_to_{target}"
                    f"__source_run_{source_run:02d}"
                    f"__run_{finetune_run:02d}"
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
                        "finetune_run": finetune_run,
                        "init_checkpoint": checkpoint,
                    }
                )
    return tasks


def default_csv_path(state_path: str) -> str:
    csv_path = state_path.replace("grid_state__", "experiments__")
    if csv_path.endswith(".yaml"):
        return f"{csv_path[:-5]}.csv"
    if csv_path.endswith(".yml"):
        return f"{csv_path[:-4]}.csv"
    return f"{csv_path}.csv"


def show_plan(args: argparse.Namespace) -> None:
    active_models = [
        model for model in args.models if model not in args.skip_models
    ]
    combinations = len(active_models) * len(args.grid_values)
    scratch = combinations * len(args.datasets) * args.runs
    cross = scratch * (len(args.datasets) - 1)
    print("Configuracoes:")
    for grid in args.grid_values:
        for model_key in active_models:
            spec = MODEL_SPECS[model_key]
            print(
                f"  {model_key}: g={grid}, k={spec['k']}, "
                f"config={spec['config']}"
            )
    if args.skip_models:
        print(
            "Modelos ignorados nesta execucao: "
            + ", ".join(args.skip_models)
        )
    print(
        f"Plano: {scratch} treinos scratch + {cross} fine-tunings cross "
        f"= {scratch + cross} experimentos."
    )
    print(
        f"Escalonamento dinamico: {len(args.gpu)} GPU(s) consumindo "
        "uma fila global por fase."
    )


def all_train_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Monta uma unica fila com todos os treinos-base selecionados."""
    return [
        task
        for grid in args.grid_values
        for model_key in args.models
        if model_key not in args.skip_models
        for task in train_tasks(
            model_key,
            grid,
            args.datasets,
            args.runs,
        )
    ]


def all_cross_tasks(
    args: argparse.Namespace,
    state: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Monta uma unica fila cross depois que a fase scratch terminou."""
    return [
        task
        for grid in args.grid_values
        for model_key in args.models
        if model_key not in args.skip_models
        for task in cross_tasks(
            model_key,
            grid,
            args.datasets,
            args.runs,
            state,
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["train", "cross", "all"],
        default="all",
        help="train=scratch; cross=transferencias; all=as duas fases.",
    )
    parser.add_argument("--gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--skip-models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=[],
        help=(
            "Modelos a marcar como skipped e retirar das filas. "
            "Use para pausar modelos demorados sem apagar o estado."
        ),
    )
    parser.add_argument(
        "--grid-values",
        nargs="+",
        type=int,
        choices=DEFAULT_GRIDS,
        default=DEFAULT_GRIDS,
        help="Valores de g. O k e fixado pelo modelo.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(engine.DATASET_CONFIG_BY_NAME),
        default=DEFAULT_DATASETS,
    )
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument(
        "--state-file",
        default="grid_state__kan_best_k_g10_g15_cross.yaml",
    )
    parser.add_argument("--csv-file", default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs deve ser maior que zero")
    args.models = list(dict.fromkeys(args.models))
    args.skip_models = list(dict.fromkeys(args.skip_models))
    args.grid_values = list(dict.fromkeys(args.grid_values))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.gpu = list(dict.fromkeys(args.gpu))
    for grid in args.grid_values:
        for model_key in args.models:
            k = int(MODEL_SPECS[model_key]["k"])
            if grid <= k:
                parser.error(
                    f"A condicao g > k falhou para {model_key}: g={grid}, k={k}"
                )
    return args


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    state_path = args.state_file
    csv_path = args.csv_file or default_csv_path(state_path)

    if args.status:
        engine.show_status(state_path)
        return

    if args.skip_models:
        state = engine.load_state(state_path)
        for experiment_id, entry in state.items():
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("model_key", ""))
            matches_skipped_model = model_key in args.skip_models or any(
                f"__{skipped_model}__" in experiment_id
                for skipped_model in args.skip_models
            )
            if (
                matches_skipped_model
                and entry.get("status") not in {"done", "success"}
            ):
                entry["status"] = "skipped"
                entry["observation"] = (
                    "SKIPPED: modelo pausado por solicitacao do usuario; "
                    "metricas nao devem ser interpretadas como sucesso."
                )
        engine.save_state(state_path, state)

    show_plan(args)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"GPUs: {args.gpu}")
    if args.dry_run:
        return

    if args.phase in {"train", "all"}:
        print("\n=== FILA GLOBAL: TREINOS SCRATCH ===")
        run_batch(
            all_train_tasks(args),
            args,
            state_path,
            csv_path,
        )

    if args.phase in {"cross", "all"}:
        # A barreira entre as duas chamadas garante que nenhum fine-tuning
        # comece antes de todos os checkpoints scratch estarem finalizados.
        state = engine.load_state(state_path)
        print("\n=== FILA GLOBAL: CROSS-DATASET ===")
        run_batch(
            all_cross_tasks(args, state),
            args,
            state_path,
            csv_path,
        )

    print(f"\nResultados CSV: {csv_path}")
    print(f"Estado YAML:    {state_path}")


if __name__ == "__main__":
    main()
