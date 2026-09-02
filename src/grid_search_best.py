import os
import sys
import argparse
import itertools
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import lightning as L
import multiprocessing as mp
import queue
import warnings
import signal
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning:pydantic")

sys.path.append(os.path.dirname(__file__))

from main import run_train
from eval.inference import run_inference

# ─────────────────────────────────────────────────────────────────────────────
# SAÍDAS
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV = "experiments.csv"   # único CSV consolidado
STATE_FILE = "grid_state.yaml"   # banco de estado (resume/retry)

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTOS SELECIONADOS  (modo --mode selected)
# Formato: (dataset_config, model_config, learning_rate, mlp_dropout)
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_EXPERIMENTS = [
    ("config/datasets/bvcc.yaml", "config/models/wav2vec2_300m.yaml", 1e-3, 0.1),
    ("config/datasets/bvcc.yaml", "config/models/wav2vec2_300m.yaml", 5e-3, 0.2),

    ("config/datasets/bvcc.yaml", "config/models/mHuBERT-147.yaml",   1e-3, 0.3),
    ("config/datasets/bvcc.yaml", "config/models/mHuBERT-147.yaml",   5e-3, 0.3),
    ("config/datasets/bvcc.yaml", "config/models/mHuBERT-147.yaml",   5e-4, 0.3),

    ("config/datasets/bvcc.yaml", "config/models/wav2vec2_1b.yaml",   1e-3, 0.1),
    ("config/datasets/bvcc.yaml", "config/models/wav2vec2_1b.yaml",   5e-3, 0.1),
    ("config/datasets/bvcc.yaml", "config/models/wav2vec2_1b.yaml",   5e-4, 0.2),
]

EXPERIMENTOS_IGNORADOS = []


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────
def format_duration(start: datetime, end: datetime) -> str:
    total = int((end - start).total_seconds())
    h, rem = divmod(total, 3600)
    m, _   = divmod(rem, 60)
    return f"{h}:{m:02d}"


def format_hp_for_name(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        key = k.split(".")[-1]
        # Adicionado mapeamento para o k e grid
        key = {
            "learning_rate": "lr", 
            "weight_decay": "wd",
            "relukan_k": "k",
            "relukan_grid": "g"
        }.get(key, key)
        val = f"{v:.1e}".replace(".0e", "e").replace("-0", "-") if isinstance(v, float) and v < 0.01 else str(v)
        parts.append(f"{key}{val}")
    return "_".join(parts)


def append_to_csv(path: str, entry: dict, lock: mp.Lock) -> None:
    """Acrescenta uma linha ao CSV de forma thread-safe."""
    with lock:
        df = pd.DataFrame([entry])
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            df.to_csv(f, header=write_header, index=False, lineterminator="\n")


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO (YAML)  — leitura/escrita atômica com lock
# ─────────────────────────────────────────────────────────────────────────────
def load_state(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_state(yaml_path: str, state: dict, lock: mp.Lock) -> None:
    with lock:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def patch_state(yaml_path: str, exp_id: str, updates: dict, lock: mp.Lock) -> None:
    """Lê → aplica patch → salva. Operação atômica via lock."""
    with lock:
        state = load_state(yaml_path)
        state.setdefault(exp_id, {}).update(updates)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSO FILHO — executa um único experimento em isolamento de memória
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_experiment(exp_data: tuple, args_dict: dict, seed: int, lock: mp.Lock) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C gerenciado pelo pai
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_path, config_dict, real_gpu_id = exp_data

    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_id)
    L.seed_everything(seed + run_idx - 1)

    dataset_name = os.path.splitext(os.path.basename(ds_path))[0]
    model_name   = os.path.splitext(os.path.basename(mod_path))[0]

    # Reconstrói o config a partir do dict passado pela fila (sem tocar em disco)
    config = OmegaConf.create(config_dict)

    class _Args:
        pass
    local_args = _Args()
    for k, v in args_dict.items():
        setattr(local_args, k, v)
    local_args.gpu = 0   # dentro do processo filho CUDA_VISIBLE_DEVICES já isola a GPU

    patch_state(yaml_path, exp_id, {
        "status": "running",
        "gpu": real_gpu_id,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, lock)

    start_train = datetime.now()

    try:
        # ── Treino ────────────────────────────────────────────────────────────
        # run_train deve devolver:
        #   metrics      → dict com val/mse, val/pearson, val/spearman
        #   best_ckpt    → trainer.checkpoint_callback.best_model_path  (path absoluto)
        #   wandb_run_id → run.id do WandB (ou "" se não usar WandB)
        metrics, best_ckpt, wandb_run_id = run_train(config, local_args)
        end_train = datetime.now()

        # Salva o config resolvido junto ao checkpoint (sem pastas temporárias)
        if best_ckpt and os.path.isfile(best_ckpt):
            OmegaConf.save(config, os.path.join(os.path.dirname(best_ckpt), "config.yaml"))
        else:
            best_ckpt = "N/A"
            print(f"[{exp_id}] ⚠️  run_train não retornou um checkpoint válido.")

        # ── Inferência ────────────────────────────────────────────────────────
        inf_metrics = {}
        dur_inf = "N/A"
        if best_ckpt != "N/A":
            print(f"[{exp_id}] Iniciando inferência | ckpt: {best_ckpt}")
            start_inf   = datetime.now()
            inf_metrics = run_inference(config, best_ckpt, 0)
            dur_inf     = format_duration(start_inf, datetime.now())

        # ── Linha única no CSV consolidado ────────────────────────────────────
        entry = {
            # Identificação
            "experiment_id":  exp_id,
            "config_id":      f"config_{config_id:03d}",
            "run_idx":        run_idx,
            "dataset":        dataset_name,
            "model":          model_name,
            "mode":           args_dict.get("mode"),
            "wandb_run_id":   wandb_run_id or "N/A",
            # Configs
            "dataset_config": ds_path,
            "model_config":   mod_path,
            **{f"hp_{k.split('.')[-1]}": v for k, v in combo.items()},
            # Tempos
            "inicio_treino":      start_train.strftime("%Y-%m-%d %H:%M:%S"),
            "fim_treino":         end_train.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_treino":     format_duration(start_train, end_train),
            "duracao_inferencia": dur_inf,
            # Artefatos
            "checkpoint_path": best_ckpt,
            "dataset_teste":   config.datasets.test[0].metadata_path,
            # Métricas de validação
            "val_mse":      float(metrics.get("val/mse",      "nan")),
            "val_pearson":  float(metrics.get("val/pearson",  "nan")),
            "val_spearman": float(metrics.get("val/spearman", "nan")),
            # Métricas de teste
            "test_mse":      inf_metrics.get("MSE"),
            "test_pearson":  inf_metrics.get("LCC"),
            "test_spearman": inf_metrics.get("SRCC"),
            "status": "success",
        }
        append_to_csv(OUTPUT_CSV, entry, lock)

        patch_state(yaml_path, exp_id, {
            "status":   "done",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, lock)

    except Exception as e:
        import traceback
        end_train = datetime.now()
        print(f"ERRO em {exp_id} [GPU {real_gpu_id}]: {e}")
        traceback.print_exc()

        append_to_csv(OUTPUT_CSV, {
            "experiment_id": exp_id,
            "config_id":     f"config_{config_id:03d}",
            "run_idx":       run_idx,
            "dataset":       dataset_name,
            "model":         model_name,
            "mode":          args_dict.get("mode"),
            "inicio_treino": start_train.strftime("%Y-%m-%d %H:%M:%S"),
            "fim_treino":    end_train.strftime("%Y-%m-%d %H:%M:%S"),
            "status":        f"ERROR: {e}",
        }, lock)

        patch_state(yaml_path, exp_id, {
            "status":    "error",
            "error_msg": str(e),
            "end_time":  end_train.strftime("%Y-%m-%d %H:%M:%S"),
        }, lock)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER — consome a fila e despacha processos filhos sequencialmente
# ─────────────────────────────────────────────────────────────────────────────
def worker_gpu(real_gpu_id: int, task_queue: mp.Queue, stop_event: mp.Event,
               args_dict: dict, lock: mp.Lock) -> None:
    print(f"[Worker GPU {real_gpu_id}] Iniciado.")
    while not stop_event.is_set():
        try:
            exp_data_raw = task_queue.get(timeout=3)
        except queue.Empty:
            break

        exp_id = exp_data_raw[0]
        print(f"\n--- {exp_id} [GPU {real_gpu_id}] ---")

        p = mp.Process(
            target=_run_single_experiment,
            args=((*exp_data_raw, real_gpu_id), args_dict, args_dict["seed"], lock),
        )
        p.start()
        p.join()

        status = "concluído" if p.exitcode == 0 else f"encerrou com código {p.exitcode}"
        print(f"[Worker GPU {real_gpu_id}] {exp_id} {status}.")

    print(f"[Worker GPU {real_gpu_id}] Encerrando.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",  nargs="+", default=[5], type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-r", "--runs", default=1,  type=int)
    parser.add_argument("-m", "--mode", choices=["resume", "scratch", "selected"], default="scratch")
    parser.add_argument("--state-file",   default=STATE_FILE, type=str)
    parser.add_argument("--status",       action="store_true", help="Mostra o status e sai")
    parser.add_argument("--retry-errors", action="store_true", help="Re-executa experimentos com erro")
    args = parser.parse_args()

    yaml_path = args.state_file

    # ── Definição do espaço de experimentos ───────────────────────────────────
    DATASET_CONFIGS = [
        # "config/datasets/brspeech.yaml",
        # "config/datasets/bvcc.yaml",
        # "config/datasets/singmos.yaml",
        "config/datasets/tmhint.yaml"
    ]
    MODEL_CONFIGS = [
        # "config/models/wav2vec2_300m.yaml",
        # "config/models/mHuBERT-147.yaml",
        # "config/models/wav2vec2_1b.yaml",
        "config/models/wav2vec2_300m_kan.yaml"
    ]
    # GRID = {
    #     "optimizer.params.learning_rate": [1e-3, 5e-3, 1e-4, 5e-4],
    #     "model.mlp_dropout":              [0.1, 0.2, 0.3],
    # }
    GRID = {
        "model.relukan_k": [1, 2, 3, 4, 5, 6],
        "model.relukan_grid": [1, 2, 3, 4, 5, 6],
    }
    if args.mode == "selected":
        combos = [
            {"model.relukan_k": k, "model.relukan_grid": g}
            for _, _, k, g in SELECTED_EXPERIMENTS
        ]
        all_experiments = [
            (ds, mod, combo, run_idx)
            for (ds, mod, _, __), combo in zip(SELECTED_EXPERIMENTS, combos)
            for run_idx in range(1, args.runs + 1)
        ]
    else:
        keys, vals = zip(*GRID.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        all_experiments = [
            (ds, mod, combo, run_idx)
            for ds, mod, combo, run_idx in itertools.product(
                DATASET_CONFIGS, MODEL_CONFIGS, combos, range(1, args.runs + 1)
            )
        ]

    # ── Modo --status ─────────────────────────────────────────────────────────
    if args.status:
        state = load_state(yaml_path)
        if not state:
            print(f"Nenhum estado encontrado em '{yaml_path}'.")
            return
        counts = {}
        for v in state.values():
            s = v.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        done = counts.get("done", 0) + counts.get("success", 0)
        print(f"\n--- {yaml_path} ---")
        print(f"✅ Concluídos: {done}  ❌ Erros: {counts.get('error', 0)}  "
              f"⏳ Pendentes: {counts.get('pending', 0)}  🚀 Rodando: {counts.get('running', 0)}")
        if counts.get("error", 0):
            print("\n⚠️  Experimentos com erro:")
            for k, v in state.items():
                if v.get("status") == "error":
                    print(f"  - {k}: {v.get('error_msg')}")
        return

    # ── Scratch: limpa saídas anteriores ─────────────────────────────────────
    if args.mode == "scratch":
        for path in [OUTPUT_CSV, yaml_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[Limpeza] Removido: {path}")

    # ── Monta a fila de tarefas ───────────────────────────────────────────────
    state    = load_state(yaml_path)
    lock     = mp.Lock()
    task_q   = mp.Queue()
    n_queued = 0

    print("\nMapeando experimentos...")
    for i, (ds_path, mod_path, combo, run_idx) in enumerate(all_experiments):
        config_id = (i // args.runs) + 1
        hp_str    = format_hp_for_name(combo)
        ds_name   = os.path.splitext(os.path.basename(ds_path))[0]
        exp_id    = f"config_{config_id:03d}_{hp_str}__{ds_name}_run_{run_idx:02d}"

        if exp_id in EXPERIMENTOS_IGNORADOS:
            continue

        # Mescla as configs em memória — sem arquivos temporários em disco
        config = OmegaConf.merge(OmegaConf.load(ds_path), OmegaConf.load(mod_path))
        for path, value in combo.items():
            OmegaConf.update(config, path, value, merge=True)
        config.dataset_name = ds_name
        config.title = f"{config.title}-{ds_name}-{hp_str}-(run-{run_idx:02d})"
        config_dict = OmegaConf.to_container(config, resolve=True)   # dict serializável

        if exp_id not in state:
            state[exp_id] = {"status": "pending"}

        st = state[exp_id].get("status", "pending")

        if st in ("done", "success"):
            print(f"  Pulando {exp_id} (concluído)")
            continue
        if st == "error" and not args.retry_errors:
            print(f"  Pulando {exp_id} (erro anterior — use --retry-errors)")
            continue
        if st == "running":
            state[exp_id]["status"] = "pending"

        task_q.put((exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_path, config_dict))
        n_queued += 1

    save_state(yaml_path, state, lock)
    print(f"\nTotal: {len(all_experiments)} | Enfileirados: {n_queued}")

    if n_queued == 0:
        print("Nenhum experimento a executar. Saindo.")
        return

    # ── Despacha workers (um por GPU) ─────────────────────────────────────────
    args_dict  = vars(args)
    stop_event = mp.Event()
    processes  = [
        mp.Process(target=worker_gpu, args=(gpu_id, task_q, stop_event, args_dict, lock))
        for gpu_id in args.gpu
    ]
    for p in processes:
        p.start()

    # ── Graceful shutdown (Ctrl+C) ────────────────────────────────────────────
    try:
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — aguardando fim dos experimentos em andamento...")
        stop_event.set()
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("[!] Saída forçada. Progresso pode ter sido perdido.")
            sys.exit(1)

    print(f"\nConcluído. Resultados em '{OUTPUT_CSV}' | Estado em '{yaml_path}'.")


if __name__ == "__main__":
    main()