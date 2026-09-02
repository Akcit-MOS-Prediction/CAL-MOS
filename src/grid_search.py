import os
import sys
import argparse
import itertools
import gc
import torch
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import lightning as L
import glob
import multiprocessing as mp
import queue
import copy
import shutil
import warnings
import signal
import yaml
import re

# ─────────────────────────────────────────────────────────────────────────────
# SUPRESSÃO DE WARNINGS GLOBAL
# O ambiente garante que bibliotecas instanciadas nasçam silenciosas.
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning:pydantic")

sys.path.append(os.path.dirname(__file__))

from main import run_train
from eval.inference import run_inference

CONSOLIDATED_TREINO_CSV     = "all_experiments_treino.csv"
CONSOLIDATED_INFERENCIA_CSV = "all_experiments_inferencia.csv"

EXPERIMENTOS_IGNORADOS = []

GRID_META = {
    "grid_start_dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}


def format_duration(start: datetime, end: datetime) -> str:
    total_seconds = int((end - start).total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}"


def format_hp_for_name(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        key_name = k.split('.')[-1]
        if key_name == "learning_rate": key_name = "lr"
        elif key_name == "weight_decay": key_name = "wd"
        if isinstance(v, float) and v < 0.01:
            val_str = f"{v:.1e}".replace(".0e", "e").replace("-0", "-")
        else:
            val_str = str(v)
        parts.append(f"{key_name}{val_str}")
    return "_".join(parts)


def append_to_csv(csv_path: str, new_entry: dict, lock: mp.Lock):
    with lock:
        df = pd.DataFrame([new_entry])
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            df.to_csv(f, header=write_header, index=False, lineterminator='\n')


def update_yaml_state(yaml_path: str, exp_id: str, updates: dict, lock: mp.Lock):
    """Atualiza o estado de um experimento no banco de dados YAML com segurança."""
    with lock:
        state = {}
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    state = yaml.safe_load(f) or {}
            except Exception:
                pass

        if exp_id not in state:
            state[exp_id] = {}

        state[exp_id].update(updates)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO ROBUSTA DO MELHOR CHECKPOINT
# Hierarquia de critérios:
#   1. Métrica extraída do nome do arquivo  (ex: val_pearson=0.91)
#   2. Data de modificação do arquivo       (mais recente = mais treinado)
#   3. Posição no glob                      (último recurso)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_metric_from_ckpt_name(path: str) -> float:
    """
    Tenta extrair uma métrica numérica do nome do checkpoint para ordenação.
    Padrões suportados (do mais ao menos específico):
      - pearson=0.9123
      - spearman=0.8900
      - val_pearson=0.91
      - epoch=05-val_mse=0.0034
    Retorna o valor como float (maior = melhor), ou -inf se nenhum padrão for encontrado.
    """
    name = os.path.basename(path)

    # Prioridade 1: pearson/spearman (maior é melhor)
    for pattern in [r"pearson[=_]([\d.]+)", r"spearman[=_]([\d.]+)"]:
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Prioridade 2: mse/loss (menor é melhor → negativo para inversão)
    for pattern in [r"mse[=_]([\d.]+)", r"loss[=_]([\d.]+)"]:
        m = re.search(pattern, name, re.IGNORECASE)
        if m:
            try:
                return -float(m.group(1))
            except ValueError:
                pass

    return float("-inf")


def find_best_checkpoint(search_dirs: list[str]) -> str:
    """
    Encontra o melhor checkpoint em uma lista de diretórios candidatos.

    Critérios (em ordem de prioridade):
      1. Métrica extraída do nome do arquivo
      2. Data de modificação (mais recente primeiro)
      3. Primeiro resultado do glob (fallback)

    Retorna o path do melhor checkpoint ou "N/A" se nenhum for encontrado.
    """
    candidates = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        found = glob.glob(os.path.join(directory, "**", "*.ckpt"), recursive=True)
        found = [p for p in found if "last.ckpt" not in os.path.basename(p)]
        candidates.extend(found)

    if not candidates:
        return "N/A"

    # Ordena: métrica extraída do nome (desc) → mtime (desc)
    candidates.sort(
        key=lambda p: (_extract_metric_from_ckpt_name(p), os.path.getmtime(p)),
        reverse=True,
    )

    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSO FILHO DO EXPERIMENTO (ISOLAMENTO DE MEMÓRIA)
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_experiment(exp_data, args_gpu_local, seed, lock):
    # Ignora Ctrl+C no processo filho para o shutdown gracioso no processo pai agir
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Restaura o silenciamento de warnings dentro do processo filho ('spawn')
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_path, temp_config_path, real_gpu_id = exp_data

    # Isolamento de GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_id)

    dataset_name   = os.path.splitext(os.path.basename(ds_path))[0]
    model_name     = os.path.splitext(os.path.basename(mod_path))[0]
    csv_name       = f"{dataset_name}_{model_name}"
    treino_csv     = f"{csv_name}_treino.csv"
    inferencia_csv = f"{csv_name}_inferencia.csv"

    L.seed_everything(seed + run_idx - 1)

    # Atualiza status no YAML para 'running'
    update_yaml_state(yaml_path, exp_id, {
        "status": "running",
        "gpu": real_gpu_id,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, lock)

    # Carrega o config pre-criado no inicio da execução global
    config = OmegaConf.load(temp_config_path)
    hp_columns = {f"hp_{k.split('.')[-1]}": v for k, v in combo.items()}

    class _Args:
        pass
    local_args = _Args()
    local_args.gpu  = 0
    local_args.mode = args_gpu_local["mode"]
    for k, v in args_gpu_local.items():
        setattr(local_args, k, v)
    local_args.gpu = 0

    base_dir = getattr(local_args, "checkpoint_dir", None) or config.model_checkpoint.get("dirpath", "../checkpoints/mos-prediction")
    base_dir = os.path.abspath(base_dir)  # ← resolve ../checkpoints → /home/.../CAL-MOS/checkpoints
    old_ckpt_dir = os.path.join(base_dir, config.title)

    if local_args.mode == "scratch" and os.path.exists(old_ckpt_dir):
        shutil.rmtree(old_ckpt_dir)

    start_time_train = datetime.now()

    try:
        # ── Treino ────────────────────────────────────────────────────────────
        metrics, best_ckpt, wandb_run_id = run_train(config, local_args)
        end_time_train = datetime.now()

        wandb_folder = wandb_run_id if wandb_run_id and wandb_run_id not in ("", "N/A") else "no-wandb-id"
        new_ckpt_dir = os.path.join(base_dir, wandb_folder)

        # ── Resolução Robusta do Diretório de Checkpoints ─────────────────────
        #
        # Casos possíveis:
        #   A) old existe, new NÃO existe → renomeia e atualiza best_ckpt
        #   B) old existe, new JÁ existe  → checkpoint pode estar em qualquer um
        #   C) old NÃO existe, new existe → run já foi movida anteriormente
        #   D) nenhum existe              → new_ckpt_dir = old_ckpt_dir (fallback)
        #
        with lock:
            old_exists = os.path.exists(old_ckpt_dir)
            new_exists = os.path.exists(new_ckpt_dir)

            if old_exists and not new_exists:
                # Caso A: renomeia e corrige o path do best_ckpt
                os.rename(old_ckpt_dir, new_ckpt_dir)
                if best_ckpt and old_ckpt_dir in best_ckpt:
                    best_ckpt = best_ckpt.replace(old_ckpt_dir, new_ckpt_dir)

            elif old_exists and new_exists:
                # Caso B: ambos existem → corrige o path se necessário
                if best_ckpt and old_ckpt_dir in best_ckpt:
                    best_ckpt = best_ckpt.replace(old_ckpt_dir, new_ckpt_dir)

            elif not old_exists and not new_exists:
                # Caso D: nenhum encontrado, usa old como referência de fallback
                new_ckpt_dir = old_ckpt_dir

            # Caso C: old não existe mas new existe → new_ckpt_dir já está correto

        # ── Seleção do Melhor Checkpoint ──────────────────────────────────────
        #
        # Se best_ckpt retornado pelo run_train não é válido no disco,
        # usamos find_best_checkpoint que varre AMBOS os diretórios e
        # escolhe pelo critério: métrica no nome → mtime → primeiro do glob.
        #
        search_dirs = list({new_ckpt_dir, old_ckpt_dir})
        best_ckpt_by_metric = find_best_checkpoint(search_dirs)

        if best_ckpt_by_metric != "N/A":
            best_ckpt = best_ckpt_by_metric
            print(f"[{exp_id}] Checkpoint selecionado por métrica: {best_ckpt}")
        elif not (best_ckpt and best_ckpt != "N/A" and os.path.isfile(best_ckpt)):
            print(f"[{exp_id}] ⚠️  Nenhum checkpoint encontrado na varredura nem pelo run_train.")
            best_ckpt = "N/A"
                # ── Movimentação da Configuração (Antes da Validação) ─────────────────
        resolved_config_path = os.path.join(
            new_ckpt_dir if best_ckpt != "N/A" else ".", "resolved_config.yaml"
        )

        with lock:
            if os.path.exists(temp_config_path):
                shutil.move(temp_config_path, resolved_config_path)
            else:
                resolved_config_path = temp_config_path

        # Atualiza YAML com o caminho definitivo da configuração
        update_yaml_state(yaml_path, exp_id, {"config_path": resolved_config_path}, lock)

        treino_entry = {
            "experiment_id":   exp_id,
            "config_id":       f"config_{config_id:03d}",
            "run_idx":         run_idx,
            "dataset":         dataset_name,
            "model":           model_name,
            **GRID_META,
            "wandb_run_id":    wandb_run_id,
            "dataset_config":  ds_path,
            "model_config":    mod_path,
            **hp_columns,
            "data_inicio":     start_time_train.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fim":        end_time_train.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_treino":  format_duration(start_time_train, end_time_train),
            "checkpoint_path": best_ckpt,
            "resolved_config": resolved_config_path,
            "status":          "success",
            "val_mse":         float(metrics["val/mse"])      if "val/mse"      in metrics else None,
            "val_pearson":     float(metrics["val/pearson"])  if "val/pearson"  in metrics else None,
            "val_spearman":    float(metrics["val/spearman"]) if "val/spearman" in metrics else None,
        }

        append_to_csv(treino_csv, treino_entry, lock)
        append_to_csv(CONSOLIDATED_TREINO_CSV, treino_entry, lock)

        # ── Inferência ────────────────────────────────────────────────────────
        if best_ckpt != "N/A" and os.path.isfile(best_ckpt):
            print(f"Iniciando inferência para {exp_id} na GPU {real_gpu_id} | ckpt: {best_ckpt}")
            start_time_inf = datetime.now()
            inf_metrics    = run_inference(config, best_ckpt, 0)
            end_time_inf   = datetime.now()

            inf_entry = {
                "experiment_id":          exp_id,
                "config_id":              f"config_{config_id:03d}",
                "run_idx":                run_idx,
                "dataset":                dataset_name,
                "model":                  model_name,
                **GRID_META,
                "wandb_run_id":           wandb_run_id,
                "dataset_config":         ds_path,
                "model_config":           mod_path,
                **hp_columns,
                "data_inicio_inferencia": start_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                "data_fim_inferencia":    end_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                "duracao_inferencia":     format_duration(start_time_inf, end_time_inf),
                "modelo_path":            best_ckpt,
                "resolved_config":        resolved_config_path,
                "dataset_teste":          config.datasets.test[0].metadata_path,
                "test_mse":               inf_metrics.get("MSE"),
                "test_pearson":           inf_metrics.get("LCC"),
                "test_spearman":          inf_metrics.get("SRCC"),
            }
            append_to_csv(inferencia_csv, inf_entry, lock)
            append_to_csv(CONSOLIDATED_INFERENCIA_CSV, inf_entry, lock)
        else:
            print(f"[Aviso] Inferência pulada para {exp_id}. Checkpoint não encontrado: {best_ckpt}")

        # Finalizado com Sucesso
        update_yaml_state(yaml_path, exp_id, {
            "status": "done",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, lock)

    except Exception as e:
        import traceback
        end_time_train = datetime.now()
        error_msg = str(e)
        print(f"ERRO no experimento {exp_id} [GPU {real_gpu_id}]: {error_msg}")
        traceback.print_exc()

        error_entry = {
            "experiment_id":   exp_id,
            "config_id":       f"config_{config_id:03d}",
            "run_idx":         run_idx,
            "dataset":         dataset_name,
            "model":           model_name,
            **GRID_META,
            "wandb_run_id":    "N/A",
            "dataset_config":  ds_path,
            "model_config":    mod_path,
            **hp_columns,
            "data_inicio":     start_time_train.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fim":        end_time_train.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_treino":  format_duration(start_time_train, end_time_train),
            "checkpoint_path": "N/A",
            "resolved_config": temp_config_path,
            "status":          f"ERROR: {error_msg}",
            "val_mse":         "error",
            "val_pearson":     "error",
            "val_spearman":    "error",
        }
        append_to_csv(treino_csv, error_entry, lock)
        append_to_csv(CONSOLIDATED_TREINO_CSV, error_entry, lock)

        update_yaml_state(yaml_path, exp_id, {
            "status": "error",
            "error_msg": error_msg,
            "end_time": end_time_train.strftime("%Y-%m-%d %H:%M:%S")
        }, lock)


def worker_gpu(real_gpu_id, task_queue, stop_event, args_dict, lock):
    print(f"[Worker GPU {real_gpu_id}] Iniciado.")

    while not stop_event.is_set():
        try:
            exp_data_raw = task_queue.get(timeout=3)
        except queue.Empty:
            break

        exp_id = exp_data_raw[0]
        # Repassa todos os dados brutos + GPU local para o processo filho
        exp_data_with_gpu = (*exp_data_raw, real_gpu_id)

        print(f"\n--- Iniciando Experimento: {exp_id} [Worker GPU: {real_gpu_id}] ---")

        p = mp.Process(
            target=_run_single_experiment,
            args=(exp_data_with_gpu, args_dict, args_dict["seed"], lock),
        )
        p.start()
        p.join()  # Aguarda a conclusão completa para liberar RAM/VRAM

        if p.exitcode != 0:
            print(f"[Worker GPU {real_gpu_id}] Experimento {exp_id} encerrou com código {p.exitcode}.")
        else:
            print(f"[Worker GPU {real_gpu_id}] Experimento {exp_id} concluído localmente.")

    print(f"[Worker GPU {real_gpu_id}] Encerrando...")


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",  nargs="+", default=[1,4], type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-r", "--runs", default=1,  type=int)
    parser.add_argument("-m", "--mode", choices=["resume", "scratch"], default="scratch")
    parser.add_argument("--state-file", default="", type=str, help="Caminho/nome do arquivo YAML de state")
    parser.add_argument("--status", action="store_true", help="Mostra o status atual sem rodar o grid")
    parser.add_argument("--retry-errors", action="store_true", help="Força a re-execução de experimentos com erro")
    args = parser.parse_args()

    DATASET_CONFIGS = [
        # "config/datasets/brspeech.yaml",
        "config/datasets/bvcc.yaml",
        # "config/datasets/singmos.yaml",
        # "config/datasets/tmhint.yaml",
    ]

    MODEL_CONFIGS = [
    #    "config/models/wav2vec2_300m.yaml",
    #    "config/models/mHuBERT-147.yaml",
       "config/models/wav2vec2_1b.yaml",
    ]

    grid = {
        "optimizer.params.learning_rate": [1e-3, 5e-3, 1e-4, 5e-4],
        "model.mlp_dropout": [0.1, 0.2, 0.3]
    }

    keys, values = zip(*grid.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    runs_list       = list(range(1, args.runs + 1))
    all_experiments = list(itertools.product(DATASET_CONFIGS, MODEL_CONFIGS, hp_combinations, runs_list))

    # Define o nome do arquivo YAML do status do grid
    if not args.state_file:
        if all_experiments:
            first_ds  = os.path.splitext(os.path.basename(all_experiments[0][0]))[0]
            first_mod = os.path.splitext(os.path.basename(all_experiments[0][1]))[0]
            yaml_name = f"grid_state_{first_ds}_{first_mod}.yaml"
        else:
            yaml_name = "grid_state.yaml"
    else:
        yaml_name = args.state_file

    # ── Modo Status Visualizador ─────────────────────────────────────────────
    if args.status:
        if os.path.exists(yaml_name):
            with open(yaml_name, 'r', encoding='utf-8') as f:
                state_db = yaml.safe_load(f) or {}

            pending = sum(1 for v in state_db.values() if v.get("status") == "pending")
            running = sum(1 for v in state_db.values() if v.get("status") == "running")
            done    = sum(1 for v in state_db.values() if v.get("status") in ["done", "success"])
            errors  = sum(1 for v in state_db.values() if v.get("status") == "error")

            print(f"\n--- Resumo do Grid: {yaml_name} ---")
            print(f"✅ Concluídos: {done}")
            print(f"❌ Erros:      {errors}")
            print(f"⏳ Pendentes:  {pending}")
            print(f"🚀 Rodando:    {running}")

            if errors > 0:
                print("\n⚠️  Experimentos com erro:")
                for k, v in state_db.items():
                    if v.get("status") == "error":
                        print(f"  - {k} | Motivo: {v.get('error_msg')}")
        else:
            print(f"Arquivo de status '{yaml_name}' não encontrado. Nenhum grid parece ter rodado.")
        return

    # ── Scratch Clean-up ──────────────────────────────────────────────────────
    if args.mode == "scratch":
        print("\n[Aviso] Modo 'scratch': limpando CSVs e YAML antigos...")
        csvs_para_deletar = {CONSOLIDATED_TREINO_CSV, CONSOLIDATED_INFERENCIA_CSV, yaml_name}
        for ds_path, mod_path in itertools.product(DATASET_CONFIGS, MODEL_CONFIGS):
            d = os.path.splitext(os.path.basename(ds_path))[0]
            m = os.path.splitext(os.path.basename(mod_path))[0]
            csvs_para_deletar.add(f"{d}_{m}_treino.csv")
            csvs_para_deletar.add(f"{d}_{m}_inferencia.csv")
        for f in csvs_para_deletar:
            if os.path.exists(f):
                os.remove(f)
                print(f" - Excluído: {f}")

    # ── Pre-Criação de Configurações e Atualização YAML ───────────────────────
    os.makedirs("temp_configs", exist_ok=True)

    state_db = {}
    if os.path.exists(yaml_name):
        with open(yaml_name, 'r', encoding='utf-8') as f:
            state_db = yaml.safe_load(f) or {}

    task_queue = mp.Queue()
    total_validos = 0

    print("\nVerificando/Criando Configurações Prévias...")
    for i, (ds_path, mod_path, combo, run_idx) in enumerate(all_experiments):
        config_id    = (i // args.runs) + 1
        dataset_name = os.path.splitext(os.path.basename(ds_path))[0]
        model_name   = os.path.splitext(os.path.basename(mod_path))[0]
        hp_str       = format_hp_for_name(combo)
        exp_id       = f"config_{config_id:03d}_{hp_str}__{dataset_name}_run_{run_idx:02d}"

        temp_config_path = f"temp_configs/{exp_id}.yaml"

        # Se o yaml temporário da configuração não existe, geramos agora:
        if args.mode == "scratch" and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        if not os.path.exists(temp_config_path) and exp_id not in EXPERIMENTOS_IGNORADOS:
            config = OmegaConf.merge(
                OmegaConf.load(ds_path),
                OmegaConf.load(mod_path)
            )
            for path, value in combo.items():
                OmegaConf.update(config, path, value, merge=True)

            config.dataset_name = dataset_name
            config.title = f"{config.title}-{dataset_name}-{hp_str}-(run-{run_idx:02d})"
            OmegaConf.save(config, temp_config_path)

        # Regista no state caso ainda não exista
        if exp_id not in state_db:
            state_db[exp_id] = {
                "status": "pending",
                "gpu": None,
                "start_time": None,
                "end_time": None,
                "error_msg": None,
                "config_path": temp_config_path
            }

        st = state_db[exp_id].get("status", "pending")

        if exp_id in EXPERIMENTOS_IGNORADOS:
            continue

        if st in ["done", "success"]:
            print(f"Pulando {exp_id} (Já concluído)")
            continue

        if st == "error" and not args.retry_errors:
            print(f"Pulando {exp_id} (Falhou anteriormente - Use --retry-errors)")
            continue

        # Se parou no meio (ex: crachou a máquina), a status ficou em running. Nós retornamos para pending.
        if st == "running":
            state_db[exp_id]["status"] = "pending"

        task_queue.put((exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_name, temp_config_path))
        total_validos += 1

    # Salva o arquivo YAML antes de rodar os workers
    with open(yaml_name, 'w', encoding='utf-8') as f:
        yaml.dump(state_db, f, default_flow_style=False, sort_keys=False)

    print(f"\nTotal mapeado: {len(all_experiments)} | A rodar (Enfileirados): {total_validos}")

    if total_validos == 0:
        print("Nenhum experimento restante. Finalizando.")
        return

    # Passa args como dict simples para ser iterável sem problemas de pickle no spawn
    args_dict = vars(args)

    lock = mp.Lock()
    stop_event = mp.Event()
    processes = []

    # ── Despacho dos Workers ──────────────────────────────────────────────────
    for real_gpu_id in args.gpu:
        p = mp.Process(target=worker_gpu, args=(real_gpu_id, task_queue, stop_event, args_dict, lock))
        p.start()
        processes.append(p)

    # ── Tratamento Gracioso (Ctrl+C Seguro) ───────────────────────────────────
    try:
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[!] 🛑 Ctrl+C Detectado! Enviando sinal de Shutdown Gracioso...")
        stop_event.set()
        print("[!] Os workers irão parar imediatamente APÓS finalizarem os experimentos em andamento.")
        print("[!] Pressione Ctrl+C novamente caso precise abortar imediatamente (Atenção: Progresso será perdido e CSVs/YAML corrompidos).")
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[!] 💀 Saída forçada pelo usuário. O script foi abortado no meio de processos em execução.")
            sys.exit(1)

    print(f"\nFim da execução do Grid Search. Verifique '{yaml_name}' para o status de todos.")


if __name__ == "__main__":
    main()