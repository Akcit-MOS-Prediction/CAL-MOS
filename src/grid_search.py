import os
import sys
import argparse
import itertools
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import lightning as L
import glob

sys.path.append(os.path.dirname(__file__))

from main import run_train
from eval.inference import run_inference

def get_experiment_id(index):
    return f"exp_{index:03d}"

def update_config_at_path(config, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        config = config.get(part)
    config[parts[-1]] = value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU device")
    parser.add_argument("-s", "--seed", default=42, type=int, help="Seed base para reprodutibilidade")
    parser.add_argument("-r", "--runs", default=1, type=int, help="Quantidade de vezes para rodar a mesma configuração")

    args = parser.parse_args()

    DATASET_CONFIGS = [
        "config/datasets/brspeech.yaml",
        "config/datasets/bvcc.yaml",
        "config/datasets/singmos.yaml",
        "config/datasets/tmhint.yaml",
    ]

    MODEL_CONFIGS = [
        "config/models/wav2vec2_300m.yaml",
        # "config/models/wav2vec2_1b.yaml",
    ]

    grid = {
        "optimizer.params.learning_rate": [1e-5],
        # "optimizer.params.weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
    }
    # ==========================================

    keys, values = zip(*grid.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    runs_list = list(range(1, args.runs + 1))
    all_experiments = list(itertools.product(DATASET_CONFIGS, MODEL_CONFIGS, hp_combinations, runs_list))

    print(f"Total de experimentos a serem rodados: {len(all_experiments)}")

    for i, (ds_path, mod_path, combo, run_idx) in enumerate(all_experiments):
        config_id = (i // args.runs) + 1
        exp_id = f"config_{config_id:03d}_run_{run_idx:02d}"

        print(f"\n--- Iniciando Experimento: {exp_id} ---")
        print(f"Dataset: {ds_path} | Modelo: {mod_path}")
        print(f"Run: {run_idx}/{args.runs} | Hiperparâmetros: {combo}")

        # Nome dos CSVs baseado no dataset e modelo
        dataset_name = os.path.splitext(os.path.basename(ds_path))[0]   # ex: "brspeech"
        model_name = os.path.splitext(os.path.basename(mod_path))[0]     # ex: "wav2vec2_300m"
        csv_name = f"{dataset_name}_{model_name}"
        treino_csv = f"{csv_name}_treino.csv"
        inferencia_csv = f"{csv_name}_inferencia.csv"

        treino_results = pd.read_csv(treino_csv).to_dict("records") if os.path.exists(treino_csv) else []
        inferencia_results = pd.read_csv(inferencia_csv).to_dict("records") if os.path.exists(inferencia_csv) else []

        if treino_results:
            print(f"[treino] Retomando com {len(treino_results)} entradas já existentes em {treino_csv}")
        if inferencia_results:
            print(f"[inferencia] Retomando com {len(inferencia_results)} entradas já existentes em {inferencia_csv}")

        L.seed_everything(args.seed + run_idx - 1)

        # Merge base: dataset + modelo
        config = OmegaConf.merge(
            OmegaConf.load(ds_path),
            OmegaConf.load(mod_path)
        )

        # Aplica hiperparâmetros do grid
        for path, value in combo.items():
            update_config_at_path(config, path, value)
        config.dataset_name = dataset_name
        config.title = f"{config.title}-{dataset_name}-(run-{run_idx:02d})"

        start_time_train = datetime.now()
        try:
            metrics, best_ckpt, wandb_run_id = run_train(config, args)
            end_time_train = datetime.now()

            # --- Renomeia o diretório do checkpoint para incluir o wandb_run_id ---
            base_dir = getattr(args, "checkpoint_dir", None) or config.model_checkpoint.get("dirpath", "../checkpoints/mos-prediction")
            old_ckpt_dir = os.path.join(base_dir, config.title)
            wandb_suffix = wandb_run_id if wandb_run_id and wandb_run_id not in ("", "N/A") else "no-wandb-id"
            new_ckpt_dir = os.path.join(base_dir, f"{config.title}-{wandb_suffix}")

            if os.path.exists(old_ckpt_dir) and not os.path.exists(new_ckpt_dir):
                os.rename(old_ckpt_dir, new_ckpt_dir)
                print(f"[ckpt] Diretório renomeado para incluir wandb_run_id: {new_ckpt_dir}")
                if best_ckpt and old_ckpt_dir in best_ckpt:
                    best_ckpt = best_ckpt.replace(old_ckpt_dir, new_ckpt_dir)
            elif os.path.exists(new_ckpt_dir):
                pass
            else:
                new_ckpt_dir = old_ckpt_dir  # fallback

            # Fallback: busca recursiva por checkpoints se best_ckpt não for válido
            if best_ckpt == "" or best_ckpt == "N/A" or not os.path.exists(best_ckpt):
                checkpoint_paths = glob.glob(os.path.join(new_ckpt_dir, "**", "*.ckpt"), recursive=True)
                checkpoint_paths = [p for p in checkpoint_paths if "last.ckpt" not in p]
                best_ckpt = checkpoint_paths[0] if checkpoint_paths else "N/A"

            # Salva o resolved_config.yaml no diretório do checkpoint
            resolved_config_path = os.path.join(new_ckpt_dir if best_ckpt != "N/A" else ".", "resolved_config.yaml")
            OmegaConf.save(config, resolved_config_path)
            print(f"[config] resolved_config.yaml salvo em: {resolved_config_path}")

            # Registrar treino
            treino_entry = {
                "experiment_id": exp_id,
                "config_id": f"config_{config_id:03d}",
                "run_idx": run_idx,
                "wandb_run_id": wandb_run_id,
                "dataset_config": ds_path,
                "model_config": mod_path,
                "data_inicio": start_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "data_fim": end_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": best_ckpt,
                "resolved_config": resolved_config_path,
                "status": "success",
                "val_mse": float(metrics["val/mse"]) if "val/mse" in metrics else None,
                "val_pearson": float(metrics["val/pearson"]) if "val/pearson" in metrics else None,
                "val_spearman": float(metrics["val/spearman"]) if "val/spearman" in metrics else None,
            }

            treino_results.append(treino_entry)
            pd.DataFrame(treino_results).to_csv(treino_csv, index=False)
            print(f"[treino] Salvo em {treino_csv} ({len(treino_results)} entradas)")

            # --- INFERÊNCIA: usa o mesmo config do merge + grid ---
            if best_ckpt != "N/A":
                print(f"Iniciando inferência para {exp_id}...")
                start_time_inf = datetime.now()
                inf_metrics = run_inference(config, best_ckpt, args.gpu)
                end_time_inf = datetime.now()

                inf_entry = {
                    "experiment_id": exp_id,
                    "config_id": f"config_{config_id:03d}",
                    "run_idx": run_idx,
                    "wandb_run_id": wandb_run_id,
                    "dataset_config": ds_path,
                    "model_config": mod_path,
                    "data_inicio_inferencia": start_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                    "data_fim_inferencia": end_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                    "modelo_path": best_ckpt,
                    "resolved_config": resolved_config_path,
                    "dataset_teste": config.datasets.test[0].metadata_path,
                    "test_mse": inf_metrics.get("MSE"),
                    "test_pearson": inf_metrics.get("LCC"),
                    "test_spearman": inf_metrics.get("SRCC"),
                }
                inferencia_results.append(inf_entry)
                pd.DataFrame(inferencia_results).to_csv(inferencia_csv, index=False)
                print(f"[inferencia] Salvo em {inferencia_csv} ({len(inferencia_results)} entradas)")

        except Exception as e:
            end_time_train = datetime.now()
            print(f"ERRO no experimento {exp_id}: {e}")

            error_entry = {
                "experiment_id": exp_id,
                "config_id": f"config_{config_id:03d}",
                "run_idx": run_idx,
                "wandb_run_id": "N/A",
                "dataset_config": ds_path,
                "model_config": mod_path,
                "data_inicio": start_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "data_fim": end_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": "N/A",
                "resolved_config": "N/A",
                "status": f"ERROR: {e}",
                "val_mse": "error",
                "val_pearson": "error",
                "val_spearman": "error",
            }
            treino_results.append(error_entry)
            pd.DataFrame(treino_results).to_csv(treino_csv, index=False)
            print(f"[treino] Erro registrado em {treino_csv} ({len(treino_results)} entradas)")

            if inferencia_results:
                pd.DataFrame(inferencia_results).to_csv(inferencia_csv, index=False)
            continue

    print("\nGrid Search concluído com sucesso!")
    print("Resultados salvos nos CSVs por dataset/modelo.")

if __name__ == "__main__":
    main()