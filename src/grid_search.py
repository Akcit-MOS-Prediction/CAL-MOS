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
    parser.add_argument("-r", "--runs", default=3, type=int, help="Quantidade de vezes para rodar a mesma configuração")
    
    args = parser.parse_args()
    
    # DEFINIÇÃO DAS LISTAS DE CONFIGURAÇÕES
    BASE_CONFIG = "config/default.yaml"
    
    DATASET_CONFIGS = [
        "config/datasets/brspeech_v2.yaml",
    ]
    
    MODEL_CONFIGS = [
        "config/models/wav2vec2_base.yaml",
    ]

    grid = {
        "optimizer.params.learning_rate": [1e-5, 5e-5],
        "optimizer.params.weight_decay": [1e-6, 1e-4],
        "train.batch_size": [16, 32],
        "model.relukan_k": [1,2,3,4,5,6],
        "model.relukan_grid": [1,2,3,4,5,6] 
    }
    # ==========================================
    
    keys, values = zip(*grid.items())
    hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Lista de runs (1 até N)
    runs_list = list(range(1, args.runs + 1))
    
    # Combina datasets, modelos, hiperparâmetros e número da run
    all_experiments = list(itertools.product(DATASET_CONFIGS, MODEL_CONFIGS, hp_combinations, runs_list))
    
    print(f"Total de experimentos a serem rodados: {len(all_experiments)}")
    
    treino_results = []
    inferencia_results = []
    
    for i, (ds_path, mod_path, combo, run_idx) in enumerate(all_experiments):
        # Cada configuração única ganha um ID, ex: config_001
        config_id = (i // args.runs) + 1
        exp_id = f"config_{config_id:03d}_run_{run_idx:02d}"
        
        print(f"\n--- Iniciando Experimento: {exp_id} ---")
        print(f"Dataset: {ds_path} | Modelo: {mod_path}")
        print(f"Run: {run_idx}/{args.runs} | Hiperparâmetros: {combo}")
        
        # Mudamos a seed sutilmente a cada run para variações estatísticas, caso desejado
        L.seed_everything(args.seed + run_idx - 1)
        
        # Merge do config final combinando base + dataset + modelo
        config = OmegaConf.merge(
            OmegaConf.load(BASE_CONFIG),
            OmegaConf.load(ds_path),
            OmegaConf.load(mod_path)
        )
        
        # Aplica hiperparâmetros do grid
        for path, value in combo.items():
            update_config_at_path(config, path, value)
            
        # Atualiza o título para ser único no wandb
        config.title = f"{exp_id}_{config.title}"
        
        start_time_train = datetime.now()
        try:
            metrics, checkpoint_dir, wandb_run_id = run_train(config, args)
            end_time_train = datetime.now()
            
            # Encontrar o melhor checkpoint (não o last.ckpt)
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            checkpoint_paths = [p for p in checkpoint_paths if "last.ckpt" not in p]
            best_ckpt = checkpoint_paths[0] if checkpoint_paths else "N/A"
            
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
                "resolved_config": os.path.join(checkpoint_dir, "resolved_config.yaml")
            }
            
            # Adiciona apenas as métricas de validação desejadas
            if "val/mse" in metrics:
                treino_entry["val_mse"] = float(metrics["val/mse"])
            if "val/pearson" in metrics:
                treino_entry["val_pearson"] = float(metrics["val/pearson"])
            if "val/spearman" in metrics:
                treino_entry["val_spearman"] = float(metrics["val/spearman"])
                
            treino_results.append(treino_entry)
            
            # --- INFERÊNCIA ---
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
                    "dataset_teste": config.datasets.test[0].metadata_path,
                    "test_mse": inf_metrics.get("MSE"),
                    "test_pearson": inf_metrics.get("LCC"),
                    "test_spearman": inf_metrics.get("SRCC")
                }
                inferencia_results.append(inf_entry)
            
        except Exception as e:
            print(f"ERRO no experimento {exp_id}: {e}")
            continue
        
        # Salva resultados parciais para não perder dados se o script cair
        pd.DataFrame(treino_results).to_csv("treino.csv", index=False)
        pd.DataFrame(inferencia_results).to_csv("inferencia.csv", index=False)

    print("\nGrid Search concluído com sucesso!")
    print("Resultados salvos em treino.csv e inferencia.csv")

if __name__ == "__main__":
    main()
