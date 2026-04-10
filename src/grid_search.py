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
    parser.add_argument("-s", "--seed", default=42, type=int, help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    L.seed_everything(args.seed)
    
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
    
    # Combina todos os datasets, modelos e hiperparâmetros
    all_experiments = list(itertools.product(DATASET_CONFIGS, MODEL_CONFIGS, hp_combinations))
    
    print(f"Total de experimentos a serem rodados: {len(all_experiments)}")
    
    treino_results = []
    inferencia_results = []
    
    for i, (ds_path, mod_path, combo) in enumerate(all_experiments):
        exp_id = get_experiment_id(i+1)
        print(f"\n--- Iniciando Experimento: {exp_id} ---")
        print(f"Dataset: {ds_path} | Modelo: {mod_path}")
        print(f"Hiperparâmetros: {combo}")
        
        # Merge do config final combinando base + dataset + modelo
        config = OmegaConf.merge(
            OmegaConf.load(BASE_CONFIG),
            OmegaConf.load(ds_path),
            OmegaConf.load(mod_path)
        )
        
        # Aplica hiperparâmetros do grid
        for path, value in combo.items():
            update_config_at_path(config, path, value)
            
        # Atualiza o título para ser único
        config.title = f"{exp_id}_{config.title}"
        
        start_time_train = datetime.now()
        try:
            metrics, checkpoint_dir = run_train(config, args)
            end_time_train = datetime.now()
            duration_train = (end_time_train - start_time_train).total_seconds()
            
            # Encontrar o melhor checkpoint (não o last.ckpt)
            checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            checkpoint_paths = [p for p in checkpoint_paths if "last.ckpt" not in p]
            best_ckpt = checkpoint_paths[0] if checkpoint_paths else "N/A"
            
            # Registrar treino
            treino_entry = {
                "experiment_id": exp_id,
                "dataset_config": ds_path,
                "model_config": mod_path,
                "data_inicio": start_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "data_fim": end_time_train.strftime("%Y-%m-%d %H:%M:%S"),
                "duracao_segundos": duration_train,
                "checkpoint_path": best_ckpt,
                "resolved_config": os.path.join(checkpoint_dir, "resolved_config.yaml")
            }
            # Adiciona métricas do treino
            for k, v in metrics.items():
                treino_entry[f"train_{k}"] = float(v)
            # Adiciona os hiperparâmetros testados
            treino_entry.update(combo)
            treino_results.append(treino_entry)
            
            # --- INFERÊNCIA ---
            if best_ckpt != "N/A":
                print(f"Iniciando inferência para {exp_id}...")
                start_time_inf = datetime.now()
                inf_metrics = run_inference(config, best_ckpt, args.gpu)
                end_time_inf = datetime.now()
                duration_inf = (end_time_inf - start_time_inf).total_seconds()
                
                inf_entry = {
                    "experiment_id": exp_id,
                    "dataset_config": ds_path,
                    "model_config": mod_path,
                    "data_inicio_inferencia": start_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                    "data_fim_inferencia": end_time_inf.strftime("%Y-%m-%d %H:%M:%S"),
                    "duracao_inferencia": duration_inf,
                    "modelo_path": best_ckpt,
                    "dataset_teste": config.datasets.test[0].metadata_path
                }
                inf_entry.update(inf_metrics)
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
