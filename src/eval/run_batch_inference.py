"""
run_batch_inference.py
----------------------
Lê um CSV de experimentos de treino e roda inferência para cada linha,
gerando um CSV de saída com as métricas de teste.

Uso:
    python run_batch_inference.py \
        --input_csv  /home/user_danielcasanova/mos/CAL-MOS/tmhint_mHuBERT-147_treino.csv \
        --output_csv /home/user_danielcasanova/mos/CAL-MOS/tmhint_mHuBERT-147_inferencia.csv \
        --gpu 0 \
        [--dataset_teste <nome_do_split>]   # opcional; padrão: "test"

Colunas geradas no CSV de saída:
    experiment_id, config_id, run_idx, dataset, model,
    grid_start_dt, wandb_run_id, dataset_config, model_config,
    hp_learning_rate, hp_mlp_dropout,
    data_inicio_inferencia, data_fim_inferencia, duracao_inferencia,
    modelo_path, resolved_config,
    dataset_teste, test_mse, test_pearson, test_spearman
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import scipy.stats
import torch

from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.dataloader import (
    DynamicDataset,
    AllLayersEmbeddingCollate,
    OneLayerEmbeddingCollate,
    DynamicCollate,
    DynamicAudioCollate,
)
from models.calmos_wrapper import CALMOSWrapper
from transformers import AutoFeatureExtractor


# ──────────────────────────────────────────────
# Core inference helpers (reusados do original)
# ──────────────────────────────────────────────

@torch.no_grad()
def _inference_loop(model, dataloader, device):
    model.eval()
    predictions, targets = [], []

    for batch in tqdm(dataloader, desc="Inference", leave=False):
        input_features, target = batch

        if isinstance(input_features, (list, tuple)):
            processed = []
            for i in input_features:
                if isinstance(i, dict):
                    processed.append({k: v.to(device) for k, v in i.items()})
                else:
                    processed.append(i.to(device))
            input_features = tuple(processed)
        elif isinstance(input_features, dict):
            input_features = {k: v.to(device) for k, v in input_features.items()}
        else:
            input_features = input_features.to(device)

        logits = model(input_features)
        predictions.append(logits.cpu().numpy())
        targets.append(target.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(targets)


def run_inference(config, checkpoint_path, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  ▶ Checkpoint : {checkpoint_path}")
    print(f"  ▶ Device     : {device}")

    model = CALMOSWrapper.load_from_checkpoint(
        checkpoint_path, config=config, map_location=device, strict=False
    )
    model = model.to(device)

    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

    test_dataset = DynamicDataset(
        data=test_data,
        filename_column=config.datasets.test[0].filename_column,
        target_column=config.datasets.test[0].target_column,
        sr_column=config.datasets.test[0].get("sr_column", None),
        sr_dictionary=config.data.get("sr_dictionary", None),
        base_dir=config.datasets.test[0].base_dir,
        data_type="test",
    )

    model_type = config.model.model_type.lower()

    if model_type in ("dynamic", "dynamic_kan"):
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DynamicCollate(target_sr=config.data.target_sr, processor=processor)
    elif model_type in ("dynamic_melspec", "dynamic_kan_melspec"):
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DynamicAudioCollate(target_sr=config.data.target_sr, processor=processor)
    elif model_type == "all_layers_embedding":
        collate_fn = AllLayersEmbeddingCollate()
    elif model_type == "one_layer_embedding":
        collate_fn = OneLayerEmbeddingCollate()
    else:
        raise ValueError(f"Tipo de modelo inválido: {config.model.model_type}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    predictions, targets = _inference_loop(model, test_dataloader, device)

    mse  = float(np.mean((targets - predictions) ** 2))
    pearson  = float(np.corrcoef(targets, predictions)[0][1])
    spearman = float(scipy.stats.spearmanr(targets, predictions)[0])

    return {"test_mse": mse, "test_pearson": pearson, "test_spearman": spearman}


# ──────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────

OUTPUT_COLS = [
    "experiment_id", "config_id", "run_idx", "dataset", "model",
    "grid_start_dt", "wandb_run_id", "dataset_config", "model_config",
    "hp_learning_rate", "hp_mlp_dropout",
    "data_inicio_inferencia", "data_fim_inferencia", "duracao_inferencia",
    "modelo_path", "resolved_config",
    "dataset_teste", "test_mse", "test_pearson", "test_spearman",
]


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def batch_inference(input_csv: str, output_csv: str, gpu: int, dataset_teste: str):
    df_in = pd.read_csv(input_csv)

    # Filtra apenas experimentos com status 'success', se a coluna existir
    if "status" in df_in.columns:
        df_in = df_in[df_in["status"] == "success"].reset_index(drop=True)
        print(f"[INFO] {len(df_in)} experimento(s) com status=success encontrado(s).")
    else:
        print(f"[INFO] {len(df_in)} experimento(s) encontrado(s) (sem coluna 'status').")

    rows = []

    for idx, row in df_in.iterrows():
        exp_id       = row["experiment_id"]
        checkpoint   = row["checkpoint_path"]
        resolved_cfg = row["resolved_config"]

        print(f"\n[{idx+1}/{len(df_in)}] Experimento: {exp_id}")

        # Carrega o resolved config do checkpoint
        if not os.path.isfile(resolved_cfg):
            print(f"  ✗ resolved_config não encontrado: {resolved_cfg} — pulando.")
            continue

        config = OmegaConf.load(resolved_cfg)

        t_start = datetime.now()
        try:
            metrics = run_inference(config, checkpoint, gpu)
            status_ok = True
        except Exception as e:
            print(f"  ✗ Erro na inferência: {e}")
            metrics = {"test_mse": None, "test_pearson": None, "test_spearman": None}
            status_ok = False
        t_end = datetime.now()

        duration = _fmt_duration((t_end - t_start).total_seconds())

        out_row = {
            "experiment_id"         : exp_id,
            "config_id"             : row.get("config_id", ""),
            "run_idx"               : row.get("run_idx", ""),
            "dataset"               : row.get("dataset", ""),
            "model"                 : row.get("model", ""),
            "grid_start_dt"         : row.get("grid_start_dt", ""),
            "wandb_run_id"          : row.get("wandb_run_id", ""),
            "dataset_config"        : row.get("dataset_config", ""),
            "model_config"          : row.get("model_config", ""),
            "hp_learning_rate"      : row.get("hp_learning_rate", ""),
            "hp_mlp_dropout"        : row.get("hp_mlp_dropout", ""),
            "data_inicio_inferencia": t_start.strftime("%Y-%m-%d %H:%M:%S"),
            "data_fim_inferencia"   : t_end.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_inferencia"    : duration,
            "modelo_path"           : checkpoint,
            "resolved_config"       : resolved_cfg,
            "dataset_teste"         : dataset_teste,
            **metrics,
        }

        rows.append(out_row)

        if status_ok:
            print(f"  ✔ MSE={metrics['test_mse']:.4f}  "
                  f"Pearson={metrics['test_pearson']:.4f}  "
                  f"Spearman={metrics['test_spearman']:.4f}  "
                  f"({duration})")

    df_out = pd.DataFrame(rows, columns=OUTPUT_COLS)
    df_out.to_csv(output_csv, index=False)
    print(f"\n[✔] CSV de saída salvo em: {output_csv}")
    print(df_out.to_string(index=False))


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Roda inferência em batch a partir de um CSV de experimentos."
    )
    parser.add_argument(
        "--input_csv", "-i",
        required=True,
        type=str,
        help="CSV de experimentos de treino (ex: tmhint_mHuBERT-147_treino.csv)",
    )
    parser.add_argument(
        "--output_csv", "-o",
        required=True,
        type=str,
        help="Caminho do CSV de saída com as métricas de teste",
    )
    parser.add_argument(
        "--gpu", "-g",
        required=True,
        type=int,
        help="Índice da GPU (ex: 0)",
    )
    parser.add_argument(
        "--dataset_teste",
        default="test",
        type=str,
        help="Nome do split de teste (padrão: 'test')",
    )

    args = parser.parse_args()

    batch_inference(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        gpu=args.gpu,
        dataset_teste=args.dataset_teste,
    )