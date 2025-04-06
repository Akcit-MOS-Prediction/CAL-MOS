import os
import sys
import argparse

# Adiciona os diretórios raiz para encontrar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau

# Importa as classes para lidar com embeddings e o modelo
from utils.dataloader import EmbeddingDataset, OneLayerEmbeddingCollate
from models.calmos_wrapper import CALMOSWrapper

@torch.no_grad()
def inference(model, dataloader, device):
    """
    Executa a inferência do modelo sobre um dataloader e retorna as predições e os targets.
    """
    model.eval()
    predictions = []
    targets = []
    for batch in tqdm(dataloader, desc="Inference"):
        input_features, target = batch
        input_features, target = input_features.to(device), target.to(device)
        logits = model(input_features)
        predictions.append(logits.cpu().numpy())
        targets.append(target.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)

def run_inference(config_path, gpu, checkpoint_path, dataset="test", batch_size=16):
    """
    Carrega a configuração, o modelo a partir do checkpoint e executa a inferência sobre o dataset especificado.

    Parâmetros:
        config_path (str): Caminho para o arquivo YAML de configuração.
        gpu (int): Índice da GPU a ser utilizada.
        checkpoint_path (str): Caminho para o checkpoint já treinado.
        dataset (str): Conjunto de dados a ser avaliado ("test" ou "val").
        batch_size (int): Tamanho do batch para o DataLoader.
    
    Retorna:
        mse, lcc, srcc, tau (float): Métricas de avaliação.
    """
    config = OmegaConf.load(config_path)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    print("Carregando modelo do checkpoint:", checkpoint_path)
    model = CALMOSWrapper.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location=device,
        strict=False
    )
    model = model.to(device)
    
    if dataset == "test":
        dataset_config = config.datasets.test[0]
    elif dataset == "val":
        dataset_config = config.datasets.val[0]
    else:
        raise ValueError("Dataset inválido. Escolha 'test' ou 'val'.")
    
    data_df = pd.read_csv(dataset_config.metadata_path)
    inference_dataset = EmbeddingDataset(
        data=data_df,
        filename_column=dataset_config.filename_column,
        target_column=dataset_config.target_column,
        base_dir=dataset_config.base_dir,
        use_seqaug=config.data.get("use_seqaug", False),
        data_type=dataset
    )
    
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=OneLayerEmbeddingCollate()
    )
    
    predictions, targets = inference(model, inference_dataloader, device)
    
    mse = np.mean((targets - predictions) ** 2)
    lcc = np.corrcoef(targets, predictions)[0, 1]
    srcc = spearmanr(targets, predictions)[0]
    tau = kendalltau(targets, predictions)[0]
    
    return mse, lcc, srcc, tau

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", required=True, type=str,
        help="Arquivo YAML com as configurações"
    )
    parser.add_argument(
        "-g", "--gpu", required=True, type=int,
        help="Dispositivo GPU a ser usado"
    )
    parser.add_argument(
        "-ckpt", "--checkpoint-path", required=True, type=str,
        help="Caminho para o checkpoint já treinado"
    )
    parser.add_argument(
        "--dataset", type=str, default="test", choices=["test", "val"],
        help="Conjunto de dados a ser avaliado (default: test)"
    )
    
    args = parser.parse_args()
    
    mse, lcc, srcc, tau = run_inference(
        args.config_path,
        args.gpu,
        args.checkpoint_path,
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    
    print(f"MSE: {mse:.4f}")
    print(f"LCC: {lcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"KTAU: {tau:.4f}")
