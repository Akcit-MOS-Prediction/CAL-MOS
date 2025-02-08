import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr, kendalltau

from utils.dataloader import DynamicDataset, DynamicCollate
from models.calmos_wrapper import CALMOSWrapper
from transformers import AutoFeatureExtractor

@torch.no_grad
def inference(model, dataloader, device):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=True,
        type=int
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint-path",
        required=False,
        type=str,
        default="../checkpoints/mos-prediction",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = CALMOSWrapper.load_from_checkpoint(args.checkpoint_path, config=config, map_location=device, strict=False)
    
    model = model.to(device)

    # print(model.model.layer_weights[0])

    # layer_weights = []

    # for layer_weight in model.model.layer_weights:
    #     item = layer_weight.cpu().detach().item()
    #     layer_weights.append(item)

    # # pass through a softmax layer
    # layer_weights = np.exp(layer_weights) / np.sum(np.exp(layer_weights))

    # layer_numbers = range(1, len(layer_weights) + 1)

    # # Plot the layer weights as a line plot
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x=list(layer_numbers), y=layer_weights, palette="viridis")
    # plt.xlabel("Layer")
    # plt.ylabel("Weight")
    # plt.title("Layer Weights Bar Chart")
    # plt.xticks(layer_numbers)
    # plt.tight_layout()
    # plt.savefig("layer_weights_bar.png")
    # plt.show()

    # exit()

    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

    test_dataset = DynamicDataset(
        data=test_data,
        filename_column=config.datasets.test[0].filename_column,
        target_column=config.datasets.test[0].target_column,
        base_dir=config.datasets.test[0].base_dir,
        data_type="test",
    )

    processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
     
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=DynamicCollate(processor=processor)
    )

    predictions, targets = inference(model, test_dataloader, device)

    # mse = mean_squared_error(targets, predictions)
    mse = np.mean((targets-predictions)**2)
    lcc = np.corrcoef(targets, predictions)[0, 1]
    srcc = spearmanr(targets, predictions)[0]
    tau = kendalltau(targets, predictions)[0]

    print(f"MSE: {mse:.4f}")
    print(f"LCC: {lcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"KTAU: {tau:.4f}")
