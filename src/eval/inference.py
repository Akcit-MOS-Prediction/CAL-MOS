import os
import sys
import glob
import argparse
import numpy as np
import scipy.stats
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import scipy.stats
import torch
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# from scipy.stats import spearmanr, pearsonr, kendalltau

from utils.dataloader import (
    DynamicDataset,
    AugmentationDataset,
    AllLayersEmbeddingCollate,
    OneLayerEmbeddingCollate,
    DynamicCollate,
    DynamicAudioCollate,
    DiynamicAugmentationCollate
)
from models.calmos_wrapper import CALMOSWrapper
from transformers import AutoFeatureExtractor


@torch.no_grad
def inference(model, dataloader, device):
    model.eval()

    predictions = []
    targets = []

    for batch in tqdm(dataloader, desc="Inference"):
        input_features, target = batch

        if isinstance(input_features, (list, tuple)):
            # If inputs is a list or tuple, process each element based on its type
            processed_inputs = []
            for i in input_features:
                if i is None:
                    processed_inputs.append(None)
                elif isinstance(i, dict):
                    # If the element is a dictionary, move each tensor to the device
                    processed_dict = {k: v.to(device) for k, v in i.items()}
                    processed_inputs.append(processed_dict)
                else:
                    # If the element is a tensor, move it to the device
                    processed_inputs.append(i.to(device))
            input_features = tuple(processed_inputs)
        elif isinstance(input_features, dict):
            # If inputs is a dictionary, move each tensor to the device
            input_features = {k: v.to(device) for k, v in input_features.items()}
        else:
            input_features = input_features.to(device)

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

    checkpoint_paths = glob.glob(os.path.join(args.checkpoint_path, "**", "*.ckpt"), recursive=True)
    # remove "last.ckpt" from the list
    checkpoint_paths = [path for path in checkpoint_paths if "last.ckpt" not in path]
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    print(f"Using checkpoint: {checkpoint_path}")

    model = CALMOSWrapper.load_from_checkpoint(checkpoint_path, config=config, map_location=device, strict=False)

    model = model.to(device)

    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

    if config.model.model_type.lower() == "augmentation":
        test_dataset = AugmentationDataset(
            data=test_data,
            base_dir=config.datasets.test[0].base_dir,
            filename_column=config.datasets.test[0].filename_column,
            target_column=config.datasets.test[0].target_column,
            target_sr=config.data.target_sr,
            data_type="test",
            augmentation_config=None,  
        )
    else:
        test_dataset = DynamicDataset(
            data=test_data,
            filename_column=config.datasets.test[0].filename_column,
            target_column=config.datasets.test[0].target_column,
            sr_column=config.datasets.test[0].get("sr_column", None), # Backward compatibility
            sr_dictionary=config.data.get("sr_dictionary", None), # Backward compatibility
            base_dir=config.datasets.test[0].base_dir,
            data_type="test",
        )

    if config.model.model_type.lower() == "dynamic" or config.model.model_type.lower() == "dynamic_kan":
        print("Dynamic model")
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DynamicCollate(
            target_sr=config.data.target_sr,
            processor=processor,
        )
    elif config.model.model_type.lower() == "dynamic_melspec" or config.model.model_type.lower() == "dynamic_kan_melspec":
        print("Dynamic melspec")
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DynamicAudioCollate(
            target_sr=config.data.target_sr,
            processor=processor,
        )
    elif config.model.model_type.lower() == "all_layers_embedding":
        collate_fn = AllLayersEmbeddingCollate()
    elif config.model.model_type.lower() == "one_layer_embedding":
        collate_fn = OneLayerEmbeddingCollate()
    elif config.model.model_type.lower() == "augmentation":
        print("Dynamic Augmentation Model")
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate_fn = DiynamicAugmentationCollate(
            target_sr=config.data.target_sr,
            processor=processor,
        )
    else:
        raise ValueError(f"Invalid model type: {config.model.model_type}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    predictions, targets = inference(model, test_dataloader, device)

    true_mean_scores = targets
    predict_mean_scores = predictions
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    
    print(f"{os.path.basename(args.config_path)}\t{MSE:.4f}\t{LCC:.4f}\t{SRCC:.4f}\t{KTAU:.4f}".replace(".", ","))
