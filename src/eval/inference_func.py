import os
import sys
import glob
import argparse
import numpy as np
import scipy.stats
from collections import defaultdict

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

from utils.dataloader import (
    DynamicDataset,
    AllLayersEmbeddingCollate,
    OneLayerEmbeddingCollate,
    DynamicCollate,
    DynamicAudioCollate,
    EmbeddingDataset,
    MultiLayerEmbeddingDataset
)
from models.calmos_wrapper import CALMOSWrapper
from transformers import AutoFeatureExtractor


@torch.no_grad
def get_predictions(model, dataloader, device):
    model.eval()

    predictions = []
    targets = []

    for batch in tqdm(dataloader, desc="Inference"):
        input_features, target = batch

        if isinstance(input_features, (list, tuple)):
            # If inputs is a list or tuple, process each element based on its type
            processed_inputs = []
            for i in input_features:
                if isinstance(i, dict):
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

        out = model(input_features)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            logits, aux_loss = out
        else:
            logits, aux_loss = out, None

        predictions.append(logits.cpu().numpy())
        targets.append(target.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(targets)


def inference(config: OmegaConf, checkpoint_path: str, device: torch.device) -> None:
    sys_id_column = config.datasets.test[0].get("sys_id_column", "system_id")
    eval_sys_results = defaultdict(lambda: defaultdict(list))

    sys_true_means = []
    sys_pred_means = []

    model = CALMOSWrapper.load_from_checkpoint(checkpoint_path, config=config, map_location=device, strict=False)

    model = model.to(device)

    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

    if config.model.model_type.lower() == "dynamic" \
        or config.model.model_type.lower() == "dynamic_kan" \
        or config.model.model_type.lower() == "dynamic_melspec" \
        or config.model.model_type.lower() == "dynamic_kan_melspec":
        print("\n\nDynamic model\n\n")
        test_dataset = DynamicDataset(
            data=test_data,
            filename_column=config.datasets.test[0].filename_column,
            target_column=config.datasets.test[0].target_column,
            sr_column=config.datasets.test[0].get("sr_column", None), # Backward compatibility
            sr_dictionary=config.data.get("sr_dictionary", None), # Backward compatibility
            base_dir=config.datasets.test[0].base_dir,
            data_type="test",
        )
    elif config.model.model_type.lower() == "one_layer_embedding":
        print("\n\nOne layer embedding model\n\n")
        test_dataset = EmbeddingDataset(
            data=test_data,
            filename_column=config.datasets.test[0].filename_column,
            target_column=config.datasets.test[0].target_column,
            base_dir=config.datasets.test[0].base_dir,
            data_type="test",
        )

    elif config.model.model_type.lower() == "multiple_layer_embedding":
        print("\n\nMultiple layer embedding model\n\n")
        test_dataset = MultiLayerEmbeddingDataset(
            data=test_data,
            filename_column=config.datasets.test[0].filename_column,
            target_column=config.datasets.test[0].target_column,
            base_dir=config.datasets.test[0].base_dir,
            target_dir=config.datasets.test[0].get("target_dir", None),
            layer_id=config.model.specific_layer_idx,
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
    elif config.model.model_type.lower() == "one_layer_embedding" or \
        config.model.model_type.lower() == "multiple_layer_embedding":
        collate_fn = OneLayerEmbeddingCollate()
    else:
        raise ValueError(f"Invalid model type: {config.model.model_type}")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    targets_metadata = test_data[config.datasets.test[0].target_column].values
    predictions, targets = get_predictions(model, test_dataloader, device)

    targets_metadata = np.asarray(targets_metadata).reshape(-1).astype(np.float64)
    predictions = np.asarray(predictions).reshape(-1).astype(np.float64)
    targets = np.asarray(targets).reshape(-1).astype(np.float64)

    # clip predictions to [1, 5]
    predictions = np.clip(predictions, 1.0, 5.0)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    assert targets.ndim == 1 and predictions.ndim == 1
    assert len(targets) == len(test_data) == len(predictions)
    assert np.isfinite(targets).all() and np.isfinite(predictions).all()

    # assert that targets_metadata and targets are the same
    assert np.allclose(targets_metadata, targets), "Targets from metadata and dataloader do not match!"

    true_mean_scores = targets
    predict_mean_scores = predictions

    # Use zip to align the DataFrame row safely with the numpy array position
    for (idx, row), model_pred, target in zip(test_data.iterrows(), predict_mean_scores, true_mean_scores):
        sys_id = row[sys_id_column]

        # Populate system-level lists
        eval_sys_results["predictions"][sys_id].append(model_pred)
        eval_sys_results["targets"][sys_id].append(target)

    # Iterate over keys to ensure order consistency between true and pred lists
    for sys_id in eval_sys_results["predictions"].keys():
        sys_true_mean = np.mean(eval_sys_results["targets"][sys_id])
        sys_pred_mean = np.mean(eval_sys_results["predictions"][sys_id])
        sys_true_means.append(sys_true_mean)
        sys_pred_means.append(sys_pred_mean)

    sys_true_means = np.array(sys_true_means)
    sys_pred_means = np.array(sys_pred_means)

    sys_MSE = np.mean((sys_true_means - sys_pred_means) ** 2)
    sys_LCC = np.corrcoef(sys_true_means, sys_pred_means)[0][1]
    sys_SRCC = scipy.stats.spearmanr(sys_true_means, sys_pred_means)[0]
    sys_KTAU = scipy.stats.kendalltau(sys_true_means, sys_pred_means)[0]

    utt_MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    utt_LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    utt_SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    utt_KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]

    print(f"System-level Results:")
    print(f"MSE: {sys_MSE:.3f}")
    print(f"LCC: {sys_LCC:.3f}")
    print(f"SRCC: {sys_SRCC:.3f}")
    print(f"KTAU: {sys_KTAU:.3f}")

    print(f"Utterance-level Results:")
    print(f"MSE: {utt_MSE:.3f}")
    print(f"LCC: {utt_LCC:.3f}")
    print(f"SRCC: {utt_SRCC:.3f}")
    print(f"KTAU: {utt_KTAU:.3f}")

    results = {
        "system_level": {
            "MSE": sys_MSE,
            "LCC": sys_LCC,
            "SRCC": sys_SRCC,
            "KTAU": sys_KTAU,
        },
        "utterance_level": {
            "MSE": utt_MSE,
            "LCC": utt_LCC,
            "SRCC": utt_SRCC,
            "KTAU": utt_KTAU,
        },
    }

    return results, predict_mean_scores, true_mean_scores, sys_pred_means, sys_true_means