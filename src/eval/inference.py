import os
import sys
import glob
import argparse
import time
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
import torch.nn as nn
# from sklearn.metrics import mean_squared_error
# from scipy.stats import spearmanr, pearsonr, kendalltau

from utils.dataloader import (
    DynamicDataset,
    AllLayersEmbeddingCollate,
    OneLayerEmbeddingCollate,
    DynamicCollate,
    DynamicAudioCollate,
)
from models.calmos_wrapper import CALMOSWrapper
from transformers import AutoFeatureExtractor


def move_to_device(input_features, device):
    if isinstance(input_features, (list, tuple)):
        processed_inputs = []
        for i in input_features:
            if isinstance(i, dict):
                processed_inputs.append({k: v.to(device) for k, v in i.items()})
            else:
                processed_inputs.append(i.to(device))
        return tuple(processed_inputs)
    if isinstance(input_features, dict):
        return {k: v.to(device) for k, v in input_features.items()}
    return input_features.to(device)


def describe_input(input_features):
    if isinstance(input_features, dict):
        return {
            key: list(value.shape) if hasattr(value, "shape") else str(type(value))
            for key, value in input_features.items()
        }
    if isinstance(input_features, (list, tuple)):
        return [
            describe_input(value) if isinstance(value, dict)
            else list(value.shape) if hasattr(value, "shape") else str(type(value))
            for value in input_features
        ]
    return list(input_features.shape) if hasattr(input_features, "shape") else str(type(input_features))


def get_batch_size(input_features):
    if isinstance(input_features, dict):
        for value in input_features.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
    if isinstance(input_features, (list, tuple)):
        for value in input_features:
            batch_size = get_batch_size(value)
            if batch_size:
                return batch_size
    if hasattr(input_features, "shape") and len(input_features.shape) > 0:
        return int(input_features.shape[0])
    return 0


def _linear_macs(module, output):
    output_elements = output.numel() if torch.is_tensor(output) else 0
    return int(output_elements * module.in_features)


def _conv_macs(module, output):
    if not torch.is_tensor(output):
        return 0
    output_elements = output.numel()
    kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
    return int(output_elements * kernel_ops)


@torch.no_grad()
def profile_macs_flops(model, input_features):
    totals = {"macs": 0}
    handles = []

    def register(module):
        def hook(mod, _inputs, output):
            if isinstance(mod, nn.Linear):
                totals["macs"] += _linear_macs(mod, output)
            elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                totals["macs"] += _conv_macs(mod, output)
        return hook

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(register(module)))

    try:
        model.eval()
        _ = model(input_features)
    finally:
        for handle in handles:
            handle.remove()

    batch_size = get_batch_size(input_features)
    macs = int(totals["macs"])
    flops = int(2 * macs)
    return {
        "MACS": macs,
        "FLOPS": flops,
        "MACS_PER_SAMPLE": float(macs / batch_size) if batch_size else None,
        "FLOPS_PER_SAMPLE": float(flops / batch_size) if batch_size else None,
        "PROFILE_BATCH_SIZE": batch_size,
        "PROFILE_INPUT_SHAPE": str(describe_input(input_features)),
        "PROFILE_NOTE": "Estimated with Linear/Conv forward hooks on the first inference batch; FLOPs=2*MACs.",
    }


@torch.no_grad
def inference(model, dataloader, device):
    model.eval()

    predictions = []
    targets = []
    profile_metrics = {}

    for batch in tqdm(dataloader, desc="Inference"):
        input_features, target = batch
        input_features = move_to_device(input_features, device)

        if not profile_metrics:
            profile_metrics = profile_macs_flops(model, input_features)

        logits = model(input_features)

        predictions.append(logits.cpu().numpy())
        targets.append(target.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(targets), profile_metrics


def run_inference(config, checkpoint_path, gpu, return_predictions: bool = False):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using checkpoint: {checkpoint_path}")

    model = CALMOSWrapper.load_from_checkpoint(checkpoint_path, config=config, map_location=device, strict=False)
    model = model.to(device)

    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

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

    start_inference = time.perf_counter()
    predictions, targets, profile_metrics = inference(model, test_dataloader, device)
    inference_duration_seconds = time.perf_counter() - start_inference

    true_mean_scores = targets
    predict_mean_scores = predictions
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    
    metrics = {
        "MSE": float(MSE),
        "LCC": float(LCC),
        "SRCC": float(SRCC),
        "KTAU": float(KTAU),
        "N_SAMPLES": int(len(targets)),
        "INFERENCE_DURATION_SECONDS": float(inference_duration_seconds),
        **profile_metrics,
    }

    if return_predictions:
        filename_column = config.datasets.test[0].filename_column
        sample_id = (
            test_data["sample_id"].values if "sample_id" in test_data.columns
            else test_data["id"].values if "id" in test_data.columns
            else np.arange(len(test_data))
        )
        labels = np.asarray(targets).reshape(-1)
        predictions_flat = np.asarray(predictions).reshape(-1)
        audio_paths = [
            path if os.path.isabs(str(path))
            else os.path.join(config.datasets.test[0].base_dir, str(path))
            for path in test_data[filename_column].values
        ]
        predictions_df = pd.DataFrame({
            "sample_id": sample_id,
            "audio_path": audio_paths,
            "label": labels,
            "prediction": predictions_flat,
            "absolute_error": np.abs(labels - predictions_flat),
            "squared_error": (labels - predictions_flat) ** 2,
        })
        return metrics, predictions_df

    return metrics

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
        default=None,
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    
    if args.checkpoint_path is None:
        checkpoint_dir = config.model_checkpoint.get("dirpath", "../checkpoints/mos-prediction")
        checkpoint_dir = os.path.join(checkpoint_dir, config.title)
    else:
        checkpoint_dir = args.checkpoint_path

    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "**", "*.ckpt"), recursive=True)
    # remove "last.ckpt" from the list
    checkpoint_paths = [path for path in checkpoint_paths if "last.ckpt" not in path]
    assert len(checkpoint_paths) >= 1, f"No checkpoint found in {checkpoint_dir}"
    checkpoint_path = checkpoint_paths[0]
    
    metrics = run_inference(config, checkpoint_path, args.gpu)
    print(f"{os.path.basename(args.config_path)}\t{metrics['MSE']:.4f}\t{metrics['LCC']:.4f}\t{metrics['SRCC']:.4f}\t{metrics['KTAU']:.4f}".replace(".", ","))
