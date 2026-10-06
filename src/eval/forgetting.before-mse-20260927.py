#!/usr/bin/env python3
"""Evaluate one checkpoint on a test split, including latency and partial FLOPs."""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from transformers import AutoFeatureExtractor

from models.calmos_wrapper import CALMOSWrapper
from utils.dataloader import (
    AllLayersEmbeddingCollate,
    DynamicAudioCollate,
    DynamicCollate,
    DynamicDataset,
    EmbeddingDataset,
    OneLayerEmbeddingCollate,
)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    # Hugging Face feature extractors return BatchFeature (a UserDict), not dict.
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, sample, device, batch_size, warmup, iterations):
    for _ in range(warmup):
        model(sample)
    synchronize(device)
    latencies = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter()
        model(sample)
        synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies)
    median_ms = statistics.median(ordered)
    result = {
        "latency_mean_ms": statistics.mean(ordered),
        "latency_median_ms": median_ms,
        "latency_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "latency_per_sample_ms": median_ms / batch_size,
        "latency_iterations": iterations,
        "latency_warmup_iterations": warmup,
        "inference_batch_size_actual": batch_size,
        "gpu_peak_inference_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "flops_scope": "partial_matmul_and_conv2d_only",
        "flops_per_batch_estimate": None,
        "flops_per_sample_estimate": None,
        "flops_status": "unavailable",
        "flops_error": None,
    }
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True, with_flops=True) as profiler:
            model(sample)
            synchronize(device)
        flops = sum(int(event.flops or 0) for event in profiler.key_averages())
        if flops > 0:
            result["flops_per_batch_estimate"] = flops
            result["flops_per_sample_estimate"] = flops / batch_size
            result["flops_status"] = "partial"
        else:
            result["flops_error"] = "Profiler returned no counted FLOPs for supported operators"
    except Exception as exc:
        result["flops_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-iterations", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.latency_warmup < 0 or args.latency_iterations < 1:
        parser.error("batch-size and latency-iterations must be positive; latency-warmup >= 0")

    config = OmegaConf.load(args.config_path)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint is not a file: {checkpoint}")
    test = config.datasets.test[0]
    data = pd.read_csv(test.metadata_path)
    model_type = config.model.model_type.lower()
    dynamic_types = {"dynamic", "dynamic_kan", "dynamic_melspec", "dynamic_kan_melspec"}
    if model_type in dynamic_types:
        dataset = DynamicDataset(
            data=data,
            filename_column=test.filename_column,
            target_column=test.target_column,
            sr_column=test.get("sr_column", None),
            sr_dictionary=config.data.get("sr_dictionary", None),
            base_dir=test.base_dir,
            data_type="test",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
        processor = AutoFeatureExtractor.from_pretrained(config.model.model_name)
        collate = (
            DynamicAudioCollate(target_sr=config.data.target_sr, processor=processor)
            if "melspec" in model_type else
            DynamicCollate(target_sr=config.data.target_sr, processor=processor)
        )
    elif model_type in {"all_layers_embedding", "one_layer_embedding"}:
        dataset = EmbeddingDataset(
            data=data,
            filename_column=test.filename_column,
            target_column=test.target_column,
            base_dir=test.base_dir,
            data_type="test",
        )
        collate = (AllLayersEmbeddingCollate() if model_type == "all_layers_embedding"
                   else OneLayerEmbeddingCollate())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = CALMOSWrapper.load_from_checkpoint(
        str(checkpoint), config=config, map_location="cpu", strict=True
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions = []
    targets = []
    benchmark_input = None
    benchmark_batch_size = None
    with torch.inference_mode():
        for features, target in loader:
            features = move_to_device(features, device)
            if benchmark_input is None or (
                benchmark_batch_size < args.batch_size and target.shape[0] == args.batch_size
            ):
                benchmark_input = features
                benchmark_batch_size = int(target.shape[0])
            output = model(features)
            predictions.extend(output.detach().reshape(-1).cpu().tolist())
            targets.extend(target.reshape(-1).cpu().tolist())
        if len(predictions) < 2 or len(predictions) != len(targets):
            raise ValueError(f"Invalid test predictions: {len(predictions)} predictions, {len(targets)} targets")
        srcc = float(spearmanr(targets, predictions)[0])
        if not math.isfinite(srcc):
            raise ValueError("SRCC is not finite; check predictions and test labels")
        performance = benchmark(
            model, benchmark_input, device, benchmark_batch_size,
            args.latency_warmup, args.latency_iterations,
        )

    result = {
        "srcc": srcc,
        "num_examples": len(targets),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "test_metadata_path": str(test.metadata_path),
        "config_path": str(Path(args.config_path).resolve()),
        "total_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.model.parameters()
                                    if parameter.requires_grad),
        "inference_batch_size_configured": args.batch_size,
        **performance,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, output_path)
    print(f"Test SRCC={srcc:.6f}; n={len(targets)}; latency={performance['latency_median_ms']:.3f} ms/batch; output={output_path}",
          flush=True)


if __name__ == "__main__":
    main()
