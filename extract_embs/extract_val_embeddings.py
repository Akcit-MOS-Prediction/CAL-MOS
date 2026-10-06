#!/usr/bin/env python3
"""Frozen last-layer embeddings, validation only, with two post-pooling LayerNorms.

No model/head training is performed. Backbone-internal LayerNorms stay unchanged.
By default affine gamma=1, beta=0: affine and non-affine outputs are identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import traceback
from pathlib import Path

DATASETS = ("brspeech", "bvcc", "singmos", "tmhint")
ALIASES = {"mms1b": "mms_1b", "mms300m": "mms_300m", "w2vbert": "w2vbert_2.0"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root, selected):
    import yaml
    grouped = {}
    for path in sorted((root / "config/models").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        identifier = config["model_name"]
        item = grouped.setdefault(identifier, {"id": identifier, "configs": [], "eps": float(config.get("layer_norm_eps", 1e-5))})
        if item["eps"] != float(config.get("layer_norm_eps", 1e-5)):
            raise ValueError(f"Different LayerNorm epsilon for duplicate backbone: {identifier}")
        item["configs"].append(path.stem)
    if not grouped:
        raise ValueError("No YAML models found")
    models = []
    for item in grouped.values():
        item["name"] = min(item["configs"], key=lambda name: (len(name), name))
        models.append(item)
    if selected:
        result = []
        for requested in selected:
            requested = ALIASES.get(requested, requested)
            matches = [m for m in models if requested == m["id"] or requested in m["configs"]]
            if len(matches) != 1:
                raise ValueError(f"Unknown/ambiguous model: {requested}")
            if matches[0] not in result:
                result.append(matches[0])
        models = result
    return models


def validation_rows(root, dataset, limit):
    import yaml
    path = root / f"config/datasets/{dataset}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    splits = config["datasets"]["val"]  # Never enumerate train or test.
    rows, sources = [], []
    for split in splits:
        metadata = Path(split["metadata_path"])
        sources.append({"metadata": str(metadata), "sha256": file_hash(metadata), "base_dir": split["base_dir"]})
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            for source_index, record in enumerate(csv.DictReader(stream)):
                audio = Path(record[split["filename_column"]])
                if not audio.is_absolute():
                    audio = Path(split["base_dir"]) / audio
                rows.append({"indice": len(rows), "metadata_csv": str(metadata), "linha_csv": source_index + 2,
                             "audio": str(audio), "mos": record.get(split.get("target_column", "mos"), "")})
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"Empty validation set: {dataset}")
    return rows, sources


def pooled_last_layer(model, processor, audio, rate, device, chunk_seconds, is_whisper):
    import torch
    # Chunk every recording completely; do not silently truncate long audio.
    chunk_samples = max(1, round(rate * chunk_seconds))
    total, frames = None, 0
    for chunk in audio.split(chunk_samples):
        # A very short final chunk is padded; actual duration remains recorded.
        length = chunk.numel()
        if length < 640:
            chunk = torch.nn.functional.pad(chunk, (0, 640 - length))
        batch = processor(chunk.cpu().numpy(), sampling_rate=rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1][0].float()
        if is_whisper:
            # Whisper pads to 30 s: exclude padded encoder positions (320 samples/frame).
            hidden = hidden[:max(1, min(hidden.shape[0], math.ceil(length / 320)))]
        chunk_sum = hidden.sum(dim=0)
        total = chunk_sum if total is None else total + chunk_sum
        frames += hidden.shape[0]
    if frames == 0:
        raise ValueError("No encoder frames")
    return (total / frames).cpu()


def affine_state(item, specification, dimension):
    import torch
    entry = specification.get(item["name"], specification.get(item["id"]))
    if entry is None:
        return torch.ones(dimension), torch.zeros(dimension), {"kind": "identity", "gamma": 1, "beta": 0}
    if "checkpoint" in entry:
        checkpoint = Path(entry["checkpoint"]).resolve(strict=True)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        prefix = entry.get("prefix")
        if prefix is None:
            candidates = [key[:-7] for key in state if key.endswith("input_layer_norm.weight")]
            if len(candidates) != 1:
                raise ValueError("Specify prefix: checkpoint must contain exactly one input_layer_norm.weight")
            prefix = candidates[0]
        weight, bias = state[prefix + ".weight"].float(), state[prefix + ".bias"].float()
        source = {"kind": "checkpoint", "path": str(checkpoint), "sha256": file_hash(checkpoint), "prefix": prefix}
    else:
        weight = torch.tensor(entry["weight"], dtype=torch.float32)
        bias = torch.tensor(entry["bias"], dtype=torch.float32)
        source = {"kind": "explicit", "weight": entry["weight"], "bias": entry["bias"]}
    if weight.shape != (dimension,) or bias.shape != (dimension,):
        raise ValueError(f"Affine parameters must have shape ({dimension},)")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Non-finite affine parameters")
    return weight, bias, source


def extract_dataset(args, item, dataset, model, processor, device, dimension, revision, specification):
    import numpy as np
    import torch
    import torchaudio
    from tqdm import tqdm
    rows, sources = validation_rows(args.root, dataset, args.limit)
    folder = args.output / item["name"] / dataset
    weight, bias, affine = affine_state(item, specification, dimension)
    request = {"model": item, "revision": revision, "dataset": dataset, "split": "val",
               "metadata_sources": sources, "limit": args.limit, "shape": [len(rows), dimension],
               "pooling": "mean_of_last_encoder_hidden_state_then_layernorm", "chunk_seconds": args.chunk_seconds,
               "sampling_rate": int(processor.sampling_rate), "affine": affine, "backbone_frozen": True,
               "whisper_pooling": "exclude_30_second_padding"}
    request_path = folder / "request.json"
    resume = request_path.exists()
    if resume:
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous != request:
            raise ValueError(f"Output belongs to a different request: {folder}; choose a new --output")
    else:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Non-empty output without request.json: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.npy" for name in ("raw", "afim", "nao_afim")}
    # Save the request first so an interrupted matrix initialization is identifiable.
    write_json(request_path, request)
    matrices = {}
    for name, path in paths.items():
        if path.exists():
            if not resume:
                raise ValueError(f"Unexpected existing matrix: {path}")
            matrix = np.load(path, mmap_mode="r+")
            if matrix.shape != (len(rows), dimension) or matrix.dtype != np.float32:
                raise ValueError(f"Invalid matrix: {path}")
        else:
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
            matrix[:] = np.nan
            matrix.flush()
        matrices[name] = matrix
    errors = []
    try:
        with torch.inference_mode():
            for row in tqdm(rows, desc=f"{item['name']} / {dataset}"):
                index = row["indice"]
                if all(np.isfinite(matrix[index]).all() for matrix in matrices.values()):
                    continue
                try:
                    audio, sr = torchaudio.load(row["audio"])
                    audio = audio.mean(dim=0)
                    if not audio.numel() or not torch.isfinite(audio).all():
                        raise ValueError("Empty/non-finite audio")
                    rate = int(processor.sampling_rate)
                    if sr != rate:
                        audio = torchaudio.functional.resample(audio, sr, rate)
                    pooled = pooled_last_layer(model, processor, audio, rate, device, args.chunk_seconds,
                                               "whisper" in item["id"].lower())
                    non_affine = torch.nn.functional.layer_norm(pooled, (dimension,), eps=item["eps"])
                    affine_embedding = torch.nn.functional.layer_norm(pooled, (dimension,), weight, bias, item["eps"])
                    if not all(torch.isfinite(value).all() for value in (pooled, non_affine, affine_embedding)):
                        raise ValueError("Non-finite embedding")
                    for name, value in (("afim", affine_embedding), ("nao_afim", non_affine), ("raw", pooled)):
                        matrices[name][index] = value.numpy()
                        matrices[name].flush()
                except Exception as exc:
                    errors.append({"indice": index, "audio": row["audio"], "error": f"{type(exc).__name__}: {exc}"})
                    print(f"ERROR {item['name']}/{dataset}/{index}: {exc}", file=sys.stderr, flush=True)
                    write_json(folder / "errors.json", errors)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        for matrix in matrices.values():
            matrix.flush()
        ok = np.logical_and.reduce([np.isfinite(matrix).all(axis=1) for matrix in matrices.values()])
        records = [{**row, "status": "succeeded" if ok[row["indice"]] else "failed"} for row in rows]
        write_csv(folder / "index.csv", ("indice", "metadata_csv", "linha_csv", "audio", "mos", "status"), records)
        write_json(folder / "status.json", {"status": "succeeded" if ok.all() else "incomplete",
                   "expected": len(rows), "succeeded": int(ok.sum()), "failed": int((~ok).sum()), "dimension": dimension})
        write_json(folder / "errors.json", errors)
    return "succeeded" if ok.all() else "incomplete"


def pair_rows(args, models):
    rows = []
    for item in models:
        for norm in ("afim", "nao_afim"):
            for a, b in itertools.combinations(args.datasets, 2):
                directories = [args.output / item["name"] / dataset for dataset in (a, b)]
                states = []
                for folder in directories:
                    path = folder / "status.json"
                    states.append(json.loads(path.read_text())["status"] if path.exists() else "pending")
                rows.append({"modelo": item["name"], "layernorm": norm, "dataset_a": a, "dataset_b": b,
                             "status": "succeeded" if states == ["succeeded", "succeeded"] else "/".join(states),
                             "embeddings_a": str(directories[0] / f"{norm}.npy"),
                             "embeddings_b": str(directories[1] / f"{norm}.npy"),
                             "indice_a": str(directories[0] / "index.csv"), "indice_b": str(directories[1] / "index.csv")})
    return rows


def load_pretrained(factory, identifier, args):
    if args.fallback_cache_dir:
        from huggingface_hub import list_repo_files, snapshot_download
        # Use a local snapshot path: AutoModel's PEFT probe in Transformers 4.55
        # ignores the outer cache/local_files_only options for a remote repo ID.
        try:
            snapshot = snapshot_download(identifier, local_files_only=True)
            return factory.from_pretrained(snapshot, local_files_only=True)
        except OSError:
            pass
        cache = str(args.fallback_cache_dir)
        try:
            snapshot = snapshot_download(identifier, cache_dir=cache, local_files_only=True)
            return factory.from_pretrained(snapshot, local_files_only=True)
        except OSError:
            if args.local_files_only:
                raise
        files = list_repo_files(identifier)
        weights = "*.safetensors" if any(name.endswith(".safetensors") for name in files) else "*.bin"
        print(f"Downloading missing backbone to writable cache: {identifier}", flush=True)
        snapshot = snapshot_download(identifier, cache_dir=cache, allow_patterns=["*.json", weights])
        return factory.from_pretrained(snapshot, local_files_only=True)
    return factory.from_pretrained(identifier, local_files_only=args.local_files_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: ROOT/resultados/embeddings_val_last_layer")
    parser.add_argument("--models", nargs="+", help="YAML stems, Hugging Face IDs, or aliases; default all unique backbones")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only: first N validation rows; 0=all")
    parser.add_argument("--affine-params", type=Path, help="JSON per model: weight/bias arrays or checkpoint/prefix")
    parser.add_argument("--local-files-only", action="store_true", help="Disable model downloads")
    parser.add_argument("--fallback-cache-dir", type=Path,
                        help="Use default cache first; download missing files into this writable cache")
    parser.add_argument("--dry-run", action="store_true", help="Print models, validation paths, and pairs; do not load models")
    args = parser.parse_args()
    if not 0 < args.chunk_seconds <= 30 or args.limit < 0 or args.gpu < 0:
        parser.error("0 < chunk-seconds <= 30, limit >= 0, gpu >= 0 required")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Duplicate datasets")
    args.root = args.root.resolve()
    args.output = (args.output or args.root / "resultados/embeddings_val_last_layer").resolve()
    models = discover(args.root, args.models)
    specification = json.loads(args.affine_params.read_text()) if args.affine_params else {}
    if args.dry_run:
        counts = {name: len(validation_rows(args.root, name, args.limit)[0]) for name in args.datasets}
        print(json.dumps({"models": models, "validation_examples": counts,
                          "layernorm": ["afim", "nao_afim"], "dataset_pairs": list(itertools.combinations(args.datasets, 2)),
                          "pair_count": len(models) * 2 * math.comb(len(args.datasets), 2),
                          "output": str(args.output)}, indent=2))
        return 0
    import gc
    import torch
    import transformers
    from transformers import AutoFeatureExtractor, AutoModel
    device = torch.device("cpu" if args.cpu else f"cuda:{args.gpu}")
    if device.type == "cuda" and (not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count()):
        parser.error("Requested GPU unavailable; use --cpu for CPU execution")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Affine LayerNorm is identity-initialized unless --affine-params provides learned gamma/beta.", flush=True)
    tasks, model_errors = [], []
    for item in models:
        model = processor = None
        try:
            processor = load_pretrained(AutoFeatureExtractor, item["id"], args)
            model = load_pretrained(AutoModel, item["id"], args)
            revision = getattr(model.config, "_commit_hash", None)
            if "whisper" in item["id"].lower():
                model = model.get_encoder()
            model = model.to(device).eval().requires_grad_(False)
            dimension = int(getattr(model.config, "hidden_size", None) or model.config.d_model)
            for dataset in args.datasets:
                try:
                    state = extract_dataset(args, item, dataset, model, processor, device, dimension, revision, specification)
                    tasks.append({"model": item["name"], "dataset": dataset, "status": state})
                except Exception as exc:
                    model_errors.append({"model": item["name"], "dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
                    traceback.print_exc()
        except Exception as exc:
            model_errors.append({"model": item["name"], "error": f"{type(exc).__name__}: {exc}"})
            traceback.print_exc()
        finally:
            del model, processor
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_csv(args.output / "pares_val.csv", ("modelo", "layernorm", "dataset_a", "dataset_b", "status",
                      "embeddings_a", "embeddings_b", "indice_a", "indice_b"), pair_rows(args, models))
            write_json(args.output / "summary.json", {"split": "val", "torch": torch.__version__,
                       "transformers": transformers.__version__, "tasks": tasks, "errors": model_errors,
                       "affine_identity_by_default": not bool(specification)})
    failed = bool(model_errors) or any(t["status"] != "succeeded" for t in tasks)
    print(f"Finished: {len(tasks)} dataset tasks; errors={len(model_errors)}; output={args.output}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
