#!/usr/bin/env python3
"""Fit a shared Yeo-Johnson transformer from pooled WavLM train embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import PowerTransformer

DATASETS = ("brspeech", "bvcc", "singmos", "tmhint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="CAL-MOS project root")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON path for fitted parameters")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing parameters: {output}")

    matrices = []
    sources = []
    dimensions = set()
    for dataset in DATASETS:
        folder = root / "resultados/embeddings_train_last_layer/wavlm_large" / dataset
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        if request.get("split") != "train":
            raise ValueError(f"Expected train embeddings, got {request.get('split')}: {folder}")
        if status.get("status") != "succeeded" or status.get("failed"):
            raise ValueError(f"Incomplete extraction: {folder}")
        path = folder / "raw.npy"
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or len(values) != int(status["expected"]):
            raise ValueError(f"Unexpected shape at {path}: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite embeddings at {path}")
        matrices.append(np.asarray(values, dtype=np.float32))
        dimensions.add(values.shape[1])
        sources.append({"dataset": dataset, "path": str(path),
                        "rows": int(len(values)), "sha256": sha256(path)})

    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dimensions}")
    train = np.concatenate(matrices, axis=0)
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    transformer.fit(train)
    scaler = transformer._scaler
    if scaler is None:
        raise RuntimeError("PowerTransformer did not fit its standardization scaler")
    check = transformer.transform(train[:min(128, len(train))])
    if not np.isfinite(check).all():
        raise ValueError("Fitted transformer produced non-finite values")

    artifact = {
        "type": "power_transformer_yeo_johnson",
        "method": "yeo-johnson",
        "standardize": True,
        "lambdas": transformer.lambdas_.astype(float).tolist(),
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "fit_split": "train",
        "fit_datasets": list(DATASETS),
        "n_train_examples": int(len(train)),
        "input_dim": int(train.shape[1]),
        "source_embeddings": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Fitted {artifact['method']} on {len(train)} TRAIN embeddings "
          f"({train.shape[1]} dimensions, {len(DATASETS)} datasets)")
    print(f"Parameters: {output}")


if __name__ == "__main__":
    main()
