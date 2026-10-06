"""Exact empirical W1 with Euclidean ground cost, all validation vectors."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import ot
from scipy.spatial.distance import cdist

MODELS = [
    ("mHuBERT-147", "mHuBERT-147"),
    ("mms1b", "mms_1b"),
    ("mms300m", "mms_300m"),
    ("w2vbert", "w2vbert_2.0"),
    ("wav2vec2_1b", "wav2vec2_1b"),
    ("wav2vec2_300m", "wav2vec2_300m"),
    ("wavlm_large", "wavlm_large"),
    ("whisper_large_v3", "whisper_large_v3"),
]
DATASETS = ["brspeech", "bvcc", "singmos", "tmhint"]


def w1(x, y):
    costs = np.ascontiguousarray(cdist(x, y, metric="euclidean"), dtype=np.float64)
    result, details = ot.emd2(np.full(len(x), 1.0 / len(x)),
                             np.full(len(y), 1.0 / len(y)), costs,
                             numItermax=10000000, log=True, numThreads=1)
    if details.get("warning") or details.get("result_code", 1) != 1:
        raise RuntimeError(f"OT solver did not converge: {details}")
    if not np.isfinite(result) or result < 0:
        raise ValueError("Invalid W1 result")
    return float(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Check the unequal-size transport and Euclidean (not squared) cost.
    assert abs(w1(np.array([[0.0], [2.0]]), np.array([[1.0]])) - 1.0) < 1e-12
    assert abs(w1(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])) - 5.0) < 1e-12
    rows = []
    provenance = []
    for display_name, model in MODELS:
        embeddings = {}
        dimensions = set()
        for dataset in DATASETS:
            folder = args.root / model / dataset
            status = json.loads((folder / "status.json").read_text())
            if status["status"] != "succeeded" or status["failed"]:
                raise ValueError(f"Incomplete extraction: {folder}")
            variants = {}
            for variant in ("raw", "afim", "nao_afim"):
                path = folder / f"{variant}.npy"
                x = np.load(path, allow_pickle=False)
                if x.ndim != 2 or len(x) != status["expected"] or not np.isfinite(x).all():
                    raise ValueError(f"Invalid embeddings: {path}")
                variants[variant] = x.astype(np.float64)
                dimensions.add(x.shape[1])
                provenance.append({"path": str(path), "shape": list(x.shape),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if not np.array_equal(variants["afim"], variants["nao_afim"]):
                raise ValueError(f"Affine/non-affine differ: {folder}")
            embeddings[dataset] = variants
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent embedding dimensions: {model}")
        for a, b in itertools.combinations(DATASETS, 2):
            raw = w1(embeddings[a]["raw"], embeddings[b]["raw"])
            ln = w1(embeddings[a]["nao_afim"], embeddings[b]["nao_afim"])
            rows.append({"modelo": display_name, "dataset_a": a, "dataset_b": b,
                         "w1_raw": raw, "w1_afim": ln, "w1_nao_afim": ln,
                         "n_a": len(embeddings[a]["raw"]), "n_b": len(embeddings[b]["raw"]),
                         "dimensao": next(iter(dimensions))})
            print(f"{display_name} {a} {b}: raw={raw:.8f} LN={ln:.8f}", flush=True)
    assert len(rows) == 48
    with (args.output / "wasserstein1_all.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for variant in ("raw", "afim", "nao_afim"):
        with (args.output / f"wasserstein1_{variant}_pt.tsv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["modelo", "dataset a", "dataset b", "wasserstein_1"])
            for row in rows:
                writer.writerow([row["modelo"], row["dataset_a"], row["dataset_b"],
                                 f"{row['w1_' + variant]:.8f}".replace(".", ",")])
    with (args.output / "wasserstein1_raw_layernorm_pt.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["modelo", "dataset a", "dataset b", "W1 bruto", "W1 LayerNorm"])
        for row in rows:
            writer.writerow([row["modelo"], row["dataset_a"], row["dataset_b"],
                             f"{row['w1_raw']:.8f}".replace(".", ","),
                             f"{row['w1_nao_afim']:.8f}".replace(".", ",")])
    metadata = {"metric": "exact multivariate Wasserstein-1", "ground_cost": "euclidean",
                "weights": "uniform per audio, each dataset has total mass 1",
                "split": "val", "sampling": "all rows, no subsampling",
                "normalization": "as saved; no extra normalization or dimensionality reduction",
                "solver": "POT emd2 network simplex", "pot_version": ot.__version__,
                "affine_equals_non_affine": True, "models": MODELS, "rows": len(rows),
                "inputs": provenance}
    (args.output / "method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"COMPLETED {len(rows)} model/dataset pairs", flush=True)


if __name__ == "__main__":
    main()
