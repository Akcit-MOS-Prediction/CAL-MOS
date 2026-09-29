#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-dataset (stratified) paired bootstrap ranking + pairwise significance vs the best,
with Holm–Bonferroni correction across the (B-1) comparisons.

Goal: pick the best 3 overall backbones (optionally collapsing strategies by taking the
best strategy per backbone first), using only ONE training run per setting.

Assumptions about your folder layout:
  experiments_dir/
    CAL-MOS-<model>-<dataset>-(FreezeBackbone-...)-(LayerStrategy-...)-(epochs-...)-(bs-...)-(LR-...)/
      predictions_utt.csv   columns: predicted_mean_scores, true_mean_scores
      predictions_sys.csv   columns: predicted_sys_mean_scores, true_sys_mean_scores

Default ranking metric: system-level SRCC, aggregated as macro-average over datasets.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Metrics
# -------------------------

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def rankdata_avg(a: np.ndarray) -> np.ndarray:
    # average ranks for ties, 1..n
    s = pd.Series(a)
    return s.rank(method="average").to_numpy(dtype=np.float64)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_avg(x)
    ry = rankdata_avg(y)
    return pearsonr_np(rx, ry)


def kendall_tau_fallback(x: np.ndarray, y: np.ndarray) -> float:
    # Try scipy; if not present, fallback to pandas kendall (may also require scipy)
    try:
        from scipy.stats import kendalltau  # type: ignore
        v = kendalltau(x, y, nan_policy="omit").correlation
        return float(v)
    except Exception:
        try:
            return float(pd.Series(x).corr(pd.Series(y), method="kendall"))
        except Exception:
            return float("nan")


def compute_metric(metric: str, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    metric = metric.upper()
    if metric == "MSE":
        return mse(y_pred, y_true)
    if metric == "LCC":
        return pearsonr_np(y_pred, y_true)
    if metric == "SRCC":
        return spearmanr_np(y_pred, y_true)
    if metric == "KTAU":
        return kendall_tau_fallback(y_pred, y_true)
    raise ValueError(f"Unknown metric: {metric}")


def higher_is_better_score(metric: str, value: float) -> float:
    """
    Convert a metric to a 'score' where higher is always better.
    For MSE, lower is better -> score = -MSE.
    For correlations, higher is better -> score = value.
    """
    metric = metric.upper()
    if math.isnan(value):
        return float("-inf")
    if metric == "MSE":
        return -value
    return value


# -------------------------
# Parsing experiment folders
# -------------------------

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RX = re.compile(
    rf"^CAL-MOS-(?P<model_name>.+?)-(?P<dataset_name>BRSpeech|BVCC|SingMOS|TMHINT-QI)"
    rf"-\(FreezeBackbone-(?P<freeze_backbone>True|False)\)"
    rf"-\(LayerStrategy-(?P<layer_strategy>[^)]*)\)"
    rf"-\(epochs-(?P<epochs>\d+)\)"
    rf"-\(bs-(?P<batch_size>\d+)\)"
    rf"-\(LR-(?P<learning_rate>{FLOAT_RE})\)$"
)


def infer_strategy_key(freeze_backbone: bool, layer_strategy: str) -> str:
    if freeze_backbone is False:
        return "FT"
    if freeze_backbone is True and layer_strategy == "per_layer":
        return "LP"  # last layer / per-layer probe (your naming)
    if freeze_backbone is True and layer_strategy == "weighted_sum":
        return "weighted-sum"
    raise ValueError(f"Invalid combo: FreezeBackbone={freeze_backbone}, LayerStrategy={layer_strategy}")


@dataclass(frozen=True, order=True)
class MethodId:
    backbone: str          # model slug from folder, e.g. facebook-wav2vec2-large
    strategy: str          # FT / LP / weighted-sum

    def as_str(self) -> str:
        return f"{self.backbone}::{self.strategy}"


# -------------------------
# Loading predictions
# -------------------------

def load_predictions(exp_dir: Path, level: str) -> Tuple[np.ndarray, np.ndarray]:
    level = level.lower()
    if level not in {"utt", "sys"}:
        raise ValueError("--level must be utt or sys")

    if level == "utt":
        p = exp_dir / "predictions_utt.csv"
        df = pd.read_csv(p)
        y_pred = df["predicted_mean_scores"].to_numpy(dtype=np.float64)
        y_true = df["true_mean_scores"].to_numpy(dtype=np.float64)
        return y_pred, y_true

    p = exp_dir / "predictions_sys.csv"
    df = pd.read_csv(p)
    y_pred = df["predicted_sys_mean_scores"].to_numpy(dtype=np.float64)
    y_true = df["true_sys_mean_scores"].to_numpy(dtype=np.float64)
    return y_pred, y_true


# -------------------------
# Bootstrap + Holm correction
# -------------------------

def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """
    Holm–Bonferroni adjusted p-values for a dict of tests.
    Input keys are test identifiers; output is adjusted p-values per key.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items, start=1):
        raw = (m - i + 1) * p
        raw = min(1.0, raw)
        running_max = max(running_max, raw)
        adj[k] = running_max
    return adj


def bootstrap_pvalue_vs_best(
    best: MethodId,
    other: MethodId,
    data_by_dataset: Dict[str, Dict[MethodId, Tuple[np.ndarray, np.ndarray]]],
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
    agg: str,
    weights: str,
) -> Tuple[float, Tuple[float, float], float, float]:
    """
    Returns:
      p_one_sided, delta_CI_95, delta_mean, delta_obs
    where delta = score(best) - score(other), higher=better score.
    """
    datasets = list(data_by_dataset.keys())

    # observed delta
    per_ds_best = []
    per_ds_other = []
    per_ds_w = []

    for ds in datasets:
        ypb, ytb = data_by_dataset[ds][best]
        ypo, yto = data_by_dataset[ds][other]

        vb = compute_metric(metric, ypb, ytb)
        vo = compute_metric(metric, ypo, yto)

        sb = higher_is_better_score(metric, vb)
        so = higher_is_better_score(metric, vo)

        per_ds_best.append(sb)
        per_ds_other.append(so)

        n = len(ytb)
        per_ds_w.append(n)

    per_ds_best = np.array(per_ds_best, dtype=np.float64)
    per_ds_other = np.array(per_ds_other, dtype=np.float64)
    per_ds_w = np.array(per_ds_w, dtype=np.float64)

    if weights == "equal":
        w = np.ones_like(per_ds_w)
    elif weights == "n":
        w = per_ds_w
    else:
        raise ValueError("--weights must be equal or n")

    def aggregate(x: np.ndarray) -> float:
        # x is per-dataset scores (already higher-is-better)
        if agg == "macro":
            return float(np.average(x, weights=w))
        raise ValueError("--agg must be macro")

    delta_obs = aggregate(per_ds_best) - aggregate(per_ds_other)

    # bootstrap deltas
    deltas = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        ds_scores_best = []
        ds_scores_other = []
        ds_w = []
        for ds in datasets:
            ypb, ytb = data_by_dataset[ds][best]
            ypo, yto = data_by_dataset[ds][other]
            n = len(ytb)
            idx = rng.integers(0, n, size=n, endpoint=False)

            vb = compute_metric(metric, ypb[idx], ytb[idx])
            vo = compute_metric(metric, ypo[idx], yto[idx])

            sb = higher_is_better_score(metric, vb)
            so = higher_is_better_score(metric, vo)

            ds_scores_best.append(sb)
            ds_scores_other.append(so)
            ds_w.append(n)

        ds_scores_best = np.array(ds_scores_best, dtype=np.float64)
        ds_scores_other = np.array(ds_scores_other, dtype=np.float64)
        ds_w = np.array(ds_w, dtype=np.float64)

        if weights == "equal":
            ww = np.ones_like(ds_w)
        else:
            ww = ds_w

        best_agg = float(np.average(ds_scores_best, weights=ww))
        other_agg = float(np.average(ds_scores_other, weights=ww))

        deltas[b] = best_agg - other_agg

    # one-sided p-value: P(delta <= 0)
    # add-one smoothing to avoid 0
    p = (1.0 + float(np.sum(deltas <= 0.0))) / (n_boot + 1.0)

    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    delta_mean = float(deltas.mean())

    return p, (lo, hi), delta_mean, delta_obs


# -------------------------
# Main ranking logic
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", type=str, default="./experiments",)
    ap.add_argument("--datasets", nargs="*", default=["BRSpeech", "BVCC", "SingMOS", "TMHINT-QI"])
    ap.add_argument("--level", choices=["sys", "utt"], default="utt", help="Use system or utterance level predictions.")
    ap.add_argument("--metric", choices=["MSE", "LCC", "SRCC", "KTAU"], default="SRCC",
                    help="Primary metric for ranking/significance.")
    ap.add_argument("--agg", choices=["macro"], default="macro", help="Across-dataset aggregation.")
    ap.add_argument("--weights", choices=["equal", "n"], default="equal",
                    help="Dataset weighting in aggregation: equal (macro) or n (by sample count).")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--collapse_strategies", action="store_true",
                    help="Rank backbones only by picking the best strategy per backbone first.")
    ap.add_argument("--require_all_datasets", action="store_true",
                    help="Drop methods missing any dataset (recommended).")
    ap.add_argument("--top_k", type=int, default=5,
                help="How many models/methods to recommend at the end.")
    args = ap.parse_args()

    exp_root = Path(args.experiments_dir)
    if not exp_root.exists():
        raise FileNotFoundError(exp_root)

    wanted_datasets = set(args.datasets)
    level = args.level
    metric = args.metric.upper()

    # Collect: data_by_dataset[dataset][MethodId] = (y_pred, y_true)
    data_by_dataset: Dict[str, Dict[MethodId, Tuple[np.ndarray, np.ndarray]]] = {}
    # Keep best run per (dataset, method) based on observed score
    best_score_seen: Dict[Tuple[str, MethodId], float] = {}

    for folder in exp_root.iterdir():
        if not folder.is_dir():
            continue
        m = RX.fullmatch(folder.name)
        if not m:
            continue

        ds = m.group("dataset_name")
        if ds not in wanted_datasets:
            continue

        model_name = m.group("model_name")
        freeze_backbone = (m.group("freeze_backbone") == "True")
        layer_strategy = m.group("layer_strategy")
        strat = infer_strategy_key(freeze_backbone, layer_strategy)

        method = MethodId(backbone=model_name, strategy=strat)

        # Need predictions file
        pred_file = folder / ("predictions_sys.csv" if level == "sys" else "predictions_utt.csv")
        if not pred_file.exists():
            continue

        y_pred, y_true = load_predictions(folder, level=level)
        v = compute_metric(metric, y_pred, y_true)
        s = higher_is_better_score(metric, v)

        key = (ds, method)
        if key in best_score_seen and s <= best_score_seen[key]:
            continue  # keep best run for this dataset+method
        best_score_seen[key] = s

        data_by_dataset.setdefault(ds, {})[method] = (y_pred, y_true)

    datasets_present = sorted(set(data_by_dataset.keys()))
    if not datasets_present:
        raise RuntimeError("No matching experiments found. Check --experiments_dir and folder naming.")

    # Determine candidate methods
    all_methods = set()
    for ds in datasets_present:
        all_methods |= set(data_by_dataset[ds].keys())

    if args.require_all_datasets:
        # intersection over datasets
        common = None
        for ds in datasets_present:
            ms = set(data_by_dataset[ds].keys())
            common = ms if common is None else (common & ms)
        methods = sorted(common) if common is not None else []
    else:
        methods = sorted(all_methods)

    if not methods:
        raise RuntimeError("No methods remain after dataset filtering (try without --require_all_datasets).")

    # Optional: collapse strategies -> choose best strategy per backbone using observed aggregate score
    if args.collapse_strategies:
        by_backbone: Dict[str, List[MethodId]] = {}
        for mth in methods:
            by_backbone.setdefault(mth.backbone, []).append(mth)

        def observed_aggregate_score(mth: MethodId) -> float:
            scores = []
            weights = []
            for ds in datasets_present:
                if mth not in data_by_dataset[ds]:
                    return float("-inf")
                yp, yt = data_by_dataset[ds][mth]
                v = compute_metric(metric, yp, yt)
                s = higher_is_better_score(metric, v)
                scores.append(s)
                weights.append(len(yt))
            scores = np.array(scores, dtype=np.float64)
            weights = np.array(weights, dtype=np.float64)
            ww = np.ones_like(weights) if args.weights == "equal" else weights
            return float(np.average(scores, weights=ww))

        chosen: List[MethodId] = []
        for bb, cand in by_backbone.items():
            best_m = max(cand, key=observed_aggregate_score)
            chosen.append(best_m)

        methods = sorted(chosen, key=lambda mth: mth.as_str())

        # Enforce intersection after collapsing (recommended)
        if args.require_all_datasets:
            methods = [mth for mth in methods if all(mth in data_by_dataset[ds] for ds in datasets_present)]

        if not methods:
            raise RuntimeError("No methods remain after collapsing strategies and filtering.")

    # Rank by observed aggregate score (higher-is-better score)
    def agg_score_observed(mth: MethodId) -> float:
        ds_scores = []
        ds_w = []
        for ds in datasets_present:
            if mth not in data_by_dataset[ds]:
                return float("-inf")
            yp, yt = data_by_dataset[ds][mth]
            v = compute_metric(metric, yp, yt)
            ds_scores.append(higher_is_better_score(metric, v))
            ds_w.append(len(yt))
        ds_scores = np.array(ds_scores, dtype=np.float64)
        ds_w = np.array(ds_w, dtype=np.float64)
        ww = np.ones_like(ds_w) if args.weights == "equal" else ds_w
        return float(np.average(ds_scores, weights=ww))

    ranked = sorted(methods, key=agg_score_observed, reverse=True)
    best = ranked[0]

    # Pairwise significance vs best (stratified paired bootstrap)
    rng = np.random.default_rng(args.seed)
    p_raw: Dict[str, float] = {}
    delta_ci: Dict[str, Tuple[float, float]] = {}
    delta_mean: Dict[str, float] = {}
    delta_obs: Dict[str, float] = {}

    for other in ranked[1:]:
        test_id = other.as_str()
        p, ci, dmean, dobs = bootstrap_pvalue_vs_best(
            best=best,
            other=other,
            data_by_dataset={ds: data_by_dataset[ds] for ds in datasets_present},
            metric=metric,
            n_boot=args.n_boot,
            rng=rng,
            agg=args.agg,
            weights=args.weights,
        )
        p_raw[test_id] = p
        delta_ci[test_id] = ci
        delta_mean[test_id] = dmean
        delta_obs[test_id] = dobs

    p_adj = holm_bonferroni(p_raw)

    # Print summary
    print("=" * 110)
    print(f"Datasets used: {datasets_present}")
    print(f"Level/Metric: {level}/{metric} | agg={args.agg} | weights={args.weights}")
    print(f"Bootstrap: n_boot={args.n_boot} seed={args.seed} | alpha={args.alpha}")
    print(f"Best method: {best.as_str()}")
    print("=" * 110)

    # show ranking with per-dataset raw metric values too
    rows = []
    for r, mth in enumerate(ranked, start=1):
        per_ds_vals = {}
        per_ds_scores = []
        per_ds_w = []
        ok = True
        for ds in datasets_present:
            if mth not in data_by_dataset[ds]:
                ok = False
                break
            yp, yt = data_by_dataset[ds][mth]
            v = compute_metric(metric, yp, yt)
            per_ds_vals[ds] = v
            per_ds_scores.append(higher_is_better_score(metric, v))
            per_ds_w.append(len(yt))
        if not ok:
            continue
        ww = np.ones_like(np.array(per_ds_w)) if args.weights == "equal" else np.array(per_ds_w, dtype=np.float64)
        agg_s = float(np.average(np.array(per_ds_scores, dtype=np.float64), weights=ww))

        row = {
            "rank": r,
            "method": mth.as_str(),
            "agg_score(higher=better)": agg_s,
        }
        for ds in datasets_present:
            row[f"{ds}:{metric}"] = per_ds_vals[ds]
        rows.append(row)

    df_rank = pd.DataFrame(rows)
    pd.set_option("display.max_columns", 999)
    pd.set_option("display.width", 220)
    print(df_rank.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Pairwise vs best
    if ranked[1:]:
        print("\n" + "-" * 110)
        print("Pairwise significance vs BEST (one-sided: P(score_best - score_other <= 0)) + Holm–Bonferroni across pairs")
        print("-" * 110)

        tests = []
        for other in ranked[1:]:
            tid = other.as_str()
            tests.append({
                "other": tid,
                "p_raw": p_raw[tid],
                "p_holm": p_adj[tid],
                "reject(best>other)": (p_adj[tid] < args.alpha),
                "delta_obs": delta_obs[tid],
                "delta_boot_mean": delta_mean[tid],
                "delta_ci_lo": delta_ci[tid][0],
                "delta_ci_hi": delta_ci[tid][1],
            })
        df_tests = pd.DataFrame(tests).sort_values("p_raw", ascending=True)
        print(df_tests.to_string(index=False, float_format=lambda x: f"{x:.6g}" if x < 1e-3 else f"{x:.6f}"))

    # Recommend top-3
    # Strategy: pick the highest-ranked methods, but *prefer* those not significantly worse than best (Holm).
    keep = [best]
    for mth in ranked[1:]:
        tid = mth.as_str()
        if p_adj.get(tid, 1.0) >= args.alpha:
            keep.append(mth)

    # If not enough, fill by rank
    k = max(1, int(args.top_k))

    recommended = keep[:k]
    if len(recommended) < k:
        for mth in ranked:
            if mth not in recommended:
                recommended.append(mth)
            if len(recommended) == k:
                break

    print("\n" + "=" * 110)
    print(f"RECOMMENDED TOP-{k} (prefer not significantly worse than best; filled by rank if needed):")
    for i, mth in enumerate(recommended, start=1):
        print(f"  {i}) {mth.as_str()}")
    print("=" * 110)

    # Notes for sys-level with very few systems
    if level == "sys":
        # quick warn if very small N for any dataset
        small = []
        for ds in datasets_present:
            # pick best method's sys sample size as representative
            n = len(data_by_dataset[ds][best][1])
            if n < 20:
                small.append((ds, n))
        if small:
            print("\n[NOTE] System-level sample sizes are small for some datasets (bootstrap may be noisy):")
            for ds, n in small:
                print(f"  - {ds}: n_systems={n}")
            print("      Consider also running the same script with --level utt for a complementary view.")


if __name__ == "__main__":
    main()
