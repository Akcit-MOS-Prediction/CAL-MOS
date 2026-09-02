import os
import sys
import argparse
import itertools
import copy
import re
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
import lightning as L
import multiprocessing as mp
import queue
import warnings
import signal
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning:pydantic")

sys.path.append(os.path.dirname(__file__))

from main import run_train
from eval.inference import run_inference

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTOS SELECIONADOS  (modo --mode selected)
# Formato: (dataset_config, model_config, learning_rate, mlp_dropout)
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_EXPERIMENTS = [
    # ("config/datasets/tmhint.yaml", "config/models/wav2vec2_300m_kan.yaml", 2, 6)
]

EXPERIMENTOS_IGNORADOS = []

DATASET_CONFIG_BY_NAME = {
    "brspeech": "config/datasets/brspeech.yaml",
    "bvcc": "config/datasets/bvcc.yaml",
    "singmos": "config/datasets/singmos.yaml",
    "tmhint": "config/datasets/tmhint.yaml",
}

# Lista histórica de um checkpoint escolhido manualmente por par. Mantida como
# referência, mas não usada pelo modo cross_finetune: a descoberta dinâmica
# abaixo seleciona duas das três runs sem depender destes caminhos.
LEGACY_CROSS_FINETUNE_SOURCES = [
    ("brspeech", "config/models/mHuBERT-147.yaml", "/checkpoints/CAL-MOS-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-(WD-1e-06)-brspeech--(run-02)/epoch=78-step=2449-val/spearman=0.7031.ckpt"),
    ("brspeech", "config/models/mHuBERT-147_kan.yaml", "/checkpoints/KAN-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-brspeech--(run-03)/epoch=58-step=1829-val/spearman=0.6930.ckpt"),
    ("brspeech", "config/models/wav2vec2_300m.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-brspeech--(run-02)/epoch=64-step=520-val/spearman=0.6700.ckpt"),
    ("brspeech", "config/models/wav2vec2_300m_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-brspeech--(run-03)/epoch=51-step=1612-val/spearman=0.6813.ckpt"),
    ("brspeech", "config/models/wav2vec2_1b.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-brspeech--(run-03)/epoch=30-step=248-val/spearman=0.6857.ckpt"),
    ("brspeech", "config/models/wav2vec2_1b_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-brspeech--(run-02)/epoch=26-step=837-val/spearman=0.6667.ckpt"),

    ("bvcc", "config/models/mHuBERT-147.yaml", "/checkpoints/CAL-MOS-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-(WD-1e-06)-bvcc--(run-03)/epoch=81-step=6396-val/spearman=0.8320.ckpt"),
    ("bvcc", "config/models/mHuBERT-147_kan.yaml", "/checkpoints/KAN-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-bvcc--(run-03)/epoch=79-step=6240-val/spearman=0.8205.ckpt"),
    ("bvcc", "config/models/wav2vec2_300m.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-bvcc--(run-03)/epoch=85-step=1720-val/spearman=0.8165.ckpt"),
    ("bvcc", "config/models/wav2vec2_300m_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-bvcc--(run-03)/epoch=58-step=4602-val/spearman=0.8073.ckpt"),
    ("bvcc", "config/models/wav2vec2_1b.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-bvcc--(run-02)/epoch=94-step=1900-val/spearman=0.8401.ckpt"),
    ("bvcc", "config/models/wav2vec2_1b_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-bvcc--(run-02)/epoch=75-step=5928-val/spearman=0.8316.ckpt"),

    ("singmos", "config/models/mHuBERT-147.yaml", "/checkpoints/CAL-MOS-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-(WD-1e-06)-singmos--(run-02)/epoch=62-step=2016-val/spearman=0.6278.ckpt"),
    ("singmos", "config/models/mHuBERT-147_kan.yaml", "/checkpoints/KAN-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-singmos--(run-02)/epoch=65-step=2112-val/spearman=0.6130.ckpt"),
    ("singmos", "config/models/wav2vec2_300m.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-singmos--(run-02)/epoch=42-step=344-val/spearman=0.5918.ckpt"),
    ("singmos", "config/models/wav2vec2_300m_kan_mean.yaml", "/checkpoints/KAN-MEAN-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-singmos--(run-03)/epoch=46-step=1504-val/spearman=0.6065.ckpt"),
    ("singmos", "config/models/wav2vec2_1b.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-singmos--(run-01)/epoch=28-step=232-val/spearman=0.6277.ckpt"),
    ("singmos", "config/models/wav2vec2_1b_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-singmos--(run-02)/epoch=46-step=1504-val/spearman=0.6411.ckpt"),

    ("tmhint", "config/models/mHuBERT-147.yaml", "/checkpoints/CAL-MOS-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-(WD-1e-06)-tmhint--(run-01)/epoch=33-step=6188-val/spearman=0.6143.ckpt"),
    ("tmhint", "config/models/mHuBERT-147_kan.yaml", "/checkpoints/KAN-utter-project/mHuBERT-147-(epochs-100)-(bs-16)-(LR-0.0005)-tmhint--(run-02)/epoch=40-step=7462-val/spearman=0.6154.ckpt"),
    ("tmhint", "config/models/wav2vec2_300m.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-tmhint--(run-03)/epoch=56-step=2622-val/spearman=0.5899.ckpt"),
    ("tmhint", "config/models/wav2vec2_300m_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-300m-(epochs-100)-(bs-16)-(LR-0.001)-tmhint--(run-02)/epoch=65-step=12012-val/spearman=0.5899.ckpt"),
    ("tmhint", "config/models/wav2vec2_1b.yaml", "/checkpoints/CAL-MOS-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-(WD-1e-06)-tmhint--(run-02)/epoch=73-step=3404-val/spearman=0.5989.ckpt"),
    ("tmhint", "config/models/wav2vec2_1b_kan.yaml", "/checkpoints/KAN-facebook/wav2vec2-xls-r-1b-(epochs-100)-(bs-16)-(LR-0.001)-tmhint--(run-01)/epoch=39-step=7280-val/spearman=0.6045.ckpt"),
]

CROSS_FINETUNE_OVERRIDES = {
    # Example:
    # "trainer.max_epochs": 30,
    # "early_stopping.patience": 10,
}

# Diretórios produzidos pelos seis modelos do estudo. Em cross_finetune, o
# melhor checkpoint de cada run é descoberto pela métrica val/spearman gravada
# no caminho. Isso evita manter dezenas de caminhos absolutos manualmente.
CROSS_FINETUNE_MODEL_LAYOUTS = [
    {
        "model_id": "mhubert_mlp",
        "model_config": "config/models/mHuBERT-147.yaml",
        "checkpoint_group": "CAL-MOS-utter-project",
        "run_prefix": "mHuBERT-147-",
    },
    {
        "model_id": "mhubert_kan",
        "model_config": "config/models/mHuBERT-147_kan.yaml",
        "checkpoint_group": "KAN-utter-project",
        "run_prefix": "mHuBERT-147-",
    },
    {
        "model_id": "wav2vec2_1b_mlp",
        "model_config": "config/models/wav2vec2_1b.yaml",
        "checkpoint_group": "CAL-MOS-facebook",
        "run_prefix": "wav2vec2-xls-r-1b-",
    },
    {
        "model_id": "wav2vec2_1b_kan",
        "model_config": "config/models/wav2vec2_1b_kan.yaml",
        "checkpoint_group": "KAN-facebook",
        "run_prefix": "wav2vec2-xls-r-1b-",
    },
    {
        "model_id": "wav2vec2_300m_mlp",
        "model_config": "config/models/wav2vec2_300m.yaml",
        "checkpoint_group": "CAL-MOS-facebook",
        "run_prefix": "wav2vec2-xls-r-300m-",
    },
    {
        "model_id": "wav2vec2_300m_kan",
        "model_config": "config/models/wav2vec2_300m_kan.yaml",
        "checkpoint_group": "KAN-facebook",
        "run_prefix": "wav2vec2-xls-r-300m-",
    },
]

ALL_CROSS_MODEL_IDS = [layout["model_id"] for layout in CROSS_FINETUNE_MODEL_LAYOUTS]
PILOT_CROSS_MODEL_IDS = ["mhubert_mlp", "mhubert_kan"]


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DOS NOMES DE SAÍDA
# ─────────────────────────────────────────────────────────────────────────────
def build_run_filenames(
    dataset_configs: list[str],
    model_configs: list[str],
    grid: dict,
    timestamp: str,
) -> tuple[str, str]:
    ds_names    = "+".join(sorted(os.path.splitext(os.path.basename(p))[0] for p in dataset_configs))
    model_names = "+".join(sorted(os.path.splitext(os.path.basename(p))[0] for p in model_configs))
    grid_keys   = "+".join(k.split(".")[-1] for k in sorted(grid.keys())) if grid else "no_grid"
    base        = f"{ds_names}__{model_names}__{grid_keys}__{timestamp}"
    csv_path    = f"experiments__{base}.csv"
    state_path  = f"grid_state__{base}.yaml"
    return csv_path, state_path


# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────
def format_duration(start: datetime, end: datetime) -> str:
    total = int((end - start).total_seconds())
    h, rem = divmod(total, 3600)
    m, _   = divmod(rem, 60)
    return f"{h}:{m:02d}"


def format_hp_for_name(combo: dict) -> str:
    parts = []
    for k, v in combo.items():
        key = k.split(".")[-1]
        key = {
            "learning_rate": "lr",
            "weight_decay":  "wd",
            "relukan_k":     "k",
            "relukan_grid":  "g",
        }.get(key, key)
        val = (
            f"{v:.1e}".replace(".0e", "e").replace("-0", "-")
            if isinstance(v, float) and v < 0.01
            else str(v)
        )
        parts.append(f"{key}{val}")
    return "_".join(parts)


def append_to_csv(path: str, entry: dict, lock: mp.Lock) -> None:
    with lock:
        df = pd.DataFrame([entry])
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as f:
            df.to_csv(f, header=write_header, index=False, lineterminator="\n")


def dataset_name_from_config(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def checkpoint_spearman(path: str) -> float | None:
    normalized = path.replace("\\", "/")
    match = re.search(
        r"(?:val/)?spearman=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\.ckpt$",
        normalized,
    )
    return float(match.group(1)) if match else None


def best_checkpoint_in_run(run_dir: str) -> tuple[str, float] | None:
    candidates = []
    for root, _, filenames in os.walk(run_dir):
        for filename in filenames:
            if not filename.endswith(".ckpt") or filename == "last.ckpt":
                continue
            checkpoint_path = os.path.join(root, filename)
            score = checkpoint_spearman(checkpoint_path)
            if score is not None:
                candidates.append((checkpoint_path, score))
    return max(candidates, key=lambda item: item[1]) if candidates else None


def discover_cross_finetune_sources(
    checkpoint_root: str,
    checkpoints_per_model_dataset: int,
    source_datasets: list[str] | None = None,
    model_ids: list[str] | None = None,
) -> list[dict]:
    if checkpoints_per_model_dataset < 1:
        raise ValueError("--checkpoints-per-model-dataset deve ser maior que zero.")

    selected_sources = []
    missing = []

    dataset_names = source_datasets or list(DATASET_CONFIG_BY_NAME)
    allowed_model_ids = set(model_ids or [])

    for layout in CROSS_FINETUNE_MODEL_LAYOUTS:
        if allowed_model_ids and layout["model_id"] not in allowed_model_ids:
            continue
        group_dir = os.path.join(checkpoint_root, layout["checkpoint_group"])
        if not os.path.isdir(group_dir):
            missing.append(f"diretório ausente: {group_dir}")
            continue

        run_directories = [
            entry
            for entry in os.scandir(group_dir)
            if entry.is_dir() and entry.name.startswith(layout["run_prefix"])
        ]

        for source_dataset in dataset_names:
            dataset_marker = f"-{source_dataset}--(run-"
            source_runs = []
            for entry in run_directories:
                if dataset_marker not in entry.name:
                    continue
                run_match = re.search(r"\(run-(\d+)\)$", entry.name)
                if not run_match:
                    continue
                best = best_checkpoint_in_run(entry.path)
                if best is None:
                    continue
                checkpoint_path, score = best
                source_runs.append({
                    "model_id": layout["model_id"],
                    "source_dataset": source_dataset,
                    "model_config": layout["model_config"],
                    "init_checkpoint": checkpoint_path,
                    "source_checkpoint_run": int(run_match.group(1)),
                    "source_checkpoint_val_spearman": score,
                })

            source_runs.sort(
                key=lambda item: (
                    item["source_checkpoint_val_spearman"],
                    -item["source_checkpoint_run"],
                ),
                reverse=True,
            )
            chosen = source_runs[:checkpoints_per_model_dataset]
            if len(chosen) < checkpoints_per_model_dataset:
                missing.append(
                    f"{layout['model_config']} / {source_dataset}: "
                    f"encontrados {len(chosen)} de {checkpoints_per_model_dataset} checkpoints"
                )
            for rank, source in enumerate(chosen, start=1):
                source["source_checkpoint_rank"] = rank
                selected_sources.append(source)

    if missing:
        print("\n[Descoberta de checkpoints] Avisos:")
        for message in missing:
            print(f"  - {message}")
    if not selected_sources:
        raise FileNotFoundError(
            f"Nenhum checkpoint compatível foi encontrado em '{checkpoint_root}'."
        )
    return selected_sources


def evaluate_on_dataset(
    config,
    dataset_config_path: str,
    checkpoint_path: str,
    gpu: int,
    return_predictions: bool = False,
):
    eval_config = copy.deepcopy(config)
    dataset_config = OmegaConf.load(dataset_config_path)
    eval_config.datasets.test = dataset_config.datasets.test
    return run_inference(eval_config, checkpoint_path, gpu, return_predictions=return_predictions)


def head_type_from_model_name(model_name: str) -> str:
    return "KAN" if "_kan" in model_name.lower() else "MLP"


def metric_delta(after, before):
    if after is None or before is None:
        return None
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


def build_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 2
    while True:
        candidate = f"{root}__v{idx:02d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def save_checkpoint_predictions(
    predictions_df: pd.DataFrame,
    metrics: dict,
    exp_id: str,
    run_idx: int,
    model_name: str,
    evaluation_role: str,
    evaluation_dataset: str,
    metadata_path: str,
    checkpoint_path: str,
    metadata: dict,
    lock: mp.Lock,
) -> str:
    if predictions_df is None or predictions_df.empty:
        return "N/A"

    run_name = f"run_{run_idx:02d}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    predictions_dir = os.path.join(os.path.dirname(checkpoint_path), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    filename = (
        f"{run_name}__{evaluation_role}__{evaluation_dataset}__{date_str}.csv"
    )
    requested_path = os.path.join(predictions_dir, filename)
    manifest_path = os.path.join(os.path.dirname(checkpoint_path), "predictions_manifest.csv")

    prediction_columns = [
        "sample_id",
        "audio_path",
        "label",
        "prediction",
        "absolute_error",
        "squared_error",
    ]

    with lock:
        output_path = build_unique_path(requested_path)
        predictions_df.loc[:, prediction_columns].to_csv(output_path, index=False, lineterminator="\n")

        manifest_entry = {
            "experiment_id": exp_id,
            "run_name": run_name,
            "dataset": evaluation_dataset,
            "evaluation_role": evaluation_role,
            "model": model_name,
            "head_type": head_type_from_model_name(model_name),
            "checkpoint_path": checkpoint_path,
            "predictions_path": output_path,
            "metadata_path": metadata_path,
            "initial_dataset": metadata.get("source_dataset", "N/A"),
            "finetune_dataset": metadata.get("target_dataset", evaluation_dataset),
            "initial_checkpoint_path": metadata.get("init_checkpoint", "N/A"),
            "mse": metrics.get("MSE"),
            "pearson": metrics.get("LCC"),
            "spearman": metrics.get("SRCC"),
            "n_samples": metrics.get("N_SAMPLES"),
            "inference_duration_seconds": metrics.get("INFERENCE_DURATION_SECONDS"),
            "mean_macs": metrics.get("MACS_PER_SAMPLE"),
            "mean_flops": metrics.get("FLOPS_PER_SAMPLE"),
        }
        manifest_df = pd.DataFrame([manifest_entry])
        write_header = not os.path.exists(manifest_path)
        with open(manifest_path, "a", encoding="utf-8") as f:
            manifest_df.to_csv(f, header=write_header, index=False, lineterminator="\n")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO (YAML)
# ─────────────────────────────────────────────────────────────────────────────
def load_state(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_state(yaml_path: str, state: dict, lock: mp.Lock) -> None:
    with lock:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def patch_state(yaml_path: str, exp_id: str, updates: dict, lock: mp.Lock) -> None:
    with lock:
        state = load_state(yaml_path)
        state.setdefault(exp_id, {}).update(updates)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSO FILHO
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_experiment(
    exp_data: tuple,
    args_dict: dict,
    seed: int,
    lock: mp.Lock,
    output_csv: str,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_path, config_dict, metadata, real_gpu_id = exp_data

    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_id)
    L.seed_everything(seed + run_idx - 1)

    dataset_name = os.path.splitext(os.path.basename(ds_path))[0]
    model_name   = os.path.splitext(os.path.basename(mod_path))[0]

    config = OmegaConf.create(config_dict)

    class _Args:
        pass
    local_args = _Args()
    for k, v in args_dict.items():
        setattr(local_args, k, v)
    local_args.gpu = 0
    local_args.checkpoint_dir = args_dict.get("checkpoint_root", "/checkpoints")

    patch_state(yaml_path, exp_id, {
        "status":     "running",
        "gpu":        real_gpu_id,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, lock)

    # Log de HPs do grid apenas se existirem no config
    if combo:
        hp_log = ", ".join(
            f"{k.split('.')[-1]}={v}" for k, v in combo.items()
        )
        print(f"[{exp_id}] {hp_log}")

    start_train = datetime.now()
    initial_source_metrics = {}
    initial_target_metrics = {}
    initial_source_predictions_path = "N/A"
    initial_target_predictions_path = "N/A"

    try:
        # ── Baseline antes do fine-tuning ────────────────────────────────────
        # Esta medição é indispensável: sem SRCC_X antes e depois, não existe
        # uma estimativa de esquecimento catastrófico.
        init_checkpoint = metadata.get("init_checkpoint")
        if init_checkpoint and metadata.get("source_dataset_config"):
            print(
                f"[{exp_id}] Baseline pré-fine-tuning | "
                f"checkpoint={init_checkpoint}"
            )
            initial_source_metrics, initial_source_predictions = evaluate_on_dataset(
                config,
                metadata["source_dataset_config"],
                init_checkpoint,
                0,
                return_predictions=True,
            )
            initial_target_metrics, initial_target_predictions = run_inference(
                config,
                init_checkpoint,
                0,
                return_predictions=True,
            )

            source_dataset_config = OmegaConf.load(metadata["source_dataset_config"])
            initial_source_predictions_path = save_checkpoint_predictions(
                initial_source_predictions,
                initial_source_metrics,
                exp_id,
                run_idx,
                model_name,
                "initial_source",
                metadata["source_dataset"],
                source_dataset_config.datasets.test[0].metadata_path,
                init_checkpoint,
                metadata,
                lock,
            )
            initial_target_predictions_path = save_checkpoint_predictions(
                initial_target_predictions,
                initial_target_metrics,
                exp_id,
                run_idx,
                model_name,
                "initial_target",
                metadata["target_dataset"],
                config.datasets.test[0].metadata_path,
                init_checkpoint,
                metadata,
                lock,
            )

        start_train = datetime.now()
        # ── Treino ────────────────────────────────────────────────────────────
        if config.get("finetune", {}).get("init_checkpoint"):
            print(
                f"[{exp_id}] Fine-tuning inicializado de checkpoint | "
                f"source={config.finetune.source_dataset} -> target={config.finetune.target_dataset} | "
                f"ckpt={config.finetune.init_checkpoint}"
            )
        metrics, best_ckpt, wandb_run_id = run_train(config, local_args)
        end_train = datetime.now()

        if best_ckpt and os.path.isfile(best_ckpt):
            OmegaConf.save(config, os.path.join(os.path.dirname(best_ckpt), "config.yaml"))
        else:
            best_ckpt = "N/A"
            print(f"[{exp_id}] ⚠️  run_train não retornou um checkpoint válido.")

        # ── Inferência ────────────────────────────────────────────────────────
        inf_metrics = {}
        source_inf_metrics = {}
        target_predictions_df = pd.DataFrame()
        source_predictions_df = pd.DataFrame()
        target_predictions_path = "N/A"
        source_predictions_path = "N/A"
        dur_inf = "N/A"
        if best_ckpt != "N/A":
            print(f"[{exp_id}] Iniciando inferência | ckpt: {best_ckpt}")
            start_inf   = datetime.now()
            inf_metrics, target_predictions_df = run_inference(
                config, best_ckpt, 0, return_predictions=True
            )
            if metadata.get("source_dataset_config"):
                source_inf_metrics, source_predictions_df = evaluate_on_dataset(
                    config,
                    metadata["source_dataset_config"],
                    best_ckpt,
                    0,
                    return_predictions=True,
                )
            dur_inf     = format_duration(start_inf, datetime.now())

            target_predictions_path = save_checkpoint_predictions(
                target_predictions_df,
                inf_metrics,
                exp_id,
                run_idx,
                model_name,
                "post_finetune_target",
                metadata.get("target_dataset", dataset_name),
                config.datasets.test[0].metadata_path,
                best_ckpt,
                metadata,
                lock,
            )

            if metadata.get("source_dataset_config"):
                source_dataset_config = OmegaConf.load(metadata["source_dataset_config"])
                source_predictions_path = save_checkpoint_predictions(
                    source_predictions_df,
                    source_inf_metrics,
                    exp_id,
                    run_idx,
                    model_name,
                    "post_finetune_source",
                    metadata.get("source_dataset", "N/A"),
                    source_dataset_config.datasets.test[0].metadata_path,
                    best_ckpt,
                    metadata,
                    lock,
                )

        # ── CSV ───────────────────────────────────────────────────────────────
        entry = {
            "experiment_id":  exp_id,
            "config_id":      f"config_{config_id:03d}",
            "run_idx":        run_idx,
            "dataset":        dataset_name,
            "source_dataset": metadata.get("source_dataset", "N/A"),
            "initial_dataset": metadata.get("source_dataset", "N/A"),
            "finetune_dataset": metadata.get("target_dataset", dataset_name),
            "model":          model_name,
            "model_id":       metadata.get("model_id", model_name),
            "mode":           args_dict.get("mode"),
            "wandb_run_id":   wandb_run_id or "N/A",
            "dataset_config": ds_path,
            "model_config":   mod_path,
            "init_checkpoint_path": metadata.get("init_checkpoint", "N/A"),
            "initial_checkpoint_path": metadata.get("init_checkpoint", "N/A"),
            "source_checkpoint_run": metadata.get("source_checkpoint_run"),
            "source_checkpoint_rank": metadata.get("source_checkpoint_rank"),
            "source_checkpoint_val_spearman": metadata.get(
                "source_checkpoint_val_spearman"
            ),
            **{f"hp_{k.split('.')[-1]}": v for k, v in combo.items()},
            "inicio_treino":      start_train.strftime("%Y-%m-%d %H:%M:%S"),
            "fim_treino":         end_train.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_treino":     format_duration(start_train, end_train),
            "duracao_inferencia": dur_inf,
            "checkpoint_path": best_ckpt,
            "dataset_teste":   config.datasets.test[0].metadata_path,
            "val_mse":      float(metrics.get("val/mse",      "nan")),
            "val_pearson":  float(metrics.get("val/pearson",  "nan")),
            "val_spearman": float(metrics.get("val/spearman", "nan")),
            "test_mse":      inf_metrics.get("MSE"),
            "test_pearson":  inf_metrics.get("LCC"),
            "test_spearman": inf_metrics.get("SRCC"),
            "test_mean_macs":  inf_metrics.get("MACS_PER_SAMPLE"),
            "test_mean_flops": inf_metrics.get("FLOPS_PER_SAMPLE"),
            "target_test_mse":      inf_metrics.get("MSE"),
            "target_test_pearson":  inf_metrics.get("LCC"),
            "target_test_spearman": inf_metrics.get("SRCC"),
            "target_test_mean_macs":  inf_metrics.get("MACS_PER_SAMPLE"),
            "target_test_mean_flops": inf_metrics.get("FLOPS_PER_SAMPLE"),
            "target_test_n_samples": inf_metrics.get("N_SAMPLES"),
            "target_inference_duration_seconds": inf_metrics.get("INFERENCE_DURATION_SECONDS"),
            "target_predictions_path": target_predictions_path,
            "initial_target_test_mse": initial_target_metrics.get("MSE"),
            "initial_target_test_pearson": initial_target_metrics.get("LCC"),
            "initial_target_test_spearman": initial_target_metrics.get("SRCC"),
            "initial_target_predictions_path": initial_target_predictions_path,
            "source_test_mse":      source_inf_metrics.get("MSE"),
            "source_test_pearson":  source_inf_metrics.get("LCC"),
            "source_test_spearman": source_inf_metrics.get("SRCC"),
            "source_test_mean_macs":  source_inf_metrics.get("MACS_PER_SAMPLE"),
            "source_test_mean_flops": source_inf_metrics.get("FLOPS_PER_SAMPLE"),
            "source_test_n_samples": source_inf_metrics.get("N_SAMPLES"),
            "source_inference_duration_seconds": source_inf_metrics.get("INFERENCE_DURATION_SECONDS"),
            "source_predictions_path": source_predictions_path,
            "initial_source_test_mse": initial_source_metrics.get("MSE"),
            "initial_source_test_pearson": initial_source_metrics.get("LCC"),
            "initial_source_test_spearman": initial_source_metrics.get("SRCC"),
            "initial_source_predictions_path": initial_source_predictions_path,
            "source_mse_delta_after_minus_before": metric_delta(
                source_inf_metrics.get("MSE"),
                initial_source_metrics.get("MSE"),
            ),
            "source_pearson_delta_after_minus_before": metric_delta(
                source_inf_metrics.get("LCC"),
                initial_source_metrics.get("LCC"),
            ),
            "source_spearman_delta_after_minus_before": metric_delta(
                source_inf_metrics.get("SRCC"),
                initial_source_metrics.get("SRCC"),
            ),
            "source_spearman_forgetting_before_minus_after": metric_delta(
                initial_source_metrics.get("SRCC"),
                source_inf_metrics.get("SRCC"),
            ),
            "target_mse_delta_after_minus_before": metric_delta(
                inf_metrics.get("MSE"),
                initial_target_metrics.get("MSE"),
            ),
            "target_pearson_delta_after_minus_before": metric_delta(
                inf_metrics.get("LCC"),
                initial_target_metrics.get("LCC"),
            ),
            "target_spearman_delta_after_minus_before": metric_delta(
                inf_metrics.get("SRCC"),
                initial_target_metrics.get("SRCC"),
            ),
            "status": "success",
        }
        append_to_csv(output_csv, entry, lock)

        patch_state(yaml_path, exp_id, {
            "status":   "done",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, lock)

    except Exception as e:
        import traceback
        end_train = datetime.now()
        print(f"ERRO em {exp_id} [GPU {real_gpu_id}]: {e}")
        traceback.print_exc()

        append_to_csv(output_csv, {
            "experiment_id": exp_id,
            "config_id":     f"config_{config_id:03d}",
            "run_idx":       run_idx,
            "dataset":       dataset_name,
            "source_dataset": metadata.get("source_dataset", "N/A"),
            "initial_dataset": metadata.get("source_dataset", "N/A"),
            "finetune_dataset": metadata.get("target_dataset", dataset_name),
            "model":         model_name,
            "model_id":      metadata.get("model_id", model_name),
            "mode":          args_dict.get("mode"),
            "dataset_config": ds_path,
            "model_config":   mod_path,
            "init_checkpoint_path": metadata.get("init_checkpoint", "N/A"),
            "initial_checkpoint_path": metadata.get("init_checkpoint", "N/A"),
            "inicio_treino": start_train.strftime("%Y-%m-%d %H:%M:%S"),
            "fim_treino":    end_train.strftime("%Y-%m-%d %H:%M:%S"),
            "status":        f"ERROR: {e}",
        }, lock)

        patch_state(yaml_path, exp_id, {
            "status":    "error",
            "error_msg": str(e),
            "end_time":  end_train.strftime("%Y-%m-%d %H:%M:%S"),
        }, lock)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────────────────────────────────────
def worker_gpu(
    real_gpu_id: int,
    task_queue: mp.Queue,
    stop_event: mp.Event,
    args_dict: dict,
    lock: mp.Lock,
    output_csv: str,
) -> None:
    print(f"[Worker GPU {real_gpu_id}] Iniciado.")
    while not stop_event.is_set():
        try:
            exp_data_raw = task_queue.get(timeout=3)
        except queue.Empty:
            break

        exp_id = exp_data_raw[0]
        print(f"\n--- {exp_id} [GPU {real_gpu_id}] ---")

        p = mp.Process(
            target=_run_single_experiment,
            args=((*exp_data_raw, real_gpu_id), args_dict, args_dict["seed"], lock, output_csv),
        )
        p.start()
        p.join()

        status = "concluído" if p.exitcode == 0 else f"encerrou com código {p.exitcode}"
        print(f"[Worker GPU {real_gpu_id}] {exp_id} {status}.")

    print(f"[Worker GPU {real_gpu_id}] Encerrando.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",  nargs="+", default=[1,4,5,6], type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-r", "--runs", default=1,  type=int)
    parser.add_argument("-m", "--mode", choices=["resume", "scratch", "selected", "cross_finetune"], default="scratch")
    parser.add_argument("--state-file",   default=None, type=str,
                        help="Força um arquivo de estado específico (útil para --mode resume)")
    parser.add_argument("--status",       action="store_true", help="Mostra o status e sai")
    parser.add_argument("--retry-errors", action="store_true", help="Re-executa experimentos com erro")
    parser.add_argument(
        "--checkpoint-root",
        default="/checkpoints",
        help="Raiz em que estão CAL-MOS-facebook, KAN-facebook etc.",
    )
    parser.add_argument(
        "--checkpoints-per-model-dataset",
        default=1,
        type=int,
        help=(
            "Quantidade de runs originais escolhidas para cada par modelo/dataset X. "
            "Por padrão, usa somente o melhor checkpoint."
        ),
    )
    parser.add_argument(
        "--cross-models",
        nargs="+",
        choices=ALL_CROSS_MODEL_IDS,
        default=ALL_CROSS_MODEL_IDS,
        help=(
            "Modelos usados em cross_finetune. Por padrão, executa o estudo "
            "completo com os seis modelos: mHuBERT, wav2vec2 1B e wav2vec2 "
            "300M, cada um com MLP e KAN. Para piloto, passe por exemplo: "
            f"{' '.join(PILOT_CROSS_MODEL_IDS)}."
        ),
    )
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG_BY_NAME),
        default=None,
        help="Restringe os datasets X no modo cross_finetune.",
    )
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG_BY_NAME),
        default=None,
        help="Restringe os datasets Y no modo cross_finetune.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Descobre checkpoints e lista o plano sem treinar.",
    )
    args = parser.parse_args()

    # ── Espaço de experimentos ────────────────────────────────────────────────
    DATASET_CONFIGS = [
        "config/datasets/brspeech.yaml",
        "config/datasets/bvcc.yaml",
        "config/datasets/singmos.yaml",
        "config/datasets/tmhint.yaml",
    ]
    MODEL_CONFIGS = [
        # "config/models/wav2vec2_300m.yaml",
        "config/models/wav2vec2_1b.yaml",
        "config/models/mHuBERT-147.yaml",
        
        
        # kan
        # "config/models/mHuBERT-147_kan.yaml",
        # "config/models/wav2vec2_300m_kan.yaml",
        # "config/models/wav2vec2_1b_kan.yaml",
        # mean
        # "config/models/mHuBERT-147_kan_mean.yaml",
        # "config/models/wav2vec2_300m_kan_mean.yaml",
        # "config/models/wav2vec2_1b_kan_mean.yaml"
    ]
    GRID = {
        # "model.relukan_k":    [1, 2, 3, 4, 5, 6],
        # "model.relukan_grid": [1, 2, 3, 4, 5, 6],
    }

    if GRID:
        keys, vals = zip(*GRID.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    else:
        combos = [{}]

    # ── Nomes dos arquivos de saída ───────────────────────────────────────────
    RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    cross_finetune_sources = []
    if args.mode == "cross_finetune":
        cross_finetune_sources = discover_cross_finetune_sources(
            args.checkpoint_root,
            args.checkpoints_per_model_dataset,
            args.source_datasets,
            args.cross_models,
        )
        requested_sources = args.source_datasets or list(DATASET_CONFIG_BY_NAME)
        requested_targets = args.target_datasets or list(DATASET_CONFIG_BY_NAME)
        planned_pairs = [
            (source, target)
            for source in requested_sources
            for target in requested_targets
            if source != target
        ]
        print(
            "\n[Cross-finetune] Plano solicitado: "
            f"{len(args.cross_models)} modelos x "
            f"{len(requested_sources)} sources x "
            f"{len(requested_targets)} targets "
            f"({len(planned_pairs)} pares source->target validos) x "
            f"{args.checkpoints_per_model_dataset} checkpoint(s)."
        )
        involved_datasets = {
            source["source_dataset"] for source in cross_finetune_sources
        }
        involved_datasets.update(
            args.target_datasets or DATASET_CONFIG_BY_NAME.keys()
        )
        DATASET_CONFIGS = [
            DATASET_CONFIG_BY_NAME[name] for name in sorted(involved_datasets)
        ]
        MODEL_CONFIGS = sorted(
            {source["model_config"] for source in cross_finetune_sources}
        )
        GRID = CROSS_FINETUNE_OVERRIDES

    if args.state_file:
        OUTPUT_CSV = args.state_file.replace("grid_state__", "experiments__").replace(".yaml", ".csv")
        STATE_FILE = args.state_file
    else:
        OUTPUT_CSV, STATE_FILE = build_run_filenames(
            DATASET_CONFIGS, MODEL_CONFIGS, GRID, RUN_TIMESTAMP
        )

    yaml_path = STATE_FILE

    print(f"\n📄 CSV   → {OUTPUT_CSV}")
    print(f"💾 State → {STATE_FILE}\n")

    # ── Montagem dos experimentos ─────────────────────────────────────────────
    if args.mode == "cross_finetune":
        combos = [{}]
        if GRID:
            keys, vals = zip(*GRID.items())
            combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]

        all_experiments = []
        target_dataset_names = (
            args.target_datasets
            if args.target_datasets
            else list(DATASET_CONFIG_BY_NAME)
        )
        for source in cross_finetune_sources:
            source_dataset = source["source_dataset"]
            mod_path = source["model_config"]
            init_checkpoint = source["init_checkpoint"]
            for target_dataset in target_dataset_names:
                target_ds_path = DATASET_CONFIG_BY_NAME[target_dataset]
                if target_dataset == source_dataset:
                    continue
                for combo in combos:
                    for run_idx in range(1, args.runs + 1):
                        all_experiments.append((
                            target_ds_path,
                            mod_path,
                            combo,
                            run_idx,
                            {
                                "source_dataset": source_dataset,
                                "target_dataset": target_dataset,
                                "model_id": source["model_id"],
                                "source_dataset_config": DATASET_CONFIG_BY_NAME[source_dataset],
                                "init_checkpoint": init_checkpoint,
                                "source_checkpoint_run": source[
                                    "source_checkpoint_run"
                                ],
                                "source_checkpoint_rank": source[
                                    "source_checkpoint_rank"
                                ],
                                "source_checkpoint_val_spearman": source[
                                    "source_checkpoint_val_spearman"
                                ],
                            },
                        ))
    elif args.mode == "selected":
        combos = [
            {"model.relukan_k": k, "model.relukan_grid": g}
            for _, _, k, g in SELECTED_EXPERIMENTS
        ]
        all_experiments = [
            (ds, mod, combo, run_idx, {})
            for (ds, mod, _, __), combo in zip(SELECTED_EXPERIMENTS, combos)
            for run_idx in range(1, args.runs + 1)
        ]
    else:
        all_experiments = [
            (ds, mod, combo, run_idx, {})
            for ds, mod, combo, run_idx in itertools.product(
                DATASET_CONFIGS, MODEL_CONFIGS, combos, range(1, args.runs + 1)
            )
        ]

    if args.dry_run:
        print(f"Plano cross-dataset: {len(all_experiments)} experimentos")
        for ds_path, mod_path, _, run_idx, metadata in all_experiments:
            source_run = metadata.get("source_checkpoint_run", "N/A")
            print(
                f"  {dataset_name_from_config(mod_path)} | "
                f"{metadata.get('source_dataset', dataset_name_from_config(ds_path))}"
                f"(source-run-{source_run}) -> "
                f"{metadata.get('target_dataset', dataset_name_from_config(ds_path))} | "
                f"finetune-run-{run_idx:02d} | "
                f"{metadata.get('init_checkpoint', 'scratch')}"
            )
        return

    # ── Modo --status ─────────────────────────────────────────────────────────
    if args.status:
        state = load_state(yaml_path)
        if not state:
            print(f"Nenhum estado encontrado em '{yaml_path}'.")
            return
        counts = {}
        for v in state.values():
            s = v.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        done = counts.get("done", 0) + counts.get("success", 0)
        print(f"\n--- {yaml_path} ---")
        print(f"✅ Concluídos: {done}  ❌ Erros: {counts.get('error', 0)}  "
              f"⏳ Pendentes: {counts.get('pending', 0)}  🚀 Rodando: {counts.get('running', 0)}")
        if counts.get("error", 0):
            print("\n⚠️  Experimentos com erro:")
            for k, v in state.items():
                if v.get("status") == "error":
                    print(f"  - {k}: {v.get('error_msg')}")
        return

    # ── Scratch: limpa state ──────────────────────────────────────────────────
    if args.mode == "scratch":
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
            print(f"[Limpeza] Removido: {yaml_path}")

    # ── Fila de tarefas ───────────────────────────────────────────────────────
    state    = load_state(yaml_path)
    lock     = mp.Lock()
    task_q   = mp.Queue()
    n_queued = 0

    print("Mapeando experimentos...")
    for i, (ds_path, mod_path, combo, run_idx, metadata) in enumerate(all_experiments):
        config_id = (i // args.runs) + 1
        hp_str    = format_hp_for_name(combo)
        ds_name   = dataset_name_from_config(ds_path)
        model_name = dataset_name_from_config(mod_path)
        if args.mode == "cross_finetune":
            exp_id = (
                f"config_{config_id:03d}_{model_name}__"
                f"source_run_{metadata['source_checkpoint_run']:02d}__"
                f"{metadata['source_dataset']}_to_{metadata['target_dataset']}"
                f"__finetune_run_{run_idx:02d}"
            )
        else:
            exp_id = f"config_{config_id:03d}_{hp_str}__{ds_name}_run_{run_idx:02d}"

        if exp_id in EXPERIMENTOS_IGNORADOS:
            continue

        config = OmegaConf.merge(OmegaConf.load(ds_path), OmegaConf.load(mod_path))
        for path, value in combo.items():
            OmegaConf.update(config, path, value, merge=True)
        config.dataset_name = ds_name
        if args.mode == "cross_finetune":
            config.finetune = {
                "source_dataset": metadata["source_dataset"],
                "target_dataset": metadata["target_dataset"],
                "init_checkpoint": metadata["init_checkpoint"],
            }
            config.title = (
                f"{config.title}-source-run-{metadata['source_checkpoint_run']:02d}"
                f"-ft-{metadata['source_dataset']}-to-"
                f"{metadata['target_dataset']}-{hp_str}"
                f"-(run-{run_idx:02d})"
            )
        else:
            config.title = f"{config.title}-{ds_name}-{hp_str}-(run-{run_idx:02d})"
        config_dict = OmegaConf.to_container(config, resolve=True)

        if exp_id not in state:
            state[exp_id] = {"status": "pending"}

        st = state[exp_id].get("status", "pending")

        if st in ("done", "success"):
            print(f"  Pulando {exp_id} (concluído)")
            continue
        if st == "error" and not args.retry_errors:
            print(f"  Pulando {exp_id} (erro anterior — use --retry-errors)")
            continue
        if st == "running":
            state[exp_id]["status"] = "pending"

        task_q.put((exp_id, config_id, ds_path, mod_path, combo, run_idx, yaml_path, config_dict, metadata))
        n_queued += 1

    save_state(yaml_path, state, lock)
    print(f"\nTotal: {len(all_experiments)} | Enfileirados: {n_queued}")

    if n_queued == 0:
        print("Nenhum experimento a executar. Saindo.")
        return

    # ── Despacha workers ──────────────────────────────────────────────────────
    args_dict  = vars(args)
    stop_event = mp.Event()
    processes  = [
        mp.Process(
            target=worker_gpu,
            args=(gpu_id, task_q, stop_event, args_dict, lock, OUTPUT_CSV),
        )
        for gpu_id in args.gpu
    ]
    for p in processes:
        p.start()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    try:
        while any(p.is_alive() for p in processes):
            for p in processes:
                p.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C — aguardando fim dos experimentos em andamento...")
        stop_event.set()
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("[!] Saída forçada. Progresso pode ter sido perdido.")
            sys.exit(1)

    print(f"\nConcluído.")
    print(f"  📄 Resultados → {OUTPUT_CSV}")
    print(f"  💾 Estado     → {STATE_FILE}")


if __name__ == "__main__":
    main()
