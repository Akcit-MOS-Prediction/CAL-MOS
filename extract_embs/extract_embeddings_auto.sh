#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
DRY_RUN=0

EMB_ROOT="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings"
PYTHON="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-extract_automodel_embeddings.py}"

MODELS_DEFAULT=(
  "microsoft/wavlm-large"
  "facebook/mms-300m"
)

DATASETS_DEFAULT=(
  # "BSpeech-MOS-Prediction"
  "BVCC"
  "SingMOS"
  "tmhint-qi"
)

declare -A BASE_DIRS=(
  ["BSpeech-MOS-Prediction"]="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/BSpeech-MOS-Prediction/BRSPEECH_MOS_DATASET_v2/data"
  ["BVCC"]="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/BVCC/main/DATA/wav"
  ["SingMOS"]="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/SingMOS/DATA/wav"
  ["tmhint-qi"]="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/tmhint-qi"
)

declare -A OUT_TAILS=(
  ["BSpeech-MOS-Prediction"]="data"
  ["BVCC"]="wav"
  ["SingMOS"]="wav"
  ["tmhint-qi"]=""
)

# -------------------------
# helpers
# -------------------------
model_dir_name() {
  # "facebook/mms-300m" -> "mms-300m"
  # "mms-300m" -> "mms-300m"
  local m="$1"
  echo "${m##*/}"
}

usage() {
  cat <<EOF
Usage: $0 [--gpu N] [--datasets a,b,c] [--models m1,m2] [--dry-run]

--gpu       GPU id for CUDA_VISIBLE_DEVICES (default: 0)
--datasets  Comma-separated dataset keys: ${DATASETS_DEFAULT[*]}
--models    Comma-separated model names (HF ids): ${MODELS_DEFAULT[*]}
--dry-run   Print commands without running
EOF
}

MODELS=("${MODELS_DEFAULT[@]}")
DATASETS=("${DATASETS_DEFAULT[@]}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="${2:?missing value for --gpu}"; shift 2 ;;
    --datasets) IFS=',' read -r -a DATASETS <<< "${2:?missing value for --datasets}"; shift 2 ;;
    --models) IFS=',' read -r -a MODELS <<< "${2:?missing value for --models}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

run_one() {
  local dataset="$1"
  local model="$2"

  local base_dir="${BASE_DIRS[$dataset]:-}"
  if [[ -z "$base_dir" ]]; then
    echo "ERROR: Unknown dataset key '$dataset' (no BASE_DIRS entry)." >&2
    exit 2
  fi

  local mdir
  mdir="$(model_dir_name "$model")"

  local tail="${OUT_TAILS[$dataset]:-}"
  local out_dir="$EMB_ROOT/$dataset/$mdir"
  if [[ -n "$tail" ]]; then
    out_dir="$out_dir/$tail"
  fi

  mkdir -p "$out_dir"

  echo "============================================================"
  echo "Dataset   : $dataset"
  echo "Base dir  : $base_dir"
  echo "Model     : $model"
  echo "Model dir : $mdir"
  echo "Output dir: $out_dir"
  echo "GPU       : $GPU_ID"
  echo "============================================================"

  local cmd=(env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT" \
    --base-dir "$base_dir" \
    --output-dir "$out_dir" \
    --model-name "$model" \
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY-RUN] '; printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$dataset" "$model"
  done
done
