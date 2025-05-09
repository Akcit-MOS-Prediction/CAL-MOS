#!/bin/bash
GPU_ID=0

checkpoints_ids=(\
    "407gplj8" \
    "mfnwde9m" \
    "ik9f5gku" \
    "84ufvv1j" \
    "s7hx3mpm" \
    "14ngh0vd" \
    "6iu1znf4" \
    "70v3w2pc" \
    "8phwgxdu" \
    "eyupcgnj" \
    "z8c2atrc" \
    "me2wt1fc" \
    "igoisrsb" \
    "adxcl1aq" \
    "gnxt7fl0" \
    "fn6yjtxl" \
    "vpgkdckp" \
    "kt3qtb5q" \
    "489q4t4x" \
    "ny40dvsl" \
    "7qele295" \
    "jb42v0rn" \
    "u2bkp9uv" \
    "xdjz3vk2" \
    "w5qke6rd" \
)

BASE_DIR="../config/wav2bert_layers"
NUM_LAYERS=$((${#checkpoints_ids[@]} - 1))
echo "Number of layers: $NUM_LAYERS"

for i in $(seq 0 $NUM_LAYERS); do
    ckpt_id="${checkpoints_ids[$i]}"
    weights_path="/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/MOS-Prediction/${ckpt_id}/checkpoints"
    config_path="${BASE_DIR}/wav2bert_layer_${i}.yaml"
    echo "Running $ckpt_id - $config_path"
    python eval/inference.py -c=$config_path -g=$GPU_ID -ckpt=$weights_path
done
