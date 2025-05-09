#!/bin/bash
GPU_ID=0

checkpoints_ids=(\
    "ev1sqg79" \
    "uq6x7ijk" \
    "zmdw3cuz" \
    "pbdu93k3" \
)

config_paths=(\
    "../config/default_dynamic_whisperv3_brspeech_ft.yaml" \
    "../config/default_dynamic_whisperv3_bvcc-ll-mp.yaml" \
    "../config/default_dynamic_whisperv3_bvcc-ws.yaml" \
    "../config/default_dynamic_whisperv3_bvcc_ft.yaml" \
)

for i in $(seq 0 $((${#checkpoints_ids[@]} - 1))); do
    ckpt_id="${checkpoints_ids[$i]}"
    weights_path="/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/MOS-Prediction/${ckpt_id}/checkpoints"
    config_path="${config_paths[$i]}"
    echo "Running $ckpt_id - $config_path"
    python eval/inference.py -c=$config_path -g=$GPU_ID -ckpt=$weights_path
done
