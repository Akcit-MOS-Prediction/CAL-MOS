#!/bin/bash
GPU_ID=0

checkpoints_ids=(\
    "fplsl3i3" \
    "f73yv7h7" \
    "cm3h77di" \
)

config_paths=(\
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_tiny.yaml" \
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_mini.yaml" \
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_small.yaml" \
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_base.yaml" \
)

for i in $(seq 0 $((${#checkpoints_ids[@]} - 1))); do
    ckpt_id="${checkpoints_ids[$i]}"
    weights_path="/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/MOS-Prediction/${ckpt_id}/checkpoints"
    config_path="${config_paths[$i]}"
    echo "Running $ckpt_id - $config_path"
    python eval/inference.py -c=$config_path -g=$GPU_ID -ckpt=$weights_path
done
