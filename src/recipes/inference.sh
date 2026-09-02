#!/bin/bash
GPU_ID=0

checkpoints_ids=(\
    "hkyzxrbs" \
    "0hbhgbxl" \
    "dnbfz8tj" \
    "tk1v46ac" \
    "uir480st" \
    "t3lod75c" \
    "ln7xzz81" \
    "g2r0havc" \
    "tzfjqfsi" \
    "l0nz0l9j" \
    "2h0wyf9d" \
    "0he6lw1g" \
    "jo51536x" \
    "bnjjyiuy" \
    "5ju42ldm" \
    "7zxywh3q" \
    "zhmymirl" \
    "x6rq3lf5" \
    "j6jud33j" \
    "p3ngf9lq" \
    "8yyibiip" \
    "pfch4hjz" \
    "cm9669kt" \
    "jzprgxc5" \
    "3eb6sgfx" \
)

BASE_DIR="../config/mms300_layers"
NUM_LAYERS=$((${#checkpoints_ids[@]} - 1))
echo "Number of layers: $NUM_LAYERS"

for i in $(seq 0 $NUM_LAYERS); do
    ckpt_id="${checkpoints_ids[$i]}"
    weights_path="/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/MOS-Prediction/${ckpt_id}/checkpoints"
    config_path="${BASE_DIR}/mms300m_layer_${i}.yaml"
    echo "Running $ckpt_id - $config_path"
    python eval/inference.py -c=$config_path -g=$GPU_ID -ckpt=$weights_path
done
