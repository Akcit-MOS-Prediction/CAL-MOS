#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/compose3_5_bv_w.yaml" \
    "config/compose3_5_bv_w.yaml" \
    "config/compose3_5_bv_w.yaml" \
    "config/compose3_1_bv_w.yaml" \
    "config/compose3_1_bv_w.yaml" \
    "config/compose3_1_bv_w.yaml" \
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
