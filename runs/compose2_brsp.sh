#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/compose2_5_br_w.yaml" \
    "config/compose2_5_br_w.yaml" \
    "config/compose2_5_br_w.yaml" \
    "config/compose2_1_br_w.yaml" \
    "config/compose2_1_br_w.yaml" \
    "config/compose2_1_br_w.yaml" \
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
