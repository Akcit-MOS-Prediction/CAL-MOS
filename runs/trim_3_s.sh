#!/bin/bash
GPU_ID=0
CONFIG_PATH=(\
    "config/trim_5_3_s.yaml" \
    "config/trim_5_3_s.yaml" \
    "config/trim_5_3_s.yaml" \
    "config/trim_1_3_s.yaml" \
    "config/trim_1_3_s.yaml" \
    "config/trim_1_3_s.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
