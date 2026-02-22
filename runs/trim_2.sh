#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/trim_5_2.yaml" \
    "config/trim_5_2.yaml" \
    "config/trim_5_2.yaml" \
    "config/trim_1_2.yaml" \
    "config/trim_1_2.yaml" \
    "config/trim_1_2.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
