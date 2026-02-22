#!/bin/bash
GPU_ID=4
CONFIG_PATH=(\
    "config/xlsr.yaml" \
    "config/xlsr.yaml" \
    "config/xlsr.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
