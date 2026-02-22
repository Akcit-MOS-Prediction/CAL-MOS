#!/bin/bash
GPU_ID=0
CONFIG_PATH=(\
    "config/mms.yaml" \
    "config/mms.yaml" \
    "config/mms.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
