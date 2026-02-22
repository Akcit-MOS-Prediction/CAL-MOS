#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/whisper.yaml" \
    "config/whisper.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
