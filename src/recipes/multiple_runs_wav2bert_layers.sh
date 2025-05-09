#!/bin/bash
GPU_ID=3
BASE_DIR="../config/wav2bert_layers"
NUM_LAYERS=24

for i in $(seq 1 $NUM_LAYERS)
do
    echo "Running layer $i - config wav2bert_layer_${i}.yaml"
    # Construct the config file path
    CONFIG_PATH="${BASE_DIR}/wav2bert_layer_${i}.yaml"
    python main.py -c=$CONFIG_PATH -g=$GPU_ID
done
