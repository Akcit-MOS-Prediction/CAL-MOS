#!/bin/bash
GPU_ID=3
BASE_DIR="../config/mms300_layers"
NUM_LAYERS=24

for i in $(seq 0 $NUM_LAYERS)
do
    echo "Running layer $i - config mms300m_layer_${i}.yaml"
    # Construct the config file path
    CONFIG_PATH="${BASE_DIR}/mms300m_layer_${i}.yaml"
    python main.py -c=$CONFIG_PATH -g=$GPU_ID
done
