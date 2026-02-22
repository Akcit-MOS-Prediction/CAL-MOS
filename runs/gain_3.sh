#!/bin/bash
GPU_ID=4
CONFIG_PATH=(\
    "config/gain_t_5_3.yaml" \
    "config/gain_t_5_3.yaml" \
    "config/gain_t_5_3.yaml" \
    "config/gain_t_1_3.yaml" \
    "config/gain_t_1_3.yaml" \
    "config/gain_t_1_3.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
