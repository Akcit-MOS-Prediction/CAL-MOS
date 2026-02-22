#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/gain_t_5.yaml" \
    "config/gain_t_5.yaml" \
    "config/gain_t_5.yaml" \
    "config/gain_t_1.yaml" \
    "config/gain_t_1.yaml" \
    "config/gain_t_1.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
