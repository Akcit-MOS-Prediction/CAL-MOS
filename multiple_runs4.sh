#!/bin/bash
GPU_ID=4
CONFIG_PATH=(\
    "config/shift_5.yaml" \
    "config/trim_5.yaml" \
    "config/time_m_5.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
