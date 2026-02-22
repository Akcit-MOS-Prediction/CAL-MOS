#!/bin/bash
GPU_ID=3
CONFIG_PATH=(\
    "config/highp_ft_5.yaml" \
    "config/lowp_ft_5.yaml" \
    "config/time_st_5.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
