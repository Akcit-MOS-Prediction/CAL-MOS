#!/bin/bash
GPU_ID=4
CONFIG_PATH=(\
    "config/highp_ft_1.yaml" \
    "config/lowp_ft_1.yaml" \
    "config/time_st_1.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
