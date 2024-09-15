#!/bin/bash
GPU_ID=6
CONFIG_PATH=("../config/default.yaml" "../config/weighted_sum.yaml" "../config/default_seqaug.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python main.py -c=$i -g=$GPU_ID
done
