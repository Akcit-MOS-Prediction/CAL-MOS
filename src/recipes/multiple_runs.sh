#!/bin/bash
GPU_ID=6
CONFIG_PATH=("../config/default_weighted_sum.yaml" "../config/default.yaml" "../config/default_transformers.yaml" "../config/default_transformers_seqaug.yaml")

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python main.py -c=$i -g=$GPU_ID
done
