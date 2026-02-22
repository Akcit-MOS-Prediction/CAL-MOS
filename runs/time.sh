#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "config/time_st_5.yaml" \
    "config/time_st_5.yaml" \
    "config/time_st_5.yaml" \
    "config/time_st_1.yaml" \
    "config/time_st_1.yaml" \
    "config/time_st_1.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
