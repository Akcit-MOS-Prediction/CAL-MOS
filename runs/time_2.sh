#!/bin/bash
GPU_ID=0
CONFIG_PATH=(\
    "config/time_st_5_2.yaml" \
    "config/time_st_5_2.yaml" \
    "config/time_st_5_2.yaml" \
    "config/time_st_1_2.yaml" \
    "config/time_st_1_2.yaml" \
    "config/time_st_1_2.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
