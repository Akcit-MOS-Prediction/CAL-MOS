#!/bin/bash
GPU_ID=2
CONFIG_PATH=(\
    "tm_5_bvcc_x.yaml" \
    "tm_1_bvcc_x.yaml" \
    "tm_5_brsp_x.yaml" \
    "tm_1_brsp_x.yaml" \
    "tm_5_sing_x.yaml" \
    "tm_1_sing_x.yaml" 
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python src/main.py -c=$i -g=$GPU_ID
done
