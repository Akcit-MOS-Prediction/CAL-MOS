#!/bin/bash
GPU_ID=3
CONFIG_PATH=(\
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms300m_small_random.yaml" \
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_wav2bert_small_random.yaml" \
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_small_random.yaml"
)

for i in "${CONFIG_PATH[@]}"
do
    echo "Running $i"
    python main.py -c=$i -g=$GPU_ID
done
