#!/bin/bash
GPU_ID=0

checkpoints_ids=(\
    # "hfx7kfj8" \
    # "7aw5joyn" \
    # "sak976ez" \
    # "r1xjhtlm" \
    # "na6wkvvz" \
    # "04rkrghe" \
    # "trzhl7n3" \
    # "b2acgvpt" \
    # "83bqqcd5" \
    "mby1cm1f" \
    "do909g57" \
)

config_paths=(\
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_wav2bert_small_random.yaml" \
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms300m_small_random.yaml" \
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_bvcc_mms1b_small_random.yaml" \

    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_wav2bert_small.yaml" \
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_wav2bert_small_random.yaml" \

    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_mms300m_small.yaml" \
    # "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_mms300m_small_random.yaml" \

    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_mms1b_small.yaml" \
    "../config/PEFT-MelSpec/default_dynamic_peft_melspec_brspeech_mms1b_small_random.yaml" \
)

for i in $(seq 0 $((${#checkpoints_ids[@]} - 1))); do
    ckpt_id="${checkpoints_ids[$i]}"
    weights_path="/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/MOS-Prediction/${ckpt_id}/checkpoints"
    config_path="${config_paths[$i]}"
    echo "Running $ckpt_id - $config_path"
    python eval/inference.py -c=$config_path -g=$GPU_ID -ckpt=$weights_path
done
