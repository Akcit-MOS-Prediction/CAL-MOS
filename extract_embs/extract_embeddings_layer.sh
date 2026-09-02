BASE_DIR="/hadatasets/alef.ferreira/MOS-Prediction/BSpeech-MOS-Prediction"
INPUT_DIR_NAME="BRSPEECH_MOS_DATASET_v2"
OUTPUT_DIR_NAME="BRSPEECH_MOS_DATASET_v2_mms_embeddings"
MODEL_NAME="mms-300m"
SPECIFIC_LAYER=0
GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --input-dir-name $INPUT_DIR_NAME \
    --output-dir-name $OUTPUT_DIR_NAME \
    --model-name $MODEL_NAME \
    --specific-layer $SPECIFIC_LAYER