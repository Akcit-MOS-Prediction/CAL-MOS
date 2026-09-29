GPU_ID=0

############### BRSpeech
BASE_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/BSpeech-MOS-Prediction/BRSPEECH_MOS_DATASET_v2/data"

## MMS-300M
MODEL_NAME="mms-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/BSpeech-MOS-Prediction/$MODEL_NAME/data"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

## XLS-R-300M
MODEL_NAME="wav2vec2-xls-r-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/BSpeech-MOS-Prediction/$MODEL_NAME/data"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME


############### BVCC
BASE_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/BVCC/main/DATA/wav"
## MMS-300M
MODEL_NAME="mms-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/BVCC/$MODEL_NAME/wav"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

## XLS-R-300M
MODEL_NAME="wav2vec2-xls-r-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/BVCC/$MODEL_NAME/wav"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

############### SingMOS
BASE_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/SingMOS/DATA/wav"
## MMS-300M
MODEL_NAME="mms-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/SingMOS/$MODEL_NAME/wav"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

## XLS-R-300M
MODEL_NAME="wav2vec2-xls-r-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/SingMOS/$MODEL_NAME/wav"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

############### TMHINT-QI
BASE_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/tmhint-qi"
## MMS-300M
MODEL_NAME="mms-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/tmhint-qi/$MODEL_NAME"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME

## XLS-R-300M
MODEL_NAME="wav2vec2-xls-r-300m"
OUTPUT_DIR="/hadatasets/alef.ferreira/MOS-Prediction/DATASETS/embeddings/tmhint-qi/$MODEL_NAME"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_wav2vec_embeddings.py \
    --base-dir $BASE_DIR \
    --output-dir $OUTPUT_DIR \
    --model-name $MODEL_NAME