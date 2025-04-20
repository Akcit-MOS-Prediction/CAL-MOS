import subprocess
import os
import sys

BASE_DIR = "/hadatasets/alef.ferreira/MOS-Prediction/BSpeech-MOS-Prediction"
INPUT_DIR_NAME = "BRSPEECH_MOS_DATASET_v2"
OUTPUT_DIR_BASE = "BRSPEECH_MOS_DATASET_v2_mms_embeddings"
MODEL_NAME = "mms-300m"
GPU_ID = "0"  

SCRIPT = "extract_wav2vec_embeddings.py" 


for layer in range(25):
    output_dir = f"{OUTPUT_DIR_BASE}_layer-{layer}"
    cmd = [
        sys.executable,     
        SCRIPT,
        "--base-dir", BASE_DIR,
        "--input-dir-name", INPUT_DIR_NAME,
        "--output-dir-name", output_dir,
        "--model-name", MODEL_NAME,
        "--specific-layer", str(layer),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print(f"\n>>> Extraindo camada {layer} → saída em: {output_dir}")
    subprocess.run(cmd, check=True, env=env)

