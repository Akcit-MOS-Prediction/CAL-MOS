import subprocess
import os
import sys

BASE_DIR = "F:\Git\CAL-MOS\data\BVCC"
INPUT_DIR_NAME = "DATA\wav"
OUTPUT_DIR_BASE = "BVCC_WAV2BERT_embeddings"
# MODEL_NAME = "mms-300m"
GPU_ID = "0"  

# SCRIPT = "extract_wav2vec_embeddings.py" 
SCRIPT = "extract_wav2bert_embeddings.py" 

for layer in range(25):
    output_dir = f"{OUTPUT_DIR_BASE}_layer-{layer}"
    cmd = [
        sys.executable,     
        SCRIPT,
        "--base-dir", BASE_DIR,
        "--input-dir-name", INPUT_DIR_NAME,
        "--output-dir-name", output_dir,
        # "--model-name", MODEL_NAME,
        "--specific-layer", str(layer),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print(f"\n>>> Extraindo camada {layer} → saída em: {output_dir}")
    subprocess.run(cmd, check=True, env=env)

