#!/usr/bin/env python3
import os
import sys
import subprocess

# --- permitir única cópia do OpenMP runtime antes de carregar quaisquer bibliotecas que usem OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

here = os.path.dirname(os.path.abspath(__file__))  
src_dir = os.path.abspath(os.path.join(here, "..")) 
sys.path.append(src_dir)

# Mapeamento de layer para código de checkpoint
layer_codes = {
    15: "cth1nc0w", 14: "lyk54sow", 13: "el8hiqgd",
    12: "ylv6om52", 11: "qzst7vck", 10: "7xt3ce55", 9: "cmbtyao4",
    8:  "64fnzc2j", 
}

# Caminho base para os arquivos YAML existentes
CONFIG_BASE_DIR = r"F:\Git\CAL-MOS\config\geral\all_layers\wav2vec2-large"
eval_script = os.path.join(here, "inference.py")

def main():
    for layer in range(15, 16):
        code = layer_codes.get(layer)
        if code is None:
            print(f"⚠️ Código de checkpoint não encontrado para a layer {layer}, pulando.")
            continue

        print(f"\n=== Avaliação para layer {layer} ===")

        config_path = os.path.join(CONFIG_BASE_DIR, f"config_w2v2_large_layer-{layer}-seqaug.yaml")
        if not os.path.isfile(config_path):
            print(f"❌ Arquivo de configuração não encontrado: {config_path}")
            continue

        ckpt_path = f"F:/Git/CAL-MOS/src/mosEmbeddings/{code}/checkpoints/last.ckpt"
        if not os.path.isfile(ckpt_path):
            print(f"❌ Checkpoint não encontrado: {ckpt_path}")
            continue

        print(f"Checkpoint: {ckpt_path}")
        print(f"Configuração: {config_path}")

        cmd = [sys.executable, eval_script, "-c", config_path, "-g", "0", "-ckpt", ckpt_path]
        try:
            subprocess.run(cmd, check=True, env=os.environ)
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao executar layer {layer}: {e}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
