#!/usr/bin/env python3
import os
# --- permitir única cópia do OpenMP runtime antes de carregar quaisquer bibliotecas que usem OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
import tempfile
import yaml

here    = os.path.dirname(os.path.abspath(__file__))  
src_dir = os.path.abspath(os.path.join(here, "..")) 
sys.path.append(src_dir)

def update_config(layer, template_path, output_path):
    """
    Lê o arquivo de configuração template e atualiza o campo base_dir para as layers,
    além do título para incluir o número da layer.
    """
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    new_base_dir = f"../../data/BVCC/BVCC_WAV2BERT_embeddings_layer-{layer}_layer-{layer}"
    if 'datasets' in config:
        if 'train' in config['datasets'] and isinstance(config['datasets']['train'], list):
            for dataset in config['datasets']['train']:
                dataset['base_dir'] = new_base_dir
        if 'val' in config['datasets'] and isinstance(config['datasets']['val'], list):
            for dataset in config['datasets']['val']:
                dataset['base_dir'] = new_base_dir
        # Adicionando o test também
        if 'test' in config['datasets'] and isinstance(config['datasets']['test'], list):
            for dataset in config['datasets']['test']:
                dataset['base_dir'] = new_base_dir

    if 'title' in config:
        config['title'] = f"BVCC-WAV2BERT-Embeddings-layer{layer}-(epochs-${{trainer.max_epochs}})-(bs-${{train.batch_size}})-(LR-${{optimizer.params.learning_rate}})"

    with open(output_path, 'w') as f:
        yaml.dump(config, f)
# mapeamento de layer para código de checkpoint
layer_codes = {
    24: "x1d5irjp", 23: "24nu67e3", 22: "88vl6zz9", 21: "l4wls9lo",
    20: "ckovg9mo", 19: "pvmngcme", 18: "u8tfpygk", 17: "08od624u",
    16: "3o2jbhmg", 15: "19cu0s1u", 14: "kzmgtjlx", 13: "8td579fy",
    12: "jhwnuomy", 11: "b4lop3r9", 10: "ctj54c8f", 9: "lhe04y7r",
    8:  "onrbtbko", 7:  "1pubrrgh", 6:  "6r5w1u1h", 5:  "voy3mwaq",
    4:  "uxf7vid3", 3:  "g7kechsw", 2:  "rsxgzdbu", 1:  "fbnjs2zc",
    0:  "3bt7urvj"
}

# caminhos base
default_config = os.path.abspath(os.path.join(here, "../../config/default_one_layer_embedding.yaml"))
eval_script   = os.path.join(here, "inference.py")

def main():
    for layer, code in sorted(layer_codes.items(), reverse=True):
        print(f"\n=== Avaliação para layer {layer} ===")
        # cria config YAML temporário
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp:
            tmp_path = tmp.name
        update_config(layer, default_config, tmp_path)

        ckpt_path = f"F:/Git/CAL-MOS/src/MOS-Prediction/{code}/checkpoints/last.ckpt"
        print(f"Checkpoint: {ckpt_path}")
        cmd = [sys.executable, eval_script, "-c", tmp_path, "-ckpt", ckpt_path]
        print("Executando:", ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True, env=os.environ)
        except subprocess.CalledProcessError as e:
            print(f"Erro na layer {layer}: {e}")
        finally:
            os.remove(tmp_path)
            print(f"Config temporário removido: {tmp_path}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
