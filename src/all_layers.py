import os
import subprocess
import yaml
import sys

def update_config(layer, template_path, output_path):
    """
    Lê o arquivo de configuração template e atualiza o campo base_dir para as layers,
    além do título para incluir o número da layer e flag de SeqAug.
    """
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    # Novo base_dir específico para a layer
    new_base_dir = (
        f"F:/Git/CAL-MOS/data/voice_mos/track3_obf/WAV2BERT/"
        f"mos_WAV2BERT_embeddings_layer-{layer}_layer-{layer}"
    )
    for split in ['train', 'val']:
        if split in config.get('datasets', {}) and isinstance(config['datasets'][split], list):
            for ds in config['datasets'][split]:
                ds['base_dir'] = new_base_dir

    # SeqAug para título
    use_seqaug = config.get('data', {}).get('use_seqaug', False)
    if 'title' in config:
        config['title'] = (
            f"Voicemos-CAL-MOS-OneLayerEmbedding-layer{layer}"
            f"-WAV2BERT-geral-SeqAug-{use_seqaug}"
            f"-(epochs-${{trainer.max_epochs}})"
            f"-(bs-${{train.batch_size}})"
        )

    with open(output_path, 'w') as f:
        yaml.dump(config, f)


def main():
    base_config = "../config/default_one_layer_embedding.yaml"
    # Carrega flag SeqAug
    cfg = yaml.safe_load(open(base_config, 'r'))
    use_seqaug = cfg.get('data', {}).get('use_seqaug', False)

    # Pasta de saída incluindo SeqAug
    output_dir = f"../config/wav2bert_embeddings-geral" + (
        f"-seqaug" if use_seqaug else ""
    )
    os.makedirs(output_dir, exist_ok=True)

    for layer in range(25):
        print(f"\n=== Layer {layer:02d} | SeqAug={use_seqaug} ===")
        yaml_filename = f"wav2bert_layer_{layer:02d}.yaml"
        temp_config_path = os.path.join(output_dir, yaml_filename)

        # Gera ou reutiliza o config YAML
        if not os.path.exists(temp_config_path):
            update_config(layer, base_config, temp_config_path)

        # Execução do experimento
        checkpoint_dir = (
            f"../checkpoints/mos-prediction/"
            f"voicemos-WAV2BERT-geral-layer{layer:02d}"
            + ("-seqaug" if use_seqaug else "")
        )
        cmd = [
            sys.executable, "main.py",
            "-c", temp_config_path,
            "-g", "0",
            "--checkpoint-dir", checkpoint_dir
        ]
        print(f"Running: {' '.join(cmd)}")

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"✔ Layer {layer:02d} done.")
        except subprocess.CalledProcessError as e:
            print(f"✘ Error at layer {layer:02d}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
