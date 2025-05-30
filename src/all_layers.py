import os
import subprocess
import yaml
import sys

def update_config(layer, template_path, output_path):
    """
    Lê o arquivo de configuração template e atualiza o campo base_dir para as layers,
    além do título para incluir o número da layer.
    """
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    new_base_dir = f"F:/Git/CAL-MOS/data/voice_mos/track3_obf/w2v-bert/w2v-bert_layer-{layer}"
    if 'datasets' in config:
        if 'train' in config['datasets'] and isinstance(config['datasets']['train'], list):
            for dataset in config['datasets']['train']:
                dataset['base_dir'] = new_base_dir
        if 'val' in config['datasets'] and isinstance(config['datasets']['val'], list):
            for dataset in config['datasets']['val']:
                dataset['base_dir'] = new_base_dir

    if 'title' in config:
        use_seqaug = config.get('data', {}).get('use_seqaug', False)
        config['title'] = (
        f"Voicemos-CAL-MOS-OneLayerEmbedding-layer{layer}"
        f"-WAV2BERT"
        f"-(epochs-${{trainer.max_epochs}})"
        f"-(bs-${{train.batch_size}})"
)

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def main():
    print("Executando")
    base_config = "../config/default_one_layer_embedding.yaml"
    output_dir = "../config/embeddings/geral/all_layers/w2v-bert"
    os.makedirs(output_dir, exist_ok=True)

    for layer in range(10, 25):
        print(f"\n=== Executando experimento para a layer {layer} ===")
        
        # Caminho fixo para salvar o yaml da layer atual
        yaml_filename = f"wav2bert_layer_{layer}.yaml"
        temp_config_path = os.path.join(output_dir, yaml_filename)

        # Se o arquivo não existir, criar e salvar
        if not os.path.exists(temp_config_path):
            print(f"Criando novo arquivo de configuração: {temp_config_path}")
            update_config(layer, base_config, temp_config_path)
        else:
            print(f"Usando configuração existente: {temp_config_path}")

        # Exibe a configuração (opcional)
        with open(temp_config_path, 'r') as f:
            updated_config = f.read()
        print(f"Configuração para a layer {layer}:\n{updated_config}")

        checkpoint_dir = f"../checkpoints/mos-prediction/voicemos-w2v-bert-layer{layer:02d}"
        command = ["python", "main.py", "-c", temp_config_path, "-g", "0", "--checkpoint-dir", checkpoint_dir]
        print(f"Executando comando: {' '.join(command)}")

        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        try:
            subprocess.run(command, cwd=".", env=env, check=True)
            print(f"Experimento para a layer {layer} finalizado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"\nErro ao executar o experimento para a layer {layer}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    print("Iniciando execução...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
