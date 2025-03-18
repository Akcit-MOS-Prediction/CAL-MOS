import os
import subprocess
import yaml
import tempfile
import sys

def update_config(layer, template_path, output_path):
    """
    Lê o arquivo de configuração template, atualiza o campo base_dir para as layers,
    o nome do checkpoint e o título para incluir o número da layer.
    """
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    new_base_dir = f"../data/BRSPEECH_MOS_DATASET_v2_mms_embeddings_layer_{layer}"
    if 'datasets' in config:
        if 'train' in config['datasets'] and isinstance(config['datasets']['train'], list):
            for dataset in config['datasets']['train']:
                dataset['base_dir'] = new_base_dir
        if 'val' in config['datasets'] and isinstance(config['datasets']['val'], list):
            for dataset in config['datasets']['val']:
                dataset['base_dir'] = new_base_dir

    if 'model_checkpoint' in config:
        original_filename = config['model_checkpoint'].get('filename', "{epoch:02d}-{step:02d}-{val/spearman:.4f}")
        config['model_checkpoint']['filename'] = f"layer{layer}_" + original_filename

    if 'title' in config:
        config['title'] = f"CAL-MOS-OneLayerEmbedding-MMS300M-layer{layer}-(epochs-${{trainer.max_epochs}})-(bs-${{train.batch_size}})-(LR-${{optimizer.params.learning_rate}})"

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

def main():
    print("Executando")
    base_config = "../config/default_one_layer_embedding.yaml"
    print(f"Usando arquivo de configuração base: {base_config}")
    
    # Itera pelas layers de 0 a 24
    for layer in range(0, 25):
        print(f"\n=== Executando experimento para a layer {layer} ===")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
            temp_config_path = temp_file.name
        
        update_config(layer, base_config, temp_config_path)
        
        
        with open(temp_config_path, 'r') as f:
            updated_config = f.read()
        print(f"Configuração para a layer {layer}:\n{updated_config}")
        
        command = ["python", "main.py", "-c", temp_config_path, "-g", "0"]
        print(f"Executando comando: {' '.join(command)}")
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
        
        try:
            subprocess.run(command, cwd=".", env=env, check=True)
            print(f"Experimento para a layer {layer} finalizado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"\nErro ao executar o experimento para a layer {layer}: {e}")
            os.remove(temp_config_path)
            print(f"Arquivo de configuração temporário removido: {temp_config_path}")
            sys.exit(1) 
        
        os.remove(temp_config_path)
        print(f"Arquivo de configuração temporário removido: {temp_config_path}")

if __name__ == "__main__":
    print("Iniciando execução...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()


