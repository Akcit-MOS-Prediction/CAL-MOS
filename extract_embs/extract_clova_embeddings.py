import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname
import pandas as pd
from tqdm import tqdm
import torch
from TTS.tts.utils.speakers import SpeakerManager

# Define o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

def load_model(model_path: str, config_path: str) -> SpeakerManager:
    """
    Carrega o modelo Clova utilizando o SpeakerManager.
    """
    encoder_manager = SpeakerManager(
        encoder_model_path=model_path,
        encoder_config_path=config_path,
        d_vectors_file_path=None,
        use_cuda=use_cuda,
    )
    return encoder_manager

def extract_clova_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_path: str,
    config_path: str
) -> None:
    """
    Extrai as features de áudio para cada arquivo na lista usando o modelo Clova.
    Mantém a estrutura de subpastas do diretório de entrada na saída.
    """
    encoder_manager = load_model(model_path, config_path)
    
    for filepath in tqdm(filelist, desc="Extraindo embeddings"):
        if not exists(filepath):
            print(f"Arquivo {filepath} não existe!")
            continue

        # Preserva a estrutura de subpastas
        rel_path = relpath(filepath, input_dir)
        sub_dir = dirname(rel_path)
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Extrai o embedding
        embedding = encoder_manager.compute_embedding_from_clip(filepath)
        embedding_tensor = torch.as_tensor(embedding)
        
        # Salva o embedding mantendo a estrutura de subpastas
        output_filename = basename(filepath).split(".")[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(embedding_tensor.cpu(), output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--base-dir",
        required=True,
        help="Caminho para o diretório base"
    )
    parser.add_argument(
        "-i", "--input-dir-name",
        help="Nome do diretório de entrada (relativo ao base-dir)"
    )
    parser.add_argument(
        "-c", "--input-csv",
        help="Caminho para o arquivo CSV com os nomes dos arquivos"
    )
    parser.add_argument(
        "--model-path",
        default="./checkpoints/clova/model_se.pth.tar",
        help="Caminho para o checkpoint do modelo"
    )
    parser.add_argument(
        "--model-config",
        default="./checkpoints/clova/config_se.json",
        help="Caminho para o arquivo de configuração do modelo"
    )
    parser.add_argument(
        "-o", "--output-dir-name",
        default="output_embeddings",
        help="Nome do diretório de saída (relativo ao base-dir)"
    )
    parser.add_argument(
        "-col", "--column-name",
        default="filename",
        help="Nome da coluna no CSV que contém os caminhos dos arquivos"
    )
    args = parser.parse_args()

    # Determina o diretório de entrada e a lista de arquivos
    input_dir = None
    filelist = None
    if args.input_dir_name:
        input_dir = join(args.base_dir, args.input_dir_name)
        filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    elif args.input_csv:
        input_csv_path = join(args.base_dir, args.input_csv)
        df = pd.read_csv(input_csv_path)
        filelist = df[args.column_name].tolist()
        if filelist and not os.path.isabs(filelist[0]):
            input_dir = args.base_dir
    else:
        print("Erro: é necessário informar input-dir-name ou input-csv!")
        exit(1)
    
    output_dir = join(args.base_dir, args.output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    extract_clova_embeddings(filelist, input_dir, output_dir, args.model_path, args.model_config)

if __name__ == "__main__":
    main()


