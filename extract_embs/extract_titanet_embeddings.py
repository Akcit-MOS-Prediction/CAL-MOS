import os
import sys

# Caminho absoluto até a pasta onde este script está
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho absoluto até a pasta 'titanet', que contém a subpasta 'src'
TITANET_DIR = os.path.join(SCRIPT_DIR, "titanet")

# Adiciona a pasta 'titanet' ao sys.path, para que possamos importar 'src.models'
sys.path.insert(0, TITANET_DIR)

import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch, torchaudio

# Para contornar aviso do OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

# Agora, como 'titanet/src' é uma subpasta, podemos importar 'src.models'
from src.models import TitaNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_size: str = "s") -> TitaNet:
    """
    Carrega o modelo TitaNet.
    
    model_size: 's', 'm' ou 'l', conforme implementado no TitaNet (TitaNet-S, TitaNet-M, TitaNet-L).
    """
    # Se TitaNet.get_titanet for a forma recomendada de criar o modelo, use-a:
    model = TitaNet.get_titanet(
        embedding_size=192,
        n_mels=80,
        model_size=model_size,
        device=device
    )
    model.eval()
    return model

def extract_titanet_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_size: str
) -> None:
    """
    Extrai o embedding global para cada arquivo .wav na lista, usando TitaNet.
    Converte o áudio para mel-spectrograma antes de passá-lo ao modelo.
    """
    model = load_model(model_size)

    # Transforma o áudio em mel-spectrograma de 80 bandas
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80
    ).to(device)
    
    for filepath in tqdm(filelist, desc="Extraindo embeddings TitaNet"):
        if not exists(filepath):
            print(f"Arquivo {filepath} não existe!")
            continue

        # Cria a mesma estrutura de pastas na saída
        rel_path = relpath(filepath, input_dir)
        sub_dir = dirname(rel_path)
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)

        # Carrega o áudio
        audio_data, sr = torchaudio.load(filepath)
        # Se for estéreo, converte para mono
        if audio_data.dim() > 1:
            audio_data = audio_data.mean(dim=0)
        # Reamostra para 16kHz, se necessário
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_data = resampler(audio_data)
        audio_data = audio_data.squeeze().to(device)

        # [T] -> [1, T]
        input_tensor = audio_data.unsqueeze(0)
        # [1, T] -> [1, n_mels, T'] via MelSpectrogram
        spectrogram = melspec_transform(input_tensor)

        with torch.no_grad():
            global_embedding = model(spectrogram)  # [1, E]

        # Salva o embedding (tensor) em arquivo .pt
        output_filename = basename(filepath).split(".")[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(global_embedding.cpu(), output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help="Caminho para o diretório base"
    )
    parser.add_argument(
        "-i",
        "--input-dir-name",
        required=True,
        help="Nome do diretório de entrada (dentro do diretório base)"
    )
    parser.add_argument(
        "-o",
        "--output-dir-name",
        default="output_embeddings",
        help="Nome do diretório de saída"
    )
    parser.add_argument(
        "-m",
        "--model-size",
        default="s",
        help="Tamanho do TitaNet: 's', 'm' ou 'l'"
    )
    parser.add_argument(
        "-c",
        "--input-csv",
        help="Caminho para o arquivo CSV com metadados (opcional)"
    )
    parser.add_argument(
        "-col",
        "--column-name",
        default="filename",
        help="Coluna do CSV que contém os nomes dos arquivos"
    )
    args = parser.parse_args()

    input_dir = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    # Gera a lista de .wav
    filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    # Se um CSV foi passado, prioriza a lista do CSV
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    os.makedirs(output_dir, exist_ok=True)
    extract_titanet_embeddings(filelist, input_dir, output_dir, args.model_size)

if __name__ == "__main__":
    main()
