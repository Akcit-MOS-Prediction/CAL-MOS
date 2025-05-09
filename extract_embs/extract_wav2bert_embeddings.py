import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor

# Usa CUDA se disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name: str = "facebook/w2v-bert-2.0"):
    assert model_name == "facebook/w2v-bert-2.0", "Only facebook/w2v-bert-2.0 is supported"
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_name, config=config).to(device)
    print(f"[DEBUG] Loaded model: {model_name} on device {device}")
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor


def extract_wav2vec_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str,
    specific_layer: int = None,
    mean: bool = False,
) -> None:
    model, processor = load_model(model_name)
    for filepath in tqdm(filelist, desc="Extracting embeddings"):
        if not exists(filepath):
            print(f"[WARNING] file {filepath} doesn't exist!")
            continue

        # Mantém estrutura de subpastas
        rel_path = relpath(filepath, input_dir)
        output_subdir = join(output_dir, dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        # Carrega e normaliza áudio
        audio_data, sr = torchaudio.load(filepath)
        if audio_data.dim() > 1:
            audio_data = audio_data.mean(dim=0)
        if sr != 16000:
            audio_data = torchaudio.transforms.Resample(sr, 16000)(audio_data)
        audio_data = audio_data.squeeze()

        # Tokenização e inferência
        input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = model(**input_features, output_hidden_states=True).hidden_states

        # Extrai embeddings
        all_layers_embeddings = torch.stack(hidden_states).squeeze(1)
        if specific_layer is not None:
            all_layers_embeddings = all_layers_embeddings[specific_layer]

        # Colapsa dimensão temporal se solicitado
        if mean:
            if all_layers_embeddings.dim() == 2:
                all_layers_embeddings = all_layers_embeddings.mean(dim=0)
            else:
                all_layers_embeddings = all_layers_embeddings.mean(dim=1)


        base_name = basename(filepath).rsplit(".", 1)[0]
        filename = base_name + ".pt"
        save_path = join(output_subdir, filename)

        torch.save(all_layers_embeddings.cpu(), save_path)
        print(f"[DEBUG] Saved embedding to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base-dir", required=True, help="Path to the base directory")
    parser.add_argument("-i", "--input-dir-name", required=True, help="Input directory name inside the base directory")
    parser.add_argument("-o", "--output-dir-name", default="output_embeddings", help="Name of output directory")
    parser.add_argument("-l", "--specific-layer", default=None, type=int, help="Extract from a specific layer (optional)")
    parser.add_argument("-m", "--model-name", default="facebook/w2v-bert-2.0", help="Model name to use")
    parser.add_argument("-c", "--input-csv", help="CSV file with a column of filenames (optional)")
    parser.add_argument("-col", "--column-name", default="filename", help="Column name in CSV file")
    parser.add_argument("--mean", action="store_true", help="Save temporal mean instead of full sequence")
    args = parser.parse_args()

    input_dir = join(args.base_dir, args.input_dir_name)
    output_dir = join(args.base_dir, args.output_dir_name)
    if args.specific_layer is not None:
        assert args.specific_layer >= 0, "Layer index must be non-negative"
        # mantém diretório com sufixo de layer, se desejar
        output_dir += f"_layer-{args.specific_layer}"

    # Lista todos os arquivos .wav (100% do dataset)
    filelist = glob.glob(join(input_dir, "**", "*.wav"), recursive=True)
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    print(f"[DEBUG] Total de arquivos encontrados: {len(filelist)}")
    os.makedirs(output_dir, exist_ok=True)

    extract_wav2vec_embeddings(
        filelist,
        input_dir,
        output_dir,
        args.model_name,
        specific_layer=args.specific_layer,
        mean=args.mean,
    )

if __name__ == "__main__":
    main()