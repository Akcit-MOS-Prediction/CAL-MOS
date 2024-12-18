import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch, torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="wavlm-large"):
    model_path = None
    if (model_name == "wavlm-large"): # torch.shape: 1, t, 1024
        model_path = "microsoft/wavlm-large"
    elif (model_name == "wavlm-base-plus"): # torch.shape: 1, t, 768
        model_path = "microsoft/wavlm-base-plus"
    elif (model_name == "wavlm-base-plus-sv"):
        model_path = "microsoft/wavlm-base-plus-sv"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = WavLMModel.from_pretrained(model_path)
    model.eval()
    model = model.to(device)
    return model, feature_extractor


def extract_wavlm_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str
) -> None:
    model, processor = load_model(model_name)
    for filepath in tqdm(filelist, desc="Extracting embeddings"):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue

        # Determine the relative path structure
        rel_path = relpath(filepath, input_dir)
        # Get the subdirectory structure
        sub_dir = dirname(rel_path)
        # Create the same subdirectory structure in output_dir
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)

        audio_data, sr = torchaudio.load(filepath)
        # If stereo, convert to mono
        if audio_data.dim() > 1:
            audio_data = audio_data.mean(dim=0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_data = resampler(audio_data)

        audio_data = audio_data.squeeze().to(device)
        # Extract Embedding
        input_features = processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            hidden_states = model(**input_features, output_hidden_states=True).hidden_states
        # Concatenate all layers
        all_layers_embeddings = torch.stack(hidden_states) # [num_layers,B,T,F], B=1
        # transform to [num_layers,T,F]
        all_layers_embeddings = all_layers_embeddings.squeeze(1)
        # Saving embedding with the same subdirectory structure
        output_filename = basename(filepath).split(".")[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(all_layers_embeddings.cpu(), output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help="Path to the base directory"
    )
    parser.add_argument(
        "-i",
        "--input-dir-name",
        required=True,
        help="Name of the input directory, inside the base directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir-name",
        default="output_embeddings",
        help="Name of output directory",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        choices=[
            "wavlm-large",
            "wavlm-base-plus",
            "wavlm-base-plus-sv"
        ],
        default="wavlm-large",
        help="Model name",
    )
    parser.add_argument(
        "-c",
        "--input-csv",
        help="Metadata filepath",
    )
    parser.add_argument(
        "-col",
        "--column-name",
        default="filename",
        help="Column name of the csv file",
    )
    args = parser.parse_args()

    input_dir = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    os.makedirs(output_dir, exist_ok=True)

    extract_wavlm_embeddings(filelist, input_dir, output_dir, args.model_name)


if __name__ == "__main__":
    main()
