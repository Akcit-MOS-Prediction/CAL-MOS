import os
import glob
import argparse
from typing import List, Tuple
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch, torchaudio
from transformers import AutoModel, AutoFeatureExtractor
from safetensors.torch import save_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name: str = None):
    print("Loading model:", model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor


@torch.inference_mode()
def extract_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str,
    specific_layer: int = None,
) -> None:
    model, processor = load_model(model_name)

    resampler = {}

    for filepath in tqdm(filelist, desc="Extracting embeddings"):
        # Load audio file
        if not exists(filepath):
            print("file {} doesnt exist!".format(filepath))
            continue

        output_path = filepath.replace(input_dir, output_dir)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio_data, sr = torchaudio.load(filepath)
        # If stereo, convert to mono
        if audio_data.size(0) > 1:
            audio_data = audio_data.mean(dim=0)

        if sr != 16000:
            if sr not in resampler:
                resampler[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_data = resampler[sr](audio_data)

        audio_data = audio_data.squeeze().numpy()
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
        all_layers_embeddings = all_layers_embeddings.squeeze().cpu().contiguous()
        if specific_layer is not None:
            all_layers_embeddings = all_layers_embeddings[specific_layer]
        # Saving embedding with the same subdirectory structure
        output_filename = basename(filepath).split(".")[0] + ".safetensors"
        output_filepath = output_path.replace(basename(output_path), output_filename)
        # torch.save(all_layers_embeddings, output_filepath)
        save_file({"embeddings": all_layers_embeddings}, output_filepath)
        # print(all_layers_embeddings.shape)
        # print(filepath, output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help="Path to the base directory"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Name of output directory",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "-l",
        "--specific-layer",
        default=None,
        type=int,
        help="Extract embeddings from a specific layer (If None, extract from all layers)",
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

    input_dir = args.base_dir
    output_dir = args.output_dir

    if args.specific_layer is not None:
        print("Using specific layer:", args.specific_layer)
        assert args.specific_layer >= 0, "Layer index should be non-negative"

        output_dir += f"_layer-{args.specific_layer}"

    filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    extract_embeddings(filelist, input_dir, output_dir, args.model_name, specific_layer=args.specific_layer)


if __name__ == "__main__":
    main()