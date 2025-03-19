import os
import glob
import argparse
from os.path import exists, basename, join, relpath, dirname, splitext
from tqdm import tqdm
import torch

def load_embedding(filepath: str) -> torch.Tensor:
    """
    Carrega o embedding salvo em um arquivo .pt.
    """
    if not exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    return torch.load(filepath)

def force_output_dim(embedding: torch.Tensor, output_dim: int) -> torch.Tensor:
    current_dim = embedding.shape[0]
    if current_dim > output_dim:
        return embedding[:output_dim]
    elif current_dim < output_dim:
        padding = torch.zeros(output_dim - current_dim, dtype=embedding.dtype)
        return torch.cat((embedding, padding), dim=0)
    else:
        return embedding

def concat_embeddings(dir1: str, dir2: str, outputdir: str, output_dim: int) -> None:
    os.makedirs(outputdir, exist_ok=True)
    filelist = glob.glob(join(dir1, "**", "*.pt"), recursive=True)

    for filepath in tqdm(filelist, desc="Concatenando embeddings"):
        rel_path = relpath(filepath, dir1)
        counterpart_path = join(dir2, rel_path)

        if not exists(counterpart_path):
            print(f"Arquivo correspondente em dir2 não encontrado para {filepath}. Pulando.")
            continue

        emb1 = load_embedding(filepath)
        emb2 = load_embedding(counterpart_path)

        if emb1.ndim > 1:
            emb1 = emb1.flatten()
        if emb2.ndim > 1:
            emb2 = emb2.flatten()

        concat_emb = torch.cat((emb1, emb2), dim=0)

        concat_emb = force_output_dim(concat_emb, output_dim)

        output_filepath = join(outputdir, rel_path)
        os.makedirs(dirname(output_filepath), exist_ok=True)
        torch.save(concat_emb.cpu(), output_filepath)

def main():
    parser = argparse.ArgumentParser(
        description="Concatena embeddings de dois diretórios (dir1 e dir2) e ajusta para ter dimensão output_dim."
    )
    parser.add_argument(
        "--dir1",
        required=True,
        help="Caminho para o primeiro diretório com embeddings (.pt)."
    )
    parser.add_argument(
        "--dir2",
        required=True,
        help="Caminho para o segundo diretório com embeddings (.pt)."
    )
    parser.add_argument(
        "--outputdir",
        required=True,
        help="Caminho para o diretório de saída."
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        required=True,
        help="Dimensão desejada para o embedding concatenado."
    )
    args = parser.parse_args()

    concat_embeddings(args.dir1, args.dir2, args.outputdir, args.output_dim)

if __name__ == "__main__":
    main()
