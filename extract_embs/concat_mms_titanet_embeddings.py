import os
import glob
import argparse
from os.path import exists, join, relpath, dirname, basename

import torch
import torch.nn as nn
from tqdm import tqdm

def maybe_linear_layer(in_dim: int, out_dim: int):
    """
    Cria uma camada linear se as dimensões forem diferentes.
    Caso contrário, retorna None.
    """
    if in_dim == out_dim:
        return None
    else:
        return nn.Linear(in_dim, out_dim, bias=False)

def load_embedding(filepath: str) -> torch.Tensor:
    """
    Carrega um tensor .pt do disco.
    """
    emb = torch.load(filepath, map_location="cpu")
    if emb.dim() == 2 and emb.size(0) == 1:
        emb = emb.squeeze(0)
    return emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titanet-dir", required=True, help="Diretório com embeddings do Titanet (arquivos .pt).")
    parser.add_argument("--mms-dir", required=True, help="Diretório com embeddings do MMS (arquivos .pt).")
    parser.add_argument("--output-dir", required=True, help="Diretório onde serão salvos os embeddings concatenados.")
    parser.add_argument("--output-dim", type=int, required=True,
                        help="Dimensão final desejada para o embedding concatenado (ex: 1024).")
    args = parser.parse_args()

    branch_dim = args.output_dim // 2

    # Detecta dimensão dos embeddings de exemplo do Titanet
    titanet_files = glob.glob(join(args.titanet_dir, "**", "*.pt"), recursive=True)
    if not titanet_files:
        raise ValueError(f"Nenhum arquivo .pt encontrado em {args.titanet_dir}")
    example_titanet = load_embedding(titanet_files[0])
    titanet_dim = example_titanet.size(-1)

    # Detecta dimensão dos embeddings de exemplo do MMS
    mms_files = glob.glob(join(args.mms_dir, "**", "*.pt"), recursive=True)
    if not mms_files:
        raise ValueError(f"Nenhum arquivo .pt encontrado em {args.mms_dir}")
    example_mms = load_embedding(mms_files[0])
    mms_dim = example_mms.size(-1)

    print(f"Dimensão dos embeddings Titanet: {titanet_dim}")
    print(f"Dimensão dos embeddings MMS: {mms_dim}")
    print(f"Cada ramo será projetado para: {branch_dim} (embedding concatenado final: {branch_dim * 2})")

    titanet_projector = maybe_linear_layer(titanet_dim, branch_dim)
    mms_projector = maybe_linear_layer(mms_dim, branch_dim)

    device = torch.device("cpu")
    if titanet_projector:
        titanet_projector.to(device)
    if mms_projector:
        mms_projector.to(device)

    for titanet_path in tqdm(titanet_files, desc="Concatenando embeddings"):
        rel_path = relpath(titanet_path, args.titanet_dir)
        mms_path = join(args.mms_dir, rel_path)
        if not exists(mms_path):
            print(f"Arquivo correspondente não encontrado no MMS: {mms_path}")
            continue

        # Carrega os embeddings
        titanet_emb = load_embedding(titanet_path).to(device)
        mms_emb = load_embedding(mms_path).to(device)

        if titanet_projector:
            titanet_emb = titanet_projector(titanet_emb.unsqueeze(0)).squeeze(0)
        if mms_projector:
            mms_emb = mms_projector(mms_emb.unsqueeze(0)).squeeze(0)

        combined_emb = torch.cat([titanet_emb, mms_emb], dim=-1)

        # Salva no output_dir
        out_path = join(args.output_dir, rel_path)
        out_dir = dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(combined_emb.cpu(), out_path)

    print("Concatenação concluída com sucesso!")

if __name__ == "__main__":
    main()

