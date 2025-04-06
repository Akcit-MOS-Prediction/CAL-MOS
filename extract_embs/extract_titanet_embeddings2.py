import argparse
import os
import torch
import nemo.collections.asr as nemo_asr
from os.path import join, exists, basename, dirname, relpath, splitext
from os import makedirs
from tqdm import tqdm
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name="titanet"):
    if model_name == "speakernet":
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="speakerverification_speakernet"
        )
    elif model_name == "titanet":
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
    else:
        raise ValueError(f"Modelo {model_name} não suportado!")
    return speaker_model

def extract_nemo_embeddings(filelist, input_dir, output_dir, model_name):
    model = load_model(model_name)

    for filepath in tqdm(filelist, desc="Extraindo embeddings"):
        if not exists(filepath):
            print(f"Arquivo {filepath} não existe!")
            continue

       
        relative_path = relpath(filepath, input_dir)

       
        relative_dir = dirname(relative_path)    

      
        output_subdir = join(output_dir, relative_dir)
        makedirs(output_subdir, exist_ok=True) 

        base_name = splitext(basename(filepath))[0]  
        output_filename = base_name + ".pt"          
        output_filepath = join(output_subdir, output_filename)

        file_embedding = model.get_embedding(filepath).cpu().detach().numpy()
        embedding = torch.tensor(file_embedding)
        torch.save(embedding, output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir',
                        default='../data/BRSPEECH_MOS_DATASET_v2',
                        help="Diretório base")
    parser.add_argument('-i', '--input_dir',
                        default='',
                        help="Pasta (ou subpasta) contendo os arquivos .wav")
    parser.add_argument('-c', '--input_csv',
                        default=None,
                        help="Caminho para o arquivo CSV com metadados (opcional)")
    parser.add_argument('-o', '--output_dir',
                        default='../BRSPEECH_MOS_DATASET_v2_TITANETv1',
                        help="Diretório de saída")
    parser.add_argument('-m', '--model_name',
                        default='titanet',
                        help="Modelos disponíveis: speakernet e titanet. Padrão: titanet")
    args = parser.parse_args()

    input_dir = join(args.base_dir, args.input_dir)
    output_dir = join(args.base_dir, args.output_dir)

    if args.input_csv is not None:
        with open(join(args.base_dir, args.input_csv), encoding="utf-8") as f:
            content_file = f.readlines()
            filelist = [line.split(",")[0] for line in content_file]
    else:
        filelist = glob(join(input_dir, '**', '*.wav'), recursive=True)

    if not filelist:
        print(f"Nenhum arquivo .wav encontrado em {input_dir} (busca recursiva).")
        return

    makedirs(output_dir, exist_ok=True)

    extract_nemo_embeddings(filelist, input_dir, output_dir, args.model_name)

if __name__ == "__main__":
    main()
