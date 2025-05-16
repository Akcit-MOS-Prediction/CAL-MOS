import pandas as pd
from sklearn.model_selection import train_test_split
import os
import re

base_path = '/hadatasets/alef.ferreira/MOS-Prediction/track3_obf/DATA'

input_files = [
    os.path.join(base_path, 'utt_48k.csv'),
    os.path.join(base_path, 'utt_24k.csv'),
    os.path.join(base_path, 'utt_16k.csv'),
]

splits = ['train', 'val', 'test']
split_sets = {s: [] for s in splits}
dataframes_with_sr = []

for file in input_files:
    sr_match = re.search(r'utt_(\d+)k\.csv', file)
    if sr_match:
        sr_khz = int(sr_match.group(1))
    else:
        raise ValueError(f"Não foi possível extrair o SR do nome do arquivo: {file}")

    df = pd.read_csv(file)
    # get the mean of "rating" considering the "uttID"
    df = df.groupby('uttID').agg({
        'rating': 'mean',
    }).reset_index()

    df["sr_khz"] = sr_khz

    dataframes_with_sr.append(df)

    utts = df['uttID'].unique()

    train_utts, temp_utts = train_test_split(
        utts, test_size=0.2, random_state=42, shuffle=True)
    val_utts, test_utts = train_test_split(
        temp_utts, test_size=0.5, random_state=42, shuffle=True)

    split_sets['train'].append(set(train_utts))
    split_sets['val'].append(set(val_utts))
    split_sets['test'].append(set(test_utts))

    for split, utt_set in zip(splits, [train_utts, val_utts, test_utts]):
        split_df = df[df['uttID'].isin(utt_set)]
        out_file = f"{os.path.splitext(os.path.basename(file))[0]}_{split}.csv"
        split_df.to_csv(os.path.join(base_path, out_file), index=False)

combined_df = pd.concat(dataframes_with_sr, ignore_index=True)

for split in splits:
    common_utts = set.union(*split_sets[split])
    out_file = f'combined_{split}.csv'
    combined_df[combined_df['uttID'].isin(common_utts)].to_csv(
        os.path.join(base_path, out_file), index=False)