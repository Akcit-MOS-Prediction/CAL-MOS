import os

from typing import Dict, List, Optional, Union

import tqdm
import torch
import torchaudio
from datasets import Dataset
import torch.nn.functional as F

import pandas as pd
from omegaconf import DictConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_path(batch, base_dir, filename_column, target_column):
    """Maps the real path to the audio files"""
    path = os.path.join(base_dir, batch[filename_column].lstrip('/').lstrip('./'))

    audio, sr = torchaudio.load(path)

    # transform the audio to mono if necessary
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    batch["audio"] = audio
    batch["target"] = batch[target_column]
    batch["sr"] = sr

    return batch


def preprocess_metadata(
    base_dir: str,
    filename_column: str,
    target_column: str,
    df: pd.DataFrame,
    num_workers: int = 8
) -> Dataset:
    """Maps the real path to the audio files"""
    df.reset_index(drop=True, inplace=True)

    columns = df.columns.tolist()

    df_dataset = Dataset.from_pandas(df)
    df_dataset = df_dataset.map(
        map_path,
        fn_kwargs={
            "base_dir": base_dir,
            "filename_column": filename_column,
            "target_column": target_column
        },
        remove_columns=columns,
        num_proc=num_workers
    )

    return df_dataset