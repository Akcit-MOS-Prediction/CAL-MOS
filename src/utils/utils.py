import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoFeatureExtractor

from utils.dataloader import EmbeddingDataset, DynamicDataset


def build_dataloaders(config):
    """Builds the dataloader for the CAL-MOS model.

    Params:

    config (DictConfig): Configuration

    Returns:

    Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    train_data = pd.read_csv(config.datasets.train[0].metadata_path)
    val_data = pd.read_csv(config.datasets.val[0].metadata_path)

    if config.model.model_type.lower() == "dynamic" or \
        config.model.model_type.lower() == "dynamic_kan" or \
        config.model.model_type.lower() == "dynamic_melspec" or \
        config.model.model_type.lower() == "dynamic_kan_melspec":
        train_dataset = DynamicDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            mixup_alpha=config.data.mixup_alpha,
            use_rand_truncation=config.data.use_rand_truncation,
            min_duration=config.data.min_duration,
            insert_white_noise=config.data.insert_white_noise,
            min_white_noise_amp=config.data.min_white_noise_amp,
            max_white_noise_amp=config.data.max_white_noise_amp,
            data_type="train",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )

        val_dataset = DynamicDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            mixup_alpha=config.data.mixup_alpha,
            data_type="val",
            class_num=config.data.num_classes,
            target_sr=config.data.target_sr,
        )
    elif config.model.model_type.lower() == "all_layers_embedding" \
        or config.model.model_type.lower() == "one_layer_embedding":
        train_dataset = EmbeddingDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            use_seqaug=config.data.use_seqaug,
            data_type="train",
        )

        val_dataset = EmbeddingDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            data_type="val",
        )

    return train_dataset, val_dataset