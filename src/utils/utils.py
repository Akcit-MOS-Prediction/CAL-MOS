import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoFeatureExtractor

from utils.dataloader import (
    EmbeddingDataset,
    DynamicDataset,
    MultiLayerEmbeddingDataset,
    MultiLayerEmbeddingWeightedSumDataset
)


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
            sr_column=config.datasets.train[0].get("sr_column", None), # Backward compatibility
            sr_dictionary=config.data.get("sr_dictionary", None), # Backward compatibility
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
            sr_column=config.datasets.train[0].get("sr_column", None), # Backward compatibility
            sr_dictionary=config.data.get("sr_dictionary", None), # Backward compatibility
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
    elif config.model.model_type.lower() == "multiple_layer_embedding":
        train_dataset = MultiLayerEmbeddingDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            target_dir=config.datasets.train[0].get("target_dir", None),
            layer_id=config.model.specific_layer_idx,
            use_seqaug=config.data.use_seqaug,
            data_type="train",
        )

        val_dataset = MultiLayerEmbeddingDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            target_dir=config.datasets.train[0].get("target_dir", None),
            layer_id=config.model.specific_layer_idx,
            data_type="val",
        )
    elif config.model.model_type.lower() == "multiple_layer_embedding_weighted_sum":
        train_dataset = MultiLayerEmbeddingWeightedSumDataset(
            data=train_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            target_dir=config.datasets.train[0].get("target_dir", None),
            use_seqaug=config.data.use_seqaug,
            data_type="train",
        )

        val_dataset = MultiLayerEmbeddingWeightedSumDataset(
            data=val_data,
            filename_column=config.datasets.train[0].filename_column,
            target_column=config.datasets.train[0].target_column,
            base_dir=config.datasets.train[0].base_dir,
            target_dir=config.datasets.train[0].get("target_dir", None),
            data_type="val",
        )

    return train_dataset, val_dataset