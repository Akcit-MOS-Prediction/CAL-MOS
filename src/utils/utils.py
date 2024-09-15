import torch
import pandas as pd
from torch.utils.data import DataLoader

from utils.dataloader import MosDataset


def build_dataloaders(config):
    """Builds the dataloader for the CAL-MOS model.

    Params:

    config (DictConfig): Configuration

    Returns:

    Tuple[DataLoader, DataLoader]: Train and validation dataloaders
    """
    train_data = pd.read_csv(config.datasets.train[0].metadata_path)
    val_data = pd.read_csv(config.datasets.val[0].metadata_path)
    test_data = pd.read_csv(config.datasets.test[0].metadata_path)

    train_dataset = MosDataset(
        data=train_data,
        filename_column=config.datasets.train[0].filename_column,
        target_column=config.datasets.train[0].target_column,
        base_dir=config.datasets.train[0].base_dir,
        aggregation_strategy=config.data.aggregation_strategy,
        use_seqaug=config.data.use_seqaug,
        data_type="train",
    )

    val_dataset = MosDataset(
        data=val_data,
        filename_column=config.datasets.val[0].filename_column,
        target_column=config.datasets.val[0].target_column,
        base_dir=config.datasets.val[0].base_dir,
        aggregation_strategy=config.data.aggregation_strategy,
        use_seqaug=config.data.use_seqaug,
        data_type="val",
    )

    test_dataset = MosDataset(
        data=test_data,
        filename_column=config.datasets.test[0].filename_column,
        target_column=config.datasets.test[0].target_column,
        base_dir=config.datasets.test[0].base_dir,
        aggregation_strategy=config.data.aggregation_strategy,
        use_seqaug=config.data.use_seqaug,
        data_type="test",
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=config.train.shuffle,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader