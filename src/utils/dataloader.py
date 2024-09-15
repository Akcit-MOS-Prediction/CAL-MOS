import os
from typing import List, Tuple, Dict, Optional

import torch
import pandas as pd
from torch.utils.data import Dataset


class MosDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
        aggregation_strategy: str = "mean",
        use_seqaug: bool = False,
        data_type: str = "train",
    ):
        """Initialization"""
        self.data = data
        self.filename_column = filename_column
        self.target_column = target_column

        self.use_seqaug = use_seqaug
        self.aggregation_strategy = aggregation_strategy

        self.base_dir = base_dir
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """Load an audio file

        Params:

        filepath (str): Path to the audio file

        Returns:

        torch.Tensor: Audio tensor
        """
        features = torch.load(filepath)

        return features

    def _layer_aggregation_strategy(self, features: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
        """Aggregate the layers of the audio tensor

        Params:

        features (torch.Tensor): Audio tensor
        strategy (str): Aggregation strategy

        Returns:

        torch.Tensor: Aggregated audio tensor
        """

        if strategy == "mean":
            features = features.mean(dim=1)
        elif strategy == "max":
            features = features.max(dim=1)
        elif strategy == "min":
            features = features.min(dim=1)
        elif strategy == "sum":
            features = features.sum(dim=1)

        return features

    def seqaug(self, input_tensor, alpha=0.2):
        """
        Applies SeqAug (https://arxiv.org/abs/2305.01954) augmentation to the input tensor.

        Params:
            input_tensor (torch.Tensor): Input tensor of shape [sequence_length, feature_size].
            alpha (float): Parameter for the Beta distribution (α ∈ [0, 1]).

        Returns:
            output_tensor (torch.Tensor): Augmented tensor of the same shape as input_tensor.
        """
        # Get dimensions
        sequence_length, feature_size = input_tensor.shape

        # Sample proportion p from Beta(α, α)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        p = beta_dist.sample()

        # Determine the number of feature addresses to sample
        num_features_to_sample = int(p.item() * feature_size)
        num_features_to_sample = max(num_features_to_sample, 1)  # Ensure at least one feature is selected

        # Randomly select feature addresses (indices) to permute
        selected_features = torch.randperm(feature_size)[:num_features_to_sample]

        # Generate a random permutation of the time indices
        perm = torch.randperm(sequence_length)

        # Create a copy of the input tensor to hold the augmented data
        output_tensor = input_tensor.clone()

        # Permute the selected features along the time axis according to the random permutation
        output_tensor[:, selected_features] = input_tensor[perm][:, selected_features]

        return output_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[str]]:
        """Get an item from the dataset

        Params:

        index (int): Index of the item to get

        Returns:

        Dict[torch.Tensor, np.ndarray]: A dictionary containing the audio and the caption
        """
        filename = self.data.iloc[index][self.filename_column][:-4] + ".pt"
        filepath = os.path.join(self.base_dir, filename)

        target = self.data.iloc[index][self.target_column]

        features = self._load_file(filepath)

        features = self._layer_aggregation_strategy(features, self.aggregation_strategy)

        if self.use_seqaug and self.data_type == "train":
            features = self.seqaug(features)

        # Convert features to float
        features = features.float()
        target = torch.tensor(target).float()

        return features, target