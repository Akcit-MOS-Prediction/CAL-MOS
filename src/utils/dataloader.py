import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor
from transformers import AutoModel, AutoFeatureExtractor


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
        use_seqaug: bool = False,
        data_type: str = "train",
    ):
        """Initialization"""
        self.data = data

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        self.use_seqaug = use_seqaug

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

    def seqaug(self, input_tensor, alpha: float = 0.2):
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
        filename = self.filenames[index]

        # Ensure the filename has the correct extension
        if not filename.endswith(".pt"):
            filename = filename.split(".")[0] + ".pt"

        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]

        features = self._load_file(filepath)

        if self.use_seqaug and self.data_type == "train":
            features = self.seqaug(features)

        # Convert features to float
        features = features.float()
        return features, target


class EmbeddingCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
        """
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features, targets = zip(*batch)
        batch_size = len(features)
        num_layers = features[0].shape[0]
        feature_dim = features[0].shape[-1]

        features = list(features)
        targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets])

        lengths = [feature.shape[1] for feature in features]
        max_length = max(lengths)

        padded_features = torch.full((batch_size, num_layers, max_length, feature_dim), self.padding_value)

        for i, feature in enumerate(features):
            length = feature.shape[1]
            padded_features[i, :, :length, :] = feature

        return padded_features, targets


class DynamicDataset(Dataset):
    def __init__(
        self,
        data,
        base_dir: str,
        filename_column: str,
        target_column: str,
        mixup_alpha: Optional[float] = 0.0,
        use_rand_truncation: bool = False,
        min_duration: Optional[float] = 0.0,
        insert_white_noise: bool = False,
        min_white_noise_amp: float = 0.01,
        max_white_noise_amp: float = 0.1,
        data_type: str = "train",
        class_num: int = 15,
        target_sr: int = 16000,
    ):
        """Initialization"""
        self.data = data
        self.base_dir = base_dir

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        # Data augmentation parameters
        self.mixup_alpha = mixup_alpha

        self.min_duration = min_duration
        self.use_rand_truncation = use_rand_truncation

        self.insert_white_noise = insert_white_noise
        self.min_white_noise_amp = min_white_noise_amp
        self.max_white_noise_amp = max_white_noise_amp

        self.data_type = data_type
        self.class_num = class_num

        self.target_sr = target_sr
        # Cache for sampling rate resamplers
        self.resamplers = {}

    def __len__(self):
        return len(self.data)

    def _random_truncation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Cut or pad an audio tensor to the desired length.
        """
        min_length = int(self.min_duration * self.target_sr)
        len_audio = audio.shape[-1]

        if self.use_rand_truncation and len_audio > min_length:
            segment_length = random.randint(min_length, len_audio)

            max_start = len_audio - segment_length
            start = random.randint(0, max_start)
            end = start + segment_length

            audio = audio[..., start:end]

        return audio

    def _load_wav(self, filepath: str):
        waveform, source_sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if source_sr != self.target_sr:
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=self.target_sr)
            waveform = self.resamplers[source_sr](waveform)

        return waveform, self.target_sr

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        main_target = self.targets[index]
        main_file = Path(self.filenames[index])

        # If using mixup and in training mode
        if self.mixup_alpha > 0.0 and self.data_type == "train":
            attempts = 0
            rand_index = index
            # Force mixup with a different targets, attempt limit is 10 to avoid infinite loop
            while attempts < 10 and self.targets[rand_index] == main_target:
                rand_index = random.randint(0, len(self.targets) - 1)
                attempts += 1

            rand_target = self.targets[rand_index]
            rand_file = Path(self.filenames[rand_index])

            original_path = self.base_dir / main_file
            original_path = original_path.resolve()

            rand_path = self.base_dir / rand_file
            rand_path = rand_path.resolve()

            audio_original, _ = self._load_wav(original_path)
            audio_rand, _ = self._load_wav(rand_path)

            # Sample lambda from beta distribution
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Audios must have the same length
            if audio_original.shape[-1] > audio_rand.shape[-1]:
                audio_rand = torch.nn.functional.pad(audio_rand, (0, audio_original.shape[-1] - audio_rand.shape[-1]))
            elif audio_original.shape[-1] < audio_rand.shape[-1]:
                audio_original = torch.nn.functional.pad(audio_original, (0, audio_rand.shape[-1] - audio_original.shape[-1]))

            # Mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # When using mixup we need to use one-hot encoding for the target
            target = torch.zeros(self.class_num)
            target[main_target] = mix_lambda
            target[rand_target] = 1 - mix_lambda
        else:
            filepath = self.base_dir / main_file
            filepath = filepath.resolve()
            audio, _ = self._load_wav(filepath)
            target = main_target

        # One-hot encoding for the target if using mixup in eval mode
        if self.mixup_alpha > 0.0 and self.data_type != "train":
            target = torch.zeros(self.class_num)
            target[main_target] = 1.0

        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)

        # White noise insertion
        if self.insert_white_noise and self.data_type == "train":
            # dynamically insert white noise
            white_noise_amp = torch.rand(1) * (self.max_white_noise_amp - self.min_white_noise_amp) + self.min_white_noise_amp
            audio = audio + white_noise_amp * torch.randn_like(audio)

        return audio.squeeze(0).numpy(), target


class DynamicCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        processor = None,
        target_sr: int = 16000
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
            processor: A processor or feature extractor to process raw audio
                       into features if desired.
        """
        self.processor = processor
        self.target_sr = target_sr
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audios, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets])

        audio_len = torch.tensor([a.shape[-1] for a in audios], dtype=torch.int64)
        max_audi_len = max([a.shape[-1] for a in audios])
        #max audio_len 
        processed = self.processor(
            audios,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        # Special case for Whisper, that expects a fixed input size of 3000 (30 seconds)
        if isinstance(self.processor, WhisperFeatureExtractor) and processed.input_features.shape[-1] < 3000:
            processed = self.processor(
                audios,
                return_tensors="pt",
                sampling_rate=self.target_sr,
            )

        return processed, targets.float(), max_audi_len