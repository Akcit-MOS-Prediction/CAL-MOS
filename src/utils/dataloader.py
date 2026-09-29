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
from safetensors.torch import load_file as load_safetensors_file


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
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".pt"

        # Remove leading "./" if present
        if filename.startswith("./"):
            filename = filename[2:]

        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]

        features = self._load_file(filepath)

        if self.use_seqaug and self.data_type == "train":
            features = self.seqaug(features)

        # Convert features to float
        features = features.float()
        return features, target


class MultiLayerEmbeddingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
        target_dir: str = None,
        layer_id: int = None,
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
        self.target_dir = target_dir
        self.data_type = data_type

        self.layer_id = layer_id

        print(f"Using layer_id: {self.layer_id} for MultiLayerEmbeddingDataset")

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """Load an audio file

        Params:

        filepath (str): Path to the audio file

        Returns:

        torch.Tensor: Audio tensor
        """
        # features = torch.load(filepath)
        features = load_safetensors_file(filepath)["embeddings"]

        return features[self.layer_id]

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

        # Handle relative paths
        if self.base_dir is not None and filename.startswith("./"):
            # print("1", filename)
            filename = filename[2:]
            # print("2", filename)
            filename = os.path.join(self.base_dir, filename)
            # print("3", filename)

        # Modify path if target_dir is provided
        if self.target_dir is not None and self.base_dir is not None:
            filename = filename.replace(self.base_dir, self.target_dir)

        # Ensure the filename has the correct extension
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".safetensors"

        target = self.targets[index]

        features = self._load_file(filename)
        if self.use_seqaug and self.data_type == "train":
            features = self.seqaug(features)

        # Convert features to float
        features = features.float()

        return features, target


class OneLayerEmbeddingCollate:
    def __init__(self, padding_value: float = 0.0):
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, targets = zip(*batch)

        features = list(features)
        batch_size = len(features)
        feature_dim = features[0].shape[-1]

        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        # Keep same dtype/device as input features
        padded_features = features[0].new_full(
            (batch_size, max_length, feature_dim),
            fill_value=self.padding_value,
        )

        # 1 for valid positions, 0 for padding
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,          # or torch.bool if you prefer
            device=padded_features.device,
        )

        for i, f in enumerate(features):
            L = f.shape[0]
            padded_features[i, :L, :] = f
            attention_mask[i, :L] = 1

        # Targets -> float tensor (keeps your original behavior)
        targets = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in targets])

        return (padded_features, attention_mask), targets


class AllLayersEmbeddingCollate:
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
        sr_column: str = None,
        sr_dictionary: Optional[Dict[str, int]] = None,
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

        if sr_column is not None:
            self.sr = self.data[sr_column].values
        else:
            self.sr = [16] * len(self.data)

        if sr_dictionary is None:
            print("Warning: No sr_dictionary provided. Using default values.")
            sr_dictionary = {"16": 0}
        self.sr = [sr_dictionary.get(str(sr), 0) for sr in self.sr]

        self.filename_column = filename_column
        self.target_column = target_column
        self.sr_column = sr_column

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
        source_sr = self.sr[index]

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

        return audio.squeeze(0).numpy(), source_sr, target


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
        audios, source_srs, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets])
        source_srs = torch.stack([torch.tensor(t) for t in source_srs])

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

        return (processed, source_srs), targets.float()


class DynamicAudioCollate:
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
        audios, _, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets])

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

        # pad audios
        max_length = max([audio.shape[-1] for audio in audios])
        padded_audios = torch.full((len(audios), max_length), 0.0)
        for i, audio in enumerate(audios):
            length = audio.shape[-1]
            padded_audios[i, :length] = torch.from_numpy(audio)

        return (processed, padded_audios), targets.float()


class MultiLayerEmbeddingWeightedSumDataset(Dataset):
    """
    For weighted-sum models: returns features with ALL layers.

    Each item returns:
      features: [L, T, F]  (float32)
      target:   scalar/float
    """
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
        target_dir: str = None,
        use_seqaug: bool = False,
        data_type: str = "train",
    ):
        self.data = data
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        self.use_seqaug = use_seqaug
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """
        Expected safetensors key: ["embeddings"] with shape [L, T, F]
        """
        feats = load_safetensors_file(filepath)["embeddings"]  # [L,T,F]
        return feats

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]

        # Handle relative paths
        if self.base_dir is not None and filename.startswith("./"):
            filename = filename[2:]
            filename = os.path.join(self.base_dir, filename)

        # Modify path if target_dir is provided
        if self.target_dir is not None and self.base_dir is not None:
            filename = filename.replace(self.base_dir, self.target_dir)

        # Ensure correct extension
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".safetensors"

        target = self.targets[index]

        features = self._load_file(filename)  # [L,T,F]

        return features.float(), torch.as_tensor(target, dtype=torch.float32)


class MultiLayerEmbeddingWeightedSumCollate:
    """
    Pads variable-length sequences on time dimension for multilayer features.

    Input items:
      features: [L,T,F]
      target: scalar

    Output:
      (padded_features, attention_mask), targets
      padded_features: [B,L,Tmax,F]
      attention_mask:  [B,Tmax]
      targets:         [B]
    """
    def __init__(self, padding_value: float = 0.0):
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features_list, targets = zip(*batch)

        features_list = list(features_list)
        batch_size = len(features_list)

        # features are [L,T,F]
        L = features_list[0].shape[0]
        Fdim = features_list[0].shape[-1]

        lengths = [x.shape[1] for x in features_list]  # time lengths
        Tmax = max(lengths)

        padded = features_list[0].new_full(
            (batch_size, L, Tmax, Fdim),
            fill_value=self.padding_value,
        )

        attention_mask = torch.zeros(
            (batch_size, Tmax),
            dtype=torch.long,
            device=padded.device,
        )

        for i, x in enumerate(features_list):
            # x: [L,Ti,F]
            Ti = x.shape[1]
            padded[i, :, :Ti, :] = x
            attention_mask[i, :Ti] = 1

        targets = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in targets])

        return (padded, attention_mask), targets