"""Place each already-normalized utterance vector in its dataset's block."""
import torch
from torch import nn

from utils.dataset_identity import (
    block_expansion_enabled, block_expansion_multiplier, block_expansion_padding_value,
    validate_block_specification,
)


class DatasetBlockExpansion(nn.Module):
    def __init__(self, input_size, specification=None):
        super().__init__()
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("Block input dimension must be positive")
        self.num_blocks = 4
        self.dataset_to_id = validate_block_specification(specification or {})
        self.output_multiplier = block_expansion_multiplier(specification or {})
        self.explicit_block_width = "block_width" in (specification or {})
        self.block_width = int((specification or {}).get("block_width", self.input_size))
        if self.block_width < self.input_size:
            raise ValueError(
                f"Block width {self.block_width} cannot hold the {self.input_size}-dimensional embedding"
            )
        self.output_size = (
            self.num_blocks * self.block_width
            if self.explicit_block_width
            else int(round(self.output_multiplier * self.input_size))
        )
        self.padding_value = block_expansion_padding_value(specification or {})
        if self.explicit_block_width:
            self.start_positions = tuple(index * self.block_width for index in range(self.num_blocks))
        elif self.output_multiplier == 1.0:
            # The 1x baseline has no extra space: route every dataset to the
            # same original-width embedding rather than dropping features.
            self.start_positions = (0,) * self.num_blocks
        else:
            if self.output_size < self.input_size + self.num_blocks - 1:
                raise ValueError(
                    "Output dimension is too small to place four distinct full-width dataset windows"
                )
            available_start = self.output_size - self.input_size
            self.start_positions = tuple(
                int(round(index * available_start / (self.num_blocks - 1)))
                for index in range(self.num_blocks)
            )
        self.windows = tuple(
            (start, start + self.input_size) for start in self.start_positions
        )

    def get_extra_state(self):
        if self.explicit_block_width:
            return {
                "version": 4, "input_size": self.input_size,
                "num_blocks": self.num_blocks, "dataset_to_id": dict(self.dataset_to_id),
                "output_multiplier": self.output_multiplier,
                "block_width": self.block_width, "output_size": self.output_size,
                "start_positions": list(self.start_positions),
                "padding_value": self.padding_value,
            }
        # Keep the exact old signature for 4x checkpoints: their routed inputs
        # and learned-head dimensions are unchanged.
        if self.output_multiplier == 4.0 and self.padding_value == 0.0:
            return {"version": 1, "input_size": self.input_size,
                    "num_blocks": self.num_blocks,
                    "dataset_to_id": dict(self.dataset_to_id)}
        state = {"version": 2, "input_size": self.input_size,
                 "num_blocks": self.num_blocks, "dataset_to_id": dict(self.dataset_to_id),
                 "output_multiplier": self.output_multiplier,
                 "output_size": self.output_size,
                 "start_positions": list(self.start_positions)}
        if self.padding_value != 0.0:
            state["version"] = 3
            state["padding_value"] = self.padding_value
        return state

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint dataset block dimensions/mapping differ from the configured architecture")

    def forward(self, embeddings, dataset_ids):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_size:
            raise ValueError(f"Expected [batch, {self.input_size}] normalized embeddings")
        if not isinstance(dataset_ids, torch.Tensor) or dataset_ids.dtype != torch.long:
            raise ValueError("Dataset block expansion requires a torch.long dataset_id per example")
        if dataset_ids.ndim != 1 or dataset_ids.shape[0] != embeddings.shape[0]:
            raise ValueError("dataset_ids must have shape [batch]")
        if ((dataset_ids < 1) | (dataset_ids > self.num_blocks)).any():
            raise ValueError("dataset_ids must be between 1 and 4")
        starts = torch.tensor(self.start_positions, device=embeddings.device,
                              dtype=torch.long)[dataset_ids.to(embeddings.device) - 1]
        offsets = torch.arange(self.input_size, device=embeddings.device, dtype=torch.long)
        indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        output = embeddings.new_full((embeddings.shape[0], self.output_size), self.padding_value)
        return output.scatter(1, indices, embeddings)


def build_block_expansion(input_size, specification, use_layer_norm, layer_norm_affine):
    if not block_expansion_enabled(specification):
        return None
    if not use_layer_norm or layer_norm_affine:
        raise ValueError("Dataset block expansion requires LayerNorm with elementwise_affine=False")
    return DatasetBlockExpansion(input_size, specification)


def unpack_dynamic_inputs(inputs):
    if len(inputs) == 2:
        features, sampling_rate_ids = inputs
        return features, sampling_rate_ids, None
    if len(inputs) == 3:
        return inputs
    raise ValueError("Expected (features, sampling_rate_ids[, dataset_ids])")
