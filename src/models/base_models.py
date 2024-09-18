from typing import List, Tuple

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim = input_size
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, output_size))

        self.layers: nn.Module = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, nonlinearity="relu")
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        logits = self.layers(x)

        return logits


class CalmosModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        intermediate_size: int = 2048,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        mlp_output_dims: List[int] = [4096, 2048, 1024],
        mlp_output_size: int = 1,
        mlp_dropout: float = 0.1,
        use_bias: bool = True,
        layer_weight_strategy: str = "transformer", # "transformer" or "weighted_sum" or "last_hidden_state"
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1, # default to the last layer
    ):
        super().__init__()

        self.mlp = MLPBase(
            input_size=hidden_size,
            output_size=mlp_output_size,
            output_dims=mlp_output_dims,
            dropout=mlp_dropout,
        )

        self.layer_weight_strategy = layer_weight_strategy

        if layer_weight_strategy == "transformer":
            self.aggregation_token = nn.Parameter(torch.randn(hidden_size))

            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    activation=hidden_act,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    bias=use_bias,
                    batch_first=True,
                )
                for _ in range(num_hidden_layers)
            ])

            # Initialize weights
            self._transformers_init_weights()

        elif layer_weight_strategy == "weighted_sum":
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
            )
        elif layer_weight_strategy == "specific_layer":
            self.specific_layer_idx = specific_layer_idx
        else:
            raise ValueError(f"Invalid layer weight strategy: {layer_weight_strategy}")

    def _transformers_init_weights(self):
        for layer in self.transformer_layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    if 'self_attn' in name or 'linear' in name:
                        # Use Xavier initialization for GELU activation
                        init.xavier_uniform_(param)
                    elif 'norm' in name:
                        # LayerNorm weights initialized to ones
                        init.ones_(param)
                elif 'bias' in name:
                    if 'norm' in name:
                        # LayerNorm biases initialized to zeros
                        init.zeros_(param)
                    else:
                        # Other biases initialized to zeros
                        init.zeros_(param)

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do a weighted sum of the layers in the sequence
        """
        # Stack the weights into a tensor
        # Apply softmax to normalize the weights
        layer_weights = torch.stack([w for w in self.layer_weights]).view(1, -1, 1)  # Shape: [1, seq_length, 1]
        layer_weights = F.softmax(layer_weights, dim=1)  # Apply softmax over the sequence dimension

        # Multiply and sum over the sequence dimension
        weighted_sum = (x * layer_weights).sum(dim=1)  # Shape: [batch_size, feature_size]
        return weighted_sum

    def _transformer_aggregation(self, x: torch.Tensor) -> torch.Tensor:
        # Aggregate the input sequence
        x = torch.cat([self.aggregation_token.repeat(x.size(0), 1).unsqueeze(1), x], dim=1)
        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)
        # Extract the aggregation token
        x = x[:, 0, :]

        return x

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]

    def forward(self, x):
        if self.layer_weight_strategy == "transformer":
            # Transformer
            x = self._transformer_aggregation(x)
        elif self.layer_weight_strategy == "weighted_sum":
            # Weighted sum
            x = self._weighted_sum(x)
        elif self.layer_weight_strategy == "specific_layer":
            # Last hidden state
            x = self._specific_layer(x, self.specific_layer_idx)
        # MLP
        logits = self.mlp(x).squeeze(-1)

        return logits