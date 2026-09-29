import math

from typing import List, Tuple
from abc import ABC, abstractmethod

import os
import sys

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from models.torch_relu_kan import ReLUKAN
# Backward compatibility with older versions of the repository
try:
    from models.add_peft import build_peft_model
except ImportError:
    print("PEFT not available. Please install the PEFT package (https://huggingface.co/docs/peft/en/install) to use it.")
    build_peft_model = None

from models.ced.ced_finetuning import FineTuneCED

ACTIVATIONS_FUNCS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}


class CKALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def centering(self, K):
        """
        Centers the kernel matrix K using the centering matrix H = I - (1/n)11^T
        Args:
            K: Kernel matrix of shape [n x n]
        Returns:
            Centered kernel matrix
        """
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - (1.0/n) * torch.ones((n, n), device=K.device)
        return H @ K @ H

    def forward(self, wav_features, rob_features):
        """
        Compute CKA loss between WavLM and RoBERTa features. This is not the frame-level output of WavLM and RoBERTa, it needs to be any kind of aggregation (attention pooling, mean pooling, etc)
        Args:
            wav_features: WavLM features after transformer [batch_size x hidden_dim]
            rob_features: RoBERTa features after transformer [batch_size x hidden_dim]
        Returns:
            CKA loss
        """
        # Compute Gram matrices
        K = wav_features @ wav_features.T  # [batch_size x batch_size]
        L = rob_features @ rob_features.T  # [batch_size x batch_size]

        # Center Gram matrices
        K_centered = self.centering(K)
        L_centered = self.centering(L)

        # Compute HSIC
        HSIC_KL = torch.trace(K_centered @ L_centered)
        HSIC_KK = torch.trace(K_centered @ K_centered)
        HSIC_LL = torch.trace(L_centered @ L_centered)

        # Compute CKA
        epsilon = 1e-8  # Small constant for numerical stability
        CKA = HSIC_KL / (torch.sqrt(HSIC_KK * HSIC_LL) + epsilon)

        # Return loss
        return CKA


class AdapterLayer(nn.Module):
    def __init__(self, feat_dim: int, adapter_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.ReLU(True),
            nn.Linear(adapter_dim, adapter_dim),
        )

    def forward(self, x):
        return self.adapter(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        # Register as a buffer so it is saved in the model state_dict but not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: [B, Seq_len, d_model]
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, start_pos:start_pos+seq_len, :]
        return x


class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        output_size: int = 7,
        dropout: float = 0.1,
        activation_func: str = "relu",
    ) -> None:
        super().__init__()

        # Validate and set the activation function
        activation_func = activation_func.lower()
        if activation_func not in ACTIVATIONS_FUNCS:
            raise ValueError(
                f"Unsupported activation function: {activation_func}. "
                f"Supported activations are: {list(ACTIVATIONS_FUNCS.keys())}"
            )
        self.activation_func = activation_func
        activation_layer = ACTIVATIONS_FUNCS[self.activation_func]

        layers: List[nn.Module] = []
        current_dim = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_size))
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights of linear layers based on the activation function:
        ReLU/GELU: Kaiming (He) initialization with a=0, Leaky ReLU: Kaiming (He) initialization with a=0.01
        Biases are initialized to zeros.
        """
        for module in self.layers:
            if isinstance(module, nn.Linear):
                if self.activation_func in ["relu", "gelu"]:
                    # Kaiming initialization with a=0
                    init.kaiming_uniform_(module.weight, a=0, nonlinearity="relu")
                elif self.activation_func == "leaky_relu":
                    # Kaiming initialization with a=0.01
                    init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        logits = self.layers(x)
        return logits


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


# class AttentiveStatisticsPooling(Pooling):
#     """
#     AttentiveStatisticsPooling
#     Paper: Attentive Statistics Pooling for Deep Speaker Embedding
#     https://arxiv.org/pdf/1803.10963.pdf

#     Input:
#     x: (batch_size, T, feat_dim)
#     Output:
#     (batch_size, feat_dim*2)
#     """
#     def __init__(self, input_size):
#         super().__init__()
#         self._indim = input_size
#         self.sap_linear = nn.Linear(input_size, input_size)
#         self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
#         torch.nn.init.normal_(self.attention, mean=0, std=1)

#     def forward(self, xs):
#         """
#         Args:
#             xs: Input tensor of shape (batch_size, T, feat_dim)
#         Returns:
#             Pooled features of shape (batch_size, feat_dim*2)
#         """
#         # Calculate attention weights
#         h = torch.tanh(self.sap_linear(xs))
#         w = torch.matmul(h, self.attention).squeeze(dim=2)
#         w = F.softmax(w, dim=1).unsqueeze(2)

#         # Calculate weighted mean
#         mu = torch.sum(xs * w, dim=1)
#         # Calculate weighted standard deviation
#         rh = torch.sqrt((torch.sum((xs**2) * w, dim=1) - mu**2).clamp(min=1e-5))
#         # Concatenate mean and standard deviation
#         pooled = torch.cat((mu, rh), dim=1)
#         return pooled


class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    https://arxiv.org/pdf/1803.10963.pdf
    Input:
      xs:       [B, T, F]
      att_mask: [B, T] (1=valid, 0=pad)  (optional)
    Output:
      [B, 2F]
    """
    def __init__(self, input_size: int):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs: torch.Tensor, att_mask: torch.Tensor = None) -> torch.Tensor:
        h = torch.tanh(self.sap_linear(xs))                  # [B,T,F]
        logits = torch.matmul(h, self.attention).squeeze(dim=2)  # [B,T]

        if att_mask is not None:
            # att_mask: 1 for valid, 0 for pad
            m = att_mask.to(dtype=torch.bool, device=xs.device)  # [B,T]

            # safety: if a sample is fully masked, force at least one valid position
            all_pad = ~m.any(dim=1)  # [B]
            if all_pad.any():
                m = m.clone()
                m[all_pad, 0] = True

            # mask logits BEFORE softmax
            logits = logits.masked_fill(~m, -1e9)

        w = F.softmax(logits, dim=1).unsqueeze(2)            # [B,T,1]

        mu = torch.sum(xs * w, dim=1)                        # [B,F]
        second = torch.sum((xs ** 2) * w, dim=1)             # [B,F]
        var = (second - mu ** 2).clamp(min=1e-5)             # [B,F]
        rh = torch.sqrt(var)                                 # [B,F]

        return torch.cat((mu, rh), dim=1)                    # [B,2F]



class BaseModel(nn.Module, ABC):
    """
    Base Model for SER that handles:
    - MLP initialization
    - Layer weight strategy: 'per_layer', 'weighted_sum', 'transformer'
    - Pooling strategy: 'mean' or 'attpool' (attpool only with per_layer)
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer", # "per_layer" or "weighted_sum"
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean", # "mean" or "attpool"
        # adapter parameters (only used if layer_weight_strategy == "layer_adapter")
        adapter_input_dim: int = 1024,
        adapter_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

        self.layer_weight_strategy = layer_weight_strategy
        self.num_feature_layers = num_feature_layers
        self.specific_layer_idx = specific_layer_idx
        self.pooling_strategy = pooling_strategy

        print(f"\n\nINSIDE BASE MODEL - SPECIFIC LAYER IDX: {specific_layer_idx}\n\n")

        if layer_weight_strategy == "transformer":
            self.aggregation_token = nn.Parameter(torch.randn(kwargs["transformer_hidden_size"]))

            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=kwargs["transformer_hidden_size"],
                    nhead=kwargs["transformer_nhead"],
                    dim_feedforward=kwargs["transformer_dim_feedforward"],
                    activation=kwargs["transformer_activation"],
                    dropout=kwargs["transformer_dropout"],
                    layer_norm_eps=kwargs["transformer_layer_norm_eps"],
                    bias=kwargs["transformer_bias"],
                    batch_first=True,
                )
                for _ in range(kwargs["transformer_num_hidden_layers"])
            ])

            self._transformers_init_weights()

            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=kwargs["transformer_hidden_size"],
                max_len=num_feature_layers+1
            )
        elif layer_weight_strategy == "weighted_sum":
            print("Using weighted sum for layer weighting")
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
            )
        elif layer_weight_strategy == "per_layer":
            if specific_layer_idx < 0:
                specific_layer_idx = num_feature_layers - 1
            self.specific_layer_idx = specific_layer_idx
        elif layer_weight_strategy == "layer_adapter":
            print("Using layer adapter for layer weighting")
            self.layer_adapters = nn.ModuleList(
                [AdapterLayer(adapter_input_dim, adapter_dim) for _ in range(num_feature_layers)]
            )
        else:
            raise ValueError(f"Invalid layer weight strategy: {layer_weight_strategy}")

        # Validate pooling_strategy
        # attpool is only allowed for per_layer
        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )

        self.attpool = None

    def _transformers_init_weights(self):
        for layer in self.transformer_layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    if "self_attn" in name or "linear" in name:
                        # Use Xavier initialization for GELU activation
                        init.xavier_uniform_(param)
                    elif "norm" in name:
                        # LayerNorm weights initialized to ones
                        init.ones_(param)
                elif "bias" in name:
                    if "norm" in name:
                        # LayerNorm biases initialized to zeros
                        init.zeros_(param)
                    else:
                        # Other biases initialized to zeros
                        init.zeros_(param)

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, F]
        B, L, T, Fdim = x.shape

        # [L, 1] -> [L]
        w = torch.stack(list(self.layer_weights), dim=0).squeeze(-1)
        w = F.softmax(w, dim=0).view(1, L, 1, 1)  # [1,L,1,1]

        return (x * w).sum(dim=1)  # [B,T,F]

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]

    def _apply_layer_adapter(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, F]
        B, L, T, Fdim = x.shape
        adapted_layers = []
        for i in range(L):
            layer_i = x[:, i, :, :]  # [B,T,F]
            adapted_layer_i = self.layer_adapters[i](layer_i)  # [B,T,adapter_dim]
            adapted_layers.append(adapted_layer_i)
        # [B,T*L,adapter_dim]
        adapted_layers = torch.stack(adapted_layers, dim=1).view(B, L*T, -1)
        return adapted_layers

    def _transformer_aggregation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer aggregation.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQ_LEN, FEAT_DIM]
        Returns:
            Aggregated tensor of shape [B, SEQ_LEN, FEAT_DIM]
        """
        # Add aggregation token
        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        # Rearrange so we process each (B, T) separately
        x = x.permute(0, 2, 1, 3) # [B, SEQ_LEN, NUM_LAYERS, FEAT_DIM]
        x = x.reshape(B * SEQ_LEN, NUM_LAYERS, FEAT_DIM) # [B*SEQ_LEN, NUM_LAYERS, FEAT_DIM]

        # Insert aggregation token at position 0
        agg_token = self.aggregation_token.unsqueeze(0).unsqueeze(0).expand(B * SEQ_LEN, 1, FEAT_DIM)
        x = torch.cat([agg_token, x], dim=1) # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Add sinusoidal positional encoding
        x = self.pos_encoder(x) # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Extract the aggregation token's output
        x_agg = x[:, 0, :]  # [B*SEQ_LEN, FEAT_DIM]

        # Reshape back to [B, SEQ_LEN, FEAT_DIM]
        x_agg = x_agg.reshape(B, SEQ_LEN, FEAT_DIM)

        print("TRANSFORMER AGG SHAPE", x_agg.shape)
        return x_agg

    @abstractmethod
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to get embeddings:
        For dynamic model: returns [B,num_layers+1,T,F]
        For embedding model: returns [B,num_feature_layers,F]
        """
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass

    @staticmethod
    def _expand_att_mask_if_needed(
        embeddings: torch.Tensor,  # [B, T', F]
        att_mask: torch.Tensor,    # [B, T]
        layer_weight_strategy: str,
    ) -> torch.Tensor:
        if att_mask is None:
            return None

        T_emb = embeddings.size(1)
        T_mask = att_mask.size(1)

        if T_emb == T_mask:
            return att_mask

        # Only expected mismatch is when we concatenated time across layers
        if layer_weight_strategy != "layer_adapter":
            raise ValueError(
                f"Attention mask length mismatch: embeddings T={T_emb} vs mask T={T_mask} "
                f"with strategy={layer_weight_strategy}"
            )

        if T_emb % T_mask != 0:
            raise ValueError(
                f"Cannot expand mask: embeddings T={T_emb} not divisible by mask T={T_mask}"
            )

        L = T_emb // T_mask  # number of layer blocks concatenated
        # Your concatenation order is [layer0 T] + [layer1 T] + ... so repeat is correct.
        return att_mask.repeat(1, L)

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        if self.layer_weight_strategy == "transformer":
            embeddings = self._transformer_aggregation(embeddings)
        elif self.layer_weight_strategy == "per_layer":
            embeddings = self._specific_layer(embeddings, self.specific_layer_idx) # [B,T,F]
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings) # [B,T,F]
        elif self.layer_weight_strategy == "layer_adapter":
            embeddings = self._apply_layer_adapter(embeddings) # [B,T*L,adapter_dim]
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")

        return embeddings

    @staticmethod
    def masked_mean_pool(embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T] with 1 for valid, 0 for padding
        returns:    [B,F]
        """
        # Ensure mask is float for multiplication
        m = att_mask.to(dtype=embeddings.dtype)  # [B,T]
        m = m.unsqueeze(-1)                      # [B,T,1]

        summed = (embeddings * m).sum(dim=1)     # [B,F]
        denom = m.sum(dim=1).clamp(min=1e-6)     # [B,1] avoid div by zero
        return summed / denom

    def _apply_pooling(self, embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T]
        """
        att_mask = self._expand_att_mask_if_needed(
            embeddings, att_mask, self.layer_weight_strategy
        )

        if self.pooling_strategy == "mean":
            return self.masked_mean_pool(embeddings, att_mask)  # [B,F]

        elif self.pooling_strategy == "attpool":
            # If your AttentiveStatisticsPooling supports masks, pass it.
            # If it doesn't, you can pre-mask padded frames by zeroing them (less ideal but works).
            input_dim = embeddings.size(-1)
            if self.attpool is None or getattr(self.attpool, "input_size", None) != input_dim:
                self.attpool = AttentiveStatisticsPooling(input_size=input_dim).to(embeddings.device)

            return self.attpool(embeddings, att_mask)  # [B, 2F]
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ReLuKANBaseModel(nn.Module, ABC):
    """
    Base Model for SER that handles:
    - MLP initialization
    - Layer weight strategy: 'per_layer', 'weighted_sum', 'transformer'
    - Pooling strategy: 'mean' or 'attpool' (attpool only with per_layer)
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        # Remove or keep unused parameters (dropout, activation_func)
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer",
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean",
        relukan_grid: int = 3,  # New parameters for ReLUKAN
        relukan_k: int = 3,
        **kwargs,
    ):
        super().__init__()
        # Construct ReLUKAN width list
        width = [mlp_input_dim] + [mlp_hidden_dim] * mlp_num_layers + [mlp_output_size]
        self.mlp = ReLUKAN(width=width, grid=relukan_grid, k=relukan_k)

        # Existing code for layer weighting strategy, pooling, etc.
        self.layer_weight_strategy = layer_weight_strategy
        self.num_feature_layers = num_feature_layers
        self.specific_layer_idx = specific_layer_idx
        self.pooling_strategy = pooling_strategy

        if layer_weight_strategy == "transformer":
            self.aggregation_token = nn.Parameter(torch.randn(kwargs["transformer_hidden_size"]))

            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=kwargs["transformer_hidden_size"],
                    nhead=kwargs["transformer_nhead"],
                    dim_feedforward=kwargs["transformer_dim_feedforward"],
                    activation=kwargs["transformer_activation"],
                    dropout=kwargs["transformer_dropout"],
                    layer_norm_eps=kwargs["transformer_layer_norm_eps"],
                    bias=kwargs["transformer_bias"],
                    batch_first=True,
                )
                for _ in range(kwargs["transformer_num_hidden_layers"])
            ])

            self._transformers_init_weights()

            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=kwargs["transformer_hidden_size"],
                max_len=num_feature_layers+1
            )
        elif layer_weight_strategy == "weighted_sum":
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
            )
        elif layer_weight_strategy == "per_layer":
            if specific_layer_idx < 0:
                specific_layer_idx = num_feature_layers - 1
            self.specific_layer_idx = specific_layer_idx
        else:
            raise ValueError(f"Invalid layer weight strategy: {layer_weight_strategy}")

        # Validate pooling_strategy
        # attpool is only allowed for per_layer
        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )

        self.attpool = None

    def _transformers_init_weights(self):
        for layer in self.transformer_layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    if "self_attn" in name or "linear" in name:
                        # Use Xavier initialization for GELU activation
                        init.xavier_uniform_(param)
                    elif "norm" in name:
                        # LayerNorm weights initialized to ones
                        init.ones_(param)
                elif "bias" in name:
                    if "norm" in name:
                        # LayerNorm biases initialized to zeros
                        init.zeros_(param)
                    else:
                        # Other biases initialized to zeros
                        init.zeros_(param)

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, F]
        B, L, T, Fdim = x.shape

        # [L, 1] -> [L]
        w = torch.stack(list(self.layer_weights), dim=0).squeeze(-1)
        w = F.softmax(w, dim=0).view(1, L, 1, 1)  # [1,L,1,1]

        return (x * w).sum(dim=1)  # [B,T,F]

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]

    def _transformer_aggregation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer aggregation.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQ_LEN, FEAT_DIM]
        Returns:
            Aggregated tensor of shape [B, SEQ_LEN, FEAT_DIM]
        """
        # Add aggregation token
        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        # Rearrange so we process each (B, T) separately
        x = x.permute(0, 2, 1, 3) # [B, SEQ_LEN, NUM_LAYERS, FEAT_DIM]
        x = x.reshape(B * SEQ_LEN, NUM_LAYERS, FEAT_DIM) # [B*SEQ_LEN, NUM_LAYERS, FEAT_DIM]

        # Insert aggregation token at position 0
        agg_token = self.aggregation_token.unsqueeze(0).unsqueeze(0).expand(B * SEQ_LEN, 1, FEAT_DIM)
        x = torch.cat([agg_token, x], dim=1) # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Add sinusoidal positional encoding
        x = self.pos_encoder(x) # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)  # [B*SEQ_LEN, NUM_LAYERS+1, FEAT_DIM]

        # Extract the aggregation token's output
        x_agg = x[:, 0, :]  # [B*SEQ_LEN, FEAT_DIM]

        # Reshape back to [B, SEQ_LEN, FEAT_DIM]
        x_agg = x_agg.reshape(B, SEQ_LEN, FEAT_DIM)

        print("TRANSFORMER AGG SHAPE", x_agg.shape)
        return x_agg

    @abstractmethod
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to get embeddings:
        For dynamic model: returns [B,num_layers+1,T,F]
        For embedding model: returns [B,num_feature_layers,F]
        """
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        if self.layer_weight_strategy == "transformer":
            embeddings = self._transformer_aggregation(embeddings)
        elif self.layer_weight_strategy == "per_layer":
            embeddings = self._specific_layer(embeddings, self.specific_layer_idx) # [B,T,F]
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings) # [B,T,F]
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")

        return embeddings

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over the time dimension.
        embeddings: [B,T,F]
        mask: [B,T] if attpool selected
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over T
            return embeddings.mean(dim=1)  # [B,F]

        elif self.pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            input_dim = embeddings.size(-1)
            self.attpool = AttentiveStatisticsPooling(input_size=input_dim).to(embeddings.device)
            return self.attpool(embeddings)

        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x.shape", x.shape)
        # Get embeddings
        embeddings = self._get_embeddings(x)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings)  # [B,F] or [B,2F]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class CalMOSDynamicModel(BaseModel):
    """
    Uses a pretrained backbone (e.g. WavLM).
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        # PEFT parameters
        use_peft: bool = False,
        lora_keys: List[str] = None,
        lora_r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        bias: str = "none",
        # Sampling Rate parameters
        use_sr_embeddings: bool = False,
        sr_embedding_dim: int = 16,
        num_sr_classes: int = 3,
        # top_k_re-init parameters (if using Fine-tuning strategy)
        use_top_k_reinit: bool = False,
        reinit_k: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)

        if freeze_backbone and use_peft:
            raise ValueError("Parameters 'freeze_backbone' and 'use_peft' cannot be 'True' at the same time.")

        assert not (use_top_k_reinit and use_peft), "Top-k re-init and PEFT cannot be used together."
        assert not (use_top_k_reinit and freeze_backbone), "Top-k re-init and freezing backbone cannot be used together."

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Whisper is an encoder-decoder model, we only need the encoder part
        if "whisper" in model_name.lower():
            self.backbone = self.backbone.encoder

        if use_top_k_reinit:
            assert reinit_k > 0, "reinit_k must be > 0 when use_top_k_reinit is True"
            print(f"Re-initializing top {reinit_k} layers of the backbone")
            self._reinit_top_k_layers(reinit_k)

        self.freeze_backbone = freeze_backbone
        if use_peft:
            self.backbone = build_peft_model(
                self.backbone,
                lora_keys=lora_keys,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
            )
        elif freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

        self.use_sr_embeddings = use_sr_embeddings
        if self.use_sr_embeddings:
            self.sr_embedding = nn.Embedding(num_sr_classes, sr_embedding_dim)

    def _reinit_top_k_layers(self, k: int):
        # Find transformer blocks for common HF models
        layers = None

        if hasattr(self.backbone, "encoder"):
            enc = self.backbone.encoder
            if hasattr(enc, "layers"):
                layers = enc.layers
            elif hasattr(enc, "layer"):
                layers = enc.layer

        if layers is None:
            # e.g., WhisperEncoder has .layers directly
            if hasattr(self.backbone, "layers"):
                layers = self.backbone.layers
            elif hasattr(self.backbone, "layer"):
                layers = self.backbone.layer

        if layers is None:
            raise ValueError("Backbone does not have expected layer structure for top-k re-init.")

        num_layers = len(layers)
        if k > num_layers:
            raise ValueError(f"Cannot re-initialize top {k} layers because backbone only has {num_layers} layers.")

        # Prefer HF init to match the rest of the pretrained model
        init_fn = getattr(self.backbone, "_init_weights", None)
        if init_fn is None:
            raise ValueError("Backbone has no _init_weights; cannot safely match HF initialization.")

        for layer in layers[-k:]:
            print(f"Layer params before re-init: {[p.data.norm().item() for p in layer.parameters()]}")
            print(f"Re-initializing layer: {layer}")
            layer.apply(init_fn)
            print(f"Layer re-initialized: {layer}")
            print(f"Layer params after re-init: {[p.data.norm().item() for p in layer.parameters()]}")
            print("="*100)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**x, output_hidden_states=True)
        else:
            outputs = self.backbone(**x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)

        feat_att = self.backbone._get_feature_vector_attention_mask(all_layers.size(2), x["attention_mask"])
        return all_layers, feat_att

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_features, sr_ids = x
        # Get embeddings
        embeddings, att_mask = self._get_embeddings(input_features)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings, att_mask)  # [B,F] or [B,2F]
        # Get the sr embeddings
        if self.use_sr_embeddings:
            sr_embeddings = self.sr_embedding(sr_ids)
            # Concatenate the embeddings
            logits_input = torch.cat((sr_embeddings, logits_input), dim=-1)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class CalMOSDynamicKANModel(ReLuKANBaseModel):
    """
    Uses a pretrained backbone (e.g. WavLM).
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Whisper is an encoder-encoder model, we only need the encoder part
        if "whisper" in model_name.lower():
            self.backbone = self.backbone.encoder

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**x, output_hidden_states=True)
        else:
            outputs = self.backbone(**x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features


class CalMOSAllLayersEmbeddingModel(BaseModel):
    """
    Speech Emotion Recognition model that uses pre-extracted embeddings as input.
    This model expects pre-computed features instead of processing raw audio through a backbone.

    Input:
        x: Pre-extracted features tensor of shape [batch_size, num_feature_layers, sequence_length, feature_dim]

    The model supports different layer weighting strategies:
    - 'per_layer': Uses features from a specific layer
    - 'weighted_sum': Learns weights to combine features from all layers

    And different pooling strategies:
    - 'mean': Simple mean pooling over the sequence dimension
    - 'attpool': Attentive Statistics Pooling (only available with 'per_layer' strategy)
    """
    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer",
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean",
    ):
        super().__init__(
            mlp_input_dim=mlp_input_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_output_size=mlp_output_size,
            mlp_dropout=mlp_dropout,
            mlp_activation_func=mlp_activation_func,
            layer_weight_strategy=layer_weight_strategy,
            num_feature_layers=num_feature_layers,
            specific_layer_idx=specific_layer_idx,
            pooling_strategy=pooling_strategy,
        )

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simply returns the pre-extracted features.

        Args:
            x: Input tensor of shape [batch_size, num_feature_layers, sequence_length, feature_dim]
                These are pre-extracted features, unlike SERDynamicModel which processes raw input
                through a backbone.

        Returns:
            The same tensor, as features are already extracted
        """
        return x

    def _get_embedding_dim(self) -> int:
        """
        Returns the dimension of the input features, which is determined by the first layer
        of the MLP.

        Returns:
            int: The feature dimension
        """
        return self.mlp.layers[0].in_features


class CalMOSDynamicMelSpecModel(CalMOSDynamicModel):
    """
    Using the same strategy as the CalMOSDynamicModel() but with the difference of using the mel-spec.
    As using the mel-spec we can extract during the training and agreggate with the audio features [ audio _features , mel_spec , mos_score]
    """

    def __init__(
        self,
        # CED params
        ced_embedding_dim: int = 768,
        ced_proj_size: int = 512,
        ced_proj_dropout: float = 0.2,
        ced_pretrained: bool = True,
        ced_freeze: bool = False,
        # Central Kernel Alignment
        use_cka_loss: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        print(ced_embedding_dim)
        print(ced_proj_size)
        print(ced_proj_dropout)
        print(ced_pretrained)
        print(ced_freeze)
        # CED Model
        self.mel_spec_encoder = FineTuneCED(
            pretrained=ced_pretrained,
            embedding_dim=ced_embedding_dim,
            proj_size=ced_proj_size,
            proj_dropout=ced_proj_dropout,
            freeze_backbone_flag=ced_freeze
        )

        self.cka_module = None
        if use_cka_loss:
            self.cka_module = CKALoss()

    def _get_mel_spec_embeddings(self, audios: torch.Tensor) -> torch.Tensor:
        mel_spec_embeddings = self.mel_spec_encoder(audios)
        return mel_spec_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_features, audios = x
        # Get embeddings
        embeddings, att_mask = self._get_embeddings(input_features)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # Apply pooling
        speech_embs = self._apply_pooling(embeddings, att_mask)  # [B,F] or [B,2F]
        # speech_embs = embeddings
        # Get the mel spec embeddings
        mel_spec_embs = self._get_mel_spec_embeddings(audios)
        # Concatenate the embeddings
        logits_input = torch.cat((speech_embs, mel_spec_embs), dim=-1)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)

        if self.cka_module is not None:
            # Compute CKA loss between audio features and mel spec features
            cka_loss = self.cka_module(speech_embs, mel_spec_embs)
            return logits, cka_loss

        return logits


class CalMOSDynamicMelSpecKANModel(CalMOSDynamicKANModel):
    """
    Using the same strategy as the CalMOSDynamicModel() but with the difference of using the mel-spec.
    As using the mel-spec we can extract during the training and agreggate with the audio features [ audio _features , mel_spec , mos_score]
    """

    def __init__(
        self,
        # CED params
        ced_embedding_dim: int = 768,
        ced_proj_size: int = 512,
        ced_proj_dropout: float = 0.2,
        ced_pretrained: bool = True,
        ced_freeze: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        print(ced_embedding_dim)
        print(ced_proj_size)
        print(ced_proj_dropout)
        print(ced_pretrained)
        print(ced_freeze)
        # CED Model
        self.mel_spec_encoder = FineTuneCED(
            pretrained=ced_pretrained,
            embedding_dim=ced_embedding_dim,
            proj_size=ced_proj_size,
            proj_dropout=ced_proj_dropout,
            freeze_backbone_flag=ced_freeze
        )

    def _get_mel_spec_embeddings(self, audios: torch.Tensor) -> torch.Tensor:
        mel_spec_embeddings = self.mel_spec_encoder(audios)
        return mel_spec_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_features, audios = x
        # Get embeddings
        embeddings = self._get_embeddings(input_features)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings)  # [B,F] or [B,2F]
        # Get the mel spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(audios)
        # Concatenate the embeddings
        logits_input = torch.cat((logits_input, mel_spec_embeddings), dim=-1)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)

        if self.cka_module is not None:
            # Compute CKA loss between audio features and mel spec features
            cka_loss = self.cka_module()
            return logits, cka_loss

        return logits


class CalMOSOneLayerEmbeddingModel(nn.Module):
    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        pooling_strategy: str = "mean",
        **kwargs,
    ):
        super().__init__()

        self.pooling_strategy = pooling_strategy
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

    def _get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Expects either:
          - (embeddings, attention_mask)   where embeddings: [B,T,F], mask: [B,T]
          - embeddings only               where embeddings: [B,T,F] (mask assumed all-ones)
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            embeddings, att_mask = batch
        else:
            embeddings = batch
            att_mask = torch.ones(
                embeddings.shape[:2],
                device=embeddings.device,
                dtype=torch.long,
            )
        return embeddings, att_mask

    @staticmethod
    def masked_mean_pool(embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T] with 1 for valid, 0 for padding
        returns:    [B,F]
        """
        # Ensure mask is float for multiplication
        m = att_mask.to(dtype=embeddings.dtype)  # [B,T]
        m = m.unsqueeze(-1)                      # [B,T,1]

        summed = (embeddings * m).sum(dim=1)     # [B,F]
        denom = m.sum(dim=1).clamp(min=1e-6)     # [B,1] avoid div by zero
        return summed / denom

    def _apply_pooling(self, embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T]
        """
        if self.pooling_strategy == "mean":
            return self.masked_mean_pool(embeddings, att_mask)  # [B,F]

        elif self.pooling_strategy == "attpool":
            # If your AttentiveStatisticsPooling supports masks, pass it.
            # If it doesn't, you can pre-mask padded frames by zeroing them (less ideal but works).
            input_dim = embeddings.size(-1)
            if self.attpool is None or getattr(self.attpool, "input_size", None) != input_dim:
                self.attpool = AttentiveStatisticsPooling(input_size=input_dim).to(embeddings.device)

            # Option A (preferred): if your attpool supports mask
            try:
                return self.attpool(embeddings, att_mask)
            except TypeError:
                # Option B: zero-out padded frames before pooling
                m = att_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
                return self.attpool(embeddings * m)

        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, batch):
        embeddings, att_mask = self._get_embeddings(batch)     # embeddings: [B,T,F], mask: [B,T]
        logits_input = self._apply_pooling(embeddings, att_mask)
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class CalMOSWeightedSumLayerEmbeddingModel(nn.Module):
    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        pooling_strategy: str = "mean",
        num_feature_layers: int = 25,
        **kwargs,
    ):
        super().__init__()

        self.layer_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
        )
        self.pooling_strategy = pooling_strategy
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

    def _get_embeddings(self, batch):
        """
        Expects either:
          - (embeddings, attention_mask) where embeddings: [B,L,T,F], mask: [B,T]
          - embeddings only             where embeddings: [B,L,T,F] (mask assumed all-ones)
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            embeddings, att_mask = batch
        else:
            embeddings = batch
            # embeddings: [B,L,T,F] -> mask should be [B,T]
            att_mask = torch.ones(
                (embeddings.size(0), embeddings.size(2)),
                device=embeddings.device,
                dtype=torch.long,
            )
        return embeddings, att_mask

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, T, F]
        B, L, T, Fdim = x.shape

        # [L, 1] -> [L]
        w = torch.stack(list(self.layer_weights), dim=0).squeeze(-1)
        w = F.softmax(w, dim=0).view(1, L, 1, 1)  # [1,L,1,1]

        return (x * w).sum(dim=1)  # [B,T,F]

    @staticmethod
    def masked_mean_pool(embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T] with 1 for valid, 0 for padding
        returns:    [B,F]
        """
        # Ensure mask is float for multiplication
        m = att_mask.to(dtype=embeddings.dtype)  # [B,T]
        m = m.unsqueeze(-1)                      # [B,T,1]

        summed = (embeddings * m).sum(dim=1)     # [B,F]
        denom = m.sum(dim=1).clamp(min=1e-6)     # [B,1] avoid div by zero
        return summed / denom

    def _apply_pooling(self, embeddings: torch.Tensor, att_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B,T,F]
        att_mask:   [B,T]
        """
        if self.pooling_strategy == "mean":
            return self.masked_mean_pool(embeddings, att_mask)  # [B,F]

        elif self.pooling_strategy == "attpool":
            # If your AttentiveStatisticsPooling supports masks, pass it.
            # If it doesn't, you can pre-mask padded frames by zeroing them (less ideal but works).
            input_dim = embeddings.size(-1)
            if self.attpool is None or getattr(self.attpool, "input_size", None) != input_dim:
                self.attpool = AttentiveStatisticsPooling(input_size=input_dim).to(embeddings.device)

            # Option A (preferred): if your attpool supports mask
            try:
                return self.attpool(embeddings, att_mask)
            except TypeError:
                # Option B: zero-out padded frames before pooling
                m = att_mask.to(dtype=embeddings.dtype).unsqueeze(-1)
                return self.attpool(embeddings * m)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, batch):
        embeddings, att_mask = self._get_embeddings(batch)     # embeddings: [B,T,F], mask: [B,T]
        print("1", embeddings.shape, att_mask.shape)
        embeddings = self._weighted_sum(embeddings)  # [B,T,F]
        print("2", embeddings.shape)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings, att_mask)
        logits = self.mlp(logits_input).squeeze(-1)
        return logits