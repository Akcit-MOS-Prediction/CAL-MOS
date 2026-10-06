import math
from typing import List, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from models.torch_relu_kan import ReLUKAN
from models.layer_selection import resolve_layer_index

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


class PowerTransformerYeoJohnson(nn.Module):
    """Apply a train-fitted sklearn Yeo-Johnson transform to pooled features."""

    def __init__(self, specification: dict):
        super().__init__()
        if specification.get("params_path"):
            import json
            from pathlib import Path

            params_path = Path(str(specification["params_path"])).expanduser().resolve(strict=True)
            with params_path.open(encoding="utf-8") as stream:
                fitted = json.load(stream)
            specification = {**dict(specification), **fitted}
        if str(specification.get("method", "")).lower() != "yeo-johnson":
            raise ValueError("input_power_transform.method must be 'yeo-johnson'")
        if not specification.get("standardize", False):
            raise ValueError("PowerTransformer Yeo-Johnson must use standardize=true")

        lambdas = torch.as_tensor(list(specification["lambdas"]), dtype=torch.float64)
        mean = torch.as_tensor(list(specification["mean"]), dtype=torch.float64)
        scale = torch.as_tensor(list(specification["scale"]), dtype=torch.float64)
        if lambdas.ndim != 1 or mean.shape != lambdas.shape or scale.shape != lambdas.shape:
            raise ValueError("Yeo-Johnson lambdas, mean and scale must be equal-length vectors")
        if (not torch.isfinite(lambdas).all() or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("Yeo-Johnson parameters must be finite and scales positive")
        self.register_buffer("lambdas", lambdas)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.lambdas.numel():
            raise ValueError(
                f"Yeo-Johnson expected {self.lambdas.numel()} features, got {values.shape[-1]}"
            )
        input_dtype = values.dtype
        x = values.to(dtype=torch.float64)
        positive_mask = x >= 0
        positive_x = x.clamp_min(0)
        negative_x = x.clamp_max(0)

        positive_log = torch.log1p(positive_x)
        positive_lambda = self.lambdas
        positive_denominator = torch.where(
            positive_lambda == 0, torch.ones_like(positive_lambda), positive_lambda
        )
        positive_power = torch.expm1(positive_lambda * positive_log) / positive_denominator
        positive = torch.where(positive_lambda == 0, positive_log, positive_power)

        negative_log = torch.log1p(-negative_x)
        negative_exponent = 2.0 - self.lambdas
        negative_denominator = torch.where(
            negative_exponent == 0, torch.ones_like(negative_exponent), negative_exponent
        )
        negative_power = -torch.expm1(negative_exponent * negative_log) / negative_denominator
        negative = torch.where(negative_exponent == 0, -negative_log, negative_power)

        transformed = torch.where(positive_mask, positive, negative)
        transformed = (transformed - self.mean) / self.scale
        return transformed.to(dtype=input_dtype)


class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        output_size: int = 7,
        dropout: float = 0.1,
        activation_func: str = "relu",
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        layer_norm_affine: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self._debug_layer_norm_logged = False

        self.input_layer_norm = (
            nn.LayerNorm(
                input_size,
                eps=layer_norm_eps,
                elementwise_affine=layer_norm_affine,
            )
            if use_layer_norm else nn.Identity()
        )

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
        normalized = self.input_layer_norm(x)
        if self.debug and not self._debug_layer_norm_logged:
            before = x[0].detach().cpu().reshape(-1).tolist()
            after = normalized[0].detach().cpu().reshape(-1).tolist()
            print(f"[DEBUG][MLP][LayerNorm] before shape={tuple(x.shape)} vector={before}")
            print(f"[DEBUG][MLP][LayerNorm] after  shape={tuple(normalized.shape)} vector={after}")
            self._debug_layer_norm_logged = True
        logits = self.layers(normalized)
        return logits


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    https://arxiv.org/pdf/1803.10963.pdf

    Input:
    x: (batch_size, T, feat_dim)
    Output:
    (batch_size, feat_dim*2)
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs):
        """
        Args:
            xs: Input tensor of shape (batch_size, T, feat_dim)
        Returns:
            Pooled features of shape (batch_size, feat_dim*2)
        """
        # Calculate attention weights
        h = torch.tanh(self.sap_linear(xs))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).unsqueeze(2)

        # Calculate weighted mean
        mu = torch.sum(xs * w, dim=1)
        # Calculate weighted standard deviation
        rh = torch.sqrt((torch.sum((xs**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        # Concatenate mean and standard deviation
        pooled = torch.cat((mu, rh), dim=1)
        return pooled


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
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        layer_norm_affine: bool = False,
        last_layer: bool = False,
        debug: bool = False,
        layer_weight_strategy: str = "per_layer", # "per_layer" or "weighted_sum"
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean", # "mean" or "attpool"
        input_power_transform: dict | None = None,
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
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            layer_norm_affine=layer_norm_affine,
            debug=debug,
        )
        self.input_transform = (
            PowerTransformerYeoJohnson(input_power_transform)
            if input_power_transform else nn.Identity()
        )

        self.last_layer = last_layer
        self.debug = debug
        self._layer_selection_logged = False
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
            print("Using weighted sum for layer weighting")
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

        if pooling_strategy == "attpool":
            if mlp_input_dim % 2:
                raise ValueError("attpool requires an even mlp_input_dim")
            self.attpool = AttentiveStatisticsPooling(input_size=mlp_input_dim // 2)
        else:
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
        """
        Weighted sum over the layers dimension.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        layer_weights = torch.stack([w for w in self.layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(NUM_LAYERS, 1, 1)

        expanded_weights = layer_weights.expand(B, NUM_LAYERS, SEQ_LEN, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, SEQ_LEN, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)

        return weighted_sum

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

    def _select_configured_layer(self, embeddings: torch.Tensor) -> torch.Tensor:
        num_layers = embeddings.shape[1]
        layer_idx = resolve_layer_index(
            num_layers=num_layers,
            specific_layer_idx=self.specific_layer_idx,
            last_layer=self.last_layer,
        )
        if not self._layer_selection_logged:
            print(
                f"[LayerSelection][{self.__class__.__name__}] "
                f"last_layer={self.last_layer}; using layer index "
                f"{layer_idx} of {num_layers} hidden-state layers"
            )
            self._layer_selection_logged = True
        return self._specific_layer(embeddings, layer_idx)

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured layer strategy. When last_layer=True, always
        select the final hidden state and bypass aggregation.
        """
        if self.last_layer:
            return self._select_configured_layer(embeddings)
        if self.layer_weight_strategy == "transformer":
            embeddings = self._transformer_aggregation(embeddings)
            strategy_description = "transformer aggregation over all hidden-state layers"
        elif self.layer_weight_strategy == "per_layer":
            return self._select_configured_layer(embeddings)
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings)
            strategy_description = "weighted sum over all hidden-state layers"
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")
        if not self._layer_selection_logged:
            print(
                f"[LayerSelection][{self.__class__.__name__}] "
                f"last_layer=False; using {strategy_description} "
                f"(num layers={embeddings.shape[1] if embeddings.ndim > 1 else 'n/a'})"
            )
            self._layer_selection_logged = True
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
        logits_input = self.input_transform(logits_input)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


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
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        layer_norm_affine: bool = False,
        last_layer: bool = False,
        debug: bool = False,
        input_power_transform: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        # Construct ReLUKAN width list
        width = [mlp_input_dim] + [mlp_hidden_dim] * mlp_num_layers + [mlp_output_size]
        self.mlp = ReLUKAN(
            width=width,
            grid=relukan_grid,
            k=relukan_k,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            layer_norm_affine=layer_norm_affine,
            debug=debug,
        )
        self.input_transform = (
            PowerTransformerYeoJohnson(input_power_transform)
            if input_power_transform else nn.Identity()
        )
        
        # Existing code for layer weighting strategy, pooling, etc.
        self.last_layer = last_layer
        self.debug = debug
        self._layer_selection_logged = False
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

        if pooling_strategy == "attpool":
            if mlp_input_dim % 2:
                raise ValueError("attpool requires an even mlp_input_dim")
            self.attpool = AttentiveStatisticsPooling(input_size=mlp_input_dim // 2)
        else:
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
        """
        Weighted sum over the layers dimension.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        layer_weights = torch.stack([w for w in self.layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(NUM_LAYERS, 1, 1)

        expanded_weights = layer_weights.expand(B, NUM_LAYERS, SEQ_LEN, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, SEQ_LEN, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)

        return weighted_sum

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

    def _select_configured_layer(self, embeddings: torch.Tensor) -> torch.Tensor:
        num_layers = embeddings.shape[1]
        layer_idx = resolve_layer_index(
            num_layers=num_layers,
            specific_layer_idx=self.specific_layer_idx,
            last_layer=self.last_layer,
        )
        if not self._layer_selection_logged:
            print(
                f"[LayerSelection][{self.__class__.__name__}] "
                f"last_layer={self.last_layer}; using layer index "
                f"{layer_idx} of {num_layers} hidden-state layers"
            )
            self._layer_selection_logged = True
        return self._specific_layer(embeddings, layer_idx)

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the configured layer strategy. When last_layer=True, always
        select the final hidden state and bypass aggregation.
        """
        if self.last_layer:
            return self._select_configured_layer(embeddings)
        if self.layer_weight_strategy == "transformer":
            embeddings = self._transformer_aggregation(embeddings)
            strategy_description = "transformer aggregation over all hidden-state layers"
        elif self.layer_weight_strategy == "per_layer":
            return self._select_configured_layer(embeddings)
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings)
            strategy_description = "weighted sum over all hidden-state layers"
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")
        if not self._layer_selection_logged:
            print(
                f"[LayerSelection][{self.__class__.__name__}] "
                f"last_layer=False; using {strategy_description} "
                f"(num layers={embeddings.shape[1] if embeddings.ndim > 1 else 'n/a'})"
            )
            self._layer_selection_logged = True
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
        logits_input = self.input_transform(logits_input)
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
        **kwargs
    ):
        super().__init__(**kwargs)

        if freeze_backbone and use_peft:
            raise ValueError("Parameters 'freeze_backbone' and 'use_peft' cannot be 'True' at the same time.")

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Whisper is an encoder-decoder model, we only need the encoder part
        if "whisper" in model_name.lower():
            self.backbone = self.backbone.encoder

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
        else:
            print("\n\t Not using PEFT...\n")
        if freeze_backbone:
            print("\n\t Freezing backbone...\n")
            self._freeze_backbone()
            self.backbone.eval()
        else:
            print("\n\t Fine-tuning backbone...\n")

        self.use_sr_embeddings = use_sr_embeddings
        if self.use_sr_embeddings:
            self.sr_embedding = nn.Embedding(num_sr_classes, sr_embedding_dim)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_features, sr_ids = x
        # Get embeddings
        embeddings = self._get_embeddings(input_features)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings)  # [B,F] or [B,2F]
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
        return self.mlp.width[0]

    def forward(self, x: tuple) -> torch.Tensor:
        # DynamicCollate returns (feature-extractor output, sample-rate IDs).
        # The KAN head does not use the IDs, as in the MLP path when
        # use_sr_embeddings=False.
        input_features, _source_srs = x
        embeddings = self._get_embeddings(input_features)
        embeddings = self._apply_layer_weighting(embeddings)
        logits_input = self._apply_pooling(embeddings)
        return self.mlp(logits_input).squeeze(-1)


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
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        layer_norm_affine: bool = False,
        last_layer: bool = False,
        debug: bool = False,
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
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            layer_norm_affine=layer_norm_affine,
            last_layer=last_layer,
            debug=debug,
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
        use_layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        layer_norm_affine: bool = False,
        last_layer: bool = False,
        debug: bool = False,
    ):
        super().__init__()

        self.pooling_strategy = pooling_strategy
        if pooling_strategy == "attpool":
            if mlp_input_dim % 2:
                raise ValueError("attpool requires an even mlp_input_dim")
            self.attpool = AttentiveStatisticsPooling(input_size=mlp_input_dim // 2)
        else:
            self.attpool = None
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
            use_layer_norm=use_layer_norm,
            layer_norm_eps=layer_norm_eps,
            layer_norm_affine=layer_norm_affine,
            debug=debug,
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
            return self.attpool(embeddings)

        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embeddings = self._get_embeddings(x)
        # Apply pooling
        logits_input = self._apply_pooling(embeddings)  # [B,F] or [B,2F]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits
