import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dataset_blocks import build_block_expansion
from utils.dataset_identity import block_expansion_enabled, kan_block_padding


class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
    def forward(self, x):
        # x shape: [batch_size, input_size]
        batch_size = x.size(0)
        
        # Expand input to [batch_size, input_size, 1]
        x_expanded = x.unsqueeze(-1)
        
        # Expand phase parameters to [1, input_size, g + k]
        phase_low = self.phase_low.unsqueeze(0)  # [1, input_size, g + k]
        phase_height = self.phase_height.unsqueeze(0)  # [1, input_size, g + k]
        
        # Compute activations with broadcasting
        x1 = torch.relu(x_expanded - phase_low)  # [batch_size, input_size, g + k]
        x2 = torch.relu(phase_height - x_expanded)  # [batch_size, input_size, g + k]
        
        # Combine and scale
        x = x1 * x2 * self.r
        x = x * x  # Square the result
        
        # Prepare for convolution: [batch_size, 1, g + k, input_size]
        x = x.permute(0, 2, 1).unsqueeze(1)  # Channel dimension added
        
        # Apply convolution
        x = self.equal_size_conv(x)  # Output: [batch_size, output_size, 1, 1]
        
        # Final output shape: [batch_size, output_size]
        return x.squeeze(-1).squeeze(-1)


class ReLUKAN(nn.Module):
    """ReLU-KAN head whose inputs always lie inside the basis grid.

    Every ReLUKANLayer only responds on [-k/g, 1 + k/g]; outside it all bases, their
    contributions and their gradients are zero. The non-affine LayerNorm yields ~N(0, 1)
    values, so the head squashes them with a sigmoid into (0, 1) before the first layer,
    and repeats LayerNorm + sigmoid before every following layer. Dataset-block padding is
    placed strictly below the grid so padded positions stay inert. The MLP head
    (MLPBase) is unaffected: it keeps LayerNorm only.
    """

    INPUT_TRANSFORM = {"version": 2, "input": "layernorm+sigmoid", "between_layers": "layernorm+sigmoid"}

    def __init__(self, width, grid, k, use_layer_norm=True, layer_norm_eps=1e-5, layer_norm_affine=False, debug=False, dataset_block_expansion=None):
        super().__init__()
        self.debug = debug
        self._debug_layer_norm_logged = False
        self.input_size = width[0]
        self.layer_norm_eps = layer_norm_eps
        self.grid = grid
        self.k = k
        self.grid_min = -k / grid
        self.grid_max = 1 + k / grid
        if any(int(size) < 2 for size in list(width)[1:-1]):
            raise ValueError("ReLU-KAN hidden widths must be >= 2: LayerNorm over one unit is constant")
        self.block_expansion = build_block_expansion(
            self.input_size,
            self._padding_outside_grid(dataset_block_expansion),
            use_layer_norm,
            layer_norm_affine,
        )
        width = list(width)
        if self.block_expansion is not None:
            width[0] = self.block_expansion.output_size
        self.width = width
        self.input_layer_norm = (
            nn.LayerNorm(
                self.input_size,
                eps=layer_norm_eps,
                elementwise_affine=layer_norm_affine,
            )
            if use_layer_norm else nn.Identity()
        )
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def _padding_outside_grid(self, specification):
        """Default the block padding below the grid; reject any padding not strictly below it."""
        if not block_expansion_enabled(specification):
            return specification
        specification = dict(specification)
        if specification.get("padding_value") is None:
            specification["padding_value"] = kan_block_padding(self.grid, self.k)
        padding = float(specification["padding_value"])
        if not padding < self.grid_min:
            raise ValueError(
                f"KAN dataset-block padding_value={padding} is not below the ReLU-KAN grid "
                f"[{self.grid_min:g}, {self.grid_max:g}]; padded positions must be inert. "
                f"Use a value below {self.grid_min:g} (default {kan_block_padding(self.grid, self.k):g})"
            )
        return specification

    def _squash(self, values):
        """Map ~N(0, 1) features into (0, 1), where the ReLU-KAN bases live."""
        return torch.sigmoid(values)

    def get_extra_state(self):
        return dict(self.INPUT_TRANSFORM)

    def set_extra_state(self, state):
        if state != self.INPUT_TRANSFORM:
            raise ValueError(
                f"Checkpoint ReLU-KAN input transform {state} differs from {self.INPUT_TRANSFORM}"
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Raised here, before the child modules load, so a pre-sigmoid checkpoint (with or
        # without dataset blocks) is reported as such, also under strict=False: its weights
        # were trained on different inputs and would load with different predictions.
        if prefix + "_extra_state" not in state_dict:
            raise RuntimeError(
                f"{prefix}_extra_state is missing: this ReLU-KAN checkpoint predates the sigmoid "
                "after LayerNorm, so its weights were trained on different inputs"
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, dataset_ids=None):
        normalized = self.input_layer_norm(x)
        squashed = self._squash(normalized)
        if self.debug and not self._debug_layer_norm_logged:
            before = x[0].detach().cpu().reshape(-1).tolist()
            after = normalized[0].detach().cpu().reshape(-1).tolist()
            inside = squashed[0].detach().cpu().reshape(-1).tolist()
            print(f"[DEBUG][KAN][LayerNorm] before shape={tuple(x.shape)} vector={before}")
            print(f"[DEBUG][KAN][LayerNorm] after  shape={tuple(normalized.shape)} vector={after}")
            print(f"[DEBUG][KAN][Sigmoid]   after  shape={tuple(squashed.shape)} vector={inside}")
            self._debug_layer_norm_logged = True
        x = self.block_expansion(squashed, dataset_ids) if self.block_expansion else squashed
        for index, rk_layer in enumerate(self.rk_layers):
            if index > 0:
                x = self._squash(F.layer_norm(x, (x.shape[-1],), eps=self.layer_norm_eps))
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x


def show_base(phase_num, step):
    rk = ReLUKANLayer(1, phase_num, step, 1)
    x = torch.Tensor([np.arange(-600, 1024+600) / 1024]).T
    x1 = torch.relu(x - rk.phase_low)
    x2 = torch.relu(rk.phase_height - x)
    y = x1 * x1 * x2 * x2 * rk.r * rk.r
    for i in range(phase_num+step):
        plt.plot(x, y[:, i:i+1].detach(), color='black')
    plt.show()
    print('1')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    is_cuda = torch.cuda.is_available()
    show_base(5, 3)
    rk = ReLUKANLayer(1, 100, 5, 2)
    x = torch.Tensor([np.arange(0, 1024) / 1024]).T
    if is_cuda:
        rk.cuda()
        x = x.cuda()
    y = rk(x).detach().cpu()
    plt.show()

