import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, width, grid, k, use_layer_norm=True, layer_norm_eps=1e-5, layer_norm_affine=False, debug=False):
        super().__init__()
        self.debug = debug
        self._debug_layer_norm_logged = False
        self.width = width
        self.grid = grid
        self.k = k
        self.input_layer_norm = (
            nn.LayerNorm(
                width[0],
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

    def forward(self, x):
        normalized = self.input_layer_norm(x)
        if self.debug and not self._debug_layer_norm_logged:
            before = x[0].detach().cpu().reshape(-1).tolist()
            after = normalized[0].detach().cpu().reshape(-1).tolist()
            print(f"[DEBUG][KAN][LayerNorm] before shape={tuple(x.shape)} vector={before}")
            print(f"[DEBUG][KAN][LayerNorm] after  shape={tuple(normalized.shape)} vector={after}")
            self._debug_layer_norm_logged = True
        x = normalized
        for rk_layer in self.rk_layers:
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

