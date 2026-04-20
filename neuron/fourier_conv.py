"""FourierInitConv2d — Conv2d layer with Fourier-initialized kernel.

The kernel weights are generated from Fourier coefficients at initialization,
then trained normally with PyTorch autograd. This gives better convergence
than random init for convolutional layers.
"""

import math
import torch
import torch.nn as nn


class FourierInitConv2d(nn.Conv2d):
    """Conv2d with Fourier-generated initial kernel weights.
    
    The kernel tensor (out_c, in_c, kH, kW) is initialized from a compact
    Fourier formula, capturing multi-scale spatial patterns that random
    init misses.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Size of convolving kernel (int or tuple)
        k: Fourier components per frequency band
        n_bands: Number of frequency bands
        layer_idx: Layer depth index
        num_layers: Total layers in network
        **kwargs: Additional Conv2d args (stride, padding, etc.)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        k: int = 64,
        n_bands: int = 3,
        layer_idx: int = 0,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._fourier_init_conv(k, n_bands, layer_idx, num_layers)
    
    def _fourier_init_conv(
        self,
        k: int,
        n_bands: int,
        layer_idx: int,
        num_layers: int,
    ):
        with torch.no_grad():
            total_k = k * n_bands
            oc, ic, kh, kw = self.weight.shape
            n_weights = oc * ic * kh * kw
            
            # Multi-scale frequency bands
            omega = torch.zeros(total_k, 4)  # (c_out, c_in, h, w) dims
            phi = torch.zeros(total_k)
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                omega[s:e] = torch.randn(k, 4) * (1 + band * 2)
                phi[s:e] = torch.randn(k) * math.pi
            
            # Build position grid for kernel weights
            # Normalized coordinates in [0, 1]
            c_out = (torch.arange(oc, dtype=torch.float32) + 1) / (oc + 1)
            c_in = (torch.arange(ic, dtype=torch.float32) + 1) / (ic + 1)
            h_idx = (torch.arange(kh, dtype=torch.float32) + 1) / (kh + 1)
            w_idx = (torch.arange(kw, dtype=torch.float32) + 1) / (kw + 1)
            
            # 4D meshgrid
            co, ci, hi, wi = torch.meshgrid(c_out, c_in, h_idx, w_idx, indexing='ij')
            l_val = (layer_idx + 1) / (num_layers + 1)
            
            pos = torch.stack([co, ci, hi, wi], dim=-1).reshape(-1, 4)
            
            # Compute Fourier features
            args = pos @ omega.T + phi  # (N, total_k)
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            
            # Amplitude coefficients
            alpha = torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float())
            
            # Generate kernel weights
            w = (features @ alpha).reshape(oc, ic, kh, kw)
            fan_in = ic * kh * kw
            fan_out = oc * kh * kw
            w *= math.sqrt(2.0 / (fan_in + fan_out))  # Kaiming-like scaling
            
            self.weight.copy_(w)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class FourierCNN(nn.Module):
    """Simple CNN with Fourier-initialized conv layers for MNIST.
    
    Architecture:
        Conv(1, 16, 3) → GELU → Conv(16, 32, 3) → GELU → Pool
        Conv(32, 64, 3) → GELU → Pool → FC(576, 10)
    
    Args:
        k: Fourier components per band
        n_bands: Number of frequency bands
    """
    
    def __init__(self, k: int = 64, n_bands: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv1 = FourierInitConv2d(1, 16, 3, k, n_bands, 0, 4, padding=1)
        self.conv2 = FourierInitConv2d(16, 32, 3, k, n_bands, 1, 4, padding=1)
        self.conv3 = FourierInitConv2d(32, 64, 3, k, n_bands, 2, 4, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.fc = nn.Linear(64 * 3 * 3, num_classes)
        
        # Fourier-init the FC layer too
        with torch.no_grad():
            in_dim, out_dim = 576, num_classes
            total_k = k * n_bands
            omega = torch.zeros(total_k, 3)
            phi = torch.zeros(total_k)
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
                phi[s:e] = torch.randn(k) * math.pi
            i_idx = (torch.arange(in_dim, dtype=torch.float32) + 1) / (in_dim + 1)
            j_idx = (torch.arange(out_dim, dtype=torch.float32) + 1) / (out_dim + 1)
            ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
            l_val = 4 / 5
            pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1).reshape(-1, 3)
            args = pos @ omega.T + phi
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            alpha = torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float())
            w = (features @ alpha).reshape(in_dim, out_dim).T * math.sqrt(2.0 / (in_dim + out_dim))
            self.fc.weight.copy_(w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = self.pool(x)  # (B, 32, 3, 3) after pool
        x = torch.nn.functional.gelu(self.conv3(x))
        x = self.pool(x)  # (B, 64, 3, 3)
        x = x.flatten(1)
        x = self.fc(x)
        return x
