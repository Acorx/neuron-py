"""FourierInitConv2d — Conv2d with Fourier-initialized kernels + CIFAR-10 CNN.

On ARM mobile, convolutions are expensive, so we use:
- Small filters (3×3) with few channels
- MaxPool instead of stride for downsampling
- BatchNorm (faster than LayerNorm for convs)
- GroupNorm as alternative if batch size is small
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class FourierInitConv2d(nn.Conv2d):
    """Conv2d with kernel weights generated from Fourier coefficients at init.
    
    After initialization, the kernel is a standard nn.Conv2d parameter
    that can be trained with PyTorch autograd — the Fourier formula is
    only used for the initial values.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of conv kernel (int or tuple)
        k: Fourier components per frequency band
        n_bands: Number of multi-scale frequency bands
        layer_idx: Layer depth index (affects phase offset)
        **kwargs: Additional Conv2d args (stride, padding, etc.)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        k: int = 32,
        n_bands: int = 3,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._fourier_init_weights(k, n_bands, layer_idx)
    
    def _fourier_init_weights(self, k: int, n_bands: int, layer_idx: int):
        """Generate initial kernel weights from Fourier formula."""
        total_k = k * n_bands
        
        # Kernel shape: (out_channels, in_channels//groups, kH, kW)
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        out_c = self.out_channels
        in_c = self.in_channels // self.groups
        
        # Position indices (normalized to [0, 1])
        i_idx = (torch.arange(in_c, dtype=torch.float32) + 1) / (in_c + 1)
        j_idx = (torch.arange(out_c, dtype=torch.float32) + 1) / (out_c + 1)
        h_idx = (torch.arange(kh, dtype=torch.float32) + 1) / (kh + 1)
        w_idx = (torch.arange(kw, dtype=torch.float32) + 1) / (kw + 1)
        
        # 4D meshgrid: (in_c, out_c, kh, kw, 4)
        ii, jj, hh, ww = torch.meshgrid(i_idx, j_idx, h_idx, w_idx, indexing='ij')
        pos = torch.stack([ii, jj, hh, ww], dim=-1).reshape(-1, 4)
        
        # Multi-scale frequency bands
        omega = torch.zeros(total_k, 4)
        phi = torch.zeros(total_k)
        for band in range(n_bands):
            scale = (band + 1) ** 2  # 1, 4, 9
            s = band * k
            e = (band + 1) * k
            omega[s:e] = torch.randn(k, 4) * scale
            phi[s:e] = torch.rand(k) * 2 * math.pi
        
        # Add layer-dependent phase shift
        phi += layer_idx * 0.7
        
        # Generate weights: W = Σ αₖ sin(ωₖ · pos + φₖ)
        args = pos @ omega.T + phi  # (N, total_k)
        sin_feats = torch.sin(args)
        alpha = torch.randn(total_k) * (1.0 / math.sqrt(total_k))
        weights_flat = sin_feats @ alpha
        
        # Reshape to kernel and assign
        weights = weights_flat.reshape(in_c, out_c, kh, kw).permute(1, 0, 2, 3)
        
        # Scale to match Kaiming-like variance
        fan_in = in_c * kh * kw
        std = math.sqrt(2.0 / fan_in)
        weights = weights * (std / max(weights.std().item(), 1e-6))
        
        with torch.no_grad():
            self.weight.copy_(weights)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FourierCNN(nn.Module):
    """CNN for CIFAR-10 with Fourier-initialized convolutions.
    
    Architecture optimized for ARM mobile:
    - 3 conv blocks (3→32→64→64 channels)
    - Global average pooling (no large FC layers)
    - Minimal parameters (~70K)
    
    Args:
        num_classes: Number of output classes
        k: Fourier components per band
        n_bands: Number of frequency bands
    """
    
    def __init__(self, num_classes: int = 10, k: int = 32, n_bands: int = 3):
        super().__init__()
        
        # Block 1: 3 → 32, 32×32 → 16×16
        self.conv1 = FourierInitConv2d(3, 32, 3, k=k, n_bands=n_bands, layer_idx=0, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 32 → 64, 16×16 → 8×8
        self.conv2 = FourierInitConv2d(32, 64, 3, k=k, n_bands=n_bands, layer_idx=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 64 → 64, 8×8 → 4×4
        self.conv3 = FourierInitConv2d(64, 64, 3, k=k, n_bands=n_bands, layer_idx=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
    
    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))  # → (B, 32, 16, 16)
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))  # → (B, 64, 8, 8)
        x = self.pool(F.gelu(self.bn3(self.conv3(x))))  # → (B, 64, 4, 4)
        x = x.flatten(1)  # → (B, 1024)
        return self.fc(x)


class XavierCNN(nn.Module):
    """Same architecture as FourierCNN but with Kaiming/Xavier init.
    
    Used for comparison benchmarks.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.pool(F.gelu(self.bn1(self.conv1(x))))
        x = self.pool(F.gelu(self.bn2(self.conv2(x))))
        x = self.pool(F.gelu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        return self.fc(x)
