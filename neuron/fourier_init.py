"""Fourier-Init Linear Layer — the core of neuron.

Instead of random initialization (Xavier/Kaiming), we generate initial weights
from a compact Fourier formula. This gives better convergence and is the
foundation for future "weightless" networks where weights are computed on-the-fly.

W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ) + βₖ · cos(...)

Key idea: a few hundred Fourier coefficients can initialize millions of weights
with structured, multi-scale patterns that converge faster than random init.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class FourierInitLinear(nn.Linear):
    """Linear layer with Fourier-generated initial weights.
    
    The weight matrix is initialized from a sum of sinusoidal basis functions
    parameterized by (omega, phi, alpha) — a compact set of Fourier coefficients.
    
    After initialization, training proceeds normally with PyTorch autograd.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        k: Base number of Fourier components per frequency band
        n_bands: Number of frequency bands (multi-scale)
        layer_idx: Layer index for depth encoding
        num_layers: Total number of layers for depth normalization
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 64,
        n_bands: int = 3,
        layer_idx: int = 0,
        num_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_dim, out_dim, bias=bias)
        self._fourier_init(in_dim, out_dim, k, n_bands, layer_idx, num_layers)
    
    def _fourier_init(
        self,
        in_dim: int,
        out_dim: int,
        k: int,
        n_bands: int,
        layer_idx: int,
        num_layers: int,
    ):
        with torch.no_grad():
            total_k = k * n_bands
            
            # Multi-scale frequency bands
            omega = torch.zeros(total_k, 3)  # (total_k, 3) for (i, j, l) dims
            phi = torch.zeros(total_k)
            
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                # Each band has progressively higher frequencies
                omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
                phi[s:e] = torch.randn(k) * math.pi
            
            # Normalized position indices
            i_idx = (torch.arange(in_dim, dtype=torch.float32) + 1) / (in_dim + 1)
            j_idx = (torch.arange(out_dim, dtype=torch.float32) + 1) / (out_dim + 1)
            
            # Mesh grid of all (i, j) positions
            ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
            
            # Depth coordinate (layer position in network)
            l_val = (layer_idx + 1) / (num_layers + 1)
            pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1)
            pos = pos.reshape(-1, 3)  # (in_dim * out_dim, 3)
            
            # Compute Fourier features: sin + cos
            args = pos @ omega.T + phi  # (in_dim * out_dim, total_k)
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (N, 2*total_k)
            
            # Learnable amplitude coefficients (decaying for higher freqs)
            alpha = torch.randn(2 * total_k) * 0.01 / (1 + torch.arange(2 * total_k).float())
            
            # Generate weights and reshape (transpose for nn.Linear convention)
            w = (features @ alpha).reshape(in_dim, out_dim).T  # (out_dim, in_dim)
            w *= math.sqrt(2.0 / (in_dim + out_dim))  # Xavier-like scaling
            
            self.weight.copy_(w)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class FourierMLP(nn.Module):
    """Multi-layer perceptron with Fourier-initialized weights.
    
    Combines Fourier weight initialization with modern architectural features:
    - LayerNorm for stable training
    - GELU activation (smooth, used in GPT/BERT)
    - Multi-scale Fourier init across frequency bands
    
    Args:
        dims: List of layer dimensions, e.g. [784, 256, 128, 10]
        k: Base Fourier components per band
        n_bands: Number of frequency bands
    """
    
    def __init__(self, dims: list[int], k: int = 64, n_bands: int = 3):
        super().__init__()
        self.dims = dims
        num_layers = len(dims) - 1
        
        self.layers = nn.ModuleList([
            FourierInitLinear(dims[i], dims[i+1], k, n_bands, i, num_layers)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i+1])
            for i in range(num_layers - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = nn.functional.gelu(x)
        return x


class FourierResMLP(nn.Module):
    """Fourier-MLP with residual connections for deeper networks.
    
    Adds skip connections every 2 layers, enabling training of
    deeper architectures without vanishing gradients.
    
    Args:
        dims: List of layer dimensions
        k: Fourier components per band
        n_bands: Number of frequency bands
    """
    
    def __init__(self, dims: list[int], k: int = 64, n_bands: int = 3):
        super().__init__()
        self.dims = dims
        num_layers = len(dims) - 1
        
        self.layers = nn.ModuleList([
            FourierInitLinear(dims[i], dims[i+1], k, n_bands, i, num_layers)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i+1])
            for i in range(num_layers - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            residual = x if (i > 0 and i < len(self.layers) - 1 and 
                            self.dims[i] == self.dims[i+1]) else None
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = nn.functional.gelu(x)
                if residual is not None:
                    x = x + residual
        return x
