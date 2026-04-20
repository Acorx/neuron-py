"""FourierWeightLinear — weights generated on-the-fly from Fourier coefficients.

The key difference from FourierInitLinear:
- FourierInitLinear: generates weights once at init, then trains normally (standard params)
- FourierWeightLinear: generates weights EVERY forward pass from trainable (omega, phi, alpha)

This means: an N×M weight matrix is represented by only 3×K Fourier params.
For a 784→256 layer: 200,704 standard params → ~600 Fourier params = 335x compression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierWeightLinear(nn.Module):
    """Linear layer where weights are generated from Fourier coefficients.
    
    Instead of storing a (out_dim, in_dim) weight matrix, we store:
    - omega: (total_k, 3) frequency vectors — trainable
    - phi: (total_k,) phase offsets — trainable  
    - alpha: (2*total_k,) amplitude coefficients — trainable
    
    Weight at position (i,j) in layer l:
        W(i,j) = Σₖ αₖ·sin(ωₖ·pos(i,j,l) + φₖ) + βₖ·cos(ωₖ·pos(i,j,l) + φₖ)
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        k: Fourier components per frequency band
        n_bands: Number of frequency bands  
        layer_idx: Layer depth index
        num_layers: Total layers in network
        block_size: Generate weights in blocks of this size (0 = no blocking)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 32,
        n_bands: int = 3,
        layer_idx: int = 0,
        num_layers: int = 1,
        block_size: int = 4096,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_k = k * n_bands
        self.block_size = block_size
        
        # Trainable Fourier parameters
        self.omega = nn.Parameter(torch.randn(self.total_k, 3) * 0.5)
        self.phi = nn.Parameter(torch.randn(self.total_k) * math.pi)
        self.alpha = nn.Parameter(
            torch.randn(2 * self.total_k) * 0.01
            / (1 + torch.arange(2 * self.total_k).float())
        )
        
        # Precompute position grid (constant, not trainable)
        l_val = (layer_idx + 1) / (num_layers + 1)
        i_idx = (torch.arange(in_dim, dtype=torch.float32) + 1) / (in_dim + 1)
        j_idx = (torch.arange(out_dim, dtype=torch.float32) + 1) / (out_dim + 1)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
        pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1)
        self.register_buffer('pos', pos.reshape(-1, 3))
        
        # Multi-scale frequency initialization
        with torch.no_grad():
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                self.omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
                self.phi[s:e] = torch.randn(k) * math.pi
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # Scaling factor
        self._scale = math.sqrt(2.0 / (in_dim + out_dim))
    
    def generate_weight(self) -> torch.Tensor:
        """Generate weight matrix from Fourier coefficients."""
        if self.block_size > 0 and self.pos.shape[0] > self.block_size:
            return self._generate_blocked()
        return self._generate_full()
    
    def _generate_full(self) -> torch.Tensor:
        args = self.pos @ self.omega.T + self.phi
        features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        w = (features @ self.alpha).reshape(self.in_dim, self.out_dim).T
        return w * self._scale
    
    def _generate_blocked(self) -> torch.Tensor:
        blocks = []
        for s in range(0, self.pos.shape[0], self.block_size):
            e = min(s + self.block_size, self.pos.shape[0])
            args = self.pos[s:e] @ self.omega.T + self.phi
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            blocks.append(features @ self.alpha)
        w = torch.cat(blocks).reshape(self.in_dim, self.out_dim).T
        return w * self._scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.generate_weight()
        return F.linear(x, w, self.bias)
    
    @property
    def compression_ratio(self) -> float:
        fourier_params = self.total_k * 3 + self.total_k + 2 * self.total_k + self.out_dim
        standard_params = self.in_dim * self.out_dim + self.out_dim
        return standard_params / fourier_params
    
    def extra_repr(self) -> str:
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'k={self.total_k}, compression={self.compression_ratio:.1f}x')


class FourierWeightMLP(nn.Module):
    """MLP with true Fourier-weight layers — weights generated on-the-fly.
    
    Every forward pass regenerates weights from compact Fourier coefficients.
    This is the core innovation: train Fourier params, not weights.
    
    Args:
        dims: Layer dimensions e.g. [784, 256, 128, 10]
        k: Fourier components per band
        n_bands: Number of frequency bands
        block_size: Block size for weight generation (0 = no blocking)
    """
    
    def __init__(self, dims: list[int], k: int = 32, n_bands: int = 3, block_size: int = 4096):
        super().__init__()
        self.dims = dims
        num_layers = len(dims) - 1
        
        self.layers = nn.ModuleList([
            FourierWeightLinear(dims[i], dims[i+1], k, n_bands, i, num_layers, block_size)
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
                x = F.gelu(x)
        return x
    
    @property
    def fourier_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if 'norm' not in n)
    
    @property
    def equivalent_standard_params(self) -> int:
        total = 0
        for i in range(len(self.dims) - 1):
            total += self.dims[i] * self.dims[i+1] + self.dims[i+1]
        return total
    
    @property
    def compression_ratio(self) -> float:
        return self.equivalent_standard_params / self.fourier_params
