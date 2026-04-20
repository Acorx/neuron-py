"""FourierNet V3 — PyTorch backend with Fourier weight generation.

Same radical concept: generate weights from a tiny Fourier formula instead of
storing billions of parameters. But now powered by PyTorch autograd for fast
backprop + multi-core CPU + GPU when available.

W(i,j,l) = Σₖ αₖ · sin(ωₖ·pos(i,j,l) + φₖ)
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class FourierLinear(nn.Module):
    """Linear layer whose weights are generated from Fourier coefficients.

    Instead of storing in_dim × out_dim weights, we store only:
    - alpha: (2*K,) Fourier amplitudes (sin + cos)
    - omega: (K, 3) learnable frequencies
    - phi: (K,) learnable phases

    Total stored: 2K + 3K + K = 6K params
    Instead of: in_dim × out_dim + out_dim params
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 64,
                 n_bands: int = 3, scale: float = 0.1,
                 learn_freq: bool = True, use_cos: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.total_k = k * n_bands
        self.n_bands = n_bands
        self.scale = scale
        self.learn_freq = learn_freq
        self.use_cos = use_cos

        # Multi-scale frequency bands
        omega = torch.zeros(self.total_k, 3)
        phi = torch.zeros(self.total_k)
        for band in range(n_bands):
            freq_scale = 1.0 + band * 3.0
            s, e = band * k, (band + 1) * k
            omega[s:e] = torch.randn(k, 3) * freq_scale
            phi[s:e] = torch.randn(k) * math.pi

        if learn_freq:
            self.omega = nn.Parameter(omega.float())
            self.phi = nn.Parameter(phi.float())
        else:
            self.register_buffer('omega', omega.float())
            self.register_buffer('phi', phi.float())

        # Fourier amplitudes — Xavier-scaled
        n_coeff = 2 * self.total_k if use_cos else self.total_k
        self.alpha = nn.Parameter(
            torch.randn(n_coeff).float() * 0.01 / (1 + torch.arange(n_coeff).float())
        )
        # Bias amplitudes
        self.alpha_bias = nn.Parameter(
            torch.randn(n_coeff).float() * 0.001
        )

        # Precompute position encodings (fixed)
        self.register_buffer('pos_weight', self._make_weight_positions())
        self.register_buffer('pos_bias', self._make_bias_positions())

        # Xavier scale
        fan_avg = (in_dim + out_dim) / 2
        self.xavier_scale = math.sqrt(2.0 / (in_dim + out_dim))

    def _make_weight_positions(self):
        i_idx = (torch.arange(self.in_dim, dtype=torch.float32) + 1) / (self.in_dim + 1)
        j_idx = (torch.arange(self.out_dim, dtype=torch.float32) + 1) / (self.out_dim + 1)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
        # Add layer info as 0 (will be overridden per-layer)
        ll = torch.zeros_like(ii)
        return torch.stack([ii, jj, ll], dim=-1).reshape(-1, 3)

    def _make_bias_positions(self):
        j_idx = (torch.arange(self.out_dim, dtype=torch.float32) + 1) / (self.out_dim + 1)
        i_idx = torch.zeros_like(j_idx)
        ll = torch.zeros_like(j_idx)
        return torch.stack([i_idx, j_idx, ll], dim=-1)

    def _layer_pos(self, layer_idx: int, num_layers: int):
        """Set the layer dimension in positions."""
        l_val = (layer_idx + 1) / (num_layers + 1)
        pw = self.pos_weight.clone()
        pw[:, 2] = l_val
        pb = self.pos_bias.clone()
        pb[:, 2] = l_val
        return pw, pb

    def generate_weights(self, layer_idx: int = 0, num_layers: int = 1):
        """Generate weight matrix and bias from Fourier coefficients."""
        pw, pb = self._layer_pos(layer_idx, num_layers)

        # Weight: sin/cos features
        args = pw @ self.omega.T + self.phi  # (in*out, total_k)
        sin_vals = torch.sin(args)
        cos_vals = torch.cos(args)

        if self.use_cos:
            features = torch.cat([sin_vals, cos_vals], dim=1)  # (in*out, 2*total_k)
        else:
            features = sin_vals

        weights = (features @ self.alpha * self.scale).reshape(self.in_dim, self.out_dim)
        weights = weights * self.xavier_scale / self.scale  # Xavier scaling

        # Bias
        bargs = pb @ self.omega.T + self.phi
        bsin = torch.sin(bargs)
        bcos = torch.cos(bargs)
        bfeatures = torch.cat([bsin, bcos], dim=1) if self.use_cos else bsin
        bias = bfeatures @ self.alpha_bias * self.scale * 0.2

        return weights, bias

    def forward(self, x, layer_idx=0, num_layers=1):
        W, b = self.generate_weights(layer_idx, num_layers)
        return x @ W + b

    def stored_params(self):
        n = self.alpha.numel() + self.alpha_bias.numel()
        if self.learn_freq:
            n += self.omega.numel() + self.phi.numel()
        return n

    def virtual_params(self):
        return self.in_dim * self.out_dim + self.out_dim


class FourierNetV3(nn.Module):
    """Full Fourier network with PyTorch backend.

    Each layer uses FourierLinear for weight generation.
    Includes LayerNorm, GELU, and residual connections.
    """

    def __init__(self, layer_dims: List[int], k: int = 64,
                 n_bands: int = 3, scale: float = 0.1,
                 learn_freq: bool = True, use_cos: bool = True,
                 residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.layer_dims = layer_dims
        self.k = k
        self.n_bands = n_bands
        self.residual = residual

        self.num_layers = len(layer_dims) - 1

        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.fourier_layers.append(
                FourierLinear(
                    in_dim=layer_dims[i],
                    out_dim=layer_dims[i + 1],
                    k=k, n_bands=n_bands, scale=scale,
                    learn_freq=learn_freq, use_cos=use_cos,
                )
            )

        # Layer norm for hidden layers
        self.layer_norms = nn.ModuleList()
        for d in layer_dims[1:-1]:
            self.layer_norms.append(nn.LayerNorm(d))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.act = nn.GELU()

    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.fourier_layers[i](x, layer_idx=i, num_layers=self.num_layers)

            if i < self.num_layers - 1:
                # Hidden layer: LayerNorm + GELU + residual
                x = self.layer_norms[i](x)
                x = self.act(x)
                if self.dropout is not None:
                    x = self.dropout(x)
                # Residual connection if dims match
                if self.residual and residual.shape[-1] == x.shape[-1]:
                    x = x + residual

        return x

    def param_count(self):
        """Total stored parameters (Fourier coefficients)."""
        return sum(p.numel() for p in self.parameters())

    def virtual_param_count(self):
        """Equivalent params if stored as regular weights."""
        count = 0
        for i in range(self.num_layers):
            count += self.layer_dims[i] * self.layer_dims[i + 1] + self.layer_dims[i + 1]
        # Layer norm params
        for d in self.layer_dims[1:-1]:
            count += d * 2
        return count

    def compression_ratio(self):
        v = self.virtual_param_count()
        s = self.param_count()
        return v / max(s, 1)
