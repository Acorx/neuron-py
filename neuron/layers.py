"""Layer abstractions for neuron."""

import numpy as np


class FourierLinear:
    """A single Fourier-generated linear layer."""
    def __init__(self, in_dim, out_dim, k=64, scale=0.1, use_cos=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.scale = scale
        self.use_cos = use_cos

        rng = np.random.RandomState(42)
        self.omega = rng.randn(k, 3).astype(np.float32) * 5.0
        self.phi = rng.randn(k).astype(np.float32) * np.pi
        n_coeff = 2 * k if use_cos else k
        self.alpha = rng.randn(n_coeff).astype(np.float32) * 0.01
        self.alpha_bias = rng.randn(n_coeff).astype(np.float32) * 0.01


class ResidualBlock:
    """Residual connection wrapper — adds skip connection when dims match."""
    def __init__(self, layer, activation='gelu'):
        self.layer = layer
        self.activation = activation

    def forward(self, x, W, b, act_fn, act_deriv):
        out = act_fn(x @ W + b)
        return out + x  # residual
