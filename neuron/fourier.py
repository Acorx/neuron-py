"""Fourier weight generation — the core of neuron.

Instead of storing billions of weights, we compute them from a tiny formula.
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)

With backpropagation through the Fourier generation.
"""

import numpy as np
from typing import List, Optional, Tuple


class FourierNet:
    """Neural network with weights generated from Fourier coefficients.

    Only the α coefficients are stored and learned.
    Weights are generated on-the-fly from the Fourier formula.

    Parameters
    ----------
    layer_dims : list of int
        Network architecture, e.g. [784, 256, 128, 10]
    k : int
        Number of Fourier components per layer
    scale : float
        Output scale for generated weights
    learn_freq : bool
        If True, Omega and Phi are also trainable
    use_cos : bool
        If True, use sin+cos pairs (doubles expressivity)
    """

    def __init__(
        self,
        layer_dims: List[int],
        k: int = 64,
        scale: float = 0.1,
        learn_freq: bool = False,
        use_cos: bool = True,
    ):
        self.layer_dims = layer_dims
        self.k = k
        self.scale = scale
        self.learn_freq = learn_freq
        self.use_cos = use_cos
        self.num_layers = len(layer_dims) - 1

        # Fixed random frequencies (Ω) and phases (φ)
        rng = np.random.RandomState(42)
        self.omega = rng.randn(self.num_layers, k, 3).astype(np.float32) * 5.0
        self.phi = rng.randn(self.num_layers, k).astype(np.float32) * np.pi

        # Learnable coefficients (α)
        # With cos: 2k per layer (sin + cos), without: k per layer
        n_coeff = 2 * k if use_cos else k
        self.alpha = [
            rng.randn(n_coeff).astype(np.float32) * 0.01
            for _ in range(self.num_layers)
        ]

        # Bias coefficients (separate alpha for bias generation)
        self.alpha_bias = [
            rng.randn(n_coeff).astype(np.float32) * 0.01
            for _ in range(self.num_layers)
        ]

        # Cache for inference
        self._weight_cache = None
        self._bias_cache = None

    def _positions(self, layer: int, in_dim: int, out_dim: int) -> np.ndarray:
        """Compute normalized positions for weight generation.

        Returns array of shape (in_dim * out_dim, 3) with (row, col, layer).
        """
        i_idx = np.arange(in_dim, dtype=np.float32) + 1
        j_idx = np.arange(out_dim, dtype=np.float32) + 1
        ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
        ii = ii.flatten() / (in_dim + 1)
        jj = jj.flatten() / (out_dim + 1)
        ll = np.full_like(ii, (layer + 1) / (self.num_layers + 1))
        return np.stack([ii, jj, ll], axis=-1)  # (N, 3)

    def _bias_positions(self, layer: int, dim: int) -> np.ndarray:
        """Compute positions for bias generation. Shape: (dim, 3)."""
        j_idx = (np.arange(dim, dtype=np.float32) + 1) / (dim + 1)
        ll = np.full_like(j_idx, (layer + 1) / (self.num_layers + 1))
        i_idx = np.zeros_like(j_idx)
        return np.stack([i_idx, j_idx, ll], axis=-1)

    def generate_weights(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate weights and bias for a layer using Fourier formula.

        Returns:
            weights: shape (in_dim, out_dim)
            bias: shape (out_dim,)
        """
        in_dim, out_dim = self.layer_dims[layer], self.layer_dims[layer + 1]
        pos = self._positions(layer, in_dim, out_dim)  # (N, 3)

        # arg[l, k] = pos[l, 0]*omega[k,0] + pos[l,1]*omega[k,1] + pos[l,2]*omega[k,2] + phi[k]
        # shape: (N, k)
        omega = self.omega[layer]  # (k, 3)
        phi = self.phi[layer]      # (k,)
        alpha = self.alpha[layer]  # (2k,) or (k,)

        args = pos @ omega.T + phi  # (N, k)
        sin_vals = np.sin(args)     # (N, k)

        if self.use_cos:
            cos_vals = np.cos(args)  # (N, k)
            features = np.concatenate([sin_vals, cos_vals], axis=1)  # (N, 2k)
        else:
            features = sin_vals

        weights = (features @ alpha * self.scale).reshape(in_dim, out_dim)

        # Bias
        bpos = self._bias_positions(layer, out_dim)  # (out_dim, 3)
        bargs = bpos @ omega.T + phi
        bsin = np.sin(bargs)
        if self.use_cos:
            bcos = np.cos(bargs)
            bfeatures = np.concatenate([bsin, bcos], axis=1)
        else:
            bfeatures = bsin
        bias = bfeatures @ self.alpha_bias[layer] * self.scale * 0.2

        return weights, bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Parameters
        ----------
        x : ndarray, shape (batch, input_dim) or (input_dim,)

        Returns
        -------
        ndarray, shape (batch, output_dim) or (output_dim,)
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        self._cache = []  # Store for backprop

        for layer in range(self.num_layers):
            W, b = self.generate_weights(layer)
            pre = x @ W + b  # (batch, out_dim)

            # Activation (GELU for hidden, none for output)
            if layer < self.num_layers - 1:
                act = _gelu(pre)
            else:
                act = pre

            self._cache.append((x, W, b, pre, act))
            x = act

        if squeeze:
            return x[0]
        return x

    def backward(self, grad_output: np.ndarray) -> dict:
        """Backpropagation through Fourier weight generation.

        Returns dict with gradients for all learnable parameters.
        """
        grads = {
            'alpha': [np.zeros_like(a) for a in self.alpha],
            'alpha_bias': [np.zeros_like(a) for a in self.alpha_bias],
        }

        grad = grad_output

        for layer in range(self.num_layers - 1, -1, -1):
            x_in, W, b, pre, act = self._cache[layer]
            in_dim, out_dim = self.layer_dims[layer], self.layer_dims[layer + 1]

            # Activation gradient
            if layer < self.num_layers - 1:
                grad_pre = grad * _gelu_deriv(pre)
            else:
                grad_pre = grad

            # Gradient w.r.t. alpha (Fourier backprop!)
            # dL/d(alpha_k) = sum over (i,j) of dL/dW[i,j] * dW/d(alpha_k)
            # dW/d(alpha_sin_k) = sin(arg) * scale
            # dW/d(alpha_cos_k) = cos(arg) * scale
            # dL/dW = x_in.T @ grad_pre  (shape: in_dim, out_dim)
            dW = x_in.T @ grad_pre  # (in_dim, out_dim)
            db = grad_pre.sum(axis=0)  # (out_dim,)

            # Compute Fourier features for this layer
            pos = self._positions(layer, in_dim, out_dim)
            omega = self.omega[layer]
            phi = self.phi[layer]
            args = pos @ omega.T + phi  # (N, k)
            sin_vals = np.sin(args)
            cos_vals = np.cos(args)

            if self.use_cos:
                features = np.concatenate([sin_vals, cos_vals], axis=1)  # (N, 2k)
            else:
                features = sin_vals

            # dL/d(alpha) = features.T @ dW.flatten() * scale
            grads['alpha'][layer] = features.T @ dW.flatten() * self.scale

            # Bias gradients
            bpos = self._bias_positions(layer, out_dim)
            bargs = bpos @ omega.T + phi
            bsin = np.sin(bargs)
            bcos = np.cos(bargs)
            if self.use_cos:
                bfeatures = np.concatenate([bsin, bcos], axis=1)
            else:
                bfeatures = bsin
            grads['alpha_bias'][layer] = bfeatures.T @ db * self.scale * 0.2

            # Propagate gradient to previous layer
            grad = grad_pre @ W.T  # (batch, in_dim)

            # Learnable frequencies
            if self.learn_freq:
                # dW/d(omega_k_d) = alpha_k * cos(arg) * pos_d
                # This is more complex but very powerful
                if 'omega' not in grads:
                    grads['omega'] = [np.zeros_like(self.omega[l]) for l in range(self.num_layers)]
                    grads['phi'] = [np.zeros_like(self.phi[l]) for l in range(self.num_layers)]

                dW_flat = dW.flatten() * self.scale
                alpha = self.alpha[layer]
                for ki in range(self.k):
                    if self.use_cos:
                        a_sin = alpha[ki]
                        a_cos = alpha[self.k + ki]
                    else:
                        a_sin = alpha[ki]
                        a_cos = 0.0

                    cos_arg = cos_vals[:, ki]  # (N,)
                    for d in range(3):
                        # d/d(omega[k,d]) of (a_sin*sin + a_cos*cos) = (a_sin*cos - a_cos*sin) * pos[d]
                        dW_domega = (a_sin * cos_arg - a_cos * sin_vals[:, ki]) * pos[:, d]
                        grads['omega'][layer][ki, d] = np.dot(dW_domega, dW_flat)

                    # d/d(phi_k)
                    dW_dphi = a_sin * cos_arg - a_cos * sin_vals[:, ki]
                    grads['phi'][layer][ki] = np.dot(dW_dphi, dW_flat)

        return grads

    def param_count(self) -> int:
        """Number of learnable parameters stored."""
        n = sum(a.size for a in self.alpha) + sum(a.size for a in self.alpha_bias)
        if self.learn_freq:
            n += self.omega.size + self.phi.size
        return n

    def virtual_param_count(self) -> int:
        """Number of virtual weights generated."""
        count = 0
        for i in range(self.num_layers):
            in_d, out_d = self.layer_dims[i], self.layer_dims[i + 1]
            count += in_d * out_d + out_d
        return count

    def compression_ratio(self) -> float:
        """Virtual params / stored params."""
        return self.virtual_param_count() / max(self.param_count(), 1)

    def cache_weights(self):
        """Cache generated weights for fast inference."""
        self._weight_cache = []
        self._bias_cache = []
        for layer in range(self.num_layers):
            W, b = self.generate_weights(layer)
            self._weight_cache.append(W)
            self._bias_cache.append(b)

    def forward_cached(self, x: np.ndarray) -> np.ndarray:
        """Fast inference using cached weights."""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        for layer in range(self.num_layers):
            x = x @ self._weight_cache[layer] + self._bias_cache[layer]
            if layer < self.num_layers - 1:
                x = _gelu(x)

        if squeeze:
            return x[0]
        return x

    def save(self, path: str):
        """Save model to .npz file."""
        data = {
            'layer_dims': np.array(self.layer_dims),
            'k': self.k,
            'scale': self.scale,
            'learn_freq': self.learn_freq,
            'use_cos': self.use_cos,
            'omega': self.omega,
            'phi': self.phi,
        }
        for i, (a, ab) in enumerate(zip(self.alpha, self.alpha_bias)):
            data[f'alpha_{i}'] = a
            data[f'alpha_bias_{i}'] = ab
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str) -> 'FourierNet':
        """Load model from .npz file."""
        data = np.load(path, allow_pickle=True)
        layer_dims = data['layer_dims'].tolist()
        net = cls(
            layer_dims=layer_dims,
            k=int(data['k']),
            scale=float(data['scale']),
            learn_freq=bool(data['learn_freq']),
            use_cos=bool(data['use_cos']),
        )
        net.omega = data['omega']
        net.phi = data['phi']
        for i in range(net.num_layers):
            net.alpha[i] = data[f'alpha_{i}']
            net.alpha_bias[i] = data[f'alpha_bias_{i}']
        return net


# === Activation functions ===

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def _gelu_deriv(x: np.ndarray) -> np.ndarray:
    """Approximate GELU derivative."""
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    sech2 = 1 - t ** 2
    inner_deriv = 0.7978845608 * (1 + 0.134145 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * sech2 * inner_deriv

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1 - t ** 2

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def _sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1 - s)


ACTIVATIONS = {
    'gelu': (_gelu, _gelu_deriv),
    'relu': (_relu, _relu_deriv),
    'tanh': (_tanh, _tanh_deriv),
    'sigmoid': (_sigmoid, _sigmoid_deriv),
}
