"""Improved Fourier network with multi-scale features and learnable frequencies."""

import numpy as np
from typing import List, Optional, Tuple


class FourierNetV2:
    """Improved FourierNet with:
    - Multi-scale frequency bands (low + high freq)
    - Learnable frequencies (omega, phi)
    - Separate alpha per position type (row, col, layer)
    - Better initialization
    - Layer normalization
    - Residual connections when dims match

    Parameters
    ----------
    layer_dims : list of int
    k : int — base Fourier components
    scale : float — weight output scale
    learn_freq : bool — make omega/phi trainable
    use_cos : bool — sin+cos pairs
    n_bands : int — number of frequency bands (multi-scale)
    residual : bool — add skip connections when dims match
    """

    def __init__(self, layer_dims, k=64, scale=0.1, learn_freq=True,
                 use_cos=True, n_bands=3, residual=True):
        self.layer_dims = layer_dims
        self.k = k
        self.scale = scale
        self.learn_freq = learn_freq
        self.use_cos = use_cos
        self.n_bands = n_bands
        self.residual = residual
        self.num_layers = len(layer_dims) - 1

        # Total components = k * n_bands (multi-scale)
        self.total_k = k * n_bands

        rng = np.random.RandomState(42)

        # Multi-scale frequencies: each band has different scale
        self.omega = np.zeros((self.num_layers, self.total_k, 3), dtype=np.float32)
        self.phi = np.zeros((self.num_layers, self.total_k), dtype=np.float32)

        for band in range(n_bands):
            freq_scale = 1.0 + band * 3.0  # 1, 4, 7, 10, ...
            start = band * k
            end = (band + 1) * k
            self.omega[:, start:end, :] = rng.randn(self.num_layers, k, 3).astype(np.float32) * freq_scale
            self.phi[:, start:end] = rng.randn(self.num_layers, k).astype(np.float32) * np.pi

        # Per-layer alpha (learnable)
        n_coeff = 2 * self.total_k if use_cos else self.total_k
        self.alpha = [
            rng.randn(n_coeff).astype(np.float32) * 0.01 / (1 + np.arange(n_coeff))
            for _ in range(self.num_layers)
        ]
        self.alpha_bias = [
            rng.randn(n_coeff).astype(np.float32) * 0.001
            for _ in range(self.num_layers)
        ]

        # Layer norm parameters (learnable scale and shift per hidden dim)
        self.ln_gamma = [np.ones(d, dtype=np.float32) for d in layer_dims[1:-1]]
        self.ln_beta = [np.zeros(d, dtype=np.float32) for d in layer_dims[1:-1]]

        self._cache = None
        self._weight_cache = None
        self._bias_cache = None

    def _positions(self, layer, in_dim, out_dim):
        """Position encodings for weight generation."""
        i_idx = (np.arange(in_dim, dtype=np.float32) + 1) / (in_dim + 1)
        j_idx = (np.arange(out_dim, dtype=np.float32) + 1) / (out_dim + 1)
        ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
        ii, jj = ii.flatten(), jj.flatten()
        ll = np.full_like(ii, (layer + 1) / (self.num_layers + 1))
        return np.stack([ii, jj, ll], axis=-1)

    def _bias_positions(self, layer, dim):
        j_idx = (np.arange(dim, dtype=np.float32) + 1) / (dim + 1)
        ll = np.full_like(j_idx, (layer + 1) / (self.num_layers + 1))
        i_idx = np.zeros_like(j_idx)
        return np.stack([i_idx, j_idx, ll], axis=-1)

    def generate_weights(self, layer):
        in_dim, out_dim = self.layer_dims[layer], self.layer_dims[layer + 1]
        pos = self._positions(layer, in_dim, out_dim)

        omega = self.omega[layer]
        phi = self.phi[layer]
        alpha = self.alpha[layer]

        args = pos @ omega.T + phi
        sin_vals = np.sin(args)
        cos_vals = np.cos(args)

        if self.use_cos:
            features = np.concatenate([sin_vals, cos_vals], axis=1)
        else:
            features = sin_vals

        weights = (features @ alpha * self.scale).reshape(in_dim, out_dim)

        # Xavier-like scaling
        fan_in, fan_out = in_dim, out_dim
        weights *= np.sqrt(2.0 / (fan_in + fan_out)) / self.scale

        # Bias
        bpos = self._bias_positions(layer, out_dim)
        bargs = bpos @ omega.T + phi
        bsin = np.sin(bargs)
        bcos = np.cos(bargs)
        bfeatures = np.concatenate([bsin, bcos], axis=1) if self.use_cos else bsin
        bias = bfeatures @ self.alpha_bias[layer] * self.scale * 0.2

        return weights, bias

    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, x_norm, mean, var

    def _layer_norm_backward(self, grad_out, x_norm, gamma, var, eps=1e-5):
        N = x_norm.shape[-1]
        d_gamma = (grad_out * x_norm).sum(axis=0)
        d_beta = grad_out.sum(axis=0)
        dx_norm = grad_out * gamma
        std_inv = 1.0 / np.sqrt(var + eps)
        dx = (1.0 / N) * std_inv * (N * dx_norm - dx_norm.sum(axis=-1, keepdims=True) - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True))
        return dx, d_gamma, d_beta

    def forward(self, x):
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        self._cache = []

        for layer in range(self.num_layers):
            W, b = self.generate_weights(layer)
            pre = x @ W + b

            if layer < self.num_layers - 1:
                # Layer norm (hidden layers only)
                ln_idx = layer  # index into ln_gamma/ln_beta
                if ln_idx < len(self.ln_gamma):
                    pre_ln, x_norm, mean, var = self._layer_norm(pre, self.ln_gamma[ln_idx], self.ln_beta[ln_idx])
                    act = self._gelu(pre_ln)
                    cache_entry = (x, W, b, pre, pre_ln, x_norm, mean, var, act)
                else:
                    act = self._gelu(pre)
                    cache_entry = (x, W, b, pre, None, None, None, None, act)

                # Residual
                if self.residual and x.shape[-1] == act.shape[-1]:
                    act = act + x
                    cache_entry = cache_entry + (True,)  # has residual
                else:
                    cache_entry = cache_entry + (False,)
            else:
                act = pre
                cache_entry = (x, W, b, pre, None, None, None, None, act, False)

            self._cache.append(cache_entry)
            x = act

        if squeeze:
            return x[0]
        return x

    def backward(self, grad_output):
        grads = {
            'alpha': [np.zeros_like(a) for a in self.alpha],
            'alpha_bias': [np.zeros_like(a) for a in self.alpha_bias],
            'ln_gamma': [np.zeros_like(g) for g in self.ln_gamma],
            'ln_beta': [np.zeros_like(b) for b in self.ln_beta],
        }
        if self.learn_freq:
            grads['omega'] = [np.zeros_like(self.omega[l]) for l in range(self.num_layers)]
            grads['phi'] = [np.zeros_like(self.phi[l]) for l in range(self.num_layers)]

        grad = grad_output

        for layer in range(self.num_layers - 1, -1, -1):
            entry = self._cache[layer]
            x_in, W, b, pre, pre_ln, x_norm, mean, var, act, has_residual = entry
            in_dim, out_dim = self.layer_dims[layer], self.layer_dims[layer + 1]

            # Residual backward
            if has_residual:
                grad_act = grad
                grad = grad.copy()  # don't modify in-place
                # grad flows both paths
            else:
                grad_act = grad

            if layer < self.num_layers - 1:
                # GELU backward
                if pre_ln is not None:
                    grad_gelu = grad_act * self._gelu_deriv(pre_ln)
                    # Layer norm backward
                    ln_idx = layer
                    grad_pre, d_gamma, d_beta = self._layer_norm_backward(
                        grad_gelu, x_norm, self.ln_gamma[ln_idx], var
                    )
                    grads['ln_gamma'][ln_idx] += d_gamma
                    grads['ln_beta'][ln_idx] += d_beta
                else:
                    grad_pre = grad_act * self._gelu_deriv(pre)
            else:
                grad_pre = grad_act

            # Weight gradients (Fourier backprop)
            dW = x_in.T @ grad_pre
            db = grad_pre.sum(axis=0)

            pos = self._positions(layer, in_dim, out_dim)
            omega = self.omega[layer]
            phi = self.phi[layer]
            args = pos @ omega.T + phi
            sin_vals = np.sin(args)
            cos_vals = np.cos(args)

            if self.use_cos:
                features = np.concatenate([sin_vals, cos_vals], axis=1)
            else:
                features = sin_vals

            # Scale factor for Xavier
            fan_in, fan_out = in_dim, out_dim
            xavier_scale = np.sqrt(2.0 / (fan_in + fan_out))

            grads['alpha'][layer] = features.T @ dW.flatten() * xavier_scale

            # Bias gradients
            bpos = self._bias_positions(layer, out_dim)
            bargs = bpos @ omega.T + phi
            bsin = np.sin(bargs)
            bcos = np.cos(bargs)
            bfeatures = np.concatenate([bsin, bcos], axis=1) if self.use_cos else bsin
            grads['alpha_bias'][layer] = bfeatures.T @ db * self.scale * 0.2

            # Learnable frequency gradients
            if self.learn_freq:
                alpha = self.alpha[layer]
                dW_flat = dW.flatten() * xavier_scale

                for ki in range(self.total_k):
                    if self.use_cos:
                        a_sin = alpha[ki]
                        a_cos = alpha[self.total_k + ki]
                    else:
                        a_sin = alpha[ki]
                        a_cos = 0.0

                    cos_arg = cos_vals[:, ki]
                    sin_arg = sin_vals[:, ki]

                    for d in range(3):
                        dW_domega = (a_sin * cos_arg - a_cos * sin_arg) * pos[:, d]
                        grads['omega'][layer][ki, d] = np.dot(dW_domega, dW_flat)

                    dW_dphi = a_sin * cos_arg - a_cos * sin_arg
                    grads['phi'][layer][ki] = np.dot(dW_dphi, dW_flat)

            # Propagate gradient
            grad = grad_pre @ W.T

            # Residual backward
            if has_residual:
                # grad already has the skip connection contribution
                pass

        return grads

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _gelu_deriv(x):
        t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
        sech2 = 1 - t ** 2
        inner = 0.7978845608 * (1 + 0.134145 * x ** 2)
        return 0.5 * (1 + t) + 0.5 * x * sech2 * inner

    def param_count(self):
        n = sum(a.size for a in self.alpha) + sum(a.size for a in self.alpha_bias)
        n += sum(g.size for g in self.ln_gamma) + sum(b.size for b in self.ln_beta)
        if self.learn_freq:
            n += self.omega.size + self.phi.size
        return n

    def virtual_param_count(self):
        count = 0
        for i in range(self.num_layers):
            count += self.layer_dims[i] * self.layer_dims[i + 1] + self.layer_dims[i + 1]
        # Plus layer norm params
        for d in self.layer_dims[1:-1]:
            count += d * 2  # gamma + beta
        return count

    def compression_ratio(self):
        return self.virtual_param_count() / max(self.param_count(), 1)

    def cache_weights(self):
        self._weight_cache = []
        self._bias_cache = []
        for layer in range(self.num_layers):
            W, b = self.generate_weights(layer)
            self._weight_cache.append(W)
            self._bias_cache.append(b)

    def forward_cached(self, x):
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        for layer in range(self.num_layers):
            x = x @ self._weight_cache[layer] + self._bias_cache[layer]
            if layer < self.num_layers - 1:
                ln_idx = layer
                if ln_idx < len(self.ln_gamma):
                    x = self.ln_gamma[ln_idx] * (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5) + self.ln_beta[ln_idx]
                x = self._gelu(x)
        if squeeze:
            return x[0]
        return x

    def save(self, path):
        data = {
            'layer_dims': np.array(self.layer_dims),
            'k': self.k, 'scale': self.scale,
            'learn_freq': self.learn_freq, 'use_cos': self.use_cos,
            'n_bands': self.n_bands, 'residual': self.residual,
            'omega': self.omega, 'phi': self.phi,
        }
        for i, (a, ab) in enumerate(zip(self.alpha, self.alpha_bias)):
            data[f'alpha_{i}'] = a
            data[f'alpha_bias_{i}'] = ab
        for i, (g, b) in enumerate(zip(self.ln_gamma, self.ln_beta)):
            data[f'ln_gamma_{i}'] = g
            data[f'ln_beta_{i}'] = b
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        net = cls(
            layer_dims=data['layer_dims'].tolist(),
            k=int(data['k']),
            scale=float(data['scale']),
            learn_freq=bool(data['learn_freq']),
            use_cos=bool(data['use_cos']),
            n_bands=int(data['n_bands']),
            residual=bool(data['residual']),
        )
        net.omega = data['omega']
        net.phi = data['phi']
        for i in range(net.num_layers):
            net.alpha[i] = data[f'alpha_{i}']
            net.alpha_bias[i] = data[f'alpha_bias_{i}']
        for i in range(len(net.ln_gamma)):
            net.ln_gamma[i] = data[f'ln_gamma_{i}']
            net.ln_beta[i] = data[f'ln_beta_{i}']
        return net
