# neuron-py v0.3 Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Prove that Fourier coefficients can *replace* weights entirely — not just initialize them — by training Fourier params directly with PyTorch autograd.

**Architecture:** FourierWeightLinear stores (omega, phi, alpha) as nn.Parameters. Weights are generated on-the-fly in forward(). For large layers, we use block-wise generation to keep memory manageable. A caching mode stores generated weights for inference.

**Tech Stack:** Python 3.13, PyTorch (CPU, 8 threads), numpy, matplotlib

---

## Phase 1: FourierWeightLinear — True Weightless Layers

### Task 1: Create FourierWeightLinear module

**Objective:** Linear layer where weights are *generated* from Fourier params, not stored.

**Files:**
- Create: `neuron/fourier_weight.py`

**Step 1: Write the module**

```python
"""FourierWeightLinear — weights generated on-the-fly from Fourier coefficients.

The key difference from FourierInitLinear:
- FourierInitLinear: generates weights once at init, then trains normally
- FourierWeightLinear: generates weights EVERY forward pass from trainable (omega, phi, alpha)

This means: N×M weight matrix is represented by only 3×K Fourier params.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierWeightLinear(nn.Module):
    """Linear layer with weights generated from Fourier coefficients.
    
    Instead of storing a (out_dim, in_dim) weight matrix, we store:
    - omega: (total_k, 3) frequency vectors — trainable
    - phi: (total_k,) phase offsets — trainable
    - alpha: (2*total_k,) amplitude coefficients — trainable
    
    Weight at position (i,j) in layer l is:
        W(i,j) = Σₖ αₖ·sin(ωₖ·pos(i,j,l) + φₖ) + βₖ·cos(ωₖ·pos(i,j,l) + φₖ)
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        k: Fourier components per frequency band
        n_bands: Number of frequency bands
        layer_idx: Layer depth index
        num_layers: Total layers in network
        block_size: If > 0, generate weights in blocks of this size (saves memory)
        cache_weights: If True, cache generated weights for repeated forward passes
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        k: int = 32,
        n_bands: int = 3,
        layer_idx: int = 0,
        num_layers: int = 1,
        block_size: int = 0,
        cache_weights: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_k = k * n_bands
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.block_size = block_size
        self.cache_weights = cache_weights
        self._cached_weight = None
        
        # Trainable Fourier parameters
        self.omega = nn.Parameter(torch.randn(self.total_k, 3) * 0.5)
        self.phi = nn.Parameter(torch.randn(self.total_k) * math.pi)
        self.alpha = nn.Parameter(torch.randn(2 * self.total_k) * 0.01 
                                  / (1 + torch.arange(2 * self.total_k).float()))
        
        # Precompute position grid (constant, not trainable)
        l_val = (layer_idx + 1) / (num_layers + 1)
        i_idx = (torch.arange(in_dim, dtype=torch.float32) + 1) / (in_dim + 1)
        j_idx = (torch.arange(out_dim, dtype=torch.float32) + 1) / (out_dim + 1)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')
        pos = torch.stack([ii, jj, torch.full_like(ii, l_val)], dim=-1)
        self.register_buffer('pos', pos.reshape(-1, 3))  # (in_dim*out_dim, 3)
        
        # Multi-scale frequency initialization
        with torch.no_grad():
            for band in range(n_bands):
                s, e = band * k, (band + 1) * k
                self.omega[s:e] = torch.randn(k, 3) * (1 + band * 3)
                self.phi[s:e] = torch.randn(k) * math.pi
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def generate_weight(self) -> torch.Tensor:
        """Generate weight matrix from Fourier coefficients."""
        if self._cached_weight is not None:
            return self._cached_weight
        
        if self.block_size > 0 and self.pos.shape[0] > self.block_size:
            w = self._generate_weight_blocked()
        else:
            w = self._generate_weight_full()
        
        if self.cache_weights:
            self._cached_weight = w.detach()
        
        return w
    
    def _generate_weight_full(self) -> torch.Tensor:
        """Generate entire weight matrix at once."""
        args = self.pos @ self.omega.T + self.phi  # (N, total_k)
        features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (N, 2*total_k)
        w = (features @ self.alpha).reshape(self.in_dim, self.out_dim).T  # (out_dim, in_dim)
        w = w * math.sqrt(2.0 / (self.in_dim + self.out_dim))
        return w
    
    def _generate_weight_blocked(self) -> torch.Tensor:
        """Generate weight matrix in blocks to save memory."""
        blocks = []
        for s in range(0, self.pos.shape[0], self.block_size):
            e = min(s + self.block_size, self.pos.shape[0])
            pos_block = self.pos[s:e]
            args = pos_block @ self.omega.T + self.phi
            features = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            blocks.append(features @ self.alpha)
        w = torch.cat(blocks).reshape(self.in_dim, self.out_dim).T
        w = w * math.sqrt(2.0 / (self.in_dim + self.out_dim))
        return w
    
    def clear_cache(self):
        self._cached_weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.generate_weight()
        return F.linear(x, w, self.bias)
    
    @property
    def compression_ratio(self) -> float:
        """How many times fewer params vs standard linear."""
        fourier_params = self.total_k * 3 + self.total_k + 2 * self.total_k + self.out_dim
        standard_params = self.in_dim * self.out_dim + self.out_dim
        return standard_params / fourier_params
    
    def extra_repr(self) -> str:
        return (f'in_dim={self.in_dim}, out_dim={self.out_dim}, '
                f'k={self.total_k}, compression={self.compression_ratio:.1f}x')


class FourierWeightMLP(nn.Module):
    """MLP with true Fourier-weight layers — weights generated on-the-fly.
    
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
```

**Step 2: Verify import**

Run: `cd ~/neuron-py && python3 -c "from neuron.fourier_weight import FourierWeightLinear, FourierWeightMLP; print('OK')"`

---

### Task 2: Benchmark FourierWeightMLP on toy datasets

**Objective:** Prove that trainable Fourier weights can solve spiral/XOR.

**Files:**
- Create: `examples/bench_fourier_weight.py`

**Step 1: Write benchmark script**

```python
#!/usr/bin/env python3
"""Benchmark FourierWeightMLP — true weightless training."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn.functional as F, numpy as np, time
torch.set_num_threads(4)

from neuron.fourier_weight import FourierWeightMLP
from neuron.data import make_spirals, make_xor, make_moons

def bench(name, X, y, dims, k=32, n_bands=3, epochs=300, lr=0.003):
    print(f"🔬 {name}")
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    net = FourierWeightMLP(dims, k=k, n_bands=n_bands)
    print(f"  Fourier params: {net.fourier_params:,}")
    print(f"  Equivalent standard: {net.equivalent_standard_params:,}")
    print(f"  Compression: {net.compression_ratio:.1f}x")
    
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(net(X_t), y_t)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = float(np.mean(net(X_t).argmax(-1).numpy() == y)) * 100
            print(f"  Ep {ep:3d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
    print(f"  ✅ {name}: {acc:.1f}% in {time.time()-t0:.1f}s\n")
    return acc

# Small toy problems — should be fast
bench("Spiral", *make_spirals(300, 3, 0.2), dims=[2, 64, 32, 3], k=32, n_bands=3, epochs=300)
bench("XOR", *make_xor(400, 0.15), dims=[2, 32, 2], k=32, n_bands=2, epochs=200)
bench("Moons", *make_moons(500, 0.15), dims=[2, 32, 2], k=32, n_bands=2, epochs=100)
```

**Step 2: Run and verify >90% on all three**

---

### Task 3: Benchmark FourierWeightMLP on MNIST

**Objective:** Test weightless training on real data.

**Files:**
- Modify: `examples/bench_fourier_weight.py` (add MNIST section)

**Step 1: Add MNIST benchmark**

Use small k (16-32) and block_size=4096 to keep it fast.

**Step 2: Run — target >95% on MNIST**

---

## Phase 2: FourierInitConv2d — CNN Support

### Task 4: Create FourierInitConv2d

**Objective:** Conv2d layer with Fourier-initialized kernel weights.

**Files:**
- Create: `neuron/fourier_conv.py`

**Key difference from linear:** The position grid for a conv kernel is 4D: (out_c, in_c, kH, kW). The layer depth `l` encodes position in the network.

```python
class FourierInitConv2d(nn.Conv2d):
    def __init__(self, in_c, out_c, kernel_size, k=64, n_bands=3, 
                 layer_idx=0, num_layers=1, **kwargs):
        super().__init__(in_c, out_c, kernel_size, **kwargs)
        self._fourier_init_conv(in_c, out_c, kernel_size, k, n_bands, layer_idx, num_layers)
```

**Step 2: Verify import**

---

### Task 5: Benchmark FourierInitConv2d on MNIST

**Objective:** Prove Fourier-init CNN works. Target >98.5%.

**Files:**
- Create: `examples/bench_fourier_conv.py`

---

## Phase 3: Comparison & Analysis

### Task 6: Head-to-head comparison script

**Objective:** Compare FourierInit vs Xavier/Kaiming init on same architectures.

**Files:**
- Create: `examples/comparison.py`

Metrics: convergence speed, final accuracy, parameter count.

---

### Task 7: Generate comparison plots

**Objective:** Visualize the difference.

**Files:**
- Create: `examples/plot_comparison.py`

Output: `docs/comparison.png` — loss curves and accuracy curves.

---

## Phase 4: Polish & Ship

### Task 8: Update package exports and __init__.py

Add FourierWeightLinear, FourierWeightMLP, FourierInitConv2d to `neuron/__init__.py`.

### Task 9: Update README with v0.3 results

Add weightless training results, Conv2d results, comparison chart.

### Task 10: Git push v0.3

```bash
cd ~/neuron-py
git add -A
git commit -m "🧠 v0.3 — Weightless training + FourierConv + benchmarks"
git push
```

---

## Key Decisions

1. **block_size for large layers**: MNIST layer 784→256 = 200K positions. Generating all at once needs (200K, total_k) float32 ≈ 25MB. Block_size=4096 reduces peak to 0.5MB.

2. **k=32 for FourierWeight vs k=128 for FourierInit**: Weightless layers regenerate every forward, so k must be smaller to stay fast. We trade some expressivity for speed.

3. **CosineAnnealing + AdamW**: Best combo for Fourier params — smooth decay helps convergence of oscillatory parameters.
