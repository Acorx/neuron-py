# neuron 🧠

> CPU-Native AI Training via Fourier Weight Generation

Instead of storing billions of weights, **neuron generates them from a tiny Fourier formula**.

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

A few hundred Fourier coefficients can initialize — and eventually replace — millions of parameters. No GPU required.

## 🚀 Quick Start

```python
from neuron import FourierMLP
import torch, torch.nn.functional as F

# Create a Fourier-initialized MLP
net = FourierMLP([784, 256, 128, 10], k=128, n_bands=4)

# Train normally — Fourier init gives faster convergence
opt = torch.optim.AdamW(net.parameters(), lr=0.001)
loss = F.cross_entropy(net(x), y)
loss.backward()
opt.step()
```

## 📊 Benchmarks

| Dataset | V1 (Go, finite-diff) | V2 (Python, Fourier-Init) | Improvement |
|---------|---------------------|---------------------------|-------------|
| **Spiral** (3-class) | 51.1% | **97.7%** | +46 pts |
| **XOR** | — | **93.2%** | — |
| **Moons** | — | **99.8%** | — |
| **MNIST** | — | **98.2%** | — |

*All benchmarks run on ARM CPU (MediaTek Helio G85, 8-core, 7.5GB RAM) — no GPU needed.*

## 🧠 How It Works

### The Problem
Traditional neural networks store every weight individually. A 784→256 layer needs **200,704 floats**. Large LLMs need **billions**.

### The Insight
Weights aren't random — they have structure. A compact Fourier formula can capture that structure with far fewer parameters:

1. **Multi-scale frequency bands** — Low frequencies capture global patterns, high frequencies capture fine details
2. **Positional encoding** — Each weight is addressed by its (row, col, layer) position
3. **Sin + Cos features** — Both sine and cosine components double expressivity

### The Formula
For a weight at position `(i, j)` in layer `l`:

```
W(i,j,l) = Σₖ αₖ · sin(ωₖ·[i, j, l] + φₖ) + βₖ · cos(ωₖ·[i, j, l] + φₖ)
```

Where:
- `ωₖ` = frequency vector (learnable, multi-scale)
- `φₖ` = phase offset (learnable)
- `αₖ, βₖ` = amplitude coefficients (learnable)
- `k` = number of Fourier components (typically 64-512)

## 🏗️ Architecture

```
neuron/
├── __init__.py           # Package exports
├── fourier_init.py       # FourierInitLinear, FourierMLP, FourierResMLP
├── fourier.py            # V1 pure-numpy FourierNet (research)
├── fourier_v2.py         # V2 with learnable frequencies (research)
├── losses.py             # MSELoss, CrossEntropyLoss (numpy)
├── optim.py              # SGD, Adam (numpy)
├── activations.py        # Activation wrappers
├── layers.py             # FourierLinear (numpy)
├── model.py              # Model trainer (numpy)
├── data.py               # Dataset utilities + MNIST loader
examples/
├── benchmarks.py         # Full benchmark suite
```

## 🔬 Evolution

| Version | Approach | Status |
|---------|----------|--------|
| **V1** (`fourier.py`) | Pure numpy, finite-diff gradients | ✅ 51% spiral — proof of concept |
| **V2** (`fourier_v2.py`) | Learnable ω/φ, multi-scale, layer norm | ⚠️ Too slow on mobile CPU |
| **V3** (`fourier_v3.py`) | PyTorch backend, on-the-fly weight gen | ⚠️ Weight gen too slow per forward |
| **V4** (`fourier_init.py`) | **Fourier init + PyTorch autograd** | ✅ **98.2% MNIST — current best** |

### Key Lesson
Generating weights on-the-fly is elegant but slow. The pragmatic approach: **use Fourier as initialization**, then train normally. Future work: hybrid where Fourier coefficients are trained alongside weights.

## 🎯 Roadmap

- [ ] **Fourier fine-tuning** — Train Fourier coefficients directly, not weights
- [ ] **Weightless inference** — On-the-fly weight generation for deployment
- [ ] **CNN support** — FourierInitConv2d layers
- [ ] **Transformer** — FourierInitAttention for LLMs
- [ ] **Compression** — Post-training: fit Fourier to trained weights, discard originals
- [ ] **ONNX export** — Deploy via NNAPI on Android/Mali GPU

## 📦 Install

```bash
cd neuron-py
pip install -e .
```

Requires: `torch`, `numpy`

## License

MIT
