# neuron 🧠

> CPU-Native AI via Fourier Weight Generation

Instead of storing billions of weights, **neuron generates them from a tiny Fourier formula**.

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

A few hundred coefficients can initialize — and eventually replace — millions of parameters.

---

## ✨ What's New in v0.4

- 🧠 **FourierTransformer** — Attention with Fourier-generated Q/K/V weights (11x compression)
- ✂️ **FourierPruning** — FFT compression that exploits spectral structure
- 🔬 **Key finding**: Fourier-init networks are **2.6x more resilient** to FFT compression

---

## 📦 Modules

| Module | What it does | Best for |
|--------|-------------|----------|
| `FourierInitLinear` | Generate init weights from Fourier, train normally | General use |
| `FourierWeightLinear` | Generate weights on-the-fly (weightless!) | Extreme compression |
| `FourierInitConv2d` | Fourier-init for Conv2d kernels | CNNs |
| `FourierWeightAttention` | Fourier-generated Q/K/V projections | Transformers |
| `FourierTransformer` | Full transformer with Fourier attention | Sequence tasks |
| `fft_compress_model` | FFT pruning of trained networks | Deployment compression |

---

## 📊 Key Results

All benchmarks on **ARM mobile** (MediaTek Helio G85, 8-core, 7.5GB RAM) — no GPU.

### Fourier-Init MLP on MNIST

| Epoch | Xavier | Fourier |
|-------|--------|---------|
| 0 | 94.9% | 94.9% |
| 4 | 97.3% | 96.9% |

→ Comparable training, but Fourier creates **spectral structure**.

### FFT Compression Resilience 🔥

| Keep % | Xavier Δ | **Fourier Δ** | Advantage |
|--------|----------|---------------|-----------|
| 50% | -0.5% | **-0.2%** | 2.5x |
| 25% | -3.3% | **-1.6%** | 2x |
| 10% | -23.2% | **-8.8%** | **2.6x** |
| 5% | -57.2% | **-36.7%** | 1.6x |

→ **Fourier-init networks lose 2.6x less accuracy** when compressed to 10% of weights.

### FourierWeight MLP (weightless)

| Config | Fourier Params | Standard Equiv | Compression |
|--------|---------------|----------------|-------------|
| k=16, n_bands=2 | 458 | 50,890 | **174x** |

### Fourier Transformer

| Component | Standard Params | Fourier Params | Compression |
|-----------|----------------|----------------|-------------|
| Q/K/V projections (d=32) | 6,336 | 576 | **11x** |

### Toy Datasets

| Dataset | Accuracy |
|---------|----------|
| 🌀 Spiral (3-class) | **97.7%** |
| 🌙 Moons (2-class) | **99.8%** |
| ✖️ XOR (2-class) | **93.2%** |

---

## 🚀 Quick Start

```python
from neuron import FourierMLP, FourierTransformer, fft_compress_model

# Fourier-initialized MLP
model = FourierMLP([784, 256, 128, 10], k=64, n_bands=3)

# Fourier Transformer (11x attention compression)
transformer = FourierTransformer(
    embed_dim=128, num_heads=4, num_layers=4, k=32, n_bands=2
)

# After training — compress with FFT pruning
compressed = fft_compress_model(model, keep_ratio=0.1)  # 10x compression
# Fourier-init models lose only 8.8% accuracy at 10% coefficients!
```

---

## 🧠 How It Works

### The Fourier Weight Formula

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

- `i, j` = weight position indices
- `l` = layer depth
- `ωₖ` = frequency vectors (multi-scale bands)
- `φₖ` = phase offsets
- `αₖ` = amplitude coefficients

### Multi-Scale Frequency Bands

```
Band 0: ω ~ N(0, 1)    → smooth, global patterns
Band 1: ω ~ N(0, 4)    → medium-frequency
Band 2: ω ~ N(0, 9)    → fine, local details
```

### Why Compression Works

Fourier-generated weights have **inherent spectral structure** — they're made of sinusoids. When you apply FFT compression, the energy is concentrated in fewer coefficients, giving near-perfect reconstruction. Random/Xavier weights spread energy across all frequencies, so compression destroys more information.

---

## 🗺️ Roadmap

- [ ] CIFAR-10 benchmarks
- [ ] Quantized Fourier coefficients (4-bit α, 8-bit ω/φ)
- [ ] Integration with [flame-tensor](https://github.com/Acorx/flame-tensor) (Rust)
- [ ] ONNX export with Fourier weight generation as custom op
- [ ] Pre-trained Fourier-weighted models for download

---

## 📜 History

| Version | Key Feature | MNIST | Finding |
|---------|------------|-------|---------|
| v0.1 (Go) | Finite-diff | ❌ | Proof of concept, 51% spiral |
| v0.2 | Fourier-init + PyTorch | 98.2% | Init matters, autograd is key |
| v0.3 | Weightless + Conv | 174x compression | On-the-fly generation works |
| v0.4 | **Transformer + Pruning** | 2.6x compression resilience | **Fourier creates compressible structure** |

---

## 📄 License

MIT
