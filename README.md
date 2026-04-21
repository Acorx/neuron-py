# neuron 🧠

> CPU-Native AI via Fourier Weight Generation

Instead of storing billions of weights, **neuron generates them from a tiny Fourier formula**.

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

A few hundred coefficients can initialize — and eventually replace — millions of parameters.

---

## ✨ What's New in v0.5

- 🖼️ **CIFAR-10 benchmarks** — FourierCNN vs XavierCNN on color images
- 📦 **CIFAR-10 loader** — binary format, no torchvision dependency
- 🔬 **FFT pruning on CNNs** — Fourier advantage at 25% keep ratio (+5.1%)

---

## 📦 Modules

| Module | What it does | Best for |
|--------|-------------|----------|
| `FourierInitLinear` | Generate init weights from Fourier, train normally | General use |
| `FourierWeightLinear` | Generate weights on-the-fly (weightless!) | Extreme compression |
| `FourierInitConv2d` | Fourier-init for Conv2d kernels | CNNs |
| `FourierCNN` | CNN for CIFAR-10 with Fourier convolutions | Image classification |
| `FourierWeightAttention` | Fourier-generated Q/K/V projections | Transformers |
| `FourierTransformer` | Full transformer with Fourier attention | Sequence tasks |
| `fft_compress_model` | FFT pruning of trained networks | Deployment compression |
| `load_cifar10` | CIFAR-10 loader (binary format) | Benchmarks |

---

## 📊 Key Results

All benchmarks on **ARM mobile** (MediaTek Helio G85, 8-core, 7.5GB RAM) — no GPU.

### MNIST (28×28 grayscale digits)

| Metric | Xavier | Fourier |
|--------|--------|---------|
| Final accuracy (5 epochs) | 97.3% | 97.1% |
| **FFT 10% compression** | **-23.2%** | **-8.8%** |
| **FFT 25% compression** | **-3.3%** | **-1.6%** |

→ **Fourier-init: 2.6x more resilient** to FFT compression at 10% keep

### CIFAR-10 (32×32 RGB images) 🔥

| Metric | XavierCNN | FourierCNN |
|--------|-----------|------------|
| Best accuracy (10 epochs) | 37.7% | 29.4% |
| FFT 25% compression | -8.4% | **-3.3%** |
| FFT 10% compression | -9.8% | **-7.3%** |

→ Fourier CNN less accurate (small kernels don't benefit as much from Fourier init),
BUT **2.5x more resilient at 25% compression** (same pattern as MNIST!)

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

## 🔑 Key Insight: Fourier Creates Compressible Structure

The main finding across all benchmarks:

> **Networks initialized with Fourier formulas have inherent spectral structure
> that makes them significantly more resilient to FFT-based compression.**

At 10% coefficients kept (10x compression):
- Xavier networks lose 23.2% accuracy
- Fourier networks lose only 8.8%
- **That's 2.6x better resilience**

This matters for edge deployment — you can compress a Fourier-init model
much more aggressively without destroying performance.

---

## 🚀 Quick Start

```python
from neuron import FourierMLP, FourierCNN, fft_compress_model

# Fourier-initialized MLP (MNIST)
model = FourierMLP([784, 256, 128, 10], k=64, n_bands=3)

# Fourier-initialized CNN (CIFAR-10)
cnn = FourierCNN(num_classes=10, k=32, n_bands=3)

# After training — compress with FFT pruning
compressed = fft_compress_model(model, keep_ratio=0.1)
# Fourier models lose much less accuracy when compressed!
```

---

## 🧠 How It Works

### The Fourier Weight Formula

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

### Multi-Scale Frequency Bands

```
Band 0: ω ~ N(0, 1²)    → smooth, global patterns
Band 1: ω ~ N(0, 2²)    → medium-frequency  
Band 2: ω ~ N(0, 3²)    → fine, local details
```

### Why Compression Works

Fourier-generated weights have **inherent spectral structure** — they're made of sinusoids. When you apply FFT compression, the energy is concentrated in fewer coefficients, giving near-perfect reconstruction. Standard (Xavier/Kaiming) weights spread energy across all frequencies, so compression destroys more information.

---

## 🗺️ Roadmap

- [ ] Larger CIFAR-10 model (need faster hardware or more patience)
- [ ] Quantized Fourier coefficients (4-bit α, 8-bit ω/φ)
- [ ] Integration with [flame-tensor](https://github.com/Acorx/flame-tensor) (Rust)
- [ ] ONNX export with Fourier weight generation as custom op
- [ ] Pre-trained Fourier-weighted models for download

---

## 📜 History

| Version | Key Feature | Finding |
|---------|------------|---------|
| v0.1 (Go) | Finite-diff | Proof of concept, 51% spiral |
| v0.2 | Fourier-init + PyTorch | 98.2% MNIST, autograd is key |
| v0.3 | Weightless + Conv | 174x compression |
| v0.4 | Transformer + Pruning | 2.6x compression resilience (MNIST) |
| v0.5 | **CIFAR-10** | Fourier resilient on CNNs too (+5.1% at 25%) |

---

## 📄 License

MIT
