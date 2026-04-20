# neuron 🧠

> CPU-Native AI Training via Fourier Weight Generation

Instead of storing billions of weights, **neuron generates them from a tiny Fourier formula**.

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

A few hundred coefficients can initialize — and in some cases replace — millions of parameters.

---

## ✨ Three Approaches

| Approach | How it works | Compression | Accuracy | Speed |
|----------|-------------|-------------|----------|-------|
| **FourierInit** | Generate init weights, then train normally | 1x (no compression) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **FourierWeight** | Generate weights on-the-fly every forward | **174x** | ⭐⭐⭐ | ⭐⭐ |
| **FourierConv** | Fourier-init for Conv2d kernels | 1x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Which to use?

- **FourierInit** → Drop-in replacement for Xavier/Kaiming. Same training speed, structured initialization.
- **FourierWeight** → When you need extreme compression. 458 params vs 50K standard on MNIST. Slower but real compression.
- **FourierConv** → For CNNs. Spatial Fourier patterns in kernel initialization.

---

## 📊 Benchmarks

All benchmarks run on **ARM mobile** (MediaTek Helio G85, 8-core, 7.5GB RAM) — no GPU.

### FourierInit MLP on MNIST

| Epoch | Xavier Init | Fourier Init |
|-------|------------|--------------|
| 0 | 94.9% | 94.9% |
| 1 | 96.2% | 95.9% |
| 2 | 97.4% | 95.8% |
| 4 | 97.3% | 96.9% |

→ Comparable performance. Fourier-init provides structured initialization.

### FourierWeight MLP (weightless!) on MNIST

| Config | Fourier Params | Standard Equiv | Compression | Accuracy |
|--------|---------------|----------------|-------------|----------|
| k=16, n_bands=2 | 458 | 50,890 | **174x** | 52.2% (9 ep) |
| k=32, n_bands=3 (toy) | 1,827 | 2,371 | 1.3x | 97.7% (spiral) |

→ Weight generation works! Compression is real but needs more epochs for full accuracy.

### Toy Datasets (FourierInit MLP)

| Dataset | Accuracy |
|---------|----------|
| 🌀 Spiral (3-class) | **97.7%** |
| 🌙 Moons (2-class) | **99.8%** |
| ✖️ XOR (2-class) | **93.2%** |

---

## 🚀 Quick Start

```python
from neuron import FourierMLP, FourierWeightMLP

# Fourier-initialized MLP (fast, general use)
model = FourierMLP([784, 256, 128, 10], k=64, n_bands=3)

# Weightless Fourier MLP (extreme compression)
model = FourierWeightMLP([784, 64, 10], k=16, n_bands=2)
# → 458 params instead of 50,890 (174x compression!)
```

### Training

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

for epoch in range(15):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

---

## 🧠 How It Works

### The Fourier Weight Formula

Instead of storing each weight `W[i][j]` individually, we compute it from a compact sum of sinusoids:

```
W(i,j,l) = Σₖ αₖ · sin(ωₖᵢ·i + ϖₖⱼ·j + ωₖₗ·l + φₖ)
```

Where:
- `i, j` = weight position indices
- `l` = layer depth
- `ωₖ` = frequency vectors (learnable or fixed, multi-scale)
- `φₖ` = phase offsets (learnable or fixed)
- `αₖ` = amplitude coefficients (always learnable)

### Multi-Scale Frequency Bands

Frequencies are organized in bands to capture patterns at different scales:

```
Band 0: ω ~ N(0, 1)    → low-frequency, smooth patterns
Band 1: ω ~ N(0, 4)    → medium-frequency, local patterns
Band 2: ω ~ N(0, 9)    → high-frequency, fine details
```

### Cosine + Sine Features

Both `sin` and `cos` are used, doubling expressivity:

```
features = [sin(ω·pos + φ), cos(ω·pos + φ)]  → 2K features per position
weights = features @ α                         → single vector multiply
```

---

## 📁 Project Structure

```
neuron/
├── __init__.py           # Public API
├── fourier_init.py       # FourierInitLinear, FourierMLP, FourierResMLP
├── fourier_weight.py     # FourierWeightLinear, FourierWeightMLP
├── fourier_conv.py       # FourierInitConv2d, FourierCNN
├── fourier.py            # V1 pure numpy (reference)
├── fourier_v2.py         # V2 learnable ω/φ (reference)
├── losses.py             # MSELoss, CrossEntropyLoss (numpy)
├── optim.py              # SGD, Adam (numpy)
├── activations.py        # ReLU, GELU, Tanh, Sigmoid
├── layers.py             # Layer abstractions
├── data.py               # MNIST loader + toy datasets
├── model.py              # Training loop utilities
examples/
├── benchmarks.py         # Full benchmark suite
```

---

## 🗺️ Roadmap

- [ ] Fourier weight generation for Transformers (attention heads)
- [ ] Pruning: extract dominant Fourier coefficients from trained standard networks
- [ ] Quantized Fourier coefficients (4-bit α, 8-bit ω/φ)
- [ ] Integration with [flame-tensor](https://github.com/Acorx/flame-tensor) (Rust)
- [ ] CIFAR-10 benchmarks
- [ ] ONNX export with weight generation as custom op

---

## 📜 History

| Version | Approach | MNIST | Key Insight |
|---------|----------|-------|-------------|
| v0.1 (Go) | Finite-diff, pure Go | ❌ | Proof of concept, 51% spiral |
| v0.2 (Python) | Fourier-init + PyTorch | 98.2% | Init matters, autograd is key |
| v0.3 (Python) | Fourier-init + Weightless + Conv | 97.7% (spiral) | 174x compression possible |

---

## 📄 License

MIT
