#!/usr/bin/env python3
"""Test V3 — PyTorch-backed FourierNet on spiral + MNIST."""

import sys
sys.path.insert(0, '/data/data/com.termux/files/home/neuron-py')

import torch
import torch.nn.functional as F
import numpy as np
import time

from neuron.fourier_v3 import FourierNetV3
from neuron.data import make_spirals, make_moons, make_circles, make_xor, load_mnist


def train_torch(net, X, y, epochs=300, batch_size=32, lr=0.003,
                 val_X=None, val_y=None, name="Test"):
    """Train FourierNetV3 using PyTorch."""
    device = next(net.parameters()).device
    n_classes = net.layer_dims[-1]

    # To tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    if val_X is not None:
        val_X_t = torch.tensor(val_X, dtype=torch.float32)
        val_y_t = torch.tensor(val_y, dtype=torch.long)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N = X.shape[0]

    print(f"🧠 neuron V3 (PyTorch) — {name}")
    print(f"   Architecture: {net.layer_dims}")
    print(f"   Fourier k={net.k} × {net.n_bands} bands = {net.fourier_layers[0].total_k} components")
    print(f"   Stored params: {net.param_count():,}")
    print(f"   Virtual params: {net.virtual_param_count():,}")
    print(f"   Compression: {net.compression_ratio():.1f}x")
    print(f"   Device: {device} | Threads: {torch.get_num_threads()}")
    print()

    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        nb = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)

            optimizer.zero_grad()
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            nb += 1

        scheduler.step()
        epoch_loss /= nb

        if epoch % 20 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                out = net(X_t.to(device))
                pred = out.argmax(dim=-1).cpu().numpy()
                acc = float(np.mean(pred == y)) * 100
            print(f"   Ep {epoch:3d} | Loss: {epoch_loss:.4f} | Acc: {acc:.1f}% | {time.time()-t0:.2f}s")

    # Final eval
    net.eval()
    with torch.no_grad():
        out = net(X_t.to(device))
        acc = float(np.mean(out.argmax(dim=-1).cpu().numpy() == y)) * 100
    return net, acc


# ====== TOY TESTS ======

print("=" * 55)
print("🔬 V3: Spiral (3 classes)")
print("=" * 55)
X, y = make_spirals(n=300, n_classes=3, noise=0.2)
net = FourierNetV3([2, 128, 64, 3], k=64, n_bands=3, residual=True)
_, acc = train_torch(net, X, y, epochs=500, lr=0.003, name="Spiral")
print(f"📊 Spiral: {acc:.1f}%\n")

print("=" * 55)
print("🔬 V3: XOR")
print("=" * 55)
X, y = make_xor(n=400, noise=0.15)
net = FourierNetV3([2, 64, 32, 2], k=64, n_bands=3, residual=True)
_, acc = train_torch(net, X, y, epochs=200, lr=0.005, name="XOR")
print(f"📊 XOR: {acc:.1f}%\n")

print("=" * 55)
print("🔬 V3: Moons")
print("=" * 55)
X, y = make_moons(n=500, noise=0.15)
net = FourierNetV3([2, 32, 2], k=64, n_bands=2, residual=False)
_, acc = train_torch(net, X, y, epochs=100, lr=0.005, name="Moons")
print(f"📊 Moons: {acc:.1f}%\n")

# ====== MNIST ======
print("=" * 55)
print("🔬 V3: MNIST")
print("=" * 55)
try:
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    net = FourierNetV3([784, 256, 128, 10], k=128, n_bands=4, residual=True, dropout=0.1)
    _, acc = train_torch(net, X_train, y_train, epochs=30, batch_size=128,
                         lr=0.001, val_X=X_test, val_y=y_test, name="MNIST")

    # Test accuracy
    net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        out = net(X_t)
        pred = out.argmax(dim=-1).numpy()
        test_acc = float(np.mean(pred == y_test)) * 100
    print(f"📊 MNIST Test: {test_acc:.1f}%")
    print(f"   Compression: {net.compression_ratio():.1f}x")
except Exception as e:
    print(f"❌ MNIST failed: {e}")
