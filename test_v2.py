#!/usr/bin/env python3
"""Test V2 — improved FourierNet on spiral, XOR, moons."""

import sys
sys.path.insert(0, '/data/data/com.termux/files/home/neuron-py')

import numpy as np
from neuron.fourier_v2 import FourierNetV2
from neuron.losses import CrossEntropyLoss, MSELoss
from neuron.optim import Adam, SGD, CosineAnnealingLR
from neuron.data import make_spirals, make_moons, make_circles, make_xor
import time


def train_v2(layer_dims, X, y, epochs=300, batch_size=32, lr=0.003,
             k=64, n_bands=3, residual=True):
    """Train a FourierNetV2 model."""
    net = FourierNetV2(
        layer_dims=layer_dims,
        k=k,
        n_bands=n_bands,
        scale=0.1,
        learn_freq=True,
        use_cos=True,
        residual=residual,
    )
    loss_fn = CrossEntropyLoss()
    opt = Adam(lr=lr, weight_decay=0.001)
    scheduler = CosineAnnealingLR(opt, t_max=epochs, eta_min=lr * 0.01)

    n_classes = layer_dims[-1]
    N = X.shape[0]
    y_onehot = np.zeros((N, n_classes), dtype=np.float32)
    y_onehot[np.arange(N), y.astype(int)] = 1

    print(f"🧠 neuron V2 — Fourier Network Training")
    print(f"   Architecture: {layer_dims}")
    print(f"   Fourier k={k} × {n_bands} bands = {net.total_k} components")
    print(f"   Residual: {residual} | Learn freq: True | Cos: True")
    print(f"   Stored params: {net.param_count():,}")
    print(f"   Virtual params: {net.virtual_param_count():,}")
    print(f"   Compression: {net.compression_ratio():.1f}x")
    print()

    for epoch in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(N)
        X_s, y_s = X[perm], y_onehot[perm]

        epoch_loss = 0.0
        nb = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            xb, yb = X_s[start:end], y_s[start:end]

            output = net.forward(xb)
            loss = loss_fn.forward(output, yb)
            grad = loss_fn.backward()
            grads = net.backward(grad)

            params = list(net.alpha) + list(net.alpha_bias)
            grad_list = list(grads['alpha']) + list(grads['alpha_bias'])

            # Layer norm params
            params += list(net.ln_gamma) + list(net.ln_beta)
            grad_list += list(grads['ln_gamma']) + list(grads['ln_beta'])

            if net.learn_freq:
                for l in range(net.num_layers):
                    params.append(net.omega[l].reshape(-1))
                    params.append(net.phi[l].reshape(-1))
                    grad_list.append(grads['omega'][l].reshape(-1))
                    grad_list.append(grads['phi'][l].reshape(-1))

            opt.step(params, grad_list)
            epoch_loss += loss
            nb += 1

        scheduler.step()
        epoch_loss /= nb

        if epoch % 10 == 0 or epoch == epochs - 1:
            output = net.forward(X)
            pred = np.argmax(output, axis=-1)
            acc = float(np.mean(pred == y.astype(int))) * 100
            print(f"   Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | Acc: {acc:.1f}% | {time.time()-t0:.2f}s | LR: {opt.lr:.6f}")

    # Final eval
    output = net.forward(X)
    pred = np.argmax(output, axis=-1)
    acc = float(np.mean(pred == y.astype(int))) * 100
    net.cache_weights()
    return net, acc


print("=" * 55)
print("🔬 TEST V2: Spiral Classification (3 classes)")
print("=" * 55)
X, y = make_spirals(n=300, n_classes=3, noise=0.2)
net, acc = train_v2([2, 128, 64, 3], X, y, epochs=500, lr=0.003, k=64, n_bands=3)
print(f"\n📊 Spiral Accuracy: {acc:.1f}%")
print(f"   Compression: {net.compression_ratio():.1f}x")

print("\n" + "=" * 55)
print("🔬 TEST V2: XOR Problem")
print("=" * 55)
X, y = make_xor(n=400, noise=0.15)
net, acc = train_v2([2, 64, 32, 2], X, y, epochs=200, lr=0.005, k=64, n_bands=3)
print(f"\n📊 XOR Accuracy: {acc:.1f}%")

print("\n" + "=" * 55)
print("🔬 TEST V2: Two Moons")
print("=" * 55)
X, y = make_moons(n=500, noise=0.15)
net, acc = train_v2([2, 32, 2], X, y, epochs=100, lr=0.005, k=64, n_bands=2)
print(f"\n📊 Moons Accuracy: {acc:.1f}%")

print("\n" + "=" * 55)
print("🔬 TEST V2: Concentric Circles")
print("=" * 55)
X, y = make_circles(n=500, noise=0.05, factor=0.5)
net, acc = train_v2([2, 32, 2], X, y, epochs=100, lr=0.005, k=64, n_bands=2)
print(f"\n📊 Circles Accuracy: {acc:.1f}%")
