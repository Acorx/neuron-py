#!/usr/bin/env python3
"""neuron benchmarks — Spiral, XOR, Moons, MNIST.

Run: python3 examples/benchmarks.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import time

torch.set_num_threads(4)

from neuron import FourierMLP
from neuron.data import make_spirals, make_xor, make_moons, load_mnist


def bench_spiral():
    print("=" * 50)
    print("🔬 Spiral Classification (3-class)")
    X, y = make_spirals(300, 3, noise=0.2)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    net = FourierMLP([2, 64, 32, 3], k=64, n_bands=3)
    opt = torch.optim.AdamW(net.parameters(), lr=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=5e-5)
    
    t0 = time.time()
    for ep in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(net(X_t), y_t)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 299:
            with torch.no_grad():
                acc = float(np.mean(net(X_t).argmax(-1).numpy() == y)) * 100
            print(f"  Ep {ep:3d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
    print(f"  ✅ Spiral: {acc:.1f}% in {time.time()-t0:.1f}s\n")
    return acc


def bench_xor():
    print("🔬 XOR Classification")
    X, y = make_xor(400, noise=0.15)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    net = FourierMLP([2, 64, 32, 2], k=64, n_bands=3)
    opt = torch.optim.AdamW(net.parameters(), lr=0.005)
    
    t0 = time.time()
    for ep in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(net(X_t), y_t)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 199:
            with torch.no_grad():
                acc = float(np.mean(net(X_t).argmax(-1).numpy() == y)) * 100
            print(f"  Ep {ep:3d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
    print(f"  ✅ XOR: {acc:.1f}% in {time.time()-t0:.1f}s\n")
    return acc


def bench_mnist():
    print("=" * 50)
    print("🔬 MNIST Digit Classification")
    (X_train, y_train), (X_test, y_test) = load_mnist()
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    
    net = FourierMLP([784, 256, 128, 10], k=128, n_bands=4)
    params = sum(p.numel() for p in net.parameters())
    print(f"  Params: {params:,}")
    
    opt = torch.optim.AdamW(net.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
    
    t0 = time.time()
    for ep in range(20):
        perm = torch.randperm(60000)
        eloss, nb = 0, 0
        for s in range(0, 60000, 256):
            e = min(s + 256, 60000)
            idx = perm[s:e]
            opt.zero_grad()
            loss = F.cross_entropy(net(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            eloss += loss.item()
            nb += 1
        sched.step()
        with torch.no_grad():
            test_acc = float(np.mean(net(X_te).argmax(-1).numpy() == y_test)) * 100
        print(f"  Ep {ep:2d} | Loss: {eloss/nb:.4f} | Test: {test_acc:.1f}%")
    print(f"  ✅ MNIST: {test_acc:.1f}% in {time.time()-t0:.0f}s\n")
    return test_acc


if __name__ == "__main__":
    print("🧠 neuron benchmarks\n")
    results = {}
    results["spiral"] = bench_spiral()
    results["xor"] = bench_xor()
    results["mnist"] = bench_mnist()
    
    print("=" * 50)
    print("📊 Summary")
    for name, acc in results.items():
        print(f"  {name:10s} | {acc:.1f}%")
