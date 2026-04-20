#!/usr/bin/env python3
"""Quick test — spiral classification (same as original Go demo)."""

import sys
sys.path.insert(0, '/data/data/com.termux/files/home/neuron-py')

import numpy as np
from neuron.model import Model
from neuron.data import make_spirals, make_moons, make_circles, make_xor


def test_spiral():
    print("=" * 55)
    print("🔬 TEST 1: Spiral Classification (3 classes)")
    print("=" * 55)

    X, y = make_spirals(n=300, n_classes=3, noise=0.2)
    n_classes = 3
    input_dim = 2

    model = Model(
        layer_dims=[input_dim, 64, 32, n_classes],
        k=64,
        loss='cross_entropy',
        optimizer='adam',
        lr=0.003,
        use_cos=True,
    )

    model.train(X, y, epochs=200, batch_size=32,
                val_X=X, val_y=y)

    acc = model.evaluate(X, y)
    print(f"\n📊 Final Accuracy: {acc:.1f}%")
    print(f"   Stored params: {model.net.param_count()}")
    print(f"   Virtual params: {model.net.virtual_param_count()}")
    print(f"   Compression: {model.net.compression_ratio():.1f}x")


def test_xor():
    print("\n" + "=" * 55)
    print("🔬 TEST 2: XOR Problem")
    print("=" * 55)

    X, y = make_xor(n=400, noise=0.1)

    model = Model(
        layer_dims=[2, 32, 16, 2],
        k=64,
        loss='cross_entropy',
        optimizer='adam',
        lr=0.005,
        use_cos=True,
    )

    model.train(X, y, epochs=150, batch_size=32,
                val_X=X, val_y=y)

    acc = model.evaluate(X, y)
    print(f"\n📊 Final Accuracy: {acc:.1f}%")


def test_moons():
    print("\n" + "=" * 55)
    print("🔬 TEST 3: Two Moons")
    print("=" * 55)

    X, y = make_moons(n=500, noise=0.1)

    model = Model(
        layer_dims=[2, 32, 2],
        k=64,
        loss='cross_entropy',
        optimizer='adam',
        lr=0.005,
        use_cos=True,
    )

    model.train(X, y, epochs=100, batch_size=32,
                val_X=X, val_y=y)

    acc = model.evaluate(X, y)
    print(f"\n📊 Final Accuracy: {acc:.1f}%")


if __name__ == '__main__':
    test_spiral()
    test_xor()
    test_moons()
