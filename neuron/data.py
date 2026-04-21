"""Dataset utilities — MNIST, CIFAR-10, and toy problem loaders."""

import numpy as np
import struct
import os
from urllib.request import urlretrieve


# ══════════════════════════════════════════
# Toy datasets
# ══════════════════════════════════════════

def make_spirals(n: int = 300, n_classes: int = 3, noise: float = 0.2):
    """2D spiral classification — classic non-linear benchmark."""
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0, 4 * np.pi, n // n_classes) + c * 2 * np.pi / n_classes
        r = t / (4 * np.pi)
        x0 = r * np.cos(t) + np.random.randn(len(t)) * noise
        x1 = r * np.sin(t) + np.random.randn(len(t)) * noise
        X.append(np.stack([x0, x1], axis=1))
        y.append(np.full(len(t), c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def make_moons(n: int = 300, noise: float = 0.1):
    """2D two-moons classification."""
    n0 = n // 2
    t = np.linspace(0, np.pi, n0)
    x0 = np.cos(t) + np.random.randn(n0) * noise
    y0 = np.sin(t) + np.random.randn(n0) * noise
    x1 = 1 - np.cos(t) + np.random.randn(n0) * noise
    y1 = 1 - np.sin(t) - np.random.randn(n0) * noise
    X = np.stack([np.concatenate([x0, x1]), np.concatenate([y0, y1])], axis=1).astype(np.float32)
    labels = np.concatenate([np.zeros(n0), np.ones(n0)]).astype(np.int64)
    perm = np.random.permutation(len(labels))
    return X[perm], labels[perm]


def make_xor(n: int = 200, noise: float = 0.1):
    """2D XOR classification."""
    nq = n // 4
    X = np.vstack([
        np.random.randn(nq, 2) * noise + [1, 1],
        np.random.randn(nq, 2) * noise + [-1, -1],
        np.random.randn(nq, 2) * noise + [1, -1],
        np.random.randn(nq, 2) * noise + [-1, 1],
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(2 * nq), np.ones(2 * nq)]).astype(np.int64)
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def make_circles(n: int = 300, noise: float = 0.05):
    """2D concentric circles classification."""
    n0 = n // 2
    t = np.linspace(0, 2 * np.pi, n0, endpoint=False)
    r0, r1 = 0.5, 1.0
    X = np.vstack([
        np.stack([r0 * np.cos(t), r0 * np.sin(t)], axis=1) + np.random.randn(n0, 2) * noise,
        np.stack([r1 * np.cos(t), r1 * np.sin(t)], axis=1) + np.random.randn(n0, 2) * noise,
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(n0), np.ones(n0)]).astype(np.int64)
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


# ══════════════════════════════════════════
# MNIST
# ══════════════════════════════════════════

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_DIR = os.path.expanduser("~/.neuron/mnist")


def _download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    ]
    import gzip
    for f in files:
        path = os.path.join(MNIST_DIR, f.replace(".gz", ""))
        if not os.path.exists(path):
            url = MNIST_URL + f
            tmp = os.path.join(MNIST_DIR, f)
            print(f"  Downloading {f}...")
            urlretrieve(url, tmp)
            with gzip.open(tmp, "rb") as fin:
                with open(path, "wb") as fout:
                    fout.write(fin.read())
            os.remove(tmp)


def load_mnist():
    """Load MNIST as (X_train, y_train), (X_test, y_test) numpy arrays."""
    _download_mnist()

    def read_images(p):
        with open(p, "rb") as f:
            magic, n, rows, cols = struct.unpack(">4I", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255

    def read_labels(p):
        with open(p, "rb") as f:
            magic, n = struct.unpack(">2I", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte")).reshape(60000, -1)
    y_train = read_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))
    X_test = read_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte")).reshape(10000, -1)
    y_test = read_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))
    return (X_train, y_train), (X_test, y_test)


# ══════════════════════════════════════════
# CIFAR-10
# ══════════════════════════════════════════

CIFAR10_DIR = os.path.expanduser("~/.neuron/cifar-10/cifar-10-batches-bin")
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_cifar10():
    """Load CIFAR-10 as (X_train, y_train), (X_test, y_test) numpy arrays.
    
    Returns images in NCHW format (N, 3, 32, 32), float32, normalized [0, 1].
    """
    def read_batch(p):
        with open(p, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # Format: 1 label + 3072 pixels per image (1024 R + 1024 G + 1024 B)
        data = data.reshape(-1, 3073)
        labels = data[:, 0].astype(np.int64)
        pixels = data[:, 1:].astype(np.float32).reshape(-1, 3, 32, 32) / 255
        return pixels, labels

    X_train, y_train = [], []
    for i in range(1, 6):
        path = os.path.join(CIFAR10_DIR, f"data_batch_{i}.bin")
        X, y = read_batch(path)
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = read_batch(os.path.join(CIFAR10_DIR, "test_batch.bin"))
    return (X_train, y_train), (X_test, y_test)
