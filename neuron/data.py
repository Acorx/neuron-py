"""Dataset utilities for neuron benchmarks."""

import numpy as np
import struct
import os
from urllib.request import urlretrieve
import gzip


# === Toy datasets ===

def make_spirals(n: int = 300, n_classes: int = 3, noise: float = 0.2) -> tuple:
    """Spiral classification dataset — famously hard for shallow networks."""
    X = []
    y = []
    for c in range(n_classes):
        r = np.linspace(0, 1, n // n_classes)
        t = np.linspace(c * 2 * np.pi / n_classes, 
                        c * 2 * np.pi / n_classes + 2 * np.pi, 
                        n // n_classes) + np.random.randn(n // n_classes) * noise
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        y += [c] * (n // n_classes)
    return np.concatenate(X), np.array(y)


def make_moons(n: int = 500, noise: float = 0.15) -> tuple:
    """Two interleaving half-circles."""
    n1 = n // 2
    theta = np.linspace(0, np.pi, n1)
    X1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    X2 = np.stack([1 - np.cos(theta), -np.sin(theta)], axis=1)
    X = np.concatenate([X1, X2]) + np.random.randn(n, 2) * noise
    y = np.array([0]*n1 + [1]*n1)
    return X, y


def make_circles(n: int = 500, noise: float = 0.05, factor: float = 0.5) -> tuple:
    """Concentric circles."""
    n1 = n // 2
    theta = np.linspace(0, 2*np.pi, n1)
    X1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    X2 = np.stack([factor*np.cos(theta), factor*np.sin(theta)], axis=1)
    X = np.concatenate([X1, X2]) + np.random.randn(n, 2) * noise
    y = np.array([0]*n1 + [1]*n1)
    return X, y


def make_xor(n: int = 400, noise: float = 0.15) -> tuple:
    """XOR problem with Gaussian noise."""
    X = np.random.randn(n, 2) * 0.5
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += np.random.randn(n, 2) * noise
    return X, y


# === MNIST ===

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

def load_mnist(data_dir: str = None) -> tuple:
    """Load MNIST dataset (download if needed).
    
    Returns:
        ((X_train, y_train), (X_test, y_test)) as float32 numpy arrays.
        Images are flattened to (N, 784) and normalized to [0, 1].
    """
    if data_dir is None:
        data_dir = os.path.expanduser("~/.neuron/mnist")
    os.makedirs(data_dir, exist_ok=True)
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    # Download if needed
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        raw_path = fpath.replace(".gz", "")
        if not os.path.exists(raw_path):
            if not os.path.exists(fpath):
                print(f"  Downloading {fname}...")
                urlretrieve(MNIST_MIRROR + fname, fpath)
            with gzip.open(fpath, "rb") as f_in:
                with open(raw_path, "wb") as f_out:
                    f_out.write(f_in.read())
    
    def read_images(path):
        with open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">4I", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        return data.astype(np.float32) / 255.0
    
    def read_labels(path):
        with open(path, "rb") as f:
            magic, n = struct.unpack(">2I", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    X_train = read_images(os.path.join(data_dir, "train-images-idx3-ubyte")).reshape(60000, -1)
    y_train = read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    X_test = read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte")).reshape(10000, -1)
    y_test = read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    
    return (X_train, y_train), (X_test, y_test)
