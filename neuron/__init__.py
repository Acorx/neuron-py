"""neuron 🧠 — CPU-Native AI via Fourier Weight Generation

Instead of storing billions of weights, neuron generates them from a tiny
Fourier formula. A few hundred coefficients can initialize — and eventually
replace — millions of parameters.
"""

from neuron.fourier_init import FourierInitLinear, FourierMLP, FourierResMLP
from neuron.fourier_weight import FourierWeightLinear, FourierWeightMLP
from neuron.fourier_conv import FourierInitConv2d, FourierCNN, XavierCNN
from neuron.fourier_transformer import (
    FourierInitAttention,
    FourierWeightAttention,
    FourierTransformerBlock,
    FourierTransformer,
)
from neuron.fourier_prune import (
    fft_compress_weight,
    fft_compress_model,
    apply_fft_compression,
)
from neuron.data import (
    make_spirals, make_moons, make_xor, make_circles,
    load_mnist, load_cifar10,
)

__version__ = "0.5.0"
__all__ = [
    # Init
    "FourierInitLinear", "FourierMLP", "FourierResMLP",
    "FourierInitConv2d", "FourierCNN", "XavierCNN",
    "FourierInitAttention",
    # Weightless
    "FourierWeightLinear", "FourierWeightMLP",
    # Transformer
    "FourierWeightAttention", "FourierTransformerBlock", "FourierTransformer",
    # Pruning
    "fft_compress_weight", "fft_compress_model", "apply_fft_compression",
    # Data
    "make_spirals", "make_moons", "make_xor", "make_circles",
    "load_mnist", "load_cifar10",
]
