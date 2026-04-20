"""neuron 🧠 — CPU-Native AI via Fourier Weight Generation

Instead of storing billions of weights, neuron generates them from a tiny
Fourier formula. A few hundred coefficients can initialize — and eventually
replace — millions of parameters.
"""

from neuron.fourier_init import FourierInitLinear, FourierMLP, FourierResMLP
from neuron.fourier_weight import FourierWeightLinear, FourierWeightMLP
from neuron.fourier_conv import FourierInitConv2d, FourierCNN
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

__version__ = "0.4.0"
__all__ = [
    # Init
    "FourierInitLinear", "FourierMLP", "FourierResMLP",
    "FourierInitConv2d", "FourierCNN",
    "FourierInitAttention",
    # Weightless
    "FourierWeightLinear", "FourierWeightMLP",
    # Transformer
    "FourierWeightAttention", "FourierTransformerBlock", "FourierTransformer",
    # Pruning
    "fft_compress_weight", "fft_compress_model", "apply_fft_compression",
]
