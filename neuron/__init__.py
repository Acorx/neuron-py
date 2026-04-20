"""Fourier Weight Generation — the complete neuron library.

Three approaches, each with different trade-offs:

1. FourierInitLinear — Generate initial weights from Fourier formula, then train normally.
   Best for: general use, same perf as Xavier but with structured initialization.
   
2. FourierWeightLinear — Generate weights on-the-fly from Fourier coefficients at EVERY forward pass.
   Best for: extreme compression (174x on MNIST), but needs more training epochs.
   
3. FourierInitConv2d — Conv2d with Fourier-generated kernel initialization.
   Best for: CNNs with structured spatial init patterns.
"""

from neuron.fourier_init import FourierInitLinear, FourierMLP, FourierResMLP
from neuron.fourier_weight import FourierWeightLinear, FourierWeightMLP
from neuron.fourier_conv import FourierInitConv2d, FourierCNN

__version__ = "0.3.0"
__all__ = [
    "FourierInitLinear", "FourierMLP", "FourierResMLP",
    "FourierWeightLinear", "FourierWeightMLP",
    "FourierInitConv2d", "FourierCNN",
]
