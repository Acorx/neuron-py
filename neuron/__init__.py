"""neuron 🧠 — CPU-Native AI via Fourier Weight Generation

Instead of storing billions of weights, neuron generates them from a tiny
Fourier formula. A few hundred coefficients can initialize — and eventually
replace — millions of parameters.
"""

from neuron.fourier_init import FourierInitLinear, FourierMLP, FourierResMLP

__version__ = "0.2.0"
__all__ = ["FourierInitLinear", "FourierMLP", "FourierResMLP"]
