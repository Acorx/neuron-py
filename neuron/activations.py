"""Activation function wrappers."""

import numpy as np
from neuron.fourier import ACTIVATIONS


class Activation:
    """Base activation class."""
    name = "base"
    def __call__(self, x): raise NotImplementedError
    def deriv(self, x): raise NotImplementedError

class ReLU(Activation):
    name = "relu"
    def __call__(self, x): return ACTIVATIONS['relu'][0](x)
    def deriv(self, x): return ACTIVATIONS['relu'][1](x)

class GELU(Activation):
    name = "gelu"
    def __call__(self, x): return ACTIVATIONS['gelu'][0](x)
    def deriv(self, x): return ACTIVATIONS['gelu'][1](x)

class Tanh(Activation):
    name = "tanh"
    def __call__(self, x): return ACTIVATIONS['tanh'][0](x)
    def deriv(self, x): return ACTIVATIONS['tanh'][1](x)

class Sigmoid(Activation):
    name = "sigmoid"
    def __call__(self, x): return ACTIVATIONS['sigmoid'][0](x)
    def deriv(self, x): return ACTIVATIONS['sigmoid'][1](x)
