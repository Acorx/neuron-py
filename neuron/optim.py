"""Optimizers for neuron — SGD and Adam."""

import numpy as np
from typing import List, Dict


class SGD:
    """Stochastic Gradient Descent with momentum and weight decay."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = None

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                if self.nesterov:
                    p -= self.lr * (g + self.momentum * self.velocity[i])
                else:
                    p -= self.lr * self.velocity[i]
            else:
                p -= self.lr * g


class Adam:
    """Adam optimizer with weight decay."""

    def __init__(self, lr: float = 0.001, betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler."""

    def __init__(self, optimizer, t_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.t_max = t_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        progress = self.last_epoch / self.t_max
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))
