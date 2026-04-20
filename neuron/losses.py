"""Loss functions for neuron."""

import numpy as np


class MSELoss:
    """Mean Squared Error loss."""

    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        self.output = output
        self.target = target
        return float(np.mean((output - target) ** 2))

    def backward(self) -> np.ndarray:
        n = self.output.size
        return 2 * (self.output - self.target) / n


class CrossEntropyLoss:
    """Cross-entropy loss with built-in softmax.

    Expects raw logits as output, one-hot or class indices as target.
    """

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        self.logits = logits
        self.target = target
        self.probs = self._softmax(logits)

        # Handle both one-hot and index targets
        if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1):
            # Index targets
            if target.ndim == 2:
                target = target.flatten()
            n = logits.shape[0]
            loss = -np.log(self.probs[np.arange(n), target.astype(int)] + 1e-12)
            self._index_target = target.astype(int)
        else:
            # One-hot targets
            loss = -np.sum(target * np.log(self.probs + 1e-12), axis=-1)
            self._index_target = None

        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        grad = self.probs.copy()
        if self._index_target is not None:
            n = grad.shape[0]
            grad[np.arange(n), self._index_target] -= 1
            grad /= n
        else:
            grad -= self.target
            grad /= grad.shape[0]
        return grad

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
