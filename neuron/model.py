"""High-level Model API — train and evaluate Fourier networks."""

import numpy as np
import time
from neuron.fourier import FourierNet
from neuron.losses import MSELoss, CrossEntropyLoss
from neuron.optim import SGD, Adam, CosineAnnealingLR


class Model:
    """High-level training interface for FourierNet.

    Example
    -------
    >>> model = Model([784, 256, 10], k=128)
    >>> model.train(X_train, y_train, epochs=20, batch_size=64)
    >>> acc = model.evaluate(X_test, y_test)
    """

    def __init__(self, layer_dims, k=64, loss='cross_entropy',
                 optimizer='adam', lr=0.001, scale=0.1,
                 learn_freq=False, use_cos=True):
        self.net = FourierNet(
            layer_dims=layer_dims,
            k=k,
            scale=scale,
            learn_freq=learn_freq,
            use_cos=use_cos,
        )

        # Loss
        if loss == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss()
        elif loss == 'mse':
            self.loss_fn = MSELoss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

        # Optimizer
        if optimizer == 'adam':
            self.opt = Adam(lr=lr)
        elif optimizer == 'sgd':
            self.opt = SGD(lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.history = {'loss': [], 'acc': []}

    def train(self, X, y, epochs=10, batch_size=32,
              val_X=None, val_y=None, scheduler=None, verbose=True):
        """Train the model.

        Parameters
        ----------
        X : ndarray, shape (N, input_dim)
        y : ndarray, shape (N,) for class indices or (N, output_dim) for one-hot
        epochs : int
        batch_size : int
        val_X, val_y : optional validation data
        scheduler : optional LR scheduler
        verbose : bool
        """
        N = X.shape[0]

        # Convert y to one-hot if needed for cross-entropy
        if self.loss_fn.__class__.__name__ == 'CrossEntropyLoss' and y.ndim == 1:
            n_classes = self.net.layer_dims[-1]
            y_onehot = np.zeros((N, n_classes), dtype=np.float32)
            y_onehot[np.arange(N), y.astype(int)] = 1
            targets = y_onehot
        else:
            targets = y

        print(f"🧠 neuron — Fourier Network Training")
        print(f"   Architecture: {self.net.layer_dims}")
        print(f"   Fourier components: {self.net.k} | cos: {self.net.use_cos}")
        print(f"   Stored params: {self.net.param_count():,}")
        print(f"   Virtual params: {self.net.virtual_param_count():,}")
        print(f"   Compression: {self.net.compression_ratio():.1f}x")
        print(f"   Samples: {N} | Batch: {batch_size} | Epochs: {epochs}")
        print()

        for epoch in range(epochs):
            t0 = time.time()

            # Shuffle
            perm = np.random.permutation(N)
            X_shuf = X[perm]
            y_shuf = targets[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                # Forward
                output = self.net.forward(xb)

                # Loss
                loss = self.loss_fn.forward(output, yb)
                epoch_loss += loss

                # Backward (loss gradient)
                grad = self.loss_fn.backward()

                # Network backward (Fourier backprop)
                grads = self.net.backward(grad)

                # Collect all params and grads
                all_params = list(self.net.alpha) + list(self.net.alpha_bias)
                all_grads = list(grads['alpha']) + list(grads['alpha_bias'])

                if self.net.learn_freq:
                    all_params += [self.net.omega.reshape(-1), self.net.phi.reshape(-1)]
                    all_grads += [grads['omega'].reshape(-1), grads['phi'].reshape(-1)]

                # Optimizer step
                self.opt.step(all_params, all_grads)
                n_batches += 1

            # LR scheduling
            if scheduler is not None:
                scheduler.step()

            epoch_loss /= n_batches
            self.history['loss'].append(epoch_loss)

            # Validation accuracy
            acc = None
            if val_X is not None:
                acc = self.evaluate(val_X, val_y)
                self.history['acc'].append(acc)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                msg = f"   Epoch {epoch:3d} | Loss: {epoch_loss:.6f}"
                if acc is not None:
                    msg += f" | Acc: {acc:.1f}%"
                msg += f" | {time.time()-t0:.2f}s"
                print(msg)

        # Cache weights for fast inference
        self.net.cache_weights()

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Predict class indices or output values."""
        if self.net._weight_cache is not None:
            forward_fn = self.net.forward_cached
        else:
            forward_fn = self.net.forward

        outputs = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            out = forward_fn(X[start:end])
            outputs.append(out)
        return np.concatenate(outputs, axis=0)

    def evaluate(self, X, y, batch_size=256):
        """Evaluate accuracy on test data."""
        output = self.predict(X, batch_size)
        pred = np.argmax(output, axis=-1)

        if y.ndim > 1 and y.shape[-1] > 1:
            target = np.argmax(y, axis=-1)
        else:
            target = y.astype(int)

        return float(np.mean(pred == target)) * 100

    def save(self, path: str):
        """Save model to file."""
        self.net.save(path)

    @classmethod
    def load(cls, path: str) -> 'Model':
        """Load model from file."""
        net = FourierNet.load(path)
        model = cls.__new__(cls)
        model.net = net
        model.loss_fn = CrossEntropyLoss()
        model.opt = Adam()
        model.history = {'loss': [], 'acc': []}
        return model
