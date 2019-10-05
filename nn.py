import numpy as np

class NN:
    def __init__(self, layer_sizes, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = [np.zeros((m, n)) for m, n in zip(layer_sizes[1:], layer_sizes)]
        self.layer_sizes = layer_sizes
        self.seed = seed
    
    def predict(self, X):
        out = X
        for W in self.weights:
            Z = out @ W.T
            out = np.tanh(Z)
        if out.shape[0] == 1 and len(out.shape) == 1:
            return out.item()
        return out

    def set_weights(self, weights, copy=False):
        if copy:
            self.weights = [np.copy(l) for l in weights]
        else:
            self.weights = weights
        
    def get_weights(self, copy=False):
        if copy:
            return [np.copy(l) for l in self.weights]
        return self.weights
