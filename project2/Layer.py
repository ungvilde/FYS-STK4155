import numpy as np

class Layer:
    def __init__(
        self,
        n_in,
        n_out
        ):

        self.n_neurons = n_out
        self.n_input = n_in
        self.weights = np.random.randn(n_in, n_out)
        self.bias = np.zeros(n_out) + 0.01
        self.activation = None
        self.z = None

        # For the gradient of the biases and weights in the layer
        self.dBias = None
        self.dWeights = None
