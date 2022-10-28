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
        #self.activation_function
        self.activation = None
        self.z = None
        self.error = None

        # For the gradient of the biases and weights in the layer
        self.dBias = None
        self.dWeights = None
    
    def set_weights(self, W):
        self.weights = W
    
    def set_bias(self, b):
        self.bias = b
    
