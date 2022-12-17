import numpy as np

class Layer:
    def __init__(
        self,
        n_in,
        n_out,
        initialization = "standard"
        ):

        self.initialization = initialization
        self.n_neurons = n_out
        self.n_input = n_in

        if self.initialization == "normalized":
            r = np.sqrt(6 / (n_in + n_out))
            self.weights = np.random.uniform(low= -r, high=r, size=(n_in, n_out))
        elif self.initialization == "standard":
            self.weights = np.random.randn(n_in, n_out)
        else:
            raise Exception("Invalid initialization scheme.")
        
        self.bias = np.zeros(n_out) + 0.01
        self.activation = None
        self.z = None

        # For the gradient of the biases and weights in the layer
        self.dBias = None
        self.dWeights = None
        self.velocity_bias = 0
        self.velocity_weights = 0
