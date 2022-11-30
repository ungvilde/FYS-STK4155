import numpy as np


# sigmoid function
def sigmoid(x):
    x = np.clip(x, -500, 500 )
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    x = np.clip(x, -500, 500 )
    f = 1/(1 + np.exp(-x))
    return 1 - f

#ReLU alternatives
def reLU(x):
    return np.maximum(0, x)

def reLU_derivative(x):
    return 1. * (x > 0)

def leakyReLU(x):
    return np.maximum(0, x) + 0.01 * np.minimum(0, x)

def leaky_reLU_derivative(x):
    return 1 * (x > 0) + 0.01 * (x < 0)

# activation for output with regression tasks
def linear(x):
    return x

def linear_derivative(x):
    return 1.0

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
            self.weights = np.random.uniform(low=-r, high=r, size=(n_in, n_out))
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
        self.velocity_bias = 0 # for computing momentum
        self.velocity_weights = 0

class FFNN:
    def __init__(
        self,
        hidden_layer_sizes,  # list of num. neurons per hidden layer
        n_epochs,
        batch_size,
        eta0, # learning rate (initial)
        lmbda=0.0, # regularization
        gamma=0.0, # moment variable
        activation_hidden="sigmoid", # possible alternatives are sigmoid, relu, leaky relu...
        initialization = "standard"
    ):

        # architecture parameters
        self.n_hidden_neurons = hidden_layer_sizes
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.initalization = initialization

        # set the activation function
        if activation_hidden == "sigmoid":
            self.activation_hidden = sigmoid
            self.activation_hidden_derivative = sigmoid_derivative
        elif activation_hidden == "reLU":
            self.activation_hidden = reLU
            self.activation_hidden_derivative = reLU_derivative
        elif activation_hidden == "leaky_reLU":
            self.activation_hidden = leakyReLU
            self.activation_hidden_derivative = leaky_reLU_derivative
        else:
            raise Exception("Invalid activation function.")
        
        self.n_output = 2
        self.activation_out = linear
        self.activation_out_derivative = linear_derivative

        # regularization
        self.lmbda = lmbda

        # parameters for SGD
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eta0 = eta0  # learning rate
        self.gamma = gamma # moment

    def initialize_biases_and_weights(self):
        # we initialize n hidden layers, as well as the weights and biases for the output layer

        self.layer = []
        self.layer.append(Layer(n_in=self.n_features,
                                n_out=self.n_hidden_neurons[0],
                                initialization=self.initalization)
                                )

        for l in range(1, self.n_hidden_layers):
            self.layer.append(Layer(n_in=self.n_hidden_neurons[l-1], 
                                    n_out=self.n_hidden_neurons[l],
                                    initialization=self.initalization)
                                    )

        self.layer.append(Layer(n_in=self.n_hidden_neurons[-1], 
                                n_out=self.n_output,
                                initialization=self.initalization)
                                )
        

    def feed_forward(self):
        a = self.X

        z = np.zeros_like(a, dtype=np.float64)
        for l in range(self.n_hidden_layers):
            weights = self.layer[l].weights
            bias = self.layer[l].bias

            z = a @ weights + bias
            a = self.activation_hidden(z) # apply activation function element-wise
            self.layer[l].z = z
            self.layer[l].activation = a 

        # the output layer
        weights = self.layer[-1].weights
        bias = self.layer[-1].bias

        z = a @ weights + bias
        self.layer[-1].z = z
        self.prediction = self.activation_out(z)

    def backpropagation(self):
        """
        Compute the gradients of weights and biases at each layer, starting at the output layer 
        and working backwards towards the first hidden layer. Gradients of weights and biases are stored
        in the Layer-objects for their respective layer.
        """
        # compute gradient wrt. output layer
        error = self.prediction - self.y
        z = self.layer[-1].z
        previous_activation = self.layer[-2].activation
        weights = self.layer[-1].weights
        #bias = self.layer[-1].bias
    
        gradient = np.multiply(error, self.activation_out_derivative(z))
    
        # Update gradient of weights and biases of output layer, including regularization term
        self.layer[-1].dBias = np.sum(gradient,axis=0) #+ self.lmbda * bias
        self.layer[-1].dWeights = previous_activation.T @ gradient + self.lmbda * weights
         
        # propagate gradient wrt lower-level hidden layer's activations
        gradient = gradient @ weights.T
        
        for l in range(self.n_hidden_layers - 1, 0, -1): 
            weights = self.layer[l].weights
            #bias = self.layer[l].bias
            z = self.layer[l].z
            previous_activation = self.layer[l-1].activation

            # Compute gradient of weights and biases in hidden layer l, including regularization term
            self.layer[l].dBias = np.sum(gradient,axis=0) #+ self.lmbda * bias
            self.layer[l].dWeights = previous_activation.T @ gradient  + self.lmbda * weights
  
            gradient = gradient @ weights.T
        
        l = 0 
        weights = self.layer[l].weights
        #bias = self.layer[l].bias
        z = self.layer[l].z
        previous_activation = self.X
        self.layer[l].dBias = np.sum(gradient,axis=0) #+ self.lmbda * bias #might be better to only regularize weights, not biases
        self.layer[l].dWeights = previous_activation.T @ gradient  + self.lmbda * weights
        self.gradient = gradient @ weights.T

    def predict(self, X):
        self.X = X
        self.feed_forward()

        return self.prediction

    def fit(self, X, y):

        # dataset parameters
        self.X_all = X # contains the whole data set
        self.y_all = np.c_[y]
        self.X = X # will be updated depending on batches
        self.y = np.c_[y]
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        
        # fill weights and biases with small random numbers
        self.initialize_biases_and_weights()

        n_batches = int(self.n_inputs/self.batch_size) # num. batches

        # linearly decaying learning schedule, taken from Goodfellow
        def learning_schedule(t): # t is the iteration
            alpha = t / (self.n_epochs*n_batches) 
            return (1-alpha) * self.eta0 + alpha * self.eta0*0.01

        indeces = np.arange(self.n_inputs)

        for epoch in range(1, self.n_epochs+1):
            random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
            batches = np.array_split(random_indeces, n_batches) # split into batches
            
            for i in range(n_batches):
                self.X = self.X_all[batches[i]]
                self.y = self.y_all[batches[i]]
                
                # update learning rate
                eta = learning_schedule(t = epoch * n_batches + i)

                #Compute the gradient using the data in minibatch k
                self.feed_forward()
                self.backpropagation()

                for l in range(len(self.layer)):
                    # compute change with momentum
                    self.layer[l].velocity_weights = self.gamma * self.layer[l].velocity_weights - eta * self.layer[l].dWeights
                    self.layer[l].velocity_bias = self.gamma * self.layer[l].velocity_bias - eta * self.layer[l].dBias
                    
                    # update gradients
                    self.layer[l].weights += self.layer[l].velocity_weights
                    self.layer[l].bias += self.layer[l].velocity_bias