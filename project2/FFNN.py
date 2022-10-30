import numpy as np
from Layer import Layer

from activation_functions import *


class FFNN:
    def __init__(
        self,
        X,
        y,
        n_hidden_neurons,  # list of num. neurons per hidden layer
        n_epochs,
        batch_size,
        eta, # learning rate (initial)
        lmbda=0.0, # regularization
        gamma=0.0, # moment variable
        activation_hidden="sigmoid", # possible alternatives are sigmoid, relu, leaky relu...
        task="regression"  # determines activation function, how many output nodes there are, the cost function
    ):

        # dataset parameters
        self.X_all = X # contains the whole data set
        self.y_all = np.c_[y]
        self.X = X # will be updated depending on batches
        self.y = np.c_[y]
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]

        # architecture parameters
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = len(n_hidden_neurons)

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
        
        if task == "regression":
            self.n_output = 1
            self.activation_out = linear
            self.activation_out_derivative = linear_derivative

        elif task == "classification":
            self.n_output = 1 # could be made flexible, but fine for now
            self.activation_out = sigmoid
            self.activation_out_derivative = sigmoid_derivative

        # regularization
        self.lmbda = lmbda

        # parameters for SGD
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eta = eta  # learning rate
        self.gamma = gamma # moment

        # fill weights and biases with small random numbers
        self.initialize_biases_and_weights()

    def initialize_biases_and_weights(self):
        # we initialize n hidden layers, as well as the weights and biases for the output layer

        self.layer = []
        self.layer.append(Layer(n_in=self.n_features,
                                n_out=self.n_hidden_neurons[0]))

        for l in range(1, self.n_hidden_layers):
            self.layer.append(Layer(n_in=self.n_hidden_neurons[l-1], 
                                    n_out=self.n_hidden_neurons[l]))

        self.layer.append(Layer(n_in=self.n_hidden_neurons[-1], 
                                n_out=self.n_output))
        

    def feed_forward(self):
        z = self.X

        for l in range(self.n_hidden_layers):
            weights = self.layer[l].weights
            bias = self.layer[l].bias

            z = z @ weights + bias
            a = self.activation_hidden(z) # apply activation function element-wise
            self.layer[l].z = z
            self.layer[l].activation = a # h_1

        # the output layer
        weights = self.layer[-1].weights
        bias = self.layer[-1].bias
        z = a @ weights + bias
        self.layer[-1].z = z
        self.prediction = self.activation_out(z)

    def backpropagation(self):
        """
        Compute the gradients of weights and biases at each layer, starting at the output layer 
        and working backwards towards the first hidden layer. Gradients of each layer are stored
        in the Layer-objects for each layer.
        """
        # compute gradient wrt output layer
        error = self.prediction - self.y
        z = self.layer[-1].z
        previous_activation = self.layer[-2].activation
        weights = self.layer[-1].weights
        bias = self.layer[-1].bias
    
        gradient = np.multiply(error, self.activation_out_derivative(z))
    
        # Update gradient of weights and biases of output layer, including regularization term
        self.layer[-1].dBias = np.sum(gradient,axis=0) + self.lmbda * bias
        self.layer[-1].dWeights = previous_activation.T @ gradient + self.lmbda * weights
         
        # propagate gradient wrt lower-level hidden layer's activations
        gradient = gradient @ weights.T
        
        for l in range(self.n_hidden_layers - 1, 0, -1): 
            weights = self.layer[l].weights
            bias = self.layer[l].bias
            z = self.layer[l].z
            previous_activation = self.layer[l-1].activation

            # Compute gradient of weights and biases in hidden layer l, including regularization term
            self.layer[l].dBias = np.sum(gradient,axis=0) + self.lmbda * bias
            self.layer[l].dWeights = previous_activation.T @ gradient  + self.lmbda * weights
  
            gradient = gradient @ weights.T
        
        l = 0 
        weights = self.layer[l].weights
        bias = self.layer[l].bias
        z = self.layer[l].z
        previous_activation = self.X
        self.layer[l].dBias = np.sum(gradient,axis=0) + self.lmbda * bias
        self.layer[l].dWeights = previous_activation.T @ gradient  + self.lmbda * weights
        self.gradient = gradient @ weights.T

    def predict(self, X):
        self.X = X
        self.feed_forward()

        return self.prediction

    def train(self):
       
        n_batches = int(self.n_inputs/self.batch_size) # num. batches
        eta0 = self.eta

        def learning_schedule(t, eta):
            alpha = t / (self.n_epochs*n_batches)
            return (1-alpha) * eta0 + alpha * eta

        eta = eta0
        indeces = np.arange(self.n_inputs)

        for epoch in range(1, self.n_epochs+1):
            
            for i in range(n_batches):
                batch_indeces = np.random.choice(indeces, size=self.batch_size, replace=True)
                self.X = self.X_all[batch_indeces]
                self.y = self.y_all[batch_indeces]
                
                #Compute the gradient using the data in minibatch k
                eta = learning_schedule(t = epoch * n_batches + i, eta = eta)

                self.feed_forward()
                self.backpropagation()

                for l in range(len(self.layer)):
                    # compute change with momentum
                    self.layer[l].velocity_weights = self.gamma * self.layer[l].velocity_weights - eta * self.layer[l].dWeights
                    self.layer[l].velocity_bias = self.gamma * self.layer[l].velocity_bias - eta * self.layer[l].dBias
                    
                    # update gradients
                    self.layer[l].weights += self.layer[l].velocity_weights
                    self.layer[l].bias += self.layer[l].velocity_bias
        
      
