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
        eta,
        lmbda=0.0,
        gamma=0.0, # moment
        # possible alternatives are sigmoid, relu, leaky relu...
        activation_hidden="sigmoid",
        # Determines output activation func., ie. can be softmax if classification task
        activation_out="linear",
        task="regression"  # should check if activation function makes sense, how many output nodes there are, the cost function
    ):

        self.X_all = X
        self.y_all = y
        self.X = X
        self.y = y
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]

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

        if activation_out == "linear":
            self.activation_out = linear
            self.activation_out_derivative = linear_derivative
            # some "assert task=regression" or something like that?

        # should set Cost function and its derivative, as they vary between classification/regression AND regularization
        self.lmbda = lmbda
        # parameters for SGD
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eta = eta  # learning rate
        self.gamma = gamma # moment

        self.initialize_biases_and_weights()

    def initialize_biases_and_weights(self):
        self.layer = []
        self.layer.append(Layer(n_in=self.n_features,
                                n_out=self.n_hidden_neurons[0]))

        for l in range(1, self.n_hidden_layers):
            self.layer.append(Layer(n_in=self.n_hidden_neurons[l-1], 
                                    n_out=self.n_hidden_neurons[l]))

        self.layer.append(Layer(n_in=self.n_hidden_neurons[-1], 
                                n_out=1))
        
        # we initialized n hidden layers, as well as the weights and biases for the output layer

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
        error = self.prediction - self.y # TODO: assert that y has shapes (n,1) for regression
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

    def predict(self):
        self.X = self.X_all
        self.y = self.y_all
        self.feed_forward()

        return self.prediction

    def train(self):
       
        n_batches = int(self.n_inputs/self.batch_size) # num. batches
        eta0 = self.eta

        def learning_schedule(t, eta):
            alpha = t / (self.n_epochs*n_batches)
            return (1-alpha) * eta0 + alpha * eta

        eta = eta0

        for epoch in range(1, self.n_epochs+1):
            
            for i in range(n_batches):
                random_index = self.batch_size*np.random.randint(n_batches)
                self.X = self.X_all[random_index:random_index+self.batch_size] 
                self.y = self.y_all[random_index:random_index+self.batch_size]

                #Compute the gradient using the data in minibatch k
                eta = learning_schedule(t = epoch * n_batches + i, eta = eta)

                self.feed_forward()
                self.backpropagation()

                for l in range(len(self.layer)):
                    self.layer[l].velocity_weights = self.gamma * self.layer[l].velocity_weights -eta * self.layer[l].dWeights
                    self.layer[l].velocity_bias = self.gamma * self.layer[l].velocity_bias -eta * self.layer[l].dBias

                    #self.layer[l].weights += -eta * self.layer[l].dWeights
                    #self.layer[l].bias += -eta * self.layer[l].dBias

                    self.layer[l].weights += self.layer[l].velocity_weights
                    self.layer[l].bias += self.layer[l].velocity_bias
        
      
