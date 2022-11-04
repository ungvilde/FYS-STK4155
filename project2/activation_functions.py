import numpy as np


# sigmoid function
def sigmoid(x):
    x = np.clip(x, -500, 500 ) # to deal with overflow, but not sure if it is a bad idea
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

