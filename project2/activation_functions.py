import numpy as np


# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    f = 1/(1 + np.exp(-x))
    return 1 - f

#ReLU alternatives
def reLU(x):
    return np.max(0, x)

def reLU_derivative(x):
    if x > 0:
        value = 1
    else:
        value = 0

def leakyReLU(x):
    if x > 0:
        value = x
    else:
        value = 0.01*x
    return value

def leaky_reLU_derivative(x):
    if x > 0:
        value = 1
    else:
        value = 0.01
        return value

# activation for output with regression tasks
def linear(x):
    return x

def linear_derivative(x):
    return 1.

# TODO: add softmax and something for binary classifiers