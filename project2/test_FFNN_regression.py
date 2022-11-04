import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from FFNN import FFNN
from common import *
from activation_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error


np.random.seed(123)
n = 1000
x = np.random.rand(n,1)
sorted_inds = np.argsort(x, axis=0).ravel()

y = 2.0+3*x +4*x*x + 0.01*np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x*x]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

eta = 0.001

# testing for basic FFNN 
network = FFNN(n_hidden_neurons=[100], n_epochs=200, batch_size=20, 
eta = eta, lmbda = 0, gamma=0.9, activation_hidden="sigmoid", task = "regression")
network.fit(Xtrain, ytrain)
pred = network.predict(Xtest)
mse = MSE(ytest, pred)
print("MSE with basic FFNN = ", mse)

# now compare w SKlearn
dnn = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', alpha=0, learning_rate_init=eta, max_iter=200, solver="sgd")
dnn.fit(Xtrain, ytrain.ravel())
pred = dnn.predict(Xtest)
mse = mean_squared_error(pred, ytest.ravel())
print("MSE with Sklearn = ", mse)
