import numpy as np
import matplotlib.pyplot as plt


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

y = 2.0+3*x +4*x*x # +np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x*x]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
y = np.c_[y]

eta = 0.001

# testing for basic FFNN 
network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="sigmoid")
network.train()
pred = network.predict()
mse = MSE(y.ravel(), pred.ravel())
print("MSE with basic FFNN= ", mse)

plt.plot(x[sorted_inds], y[sorted_inds], label = "Target")
plt.plot(x[sorted_inds], pred[sorted_inds], label ="Estimate with FFNN, Sigmoid")

# now compare w SKlearn
dnn = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', alpha=0, learning_rate_init=eta, max_iter=200, solver="sgd")
dnn.fit(X,y.ravel())
pred = dnn.predict(X)
mse = mean_squared_error(pred, y.ravel())
print("MSE with Sklearn = ", mse)

plt.plot(x[sorted_inds], pred[sorted_inds], label = "Sklearn")

# also check to see that ReLU works
network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="reLU")
network.train()
pred = network.predict()

plt.plot(x[sorted_inds], pred[sorted_inds], label ="Estimate with FFNN, ReLU")

mse = MSE(y.ravel(), pred.ravel())
print("MSE with reLU = ", mse)

# finally leaky ReLU
network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="leaky_reLU")
network.train()
pred = network.predict()
mse = MSE(y.ravel(), pred.ravel())
print("MSE with leaky reLU = ", mse)

plt.plot(x[sorted_inds], pred[sorted_inds], label ="Estimate with FFNN, Leaky ReLU")
plt.legend()
plt.show()