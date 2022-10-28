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
y = 2.0+3*x +4*x*x # +np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x*x]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
y = np.c_[y]
XT_X = X.T @ X

eta = 0.001

network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="sigmoid")
network.train()
pred = network.predict()
mse = MSE(y.ravel(), pred.ravel())
print("MSE = ", mse)

plt.plot(x, y, 'o', label = "Target")
plt.plot(x, pred, '.', label ="Estimate with FFNN")

dnn = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', alpha=0, learning_rate_init=eta, max_iter=200, solver="sgd")
dnn.fit(X,y.ravel())
pred = dnn.predict(X)
mse = mean_squared_error(pred, y.ravel())
print("MSE with Sklearn = ", mse)
plt.plot(x, pred, '.', label = "Sklearn")
plt.legend()
#plt.show()

network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="reLU")
network.train()
pred = network.predict()
mse = MSE(y.ravel(), pred.ravel())
print("MSE = ", mse)

network = FFNN(X, y, n_hidden_neurons=[100], n_epochs=100, batch_size=20, eta = eta, lmbda = 0, gamma=0.9, activation_hidden="leaky_reLU")
network.train()
pred = network.predict()
mse = MSE(y.ravel(), pred.ravel())
print("MSE = ", mse)