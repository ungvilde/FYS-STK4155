from pickletools import optimize
import numpy as np

from common import *
from LinearRegression import LinearRegression
from FFNN import FFNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Make data.
n=100
x = np.random.rand(n)
y = np.random.rand(n)
x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y)

X = FrankeDesignMatrix(x.ravel(), y.ravel(), degree=7)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = np.array(scaler.transform(X))
Xscaled[:,0] = np.ones(n*n)
Xtrain, Xtest, ztrain, ztest = train_test_split(Xscaled, z.ravel(), test_size=0.2)

# solved analytically
linreg = LinearRegression(solver="analytic", lmbda=0.0)
linreg.fit(Xtrain, ztrain)
predictions = linreg.predict(Xtest)
print(predictions.shape)
mse = MSE(predictions, ztest)
print("MSE with LinearRegression = ", mse)

# solved with GD Regression
linreg = LinearRegression(solver="gd", lmbda=0.0, gamma=0.9, max_iter=1000)
linreg.fit(Xtrain, ztrain)
predictions = linreg.predict(Xtest)
mse = MSE(predictions, ztest)
print("MSE with GD LinearRegression = ", mse)

# Solved with SGD Regression
linreg = LinearRegression(solver="sgd", lmbda=0.0, gamma=0.9, batch_size=200, optimization="adam", max_iter=1000, eta0=1e-4)
linreg.fit(Xtrain, ztrain)
predictions = linreg.predict(Xtest)
mse = MSE(y_pred=predictions, y = ztest)
print("MSE with SGD LinearRegression = ", mse)

# Solved with FFNN
network = FFNN(lmbda=0.0, n_hidden_neurons=[100], batch_size=20, 
n_epochs=200, eta=1e-4, gamma=0.9, activation_hidden="sigmoid")
network.fit(Xtrain, ztrain)
predictions = network.predict(Xtest)
mse = MSE(predictions, ztest)
print("MSE with FFNN = ", mse)

# Finally, compare with SKlearn
dnn = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', 
alpha=0, learning_rate_init=1e-4, max_iter=200, solver="sgd", learning_rate="constant")
dnn.fit(Xtrain, ztrain)
pred = dnn.predict(Xtest)
mse = MSE(pred, ztest)
print("MSE with Sklearn = ", mse)