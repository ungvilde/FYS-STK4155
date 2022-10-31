import numpy as np

from common import *
from LinearRegression import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(123)
n = 1000
x = np.random.rand(n,1)
y = 2.0 + 3*x + 4*x*x + x**3 #+ 0.01*np.random.randn(n,1)
X = np.c_[x, x*x, x*x*x]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Xscaled = np.c_[np.ones((n,1)), Xscaled]
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size=0.2)

linreg = LinearRegression(lmbda=0.0, solver="analytic")
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE analytic = ", mse)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="gd", gamma=0.9)
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE with GD = ", mse)

linreg = LinearRegression(lmbda=0, max_iter=5000, eta0=1e-1, solver="gd", gamma=0.9, optimization="adagrad")
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE with GD and adagrad =", mse)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="sgd", gamma=0.9, optimization="adam")
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE with SGD and adam = ", mse)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="sgd", gamma=0.9, optimization="adagrad")
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE with SGD and adagrad = ", mse)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="sgd", gamma=0.9, optimization="RMSprop")
linreg.fit(Xtrain, ytrain)
predicted = linreg.predict(Xtest)
mse = MSE(predicted, ytest)
print("MSE with SGD and RMSprop = ", mse)