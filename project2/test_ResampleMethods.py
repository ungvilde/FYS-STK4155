import numpy as np

from ResampleMethods import *
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
n = 1000
x = np.random.rand(n,1)
y = 2.0 + 3*x + 4*x*x + x**3 #+ 0.01*np.random.randn(n,1)

X = np.c_[x, x**2, x**3]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Xscaled = np.c_[np.ones((n,1)), Xscaled]
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size=0.2)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="analytic", gamma=0.9, optimization="adam")
mse, r2 = Bootstrap(linreg, Xtrain, ytrain, B=50)
print("Bootstrapped MSE = ", mse, " and R2 = ", r2)
linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="analytic", gamma=0.9, optimization="adam")
mse, r2 = CrossValidation(linreg, Xtrain, ytrain, k=5)
print("Cross validated MSE = ", mse, " and R2 = ", r2)
