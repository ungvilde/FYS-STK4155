import numpy as np

from ResampleMethods import *
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from FFNN import FFNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


np.random.seed(123)

print("First test on regression tasks...")
n = 1000
x = np.random.rand(n,1)
y = 2.0 + 3*x + 4*x*x + x**3 #+ 0.01*np.random.randn(n,1)

X = np.c_[x, x**2, x**3]
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Xscaled = np.c_[np.ones((n,1)), Xscaled]
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size=0.2)

print("Linear regression with analytic solution:")
linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="analytic", gamma=0.9)
mse, r2 = Bootstrap(linreg, Xtrain, ytrain, B=50)
print("Bootstrapped MSE = ", mse, " and R2 = ", r2)

linreg = LinearRegression(lmbda=0, max_iter=1000, eta0=1e-3, solver="analytic", gamma=0.9, optimization="adam")
mse, r2  = CrossValidation_regression(linreg, Xtrain, ytrain, k=5)
print("Cross validated MSE = ", mse, " and R2 = ", r2)

print("Neural network:")
network = FFNN(eta=1e-4, n_hidden_neurons=[100], n_epochs=200, batch_size=20, lmbda=0, gamma=0.9, task="regression")
mse, r2  = CrossValidation_regression(network, Xtrain, ytrain, k=5)
print("Cross validated MSE = ", mse, " and R2 = ", r2)

print("Now test with classification tasks...")
X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

print("Neural network:")
network = FFNN(n_hidden_neurons=[100], task="classification", n_epochs=500, batch_size=20, eta=1e-4, lmbda=0.0, gamma=0.9, activation_hidden="reLU")
accuracy_score  = CrossValidation_classification(network, Xtrain, ytrain, k=5)
print("Cross validated accuracy score = ", accuracy_score)

print("Logistic regression with SGD solution:")
logreg = LogisticRegression(solver="sgd", lmbda=0.0, optimization="adam", eps=1e-3)
accuracy_score  = CrossValidation_classification(logreg, Xtrain, ytrain, k=5)
print("Cross validated accuracy score = ", accuracy_score)
