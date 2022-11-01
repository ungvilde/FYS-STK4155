import numpy as np
from common import *

from GridSearch import *

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

lambda_values = np.logspace(-10, -1, 2)
eta_values = np.logspace(-8, -1, 2)

np.random.seed(123)

print("Test with classification task...")
X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

#results = GridSearch_FFNN_classifier(Xtrain, ytrain, lambda_values, eta_values, activation_hidden="sigmoid")
#print(results)

#results = GridSearch_LogReg(Xtrain, ytrain, lambda_values, eta_values)
#print(results)

print("Test with regression task...")
# data set for testing
n = 1000
x = np.random.rand(n,1)
y = 2.0 + 3*x + 4*x*x + x**3 #+ 0.01*np.random.randn(n,1)
X = np.c_[x, x*x, x*x*x]
# scale data and leave a final part of the data set out for final testing of the model.
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Xscaled = np.c_[np.ones((n,1)), Xscaled]
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size=0.2)

# do grid search across the given lambda and eta values
# we compute both mse and r2 using 5-fold cross validation, k=5
print("LinearRegression with SGD results:")
results_mse, results_r2 = GridSearch_LinReg(Xtrain, ytrain, lambda_values, eta_values, solver="sgd", k=5)
print("MSE values:")
print(results_mse)
print("R2 values")
print(results_r2)

eta_values = np.logspace(-8, -3, 2)

print("FFNN results:")
results_mse, results_r2 = GridSearch_FFNN_reg(Xtrain, ytrain, lambda_values, eta_values, k=5, activation_hidden="sigmoid")
print("MSE values:")
print(results_mse)
print("R2 values")
print(results_r2)


