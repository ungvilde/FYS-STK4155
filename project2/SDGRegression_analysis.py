import numpy as np
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression
from ResampleMethods import CrossValidation_regression
from common import *

# Here we do Linear regression using the Stochastic Gradient Descent (SGD) algorithm: 
# We will consider the following parameters/aspects of the algorithm:
# - model complexity
# - learning rate
# - regularization
# - epochs and batch size
# - optimization shcemes
# and possibly: momentum coefficients

x, y = readfile_1dpoly("datasets/Poly_degree3_sigma0.1_N1000.txt")

# set up design matrix
X = DesignMatrix(x, degree=3)
n = X.shape[0]

# scale and split data
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
Xscaled = np.c_[np.ones((n,1)), Xscaled] # add intercept column to design matrix
Xtrain, Xtest, ytrain, ytest = train_test_split(Xscaled, y, test_size=0.2)

# We will consider the following range of parameters
batch_sizes = [5, 10, 15, 20, 30, 40, 50]
epochs = np.arange(100, 1500, step = 100)
learning_rates = np.logspace(-10, -1, 10)
lambda_values = np.logspace(-10, -1, 10)

# first we look at how the number of epochs affect the model
mse_valuesSGD, r2_valuesSGD = [], []
mse_valuesGD, r2_valuesGD = [], []
for n_epochs in epochs:
    print("Cross validating using n_epochs = ", n_epochs)
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=10, gamma=0, optimization=None, eta0=1e-3)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_valuesSGD.append(mse)
    r2_valuesSGD.append(r2)

    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, gamma=0, optimization=None, eta0=1e-3)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_valuesGD.append(mse)
    r2_valuesGD.append(r2)

print(epochs)
print("With SGD:")
print(mse_valuesSGD)
print(r2_valuesSGD)
print("With GD:")
print(mse_valuesGD)
print(r2_valuesGD)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, mse_valuesSGD, label = "SGD")
plt.plot(epochs, mse_valuesGD, label = "GD")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figs/MSE_epoch_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, r2_valuesSGD, label = "SGD")
plt.plot(epochs, r2_valuesGD, label = "GD")
plt.xlabel("Epochs")
plt.ylabel("$R^2$")
plt.legend()
plt.tight_layout()
plt.savefig("figs/R2_epoch_1dpoly.pdf")

# find SGD gets best results for epoch = 500, both R2 and MSE are best then
# GD wants even more epochs, which is not feasable or interesting for further study
# should compute MSE+R2 value with error next time

# second we look at how the mini-batch size affect the model, using 500 epochs
mse_values1, r2_values1 = [], []
mse_values2, r2_values2 = [], []
mse_values3, r2_values3 = [], []

for m in batch_sizes:
    print("Cross validating using mini-batch size m = ", m)
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=1000, batch_size=m, gamma=0, optimization=None, eta0=1e-1)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values1.append(mse)
    r2_values1.append(r2)

    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=1000, batch_size=m, gamma=0, optimization=None, eta0=1e-2)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values2.append(mse)
    r2_values2.append(r2)

    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=1000, batch_size=m, gamma=0, optimization=None, eta0=1e-3)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values3.append(mse)
    r2_values3.append(r2)

print(batch_sizes)
print(f"Eta = {1e-1}")
print(mse_values1)
print(r2_values1)

print(f"Eta = {1e-2}")
print(mse_values2)
print(r2_values2)

print(f"Eta = {1e-3}")
print(mse_values3)
print(r2_values3)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(batch_sizes, mse_values1, label = "$\eta_1 = 10^{-1}$")
plt.plot(batch_sizes, mse_values2, label = "$\eta_2 = 10^{-2}$")
plt.plot(batch_sizes, mse_values3, label = "$\eta_3 = 10^{-3}$")
plt.legend()
plt.xlabel("Mini-batch size")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("figs/MSE_batchsize_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(batch_sizes, r2_values1, label = "$\eta_1 = 10^{-1}$")
plt.plot(batch_sizes, r2_values2, label = "$\eta_2 = 10^{-2}$")
plt.plot(batch_sizes, r2_values3, label = "$\eta_3 = 10^{-3}$")
plt.legend()
plt.xlabel("Mini-batch size")
plt.ylabel("$R^2$")
plt.tight_layout()
plt.savefig("figs/R2_batchsize_1dpoly.pdf")

