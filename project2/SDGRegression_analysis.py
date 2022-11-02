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
batch_sizes = [2, 4, 8, 16, 32, 64, 128]
epochs = np.arange(100, 1001, step = 100)
learning_rates = np.logspace(-10, -1, 10)
lambda_values = np.logspace(-10, -1, 10)

# strategy:
## first fit epoch and learning rate
## Then use optimal values to find smallest reasonable batch size
## Then use epoch as batch size optimal values and do grid search/find optimal eta
## Finally, consider learning rate/optimization schemes for improving the model further
## Note that, because we use a training schedule, the number of epochs should not make that big a difference

# first we look at how the number of epochs and initial training rate affect the model
# we use a batch size of 20
"""
mse_values1, r2_values1 = [], []
mse_values2, r2_values2 = [], []
mse_values3, r2_values3 = [], []

for n_epochs in epochs:
    print("Cross validating using n_epochs = ", n_epochs)
    # eta0 = 1e0
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-1)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values1.append(mse)
    r2_values1.append(r2)
    # eta0 = 1e-1
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-2)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values2.append(mse)
    r2_values2.append(r2)
    # eta0 = 1e-2
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-3)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values3.append(mse)
    r2_values3.append(r2)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, mse_values1, label = "$\eta_1 = 0.1$")
plt.plot(epochs, mse_values2, label = "$\eta_2 = 0.01$")
plt.plot(epochs, mse_values3, label = "$\eta_3 = 0.001$")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figs/MSE_epoch_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, r2_values1, label = "$\eta_1 = 0.1$")
plt.plot(epochs, r2_values2, label = "$\eta_2 = 0.01$")
plt.plot(epochs, r2_values3, label = "$\eta_3 = 0.001$")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("figs/R2_epoch_1dpoly.pdf")

"""
# results of the above analysis is that eta = 0.1 and epoch = 500 is a good starting point

eta0 = 0.1
n_epochs = 500

# now we look at how the mini-batch size affect the model, using 500 epochs and eta=0.1
mse_values, r2_values = [], []

for m in batch_sizes:
    print("Cross validating using mini-batch size m = ", m)
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=m, gamma=0, optimization=None, eta0=eta0)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values.append(mse)
    r2_values.append(r2)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(batch_sizes, mse_values, label = "$\eta_1 = 0.1,$ n_epoch$=500$")
plt.legend()
plt.xlabel("Mini-batch size")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("figs/MSE_batchsize_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(batch_sizes, r2_values, label = "$\eta_1 = 0.1,$ n_epoch$=500$")
plt.legend()
plt.xlabel("Mini-batch size")
plt.ylabel("$R^2$")
plt.tight_layout()
plt.savefig("figs/R2_batchsize_1dpoly.pdf")

