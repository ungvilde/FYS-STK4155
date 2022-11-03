import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression
from ResampleMethods import CrossValidation_regression
from common import *
from GridSearch import GridSearch_LinReg_epochs_batchsize, GridSearch_LinReg

sns.set_theme("notebook", "whitegrid")

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
batch_sizes = [5, 10, 20, 50, 100] #200]
epochs = [100, 200, 400, 600, 800, 1000]
learning_rates = np.logspace(-5, -1, 10)
lambda_values = np.logspace(-10, -1, 10)

# strategy:
## first fit epoch and learning rate
## Then use optimal values to find smallest reasonable batch size
## Then use optimal epoch and batch size, and do grid search/find optimal eta
## Finally, consider learning rate/optimization schemes for improving the model further
## Note that, because we use a training schedule, the number of epochs should not make that big a difference

# ------------------------------------------------------#

# first we look at how the number of epochs and initial training rate affect the model
# we use a batch size of 20

#results = GridSearch_LinReg_epochs_batchsize(X, y, eta=1e-1, batch_sizes=batch_sizes, n_epochs=epochs)
#print(results)

# mse_values1, r2_values1 = [], []
# mse_values2, r2_values2 = [], []
# mse_values3, r2_values3 = [], []

# for n_epochs in epochs:
#     print("Cross validating using n_epochs = ", n_epochs)
#     # eta0 = 1e0
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-1)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values1.append(mse)
#     r2_values1.append(r2)
#     # eta0 = 1e-1
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-2)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values2.append(mse)
#     r2_values2.append(r2)
#     # eta0 = 1e-2
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-3)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values3.append(mse)
#     r2_values3.append(r2)

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, mse_values1, label = "$\eta_1 = 0.1$")
# plt.plot(epochs, mse_values2, label = "$\eta_2 = 0.01$")
# plt.plot(epochs, mse_values3, label = "$\eta_3 = 0.001$")
# plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1], label="$\sigma^2=0.01$")
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/MSE_epoch_1dpoly.pdf")

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, r2_values1, label = "$\eta_1 = 0.1$")
# plt.plot(epochs, r2_values2, label = "$\eta_2 = 0.01$")
# plt.plot(epochs, r2_values3, label = "$\eta_3 = 0.001$")
# plt.hlines(y=1.0, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
# plt.xlabel("Epochs")
# plt.ylabel("$R^2$")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/R2_epoch_1dpoly.pdf")

# ------------------------------------------------------#

# # results of the above analysis is that eta = 0.1 and epoch = 500 is a good starting point

# eta0 = 0.1
# n_epochs = 500

# # now we look at how the mini-batch size affect the model, using 500 epochs and eta=0.1
# mse_values, r2_values = [], []

# for m in batch_sizes:
#     print("Cross validating using mini-batch size m = ", m)
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=m, gamma=0, optimization=None, eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values.append(mse)
#     r2_values.append(r2)

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(batch_sizes, mse_values, label = "$\eta_1 = 0.1,$ n_epoch$=500$")
# plt.legend()
# plt.xlabel("Mini-batch size")
# plt.ylabel("MSE")
# plt.tight_layout()
# plt.savefig("figs/MSE_batchsize_1dpoly.pdf")

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(batch_sizes, r2_values, label = "$\eta_1 = 0.1,$ n_epoch$=500$")
# plt.legend()
# plt.xlabel("Mini-batch size")
# plt.ylabel("$R^2$")
# plt.tight_layout()
# plt.savefig("figs/R2_batchsize_1dpoly.pdf")

# ------------------------------------------------------#

# # here we plot MSE and R2 as function of learning rate

# n_epochs = 1000
# batch_size = 20
# learning_rates = np.logspace(-4, -(0.5), 10)

# mse_values_sched = []
# r2_values_schec = []

# mse_values_mom = []
# r2_values_mom = []

# for eta in learning_rates:
#     print(f"learning rate = {eta}")
#     print("No moment")
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=batch_size, gamma=0, optimization=None, eta0=eta)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_sched.append(mse)
#     r2_values_schec.append(r2)
#     print("W/ moment")
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=batch_size, gamma=0.9, optimization=None, eta0=eta)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_mom.append(mse)
#     r2_values_mom.append(r2)


# plt.figure(figsize=(12*cm, 10*cm))
# plt.loglog(learning_rates, mse_values_sched, label = "Without moment")
# plt.loglog(learning_rates, mse_values_mom, label = "With moment, $\gamma = 0.9$")

# plt.text(x=0.6, y=0.5, s="1000 epochs\nBatch size = 20", transform=plt.gca().transAxes)
# plt.legend()
# plt.xlabel("Initial learning rate")
# plt.ylabel("MSE")
# plt.tight_layout()
# plt.savefig("figs/MSE_learning_rate_1dpoly.pdf")

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(learning_rates, r2_values_schec, label = "Without moment")
# plt.loglog(learning_rates, r2_values_mom, label = "With moment, $\gamma = 0.9$")
# plt.xscale('log')
# plt.legend()
# plt.xlabel("Initial learning rate")
# plt.ylabel("$R^2$")
# plt.tight_layout()
# plt.savefig("figs/R2_learning_rate_1dpoly.pdf")

# ------------------------------------------------------#

# mse_values_sched, r2_values_sched = [], []
# mse_values_moment, r2_values_moment = [], []
# mse_values_adagrad, r2_values_adagrad = [], []
# mse_values_rmsprop, r2_values_rmsprop = [], []
# mse_values_adam, r2_values_adam = [], []

# epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# eta0 = 0.1
# n_batches = 20

# for n_epochs in epochs:
#     print("Cross validating using n_epochs = ", n_epochs)
#     # no optimizations
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_sched.append(mse)
#     r2_values_sched.append(r2)
#     print("Moment")
#     # with moment
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0.9, optimization=None, eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_moment.append(mse)
#     r2_values_moment.append(r2)
#     # adagrad
#     print("Adagrad")
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0.0, optimization="adagrad", eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_adagrad.append(mse)
#     r2_values_adagrad.append(r2)
#     # RMSprop
#     print("RMSprop")
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0.0, optimization="RMSprop", eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_rmsprop.append(mse)
#     r2_values_rmsprop.append(r2)
#     # adam
#     print("Adam")
#     linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0.0, optimization="adam", eta0=eta0)
#     mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
#     mse_values_adam.append(mse)
#     r2_values_adam.append(r2)

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, mse_values_sched, label = "Learning schedule")
# plt.plot(epochs, mse_values_moment, label = "With moment")
# plt.plot(epochs, mse_values_adagrad, label = "Adagrad")
# plt.plot(epochs, mse_values_rmsprop, label = "RMSprop")
# plt.plot(epochs, mse_values_adam, label = "Adam")
# plt.text(x=0.6, y=0.5, s="$\eta_0 = 0.1$\nBatch size = 20", transform=plt.gca().transAxes)

# plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/MSE_optimizations_epoch_1dpoly.pdf")

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, r2_values_sched, label = "Learning schedule")
# plt.plot(epochs, r2_values_moment, label = "With moment")
# plt.plot(epochs, r2_values_adagrad, label = "Adagrad")
# plt.plot(epochs, r2_values_rmsprop, label = "RMSprop")
# plt.plot(epochs, r2_values_adam, label = "Adam")
# plt.hlines(y=1.0, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
# plt.xlabel("Epochs")
# plt.ylabel("$R^2$")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/R2_optimizations_epoch_1dpoly.pdf")

# ------------------------------------------------------#

# finally, we look at Ridge regression using varying learning rates

