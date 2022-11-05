import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from FFNN import FFNN
from ResampleMethods import CrossValidation_regression
from common import *
from GridSearch import GridSearch_FFNN_reg_architecture, GridSearch_FFNN_reg

sns.set_theme("notebook", "whitegrid")

np.random.seed(543)

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

# We explore optimal settings for:
## - Init. learning rate X
## - Num. hidden layers and num. neurons in layers X
## - Epochs X (batch size?)
## - Momentum or not X
## - Regularization
## - Activation functions
## Potentially: - Initialization (1/sqrt(n) std. in weights)

# eta = 1e-3
# n_epochs = 500
# n_neurons = [10, 30, 50, 80]
# n_hidden_layers = [1]

# results = GridSearch_FFNN_reg_architecture(
#     Xtrain, ytrain, n_layers=n_hidden_layers, n_neurons=n_neurons, lmbda=0, eta=eta, n_epochs=n_epochs, plot_grid=False)
# print(results)
# ---------------------------------------------------------------------- #

# # using optimal architecture, we now look at the learning rates and epochs

# # now we look at how the number of epochs and initial training rate affect the model
# # we use a batch size of 20

# epochs = [20, 50, 100, 200, 500, 1000]

# eta3 = 1e-4
# eta2 = 1e-3
# eta1 = 1e-2

# mse_values1, r2_values1 = [], []

# mse_values2, r2_values2 = [], []
# mse_values2_moment, r2_values2_moment = [], []

# mse_values3, r2_values3 = [], []

# for n_epochs in epochs:
#     print("Cross validating using n_epochs = ", n_epochs)
#     print("eta = ", 1e-1)
#     network = FFNN(
#         n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta1)
#     mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
#     mse_values1.append(mse)
#     r2_values1.append(r2)
    
#     print("eta = ", 1e-2)
#     network = FFNN(
#         n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta2)
#     mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
#     mse_values2.append(mse)
#     r2_values2.append(r2)

#     network = FFNN(
#         n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0.9, eta=eta2)
#     mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
#     mse_values2_moment.append(mse)
#     r2_values2_moment.append(r2)

#     print("eta = ", 1e-3)
#     network = FFNN(
#         n_hidden_neurons=[80],lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta3)
#     mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
#     mse_values3.append(mse)
#     r2_values3.append(r2)

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, mse_values1, label = f"$\eta_1 = {eta1}$", c='tab:blue')
# plt.plot(epochs, mse_values2, label = f"$\eta_2 = {eta2}$", c='tab:orange')
# plt.plot(epochs, mse_values2_moment, '--', label = f"$\eta_2 = {eta2}$, $\gamma=0.9$",c='tab:orange')
# plt.plot(epochs, mse_values3, label = f"$\eta_3 = {eta3}$", c="tab:green")
# plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1], label="$\sigma^2=0.01$")
# plt.xlabel("Epochs")
# plt.ylabel("MSE")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/FFNN_MSE_epoch_1dpoly.pdf")

# plt.figure(figsize=(12*cm, 10*cm))
# plt.plot(epochs, r2_values1, label = f"$\eta_1 = {eta1}$")
# plt.plot(epochs, r2_values2, label = f"$\eta_2 = {eta2}$")
# plt.plot(epochs, r2_values3, label = f"$\eta_3 = {eta3}$")
# plt.hlines(y=1.0, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
# plt.xlabel("Epochs")
# plt.ylabel("$R^2$")
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig("figs/FFNN_R2_epoch_1dpoly.pdf")

eta_values = [1e-3] #np.logspace(-4, -2, 7)
lmbda_values=np.logspace(-10, -1, 7)

mse_results, _ = GridSearch_FFNN_reg(Xtrain, ytrain, 
lambda_values=lmbda_values, eta_values=eta_values, activation_hidden="sigmoid", n_epochs=200, n_hidden_neurons=[80], plot_grid=False)
print(mse_results)