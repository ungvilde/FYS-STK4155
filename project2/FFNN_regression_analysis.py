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

#np.random.seed(543)
np.random.seed(123)

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
## - Regularization X
## - Activation functions
## Potentially: - Initialization (1/sqrt(n) std. in weights)

eta = 1e-4
n_epochs = 500
n_neurons = [10, 30, 50, 80]
n_hidden_layers = [1]

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

# --------------------------------------------------------- #
eta_values = np.logspace(-5, -2, 7)
lmbda_values = np.logspace(-8, 0, 7)
 
# print(eta_values)
# print(lmbda_values)

print("Sigmoid")
# mse_results, _ = GridSearch_FFNN_reg(
#     Xtrain, ytrain, 
#     lambda_values=lmbda_values, eta_values=eta_values, 
#     activation_hidden="sigmoid", 
#     n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
#     initialization="standard")
# print(mse_results)
# From printed:
# results = np.array([[1.06424510e-01, 3.66523309e-02, 3.51028381e-02, 5.36914200e-02, 7.40265331e-02, 3.81590386e-02, 1.79363057e-02],
#     [2.31568693e-02, 2.55925442e-02, 2.43291862e-02, 2.73364309e-02, 2.14012693e-02, 2.82821473e-02, 1.32969731e-02],
#  [1.45349654e-02, 1.36228431e-02, 1.37107567e-02, 1.50599751e-02, 1.32466252e-02, 1.24136933e-02, 3.66224555e-02],
#  [1.16958682e-02, 1.09975650e-02, 1.08874083e-02, 1.38949482e-02, 1.11806298e-02, 1.01217241e-02, 4.13719813e-02],
#  [1.03833739e-02, 1.02388256e-02, 1.05840702e-02, 1.03311120e-02, 1.00861589e-02, 1.13998385e-02, 3.86380209e-02],
#  [1.30241837e-02, 1.04344846e-02, 1.38405535e-02, 1.12786199e-02, 1.37031959e-02, 1.17019809e-02, 2.99419297e-02],
#  [7.26057316e-01, 1.59522294e+02, 3.35009678e-01, 2.22181332e+00, 2.82819356e-01, 3.39390313e-01, 3.42361095e-01]]
#  )

# print(np.min(results))
# i, j = np.where(results == np.min(results))
# print(eta_values[i])
# print(lmbda_values[j])
# # optimal values:
# # eta = 1e-3, lmbda = 2.15443469e-03

print("reLU")
# mse_results, _ = GridSearch_FFNN_reg(
#     Xtrain, ytrain, 
#     lambda_values=lmbda_values, eta_values=eta_values, 
#     activation_hidden="reLU", 
#     n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
#     initialization="normalized")
# print(mse_results)

## optimal was eta = 0.00046416, lmda = 1e-4

# results = np.array([[
#     0.01426618, 0.01354711, 0.0142786,  0.01505428, 0.01295214, 0.0137214, 0.02259456],
#     [0.01077542, 0.01254589, 0.01167965, 0.01245296, 0.01174404, 0.01200921, 0.0206182],
#  [0.0102088,  0.01027748, 0.01035951, 0.01040103, 0.01048774, 0.01081003, 0.02454134],
#  [0.01021436, 0.01032955, 0.01024417, 0.01007699, 0.01023868, 0.01029182, 0.02730845],
#  [0.01012351, 0.01012028, 0.01007874, 0.01015256, 0.01006711, 0.01116338, 0.02852983],
#  [0.01007835, 0.01013123, 0.01009003, 0.01033202, 0.01004878, 0.01131129, 0.02870774],
#  [       np.nan,        np.nan,        np.nan,        np.nan,        np.nan, 0.02821466, 0.02837958]])
# print(np.nanmin(results))
# i, j = np.where(results == np.nanmin(results))
# print(eta_values[i])
# print(lmbda_values[j])
## optimal values = eta = 3.16227766e-03, lambda = lmda = 1e-4

print("Leaky reLU")
# mse_results, _ = GridSearch_FFNN_reg(
#     Xtrain, ytrain, 
#     lambda_values=lmbda_values, eta_values=eta_values, 
#     activation_hidden="leaky_reLU", 
#     n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
#     initialization="normalized")
# print(mse_results)

results = np.array([[0.01470188, 0.0128659,  0.01530361, 0.01338517, 0.01473225, 0.01466999, 0.02210953],
 [0.01186763, 0.01195507, 0.01209588, 0.01115704, 0.01040109, 0.01250774, 0.02066718],
 [0.0105831, 0.01068317, 0.01060112, 0.01030054, 0.01033053, 0.01062866, 0.02442527],
 [0.01023283, 0.01035554, 0.01011767, 0.01022323, 0.01007775, 0.01049731, 0.02684194],
 [0.01010126, 0.01002976, 0.0101683,  0.01012316, 0.01005134, 0.01118842, 0.02851816],
 [0.01000177, 0.01013626, 0.01006683, 0.01013421, 0.01010335, 0.01128153, 0.02854127],
 [       np.nan,        np.nan,        np.nan,        np.nan,        np.nan,        np.nan, 0.02872754]])
print(np.nanmin(results))
i, j = np.where(results == np.nanmin(results))
print(eta_values[i])
print(lmbda_values[j])
## optimal values:
## eta = 3.16227766e-03, lambda = 1e-8

# final test

hidden_layers = [80]
n_epochs=200
batch_size=20
activation_function = "leaky_reLU"
initialization = "normalized"
eta = eta_values[i]
lmbda = lmbda_values[j]
gamma = 0.9

network = FFNN(
    n_hidden_neurons=hidden_layers,
    n_epochs=n_epochs,
    batch_size=batch_size,
    eta=eta,
    lmbda=lmbda,
    gamma=gamma,
    activation_hidden=activation_function,
    initialization=initialization
)
network.fit(Xtrain, ytrain)
ypred = network.predict(Xtest)
mse = MSE(y=ytest, y_pred=ypred)
print(mse)