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
n_neurons = [10, 30, 50, 80, 100]
n_hidden_layers = [1, 2, 3, 4, 5]

results, _ = GridSearch_FFNN_reg_architecture(
    Xtrain, ytrain, 
    n_layers=n_hidden_layers, n_neurons=n_neurons, 
    lmbda=0, eta=eta, n_epochs=n_epochs, gamma=0.0,
    plot_grid=True)
print(np.min(results))
i, j = np.where(results == np.min(results))

print(i, j)
print("Layers = ", n_hidden_layers[i[0]]) 
print("Neurons = ", n_neurons[j[0]])

# ---------------------------------------------------------------------- #

# using optimal architecture, we now look at the learning rates and epochs

# now we look at how the number of epochs and initial training rate affect the model
# we use a batch size of 20

epochs = [20, 50, 100, 200, 500, 1000]

eta3 = 1e-4
eta2 = 1e-3
eta1 = 1e-2

mse_values1, r2_values1 = [], []

mse_values2, r2_values2 = [], []
mse_values2_moment, r2_values2_moment = [], []

mse_values3, r2_values3 = [], []

for n_epochs in epochs:
    print("Cross validating using n_epochs = ", n_epochs)
    print("eta = ", 1e-1)
    network = FFNN(
        n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta1)
    mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
    mse_values1.append(mse)
    r2_values1.append(r2)
    
    print("eta = ", 1e-2)
    network = FFNN(
        n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta2)
    mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
    mse_values2.append(mse)
    r2_values2.append(r2)

    network = FFNN(
        n_hidden_neurons=[80], lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0.9, eta=eta2)
    mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
    mse_values2_moment.append(mse)
    r2_values2_moment.append(r2)

    print("eta = ", 1e-3)
    network = FFNN(
        n_hidden_neurons=[80],lmbda=0, n_epochs=n_epochs, batch_size=20, gamma=0, eta=eta3)
    mse, r2 = CrossValidation_regression(network, Xtrain, ytrain)
    mse_values3.append(mse)
    r2_values3.append(r2)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, mse_values1, label = f"$\eta_1 = {eta1}$", c='tab:blue')
plt.plot(epochs, mse_values2, label = f"$\eta_2 = {eta2}$", c='tab:orange')
plt.plot(epochs, mse_values2_moment, '--', label = f"$\eta_2 = {eta2}$, $\gamma=0.9$",c='tab:orange')
plt.plot(epochs, mse_values3, label = f"$\eta_3 = {eta3}$", c="tab:green")
plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1], label="$\sigma^2=0.01$")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/FFNN_MSE_epoch_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, r2_values1, label = f"$\eta_1 = {eta1}$")
plt.plot(epochs, r2_values2, label = f"$\eta_2 = {eta2}$")
plt.plot(epochs, r2_values3, label = f"$\eta_3 = {eta3}$")
plt.hlines(y=1.0, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
plt.xlabel("Epochs")
plt.ylabel("$R^2$")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/FFNN_R2_epoch_1dpoly.pdf")

# --------------------------------------------------------- #
eta_values = np.logspace(-5, -2, 7)
lmbda_values = np.logspace(-8, 0, 7)
 
print(eta_values)
print(lmbda_values)

print("Sigmoid")
mse_results, _ = GridSearch_FFNN_reg(
    Xtrain, ytrain, 
    lambda_values=lmbda_values, eta_values=eta_values, 
    activation_hidden="sigmoid", 
    n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
    initialization="standard")
print(mse_results)
print(np.min(results))
i, j = np.where(results == np.min(results))
print(eta_values[i])
print(lmbda_values[j])

print("reLU")
mse_results, _ = GridSearch_FFNN_reg(
    Xtrain, ytrain, 
    lambda_values=lmbda_values, eta_values=eta_values, 
    activation_hidden="reLU", 
    n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
    initialization="normalized")
print(mse_results)

print(np.nanmin(results))
i, j = np.where(results == np.nanmin(results))
print(eta_values[i])
print(lmbda_values[j])

print("Leaky reLU")
mse_results, _ = GridSearch_FFNN_reg(
    Xtrain, ytrain, 
    lambda_values=lmbda_values, eta_values=eta_values, 
    activation_hidden="leaky_reLU", 
    n_epochs=200, n_hidden_neurons=[80], plot_grid=True,
    initialization="normalized")
print(mse_results)

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

# ---------------------------------------------------- #
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