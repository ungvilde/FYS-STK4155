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

np.random.seed(199)

# Here we do Linear regression using the Gradient Descent (GD) algorithm: 
# We will consider the following parameters/aspects of the algorithm:
# - learning rate
# - epochs
# - momentum

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

# ------------------------------------------------------#

# here we plot MSE and R2 as function of learning rate, with and without momentum

n_epochs = 800
learning_rates = np.logspace(-4, -(0.5), 20)

mse_values_sched = []
r2_values_schec = []

mse_values_mom = []
r2_values_mom = []

for eta in learning_rates:
    print(f"learning rate = {eta}")
    print("No moment")
    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, gamma=0, optimization=None, eta0=eta)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values_sched.append(mse)
    r2_values_schec.append(r2)
    print("W/ moment")
    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, gamma=0.9, optimization=None, eta0=eta)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values_mom.append(mse)
    r2_values_mom.append(r2)


plt.figure(figsize=(12*cm, 10*cm))
plt.loglog(learning_rates, mse_values_sched, label = "Without moment")
plt.loglog(learning_rates, mse_values_mom, label = "With moment, $\gamma = 0.9$")
plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=learning_rates[0], xmax=learning_rates[-1], label="$\sigma^2 = 0.01$")

plt.text(x=0.6, y=0.5, s="800 epochs", transform=plt.gca().transAxes)
plt.legend()
plt.xlabel("Initial learning rate")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("figs/MSE_GD_learning_rate_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(learning_rates, r2_values_schec, label = "Without moment")
plt.loglog(learning_rates, r2_values_mom, label = "With moment, $\gamma = 0.9$")
plt.xscale('log')
plt.legend()
plt.xlabel("Initial learning rate")
plt.ylabel("$R^2$")
plt.tight_layout()
plt.savefig("figs/R2_GD_learning_rate_1dpoly.pdf")

opt_lr = np.argmin(mse_values_mom)

# ------------------------------------------------------#

# We will consider the following range of parameters
epochs = np.array([100, 200, 300, 400, 500, 600, 800, 1000])

eta0=learning_rates[opt_lr]

mse_values0, r2_values0 = [], []
mse_values_2, r2_values_2 = [], []
mse_values1, r2_values1 = [], []
mse_values2, r2_values2 = [], []
mse_values3, r2_values3 = [], []

for n_epochs in epochs:
    print("Cross validating using n_epochs = ", n_epochs)
    # eta0 = 1e1
    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=0.2)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values_2.append(mse)
    r2_values_2.append(r2)
    # eta0 = 1e0
    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, batch_size=20, gamma=0.9, optimization=None, eta0=0.2)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values0.append(mse)
    r2_values0.append(r2)
    # eta0 = 1e-1
    linreg = LinearRegression(lmbda=0, solver="gd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-1)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values1.append(mse)
    r2_values1.append(r2)
    # eta0 = 1e-2
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-2)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values2.append(mse)
    r2_values2.append(r2)
    # eta0 = 1e-3
    linreg = LinearRegression(lmbda=0, solver="sgd", max_iter=n_epochs, batch_size=20, gamma=0, optimization=None, eta0=1e-3)
    mse, r2 = CrossValidation_regression(linreg, Xtrain, ytrain)
    mse_values3.append(mse)
    r2_values3.append(r2)

plt.figure(figsize=(12*cm, 10*cm))
plt.plot(epochs, mse_values_2, label = "$\eta_1 = 0.2$")
plt.plot(epochs, mse_values0, label = "$\eta_1 = 0.2069, \gamma = 0.9$")
plt.plot(epochs, mse_values1, label = "$\eta_1 = 0.1$")
plt.plot(epochs, mse_values2, label = "$\eta_2 = 0.01$")
plt.plot(epochs, mse_values3, label = "$\eta_3 = 0.001$")
plt.hlines(y=0.01, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1], label="$\sigma^2=0.01$")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/MSE_GD_epoch_1dpoly.pdf")

plt.figure(figsize=(12*cm, 10*cm))
#plt.plot(epochs, r2_values0, label = "$\eta_1 = 1$")
plt.plot(epochs, r2_values1, label = "$\eta_1 = 0.1$")
plt.plot(epochs, r2_values2, label = "$\eta_2 = 0.01$")
plt.plot(epochs, r2_values3, label = "$\eta_3 = 0.001$")
plt.hlines(y=1.0, linestyles='dotted', colors='k', xmin=epochs[0], xmax=epochs[-1])
plt.xlabel("Epochs")
plt.ylabel("$R^2$")
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/R2_GD_epoch_1dpoly.pdf")

opt_epoch = epochs[np.argmin(mse_values_2)]

# ------------------------------------------------------#

# finally we test the model with the optimal parameters on unseen the test data

# optimal parameters
n_epochs = opt_epoch
gamma=0.9
lmbda=1e-10

gamma=0.9

linreg = LinearRegression(
    lmbda=lmbda,
    gamma=gamma,
    solver="gd",
    max_iter=n_epochs,
    optimization=None,
    eta0=eta0
)

linreg.fit(Xtrain, ytrain)
ypred = linreg.predict(Xtest)
mse1 = MSE(y=ytest, y_pred=ypred)

print("Best MSE: ", mse)
print("Best learning rate: ", eta0)
print("Best epoch: ", opt_epoch)
# Best MSE:  0.013930473199166437
# Best learning rate:  0.2069138081114788
# Best epoch:  400


# ------------------------------------------------------#