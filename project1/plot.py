from cProfile import label
from sklearn.preprocessing import normalize
from LinearRegression import LinearRegression
from OLS import OLS
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme()
np.random.seed(123)

N = 40
x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)
z = franke(x, y) + np.random.normal(loc=0, scale=0.1, size=(N,N))
d_max = 14

# code for reproducing fig 2.11
mse_test = []
mse_train = []

for i in range(1, d_max+1):
    i = int(i)
    Linreg = LinearRegression(i, x, y, z)
    X = Linreg.get_design()
    X = normalize(X)
    X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size = 0.3, random_state=42)

    # compute test error
    Linreg.set_beta(X_train, z_train)
    mse, _ = Linreg.predict(X_test, z_test)
    mse_test.append(mse)

    # compute training error
    Linreg.set_beta(X_train, z_train)
    mse, _ = Linreg.predict(X_train, z_train)
    mse_train.append(mse)

cm = 1/2.54
plt.figure(figsize = (12*cm, 10*cm))
d_values = np.arange(1, d_max+1, step=1, dtype=int)
plt.plot(d_values, mse_train, label = "Training error")
plt.plot(d_values, mse_test, label = "Test error")
plt.xlabel("Polynomial degree $d$")
plt.ylabel("MSE")
plt.xticks(np.arange(1, d_max+1, step=2, dtype=int))
plt.legend()
plt.tight_layout()
plt.savefig("figs/train_v_test_error_plot.pdf")
#plt.show()
    
###########################################
# code for plotting Bias-Variance Trade-Off
N = 40
x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)
z = franke(x, y) + np.random.normal(loc=0, scale=0.1, size=(N,N))
d_max = 15
Linreg = LinearRegression(d_max, x, y, z)
X = Linreg.get_design()
X = normalize(X)
X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size = 0.2, random_state=42)

# now we use the bootstrap

mse_list = []
bias_list = []
var_list = []

for i in range(1, d_max+1):
    i = int(i)

    Linreg = LinearRegression(i, x, y, z, scale=True)
    resampler = Resample(Linreg)
    #X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size = 0.3, random_state=42)
    #Linreg.set_beta(X_train, z_train) 

    _, mse, bias, var = resampler.bootstrap(100)

    mse_list.append(mse)
    bias_list.append(bias)
    var_list.append(var)

cm = 1/2.54
plt.figure(figsize = (12*cm, 10*cm))
d_values = np.arange(1, d_max+1, step=1, dtype=int)
plt.plot(d_values, mse_list, label = "Test error")
plt.plot(d_values, bias_list, '--', label = "Bias")
plt.plot(d_values, var_list, '--', label = "Variance")

plt.xlabel("Polynomial degree $d$")
plt.ylabel("MSE")
plt.xticks(np.arange(1, d_max+1, step=2, dtype=int))
plt.legend()
plt.tight_layout()
plt.savefig("figs/bias_variance_plot.pdf")
#plt.show()

####################################################
# code for plotting beta values with conf. intervals

N = 100
x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)
z = franke(x, y) + np.random.normal(loc=0, scale=0.1, size=(N,N))
d_max = 5

plt.figure(figsize = (12*cm, 8*cm))

for i in range(d_max, 1, -1):
    i = int(i)
    slice = int((i+1)*(i+2)/2)

    Linreg = LinearRegression(i, x, y, z)
    X = Linreg.get_design()
    X = normalize(X)

    Linreg.set_beta(X, np.ravel(z))
    Linreg.predict(X, z)
    beta = Linreg.get_beta()
    conf_int = Linreg.conf_int()

    beta_inds = range(0, len(beta))
    #plt.plot(beta_inds, beta, 'o', label=f"Order {i}")
    plt.errorbar(x=beta_inds, y=beta, yerr=conf_int, fmt='o', label=f"$d=${i}")

plt.legend()
p = (d_max+1)*(d_max+2)/2
plt.xticks(np.arange(0, p, step=2, dtype=int))
plt.xlabel(r"Index $j$")
plt.ylabel(r"$\beta_j$")
plt.tight_layout()
plt.savefig("figs/beta_coef.pdf")
