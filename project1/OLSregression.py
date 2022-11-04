from cProfile import label
from multiprocessing.resource_sharer import stop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from common import *


# part B: OLS regression
# generate dataset

N = 1000
np.random.seed(123)
x = np.random.random(N)
y = np.random.random(N)
# x = np.linspace(0, 1, N)
# y = np.linspace(0, 1, N)
z = franke_func(x, y) + np.random.randn(1)
z = test_func(x) + np.random.randn(1)

MSEvals = [[],[]] # for train and test
R2vals = [[],[]]
betavals = [[],[],[]]

for p in range(1, 6):

    X = make_X(x, y, p)
    #X = make_X_test(x, p)

    # scale data and split into training and test data (do not scale intercept!)
    X_train, X_test, z_train, z_test = train_test_split(X[:, 1:], z, test_size = 0.2)
    Ntest = X_test.shape[0]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.c_[ np.ones(Ntest), X_test_scaled ] # add intercept column
    X_train_scaled = np.c_[ np.ones(N-Ntest), X_train_scaled ]

    betahat = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train
    print(betahat)

    zpred = X_train_scaled @ betahat
    zpred_test = X_test_scaled @ betahat

    MSEvals[0].append(MSE(z_train, zpred))
    R2vals[0].append(R2(z_train, zpred))
    MSEvals[1].append(MSE(z_test, zpred_test))
    R2vals[1].append(R2(z_test, zpred_test))

    betavals[0].append(betahat[0])
    betavals[1].append(betahat[1])
    #betavals[2].append(betahat[2])

# make design matrix

# use equation to find betahat

# plot MSE, R^2, beta0, beta1, beta2
plt.plot(MSEvals[0], label="MSE train")
plt.plot(MSEvals[1], label="MSE test")
plt.legend()
plt.show()

plt.plot(R2vals[0], label=r"$R^2 train$")
plt.plot(R2vals[1], label=r"$R^2 test$")
plt.legend()
plt.show()

# plt.plot(betavals[0], label = r"$\beta_0$")
# plt.plot(betavals[1], label = r"$\beta_1$")
# plt.plot(betavals[2], label = r"$\beta_2$")
# plt.legend()
# plt.show()

