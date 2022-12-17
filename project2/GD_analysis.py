import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
from GridSearch import GridSearch_LinReg, GridSearch_LinReg_epochs_batchsize, GridSearch_LinReg_eta_epoch
from common import *

# plan:
# explore different etas and iterations in a grid
# then compare no opt, moment, adagrad, adagrad+moment
# finally do grid with eta and lambda
# use optimal hyperparams and compute test MSE
# include results in table

sns.set_theme("notebook", "whitegrid")

np.random.seed(199)
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

eta_values = np.logspace(-5, 0, 7)
epoch_values = np.arange(100, 2001, step=300) # 100, 400, 700, 1100, 1400, 1700, 2000

results, _ = GridSearch_LinReg_eta_epoch(
    X, y,
    n_epochs=epoch_values, eta_values=eta_values,
    solver="gd", optimization=None,
    plot_grid=True, gamma=0, lmbda=0
)