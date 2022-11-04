import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from FFNN import FFNN
from ResampleMethods import CrossValidation_regression
from common import *

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

