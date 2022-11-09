import numpy as np

from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import seaborn as sns
import matplotlib.pyplot as plt
from GridSearch import GridSearch_FFNN_classification_architecture
from ResampleMethods import *
from FFNN import FFNN
from activation_functions import *

sns.set_theme('notebook', 'whitegrid')

np.random.seed(123)

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
plt.figure(figsize=(12 * cm, 10 * cm))


n_layers = [1, 2, 3, 4, 5]
n_neurons = [10, 30, 50, 80, 100]
n_epochs = 400
eta = 1e-3

results = GridSearch_FFNN_classification_architecture(
    Xtrain,
    ytrain,
    n_layers,
    n_neurons,
    eta,
    n_epochs,
    lmbda=0,
    plot_grid=True,
    gamma=0.9,
    activation_hidden='sigmoid',
    batch_size=20,
    k=5,
)
print(results)

# [[0.97582418 0.96703297 0.97142857 0.95384615 0.96483516]
#  [0.96923077 0.96703297 0.97362637 0.96483516 0.97142857]
#  [0.96703297 0.97142857 0.8989011  0.81538462 0.76703297]
#  [0.96263736 0.96923077 0.89450549 0.62417582 0.62417582]
#  [0.96703297 0.96703297 0.77802198 0.62417582 0.62417582]]
# # using optimal architecture, we now look at the learning rates and lambdas


# tuning the best FFNN parameters given the architecture
# results = GridSearch_FFNN_classifier(
#      X,
#      y,
#      lmbda_values,
#      eta_values,
#      plot_grid=True,
#      gamma=0.9,
#      activation_hidden="sigmoid",
#      n_epochs=400,
#      batch_size=20,
#      n_hidden_neurons = [10],
#      k=5
#      )
# print(results)

# [[0.9648657  0.96829685 0.97539202 0.97716193 0.97357553 0.96660456
#   0.95429281]
#  [0.97365316 0.97537649 0.97363763 0.98068623 0.97890079 0.97719298
#   0.95787921]
#  [0.97540755 0.97894737 0.97894737 0.97540755 0.97539202 0.97365316
#   0.95427729]
#  [0.97540755 0.98242509 0.97716193 0.97893184 0.97542307 0.97717746
#   0.92279149]
#  [0.97886974 0.97182115 0.98070175 0.97532992 0.97890079 0.97716193
#   0.95606272]
#  [0.97542307 0.97891632 0.9806707  0.97363763 0.98594939 0.98068623
#   0.94378202]
#  [0.98245614 0.97717746 0.98240956 0.98070175 0.97717746 0.95433939
#   0.62738705]]
