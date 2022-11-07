import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from FFNN import FFNN
from common import *
from activation_functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

#np.random.seed(123)
np.random.seed(543)
X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

n_epochs = 500
m=20
eta=1e-4
lmbda = 0

network = FFNN(n_hidden_neurons=[100], task="classification", 
n_epochs=n_epochs, batch_size=m, eta=eta, lmbda=lmbda, gamma=0.9, activation_hidden="reLU")
network.fit(Xtrain, ytrain)
predictions = network.predict(Xtest)
acc = accuracy_score(ytest, predictions > 0.5)
print("For eta = ",eta," and n_epochs = ",n_epochs)
print("FFNN Accuracy = ", acc)
acc = accuracy(y=ytest, y_pred=predictions)
print("Accuracy of FFNN (my accuracy score function) = ", acc)

clf = MLPClassifier(solver="sgd", max_iter=n_epochs, alpha=lmbda)
clf.fit(Xtrain, ytrain)
pred = clf.predict(Xtest)
accuracy = accuracy_score(ytest, pred)
print("Accuracy of SKlearn=", accuracy)

network = FFNN(n_hidden_neurons=[100, 50], task="classification", 
n_epochs=n_epochs, batch_size=m, eta=eta, lmbda=lmbda, gamma=0.9, activation_hidden="reLU")
network.fit(Xtrain, ytrain)
predictions = network.predict(Xtest)
acc = accuracy_score(ytest, predictions > 0.5)
print("For eta = ",eta," and n_epochs = ",n_epochs)
print("FFNN two layers Accuracy = ", acc)

network = FFNN(n_hidden_neurons=[100, 100, 100], task="classification", 
n_epochs=100, batch_size=m, eta=eta, lmbda=lmbda, gamma=0.9, activation_hidden="leaky_reLU")
network.fit(Xtrain, ytrain)
predictions = network.predict(Xtest)
acc = accuracy_score(ytest, predictions > 0.5)
print("For eta = ",eta," and n_epochs = ",n_epochs)
print("FFNN multiple layers Accuracy = ", acc)
