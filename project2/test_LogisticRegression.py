import numpy as np

from common import *
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


np.random.seed(123)
X, y = load_breast_cancer(return_X_y=True)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

n_epochs = 300
m=20
eta=1e-4
lmbda = 0

logreg = LogisticRegression(lmbda=lmbda, solver="sgd", max_iter=n_epochs, batch_size=m, gamma=0.9)
logreg.fit(Xtrain, ytrain)
predictions = logreg.predict(Xtest)
acc = accuracy_score(ytest, predictions > 0.5)
print("For eta = ", eta, " and n_epochs = ", n_epochs)
print("LogisticRegression SGD Accuracy = ", acc)

logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=1000, gamma=0.9)
logreg.fit(Xtrain, ytrain)
predictions = logreg.predict(Xtest)
acc = accuracy_score(ytest, predictions > 0.5)
print("For eta = ", eta, " and n_epochs = ", n_epochs)
print("LogisticRegression GD Accuracy = ", acc)

clf = SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(Xtrain, ytrain)
predictions = clf.predict(Xtest)
acc = accuracy_score(ytest, predictions)
print("SKlearn Accuracy = ", acc)
