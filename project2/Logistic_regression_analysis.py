import numpy as np

from common import *
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from GridSearch import GridSearch_LogReg
from ResampleMethods import *



sns.set_theme("notebook", "whitegrid")

np.random.seed(123)

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled

# First we use the simple GD and find some good eta and n_epochs
epochs = [int(x) for x in np.linspace(100, 1000, 40)]

etas = [1e-2, 1e-3, 1e-4]
lmbda = 0
acc_values1 = [] 
acc_values2 = []
acc_values3 = []

for n_epochs in epochs:
    #eta1
    logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=0, eta0=1e-2)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values1.append(acc)

    #eta2
    logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=0, eta0=1e-3)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values2.append(acc)

    #eta3
    logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=0, eta0=1e-4)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values3.append(acc)


print("MAX LogisticRegression GD Accuracy = ", max(acc_values1))    
print("For eta = ", 1e-2, " and n_epochs = ", epochs[np.argmax(acc_values1)])

print("MAX LogisticRegression GD Accuracy = ", max(acc_values2))
print("For eta = ", 1e-3, " and n_epochs = ", epochs[np.argmax(acc_values2)])

print("MAX LogisticRegression GD Accuracy = ", max(acc_values3))
print("For eta = ", 1e-4, " and n_epochs = ", epochs[np.argmax(acc_values3)])

plt.figure(figsize=(12*cm, 10*cm))

plt.plot(epochs, acc_values1, label = f"$\eta_1 = {1e-2}$", c='tab:blue')
plt.plot(epochs, acc_values2, label = f"$\eta_2 = {1e-3}$", c='tab:orange')
plt.plot(epochs, acc_values3, label = f"$\eta_3 = {1e-4}$", c="tab:green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/logistic_acc_epoch_cancer_gamma_0.pdf")
#########################################################
# Now, with eta = 1e-3, let us try sgd with different optimizers just to see how they perform
plt.clf()
acc_values1 = [] 
acc_values2 = []
acc_values3 = []

epochs = [int(x) for x in np.linspace(20, 500, 100)]


for n_epochs in epochs:

    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization=None ,max_iter=n_epochs, batch_size=40, gamma=0.9, eta0=1e-3)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values1.append(acc)

    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='RMSprop' ,max_iter=n_epochs, batch_size=40, gamma=0.9, eta0=1e-3)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values2.append(acc)

    #logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='adagrad' ,max_iter=n_epochs, batch_size=20, gamma=0.9, eta0=1e-3)
    #acc = CrossValidation_classification(logreg, X, y, k=5)
    #acc_values3.append(acc)

    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='adam' ,max_iter=n_epochs, batch_size=40, gamma=0.9, eta0=1e-3)
    acc = CrossValidation_classification(logreg, X, y, k=5)
    acc_values3.append(acc)

plt.plot(epochs, acc_values1, label = f"No Optimization", c='tab:blue')
print("best acc no optimization", max(acc_values1))
#plt.plot(epochs, acc_values2, label = f"adagrad", c='tab:red')
plt.plot(epochs, acc_values2, label = f"RMSprop", c="tab:green")
print("best acc RMSprop", max(acc_values2))
plt.plot(epochs, acc_values3, label = f"Adam", c="tab:orange")
print("best acc Adam", max(acc_values3))

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("figs/logistic_acc_epoch_cancer_multiple_sgd.pdf")


###########################################################
# Let us do a gridsearch of lambda and eta with sgd adam
eta_values = np.logspace(-4, -1, 7)
lmbda_values = np.logspace(-8, 0, 7)


GridSearch_LogReg(
    X,
    y, 
    lmbda_values, 
    eta_values, 
    solver="sgd",
    optimization="adam",
    plot_grid=True,
    gamma=0.9,
    max_iter=500,
    batch_size=20,
    k=5
    )


# then we get the best results (lambda 10e-1.33 and eta 10e-1.5) 
# and compare to our FFNN. For this we need to train the FFNN


# Then we get the best results and compare to sklearn

