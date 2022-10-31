import numpy as np

def MSE(y, y_pred):
    N = len(y)
    return 1/N * np.sum((y - y_pred)**2)

def R2(y, y_pred):
    mean_y = np.mean(y)
    return 1 - np.sum( (y - y_pred )**2) / np.sum((mean_y - y_pred)**2)


def readfile(filename):
    xvals, datavals = [], []
    with open(filename, 'r') as f:
        for line in f:
            xvals.append(float(line.split()[0]))
            datavals.append(float(line.split()[1]))

    return np.array(xvals), np.array(datavals)

def DesignMatrix(x, degree):
    N = len(x)
    X = np.zeros((N, degree+1))

    for d in range(1, degree+1):
        X[:,d] = x**d

    X[:,0] = np.ones(N) # intercept column

    return X

def sigmoid(x):
    return 1/(1 + np.exp(-x))