import numpy as np

cm = 1/2.54

def MSE(y, y_pred):
    N = len(y)
    y = y.ravel()
    y_pred = y_pred.ravel()

    return 1/N * np.sum((y - y_pred)**2)

def R2(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    mean_y = np.mean(y)
    return 1 - np.sum( (y - y_pred )**2) / np.sum((mean_y - y_pred)**2)

def accuracy(y, y_pred):
    y = y.ravel()
    y_pred = y_pred.ravel()
    N = len(y)

    y_pred = y_pred > 0.5
    
    return np.sum( y == y_pred) / N

def readfile_1dpoly(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))

    return np.array(x), np.array(y)

def readfile_franke(filename):
    x, y, z = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
            z.append(float(line.split()[2]))

    return np.array(x), np.array(y), np.array(z)

def DesignMatrix(x, degree):
    N = len(x)
    X = np.zeros((N, degree))

    for i in range(degree):
        X[:,i] = x**(i+1)

    return X

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp( -(0.25 * (9*x - 2)**2 ) - 0.25 * ( (9*y - 2)**2) )
    term2 = 0.75 * np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
    term3 = 0.5 * np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
    term4 = -0.2 * np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
    
    return term1 + term2 + term3 + term4

def FrankeDesignMatrix(x, y, degree):
    """
    Set up design matrix for two-variable polynomial regression model of order p
    """
 
    l = int((degree + 2)*(degree + 1)/2) # number of elements in beta
    n = len(x)
    X = np.ones((n,l))

    for i in range(1, degree + 1):
        k = int(i * (i + 1)/2) # number of pairs up to order i

        for j in range(i + 1):
            X[:, k + j] = x**(i - j) * y**j # i - j + j <= p for all i,j
    
    return X
