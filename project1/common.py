import numpy as np

def franke_func(x,y):
    """
    Compute Franke function
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def test_func(x):
    return 2 + x - 3 * x**2 + 5 * x**3 + 0.01 * x**4 - 0.4 * x**5


def make_X(x, y, p):
    """
    Set up design matrix for polynomial regression model of order p
    """
 
    l = int((p+2)*(p+1)/2) # number of elements in beta
    n = len(x)
    X = np.ones((n,l))

    for i in range(1, p+1):
        k = int(i*(i+1)/2) # number of pairs up to order i

        for j in range(i+1):
            X[:,k+j] = x**(i-j) * y**j # i - j + j <= p for all i,j
    
    return X

def make_X_test(x, p):
    """
    Set up design matrix for polynomial regression model of order p
    """
 
    n = len(x)
    X = np.ones((n, p+1))

    for i in range(1, p+1):
        X[:, i] = x**i
    
    return X

def MSE(y, yhat):
    """
    Compute mean squared error
    """
    n = len(y)
    MSE = 1/n * np.sum((y - yhat)**2)
    return MSE

def R2(y, yhat):
    """
    Compute R^2
    """
    n = len(y)
    ybar = 1/n * np.sum(y)
    R2 = 1 - np.sum((y-yhat)**2) / np.sum((y-ybar)**2)

    return(R2)