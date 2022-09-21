from Regression import Regression
import numpy as np
from helper import *

class OLS():
    '''
    Subclass for OLS functionality. Call it and assign it to a variable (feed x, y, z).
    To do a fit you prepare by calling prepare_order and then you can fit. If you want to test the model on training data call fit_test

    Example on how to use the class:
    from helper import *
    import numpy as np
    from Ordinary import OLS
    import matplotlib.pyplot as plt

    N = 100

    order = 7

    x = np.sort(np.random.rand(N)).reshape((-1, 1))

    y = np.sort(np.random.rand(N)).reshape((-1, 1))
    x, y = np.meshgrid(x, y)

    z = franke(x, y) + np.random.rand(N, N)

    # we can split or create new data in this case
    # we don't really have to scale if our inputs are [0, 1]

    deg = np.linspace(1, order, order)
    mse = []
    r2 = []
    varbeta = []

    ols = OLS(x, y, z)

    for i in deg:
        ols.prepare_order(int(i))
        ols.fit()
        ols.show(figsize=(17, 13))

        mse.append(ols.mse())
        r2.append(ols.r2())
        varbeta.append(ols.var_beta(output=True))
    '''

    # declaring the 'private' variables
    _beta = None
    _design = None
    _order = None
    _fit = None
    _varBeta = None

    def __init__(self, x, y, exact):
        '''initializes the OLS class and stores the x, y and z values'''
        super().__init__()
        self._x = x
        self._y = y
        self._known = exact
    
    def prepare_order(self, order, getBeta=False):
        '''prepares the fit by creating the design matrix and computes the estimator beta.
        If you want the beta values set getBeta to True'''
        self._order = order
        self.calc_X()

        if getBeta:
            self.calc_beta(output=True)
        else:
            self.calc_beta()
    
    def fit_test(self, x, y, output=False, mse=False, r2=False):
        '''Call this function with new x and y values (test data) to test the model.
        If you want the fit: outpout=True
        If you want the r2 score: r2=True
        If you want the MSE: mse=True
        NOTE: The output will be a list with length depending on how many True values you have given (for output, mse and/or r2)
        the order will be [fit, mse, r2].
        Example with output=True, mse=False, r2=True. The return will be in the form [fit, r2]'''
        self._x = x
        self._y = y
        self.calc_X()

        out = []

        if output:
            out1 = self.fit(output=True)
            out.append(out1)
        else:
            self.fit()

        if mse:
            out2 = self.mse()
            out.append(out2)
        if r2:
            out3 = self.r2()
            out.append(out3)
        
        if output or mse or r2:
            return out

    
    def fit(self, output=False):
        '''Makes a fit of the data with the precomputed estimator.
        If you want it to return the fit fill output=False'''
        # returns the fit from the design matrix and the estimator

        self._fit = self._design @ self._beta
        N = len(self._x[:, 0])
        self._fit = np.reshape(self._fit, (N, N))

        if output:
            return self._fit
    
    '''def calc_X(self):
        with the x and y parameters as well as the maximum order of the fit computes a designmatrix that takes in all the permutations of x and y up the chosen order

        # for the franke function we see that x and y are part of a meshgrid
        # to work with this we need to ravel them
        x = np.ravel(self._x)
        y = np.ravel(self._y)

        # x and y might not be the same length
        # we find the longest which will be the n dimention of the design matrix X
        N = max(len(x), len(y))

        # initializes the designmatrix with order 0
        X = np.ones((N, triangular_number(self._order+1)))

        # keeping track of where to fill values
        place = 1

        # filling in the design matrix with the different permutations of x and y (binomial style?)
        for i in range(1, self._order+1):
            for ii in range(i+1):
                X[:, place] = x**ii * y**(i-ii)
                place += 1

        self._design = X'''
    
    def beta_ols(self, output=False):
        '''takes in the design matrix and the values that we're fitting from to create the estimator beta_hat'''

        # by the power of LINALG!
        self._beta = np.linalg.pinv(self._design.T @ self._design) @ self._design.T @ np.ravel(self._known)
        
        if output:
            return self._beta
    '''
    def mse(self):
        takes in the solution and the approximation and calculates the mean squared error of the fit
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return np.sum((exact - fit)**2) / len(exact)
    
    def r2(self):
        takes in the solution and the approximation and calculates the R2 score of the fit
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return 1 - np.sum((exact - fit)**2)/np.sum((exact - np.mean(exact))**2)
    
    def var_beta(self, output=True):
        returns the variance of the estimator in the shape of the estimator.
        If you want it to return the variance of beta fill output=True

        self._varBeta = self.mse() * np.linalg.pinv(self._design.T @ self._design)

        if output:
            return self._varBeta
    
    def get_beta(self, var=False):
        Fetch the estimator. If you also want the variance of the estimator you can add var=True and it will be returned (estimator, variance)
        if var:
            return self._beta, self.var_beta()
        else:
            return self._beta'''