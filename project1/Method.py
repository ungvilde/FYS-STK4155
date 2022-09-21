import numpy as np
from OLS import OLS
from LASSO import LASSO
from Ridge import Ridge
from Errors import Errors
from helper import *

class Method(OLS, LASSO, Ridge, Errors):
    '''
    Superclass for the regression methods, the idea is to gather the most common methods here so they're easily accessible from the subclasses.
    MSE and R2 and estimators are unique for each regression method though plotting is common.
    This can be developped as we go along, splitting and scaling can be done in this superclass for instance :)

    The names of the fits for each subclass is no _fit, it might be smart to change that to _fitOLS, _fit_Ridge and _fitLASSO so they can be distinguished and used in the superclass.
    '''

    # the 'private' variables for the superclass
    _beta_ols = None
    _beta_ridge = None
    _beta_lasso = None
    _design = None
    _order = None
    _x = None
    _y = None
    _N = None
    _fit_ols = None
    _fit_ridge = None
    _fit_lasso = None

    def __init__(self, order, x, y, known):
        '''Creates a regression object that will make a linear regression of a chosen order.'''
        self._order = order
        self._known = known
        self._x = x
        self._y = y
        self._N = len(self._x[:, 0])

        self.design()

    def __call__(self, ols=True, ridge=True, lasso=True):
        '''Makes a fit of chosen order.
        Returns ols_fit, ridge_fit, lasso_fit
        The parameteres that were False are 0'''
        self.beta(ols, ridge, lasso)

        if ols:
            self._fit_ols = self._design @ self._beta_ols
            self._fit_ols = np.reshape(self._fit_ols, (self._N, self._N))

        if ridge:
            self._fit_ridge = self._design @ self._beta_ridge
            self._fit_ridge = np.reshape(self._fit_ridge, (self._N, self._N))

        if lasso:
            self._fit_lasso = self._design @ self._beta_lasso
            self._fit_lasso = np.reshape(self._fit_lasso, (self._N, self._N))

        return self._fit_ols, self._fit_ridge, self._fit_lasso
    
    def predict(self, x, y, z, beta = None):
        '''
        Tests the model with new (test) x and y and compares it with a different (test) z.
        '''
        self._x = x
        self._y = y
        self.design()
        N = len(x[:, 0])

        if type(beta) == np.ndarray:
            own = self._design @ beta
            own = np.reshape(own (N, N))
        
        if type(self._beta_ols) == np.ndarray:
            self._fit_ols = self._design @ self._beta_ols
            self._fit_ols = np.reshape(self._fit_ols, (N, N))

        if type(self._beta_ridge) == np.ndarray:
            self._fit_ridge = self._design @ self._beta_ridge
            self._fit_ridge = np.reshape(self._fit_ridge, (N, N))

        if type(self._beta_lasso) == np.ndarray:
            self._fit_lasso = self._design @ self._beta_lasso
            self._fit_lasso = np.reshape(self._fit_lasso, (N, N))

        self.set_known(z)

        if type(beta) == np.ndarray:
            own = self._design @ beta
            own = np.reshape(own (N, N))
            out_mse = self.mse(own)
            out_r2 = self.r2(own)
        else:
            out_mse = self.mse()
            out_r2 = self.r2()

        # return self._fit_ols, self._fit_ridge, self._fit_lasso, own
        return out_mse, out_r2

    def design(self):
        '''with the x and y parameters as well as the maximum order of the fit computes a designmatrix that takes in all the permutations of x and y up the chosen order'''

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

        self._design = X
    
    def beta(self, ols=True, ridge=True, lasso=True):

        if ols == True:
            self._beta_ols = self.beta_ols(self._design, self._known)
        if ridge == True:
            self._beta_ridge = self.beta_ridge()
        if lasso == True:
            self._beta_lasso = self.beta_lasso()
    
    def get_beta(self):

        return self._beta_ols, self._beta_ridge, self._beta_lasso

    
    # VARIANCE AND SUCH

    def mse(self, own=None):
        '''takes in the solution and the approximation and calculates the mean squared error of the fit'''

        if type(own) == np.ndarray:
            out = np.zeros(4)
            self.set_fit(own)
            out[-1] = super().mse()
        else:
            out = np.zeros(3)

        if type(self._fit_ols) == np.ndarray:
            self.set_fit(self._fit_ols)
            out[0] = super().mse()
        if type(self._fit_ridge) == np.ndarray:
            self.set_fit(self._fit_ridge)
            out[1] = super().mse()
        if type(self._fit_lasso) == np.ndarray:
            self.set_fit(self._fit_lasso)
            out[2] = super().mse()
        
        return out

    def r2(self, own=None):
        '''takes in the solution and the approximation and calculates the R2 score of the fit'''
    
        if type(own) == np.ndarray:
            out = np.zeros(4)
            self.set_fit(own)
            out[-1] = super().r2()
        else:
            out = np.zeros(3)

        if type(self._fit_ols) == np.ndarray:
            self.set_fit(self._fit_ols)
            out[0] = super().r2()
        if type(self._fit_ridge) == np.ndarray:
            self.set_fit(self._fit_ridge)
            out[1] = super().r2()
        if type(self._fit_lasso) == np.ndarray:
            self.set_fit(self._fit_lasso)
            out[2] = super().r2()
        
        return out
    
    def var_beta(self):
        '''returns the variance of the estimator in the shape of the estimator.
        If you want it to return the variance of beta fill output=True'''

        # I don't really know what var beta has as a shape    

        pass