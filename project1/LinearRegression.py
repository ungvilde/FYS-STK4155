import numpy as np
from OLS import OLS
from LASSO import LASSO
from Ridge import Ridge
from Errors import Errors
from Outofbounds import OutOfBounds
from helper import *

class LinearRegression(OLS, LASSO, Ridge):
    '''
    Superclass for the regression methods, the idea is to gather the most common methods here so they're easily accessible from the subclasses.
    MSE and R2 and estimators are unique for each regression method though plotting is common.
    This can be developped as we go along, splitting and scaling can be done in this superclass for instance :)

    The names of the fits for each subclass is no _fit, it might be smart to change that to _fitOLS, _fit_Ridge and _fitLASSO so they can be distinguished and used in the superclass.
    '''

    # the 'private' variables for the superclass
    _beta = None
    _method = None
    _design = None
    _order = None
    _x = None
    _y = None
    _z = None
    _N = None
    _fit = None
    _lambda = None


    def __init__(self, order, x, y, z, method=1, lmbd=None):
        '''Creates a regression object that will make a linear regression of a chosen order.  Methods are OLS: 1, Ridge: 2 and LASSO: 3'''
        self._order = order
        self._z = z
        self._x = x
        self._y = y
        self._N = len(self._x[:, 0])
        self._lambda = lmbd

        if method not in [1, 2, 3]:
            raise OutOfBounds()

        self._method = method

        self.design()

    def __call__(self):
        '''Makes a fit of chosen order.'''

        self.beta()

        self._fit = self._design @ self._beta
        self._fit = np.reshape(self._fit, (self._N, self._N))

        return self._fit
    
    def predict_resample(self, design_train, z_train, design_test):
        self._design = design_train
        self._z = z_train
        self.beta()

        self._fit = design_test @ self._beta

        return self._fit


    def predict(self, x, y, z, beta = None):
        '''
        Tests the model with new (test) x and y and compares it with a different (test) z.
        '''
        self._x = x
        self._y = y
        temp = self._z
        self._z = z
        self.design()
        N = len(x[:, 0])

        if type(beta) == np.ndarray:
            own = self._design @ beta
            own = np.reshape(own, (N, N))

            out_mse = self.mse(own=own)
            out_r2 = self.r2(own=own)

        else:
            self._fit = self._design @ self._beta
            self._fit = np.reshape(self._fit, (N, N))

            out_mse = self.mse()
            out_r2 = self.r2()
        
        self._z = temp

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
    
    def beta(self):

        if self._method == 1:
            self._beta = self.beta_ols(self._design, self._z)
        elif self._method == 2:
            self._beta = self.beta_ridge(self._design, self._z, self._lambda)
        else:
            self._beta = self.beta_lasso(self._design, self._z, self._lambda)
    
    # VARIANCE AND SUCH

    def mse(self, own=None):
        '''takes in the solution and the approximation and calculates the mean squared error of the fit'''

        error = Errors(self._z)

        if type(own) == np.ndarray:
            error.set_fit(own)
            return error.mse()
        else:
            error.set_fit(self._fit)
            return error.mse()

    def r2(self, own=None):
        '''takes in the solution and the approximation and calculates the R2 score of the fit'''

        error = Errors(self._z)
    
        if type(own) == np.ndarray:
            error.set_fit(own)
            return error.r2()
        else:
            error.set_fit(self._fit)
            return error.r2()
   
    def var_beta(self):
        '''can only get var of ols estimator here'''

        if self._method != 1:
            raise OutOfBounds(var_beta=True)
        
        else:
            variance_ols = Errors(self._z, self._fit)

            return variance_ols.var_beta_ols(self._beta, self._design)

# GET and SET

    def set_beta(self, design, z):
        '''Sets a design matrix and calculates beta.'''
        self._design = design
        self._z = z
        self.beta()

    def set_known(self, z):
        self._z = z

    def set_order(self, order):
        self._order = order
    
    def set_design(self, design):
        self._design = design

    def get_beta(self):

        return self._beta

    def get_design(self):
        return self._design
    
    def get_known(self):
        return self._z
    
    def get_x(self):
        return self._x
    
    def get_y(self):
        return self._y