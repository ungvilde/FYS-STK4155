import numpy as np
from Errors import Errors
from LinearRegression import LinearRegression

class Resample():

    _reg = None
    _betas = None
    _fits = None

    def __init__(self, regression):
        self._reg = regression

    # The resampling methods have to store the beta to then find the bias and the variance

    def bootstrap(self, N):
        "does bootstrap"
        pass

    def cross_validation(self, k):
        "does cross validation with k folds"
        pass

    def var_beta(self):
        "finds the variance of beta set"

    def mse(self):
        "returns the mse of the performed crossvalidation"

    
    def var_model(self):
        '''Calculates the variance of the model once the resampling has been done. Returns a number!'''

        return np.var(self._fits, axis=1)


    def bias(self):
        '''Calculates the bias once the resampling has been done. Returns a number!'''

        known = self._reg.get_known()
        N = len(known)
        mean = self.estimate_fit()

        return np.sum((known - mean)**2) / N

    def estimate_fit(self):
        '''Calculates the mean of the different fits from the betas. Returns an array.'''

        return np.mean(self._fits, axis=1)