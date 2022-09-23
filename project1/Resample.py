import numpy as np
from Errors import Errors
from LinearRegression import LinearRegression

class Resample():

    _reg = None
    _betas = None
    _fits = None
    _kfolds = None

    def __init__(self, regression):
        self._reg = regression

    # The resampling methods have to store the beta to then find the bias and the variance

    def bootstrap(self, N):
        "does bootstrap"
        pass

    def cross_validation(self, k):
        "does cross validation with k folds"
    
    def k_folds(self):
        '''splits available data into chosen number of folds for cross validation. Folds are made from the design matrix!
        The design matrix is taken and the indices are shuffled and given as a an arrya of indeices so we have correspondence between the test and train data results.'''
        
        # we start by shuffling
        design = self._reg.get_design()
        z = self._reg.get_known()
        new_ind = np.random.permutation(len(design))
        mix_design = design[new_ind]
        mix_z = z[new_ind]

        # now we need to split into folds and store these

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