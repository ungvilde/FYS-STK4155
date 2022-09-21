import numpy as np

class Errors():
    _fit = None
    _known = None
    
    def set_fit(self, fit):
        self._fit = fit
    
    def set_known(self, known):
        self._known = known

    def mse(self):
        '''takes in the solution and the approximation and calculates the mean squared error of the fit'''
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return np.sum((exact - fit)**2) / len(exact)

    def r2(self):
        '''takes in the solution and the approximation and calculates the R2 score of the fit'''
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return 1 - np.sum((exact - fit)**2)/np.sum((exact - np.mean(exact))**2)
    
    def var_beta(self, X):
        '''returns the variance of the estimator in the shape of the estimator.
        If you want it to return the variance of beta fill output=True'''

        return self.mse() * np.linalg.pinv(X.T @ X)