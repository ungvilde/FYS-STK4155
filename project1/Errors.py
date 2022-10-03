import numpy as np

class Errors():
    _fit = None
    _known = None

    def __init__(self, known, fit=None):
        self._fit = fit
        self._known = known
    
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

    def var_beta_ols(self, beta, design):
        '''returns the variance of the estimator in the shape of the estimator.
        If you want it to return the variance of beta fill output=True'''
        
        variance = np.zeros(len(beta))
        temp_mse = self.mse()

        for i in range(len(beta)):
            temp_inv = np.linalg.pinv(design.T @ design)
            variance[i] = temp_mse * temp_inv[i, i]

        return variance

    def conf_int(self, beta, design):
        '''returns the confidence intervals for the betas'''
        
        n, p = np.shape(design)
        lower = np.zeros(len(beta))
        upper = np.zeros(len(beta))

        temp_mse = self.mse() / (n - p) # unbiased estimator of sigma

        for i in range(len(beta)):
            temp_inv = np.linalg.pinv(design.T @ design)
            StdErr = np.sqrt(temp_mse * temp_inv[i, i])
            lower[i] = beta[i] - 1.96*StdErr
            upper[i] = beta[i] + 1.96*StdErr

        return lower, upper