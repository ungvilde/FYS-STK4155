import numpy as np

class Errors():
    '''
    Class that handles the evaluation of the models.

    Methods
    -------
    set_fit(fit)
    set_known(known)\n
    mse()\n
    r2()\n
    conf_int(beta, design)

    Private variables
    -----------------
    The fit\n
    The expected output
    '''
    _fit = None
    _known = None

    def __init__(self, known, fit=None):
        '''
        Initializes the object.

        Parameters
        ----------
        known : numpy array
            the output we are going to compare the fit against
        
        fit : numpy array
            Default None\n
            The fit which you want to evaluate
        '''
        self._fit = fit
        self._known = known
    
    def set_fit(self, fit):
        '''
        Sets and stores the fit in the object.

        Parameters
        ---------
        fit : numpy array
            the fit which you want to evaluate
        '''
        self._fit = fit
    
    def set_known(self, known):
        '''
        Sets and stores the known output (z) in the object.

        Parameters
        ---------
        known : numpy array
            the known output which you want to compare against
        '''
        self._known = known

    def mse(self):
        '''
        Takes in the solution and the approximation and calculates the mean squared error of the fit.

        Needs
        -----
        The object must contain a fit.

        Returns
        -------
        The MSE (float) of the fit
        '''
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return np.sum((exact - fit)**2) / len(exact)

    def r2(self):
        '''
        Takes in the solution and the approximation and calculates the R2 score of the fit.

        Needs
        -----
        The object must contain a fit.

        Returns
        -------
        The R2 score (float) of the fit
        '''
        
        exact = np.ravel(self._known)
        fit = np.ravel(self._fit)

        return 1 - (np.sum((exact - fit)**2))/np.sum((exact - np.mean(exact))**2)

    def conf_int(self, beta, design):
        '''
        Calculates and returns the confidence intervals for the estimator parameters.
        
        Parameters
        ---------
        beta : numpy array (1D)
            estimator of the model
        
        design : numpy array (matrix)
            The design matrix used to calculate the estimator
        
        Returns
        -------
        lower (numpy array), upper (numpy array)
            lower and upper confidence intervals for the estimator parameters. They are the same length as beta.
        '''
        
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