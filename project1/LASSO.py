from sklearn import linear_model
import numpy as np

class LASSO():
    '''
    Class that calculates the estimator for the LASSO method for linear regression.
    Contains no stored variables.

    Methods
    -------
    beta_lasso(design, known, lambda)
    '''

    def beta_lasso(self, design, known, lmbd):
        '''
        Calculates the estimator for LASSO using the design matrix, the known output and a lambda.

        Parameters
        ----------
        design : numpy array (matrix)
            Design matrix

        known : numpy array
            output we want to estimate
        
        lmbd : float
            tuning parameter for the LASSO estimator


        The parameter known will be raveled in the method so the input can be 1D or 2D, but make sure that the raveling corresponds to the one used to create the design matrix.

        Returns
        -------
        Estimator beta as a 1D numpy array
        '''

        clf = linear_model.Lasso(alpha=lmbd, fit_intercept=False) # fit_intercept is False because the data is centered already
        clf.fit(design, np.ravel(known))    # makes a fit

        beta = clf.coef_    # extracts the estimator

        return beta