import numpy as np

class OLS():
    '''
    Class that calculates the estimator for Ordianry Least Squares method for linear regression.
    Contains no stored variables.

    Methods
    -------
    beta_ols(design, known)
    '''

    def beta_ols(self, design, known):
        '''
        Calculates the estimator for OLS using the design matrix and the known output.

        Parameters
        ----------
        design : numpy array (matrix)
            Design matrix

        known : numpy array
            output we want to estimate


        The parameter known will be raveled in the method so the input can be 1D or 2D, but make sure that the raveling corresponds to the one used to create the design matrix.

        Returns
        -------
        Estimator beta as a 1D numpy array
        '''

        # by the power of LINALG!
        beta = np.linalg.pinv(design.T @ design) @ design.T @ np.ravel(known)
        
        return beta