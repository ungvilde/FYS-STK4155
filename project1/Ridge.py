import numpy as np


class Ridge():
    '''
    Class that calculates the estimator for the Ridge method for linear regression.
    Contains no stored variables.

    Methods
    -------
    beta_ridge(design, known, lambda)
    '''

    def beta_ridge(self, design, known, lmbd):
        '''
        Calculates the estimator for Ridge using the design matrix, the known output and a lambda.

        Parameters
        ----------
        design : numpy array (matrix)
            Design matrix

        known : numpy array
            output we want to estimate
        
        lmbd : float
            tuning parameter for the Ridge estimator


        The parameter known will be raveled in the method so the input can be 1D or 2D, but make sure that the raveling corresponds to the one used to create the design matrix.

        Returns
        -------
        Estimator beta as a 1D numpy array
        '''
        num_features = np.shape(design)[1]  # finds the number of features in the design matrix

        # by the power of LINALG!
        beta = np.linalg.pinv(design.T @ design + lmbd*np.eye(num_features)) @ design.T @ np.ravel(known)

        return beta
