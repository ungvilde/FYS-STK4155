import numpy as np

class OLS():

    def beta_ols(self, design, known):
        '''takes in the design matrix and the values that we're fitting from to create the estimator beta_hat'''

        # by the power of LINALG!
        beta = np.linalg.pinv(design.T @ design) @ design.T @ np.ravel(known)
        
        return beta