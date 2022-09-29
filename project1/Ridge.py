import numpy as np
from sklearn import linear_model


class Ridge():

    def beta_ridge(self, design, known, lmbd):
        '''
        Makes Ridge estimator.
        '''
        num_features = np.shape(design)[1]
        beta = np.linalg.pinv(design.T @ design + lmbd*np.eye(num_features)) @ design.T @ np.ravel(known)

        return beta
