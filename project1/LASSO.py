from sklearn import linear_model
import numpy as np

class LASSO():

    def beta_lasso(self, design, known, lmbd):
        '''
        Makes LASSO estimator. Here we will use the sklearn Lasso model in an unsual way.
        Instead of getting the prediction right away, we are going to return only the beta.
        '''

        clf = linear_model.Lasso(alpha=lmbd, fit_intercept=False) # fit_intercept is False because the data is centered already
        clf.fit(design, np.ravel(known))

        beta = clf.coef_

        return beta