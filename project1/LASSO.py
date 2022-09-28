from sklearn import linear_model


class LASSO():

    def beta_lasso(self, design, known, lmbd):
        '''
        Makes LASSO estimator. Here we will use the sklearn Lasso model in an unsual way.
        Instead of getting the prediction right away, we are going to return only the beta.
        '''

        clf = linear_model.Lasso(alpha=lmbd)
        clf.fit(design, known)

        beta = clf.coef_

        return beta