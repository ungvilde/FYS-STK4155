class OutOfBounds(Exception):
    '''
    Our own error which we raise when the chosen method is wrong.
    '''
    def __init__(self, var_beta=False, conf_int=False):
        if var_beta:
            message = "Can only give the variance of beta for the OLS method. If you want the wvariance of beta for the other methods you have to perform resampling methods."
            super().__init__(message)
        elif conf_int:
            message = "Can only give the confidence intervals of beta for the OLS method. If you want it for the other methods you have to perform resampling methods."
            super().__init__(message)
        else:
            message = "Method number must be either 1 (OLS), 2 (Ridge) or 3 (LASSO)."
            super().__init__(message)