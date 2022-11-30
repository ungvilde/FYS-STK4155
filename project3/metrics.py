import numpy as np


def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    output_scores = 1 - (numerator / denominator)
    return np.average(output_scores)

def mean_squared_error(y_true, y_pred):
    return np.average( (np.ravel(y_pred) - np.ravel(y_true))**2 )

