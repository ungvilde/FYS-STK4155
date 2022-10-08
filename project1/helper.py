# This a file with functions that can be usefull in other purposes than a single code or hierarchy
import numpy as np

def triangular_number(n):
    '''
    Function to figure out how many permutations we have with x and y given the maximum order of the polynomial fit.

    Parameters
    ----------
    n : int
        triangular number to calculate
    
    Returns
    -------
    traingular number of parameter n : int
    '''
    return int(n*(n+1)/2)   # well-known function for finding the triangular number

def franke(x, y):
    '''
    Computes the Franke function given x and y parameters.

    Parameters
    ----------
    x : numpy array ; int ; float
        As long as it's computable (numbers basically) and the same size as y anything goes.

    y : numpy array ; int ; float
        As long as it's computable (numbers basically) and the same size as x anything goes.
    
    Returns
    -------
    Franke function evaluated at x and y (will be in the same shape as x and y)
    '''

    one = (3/4) * np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4)  # calculates the first part
    two = (3/4) * np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10)   # calculates the seccond part
    three = (1/2) * np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4)    # calculates the third part
    four = -(1/5) * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)    # calculates the fourth part

    return one + two + three + four # adding the parts together

def our_tt_split(X, y, test_size=0.33, train_size = None, random_state=None):
    '''
    Our own function to split the data into a training and testing set. You can choose either the size of the testing data or the training data. If both are chosen the splitting will be decided by the chosen size of the training data.

    Parameters
    ---------
    X : numpy array (matrix)
        the design matrix which you want to split. Has to be correctly related to y.
    
    y : numpy array
        It will be raveled.
    
    test_size : float
        Default 0.33 . Decides the percentage size of the testing data from the original data.

    train_size : float
        Default None . Decides the percentage size of the training data from the original data.

    random_state : int
        Default None . Seed from which to randomize the splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test
    '''
    y = np.ravel(y)
    assert len(X) == len(y)
    np.random.seed(random_state)

    if train_size:
        test_size = 1 - train_size
    if test_size:
        train_size = 1 - test_size


    #zip 
    zipped = list(zip(X, y))

    #shuffle
    np.random.shuffle(zipped)

    X_shuffled, y_shuffled =  [list(x) for x in zip(*zipped)]

    X_train = X_shuffled[:int(len(y)*train_size)]
    X_test = X_shuffled[int(len(y)*test_size):]
    y_train = y_shuffled[:int(len(y)*train_size)]
    y_test = y_shuffled[int(len(y)*test_size):]
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test

def our_scaler(data):
    """
    Scale our data by centering the mean to 0 and dividing by the standard deviation 
    when it is larger then a certain threshold.

    Parameters
    ---------
    data : numpy array (matrix)
        the data to center. It can be both the design matrix or the z values.

    Returns
    -------
    scaled data
    """
    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    epsilon =  10**(-2)
    denominator = [x if x > epsilon else 1 for x in np.std(data, axis=0)]
    return (data - np.mean(data, axis=0))/ denominator