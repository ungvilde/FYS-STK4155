# This a file with functions that can be usefull in other purposes than a single code or hierarchy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

def show_me_the_money(x, y, z, figsize = None):
    '''
    Plots and shows a surface from x, y and z where x and y should be mehsgrids. This functionality is given in the project description of project 1 in the course FYS-STK4155/3155 at UiO <3.

    Parameters
    ---------
    x : numpy array (meshgrid)
        used for the coordinates to place the heights in z

    y : numpy array (meshgrid)
        used for the coordinates to place the heights in z

    z : numpy array
        heights as a an array in the same shape as x and y
    
    figsize : tuple
        default is None. It sets the size of the figure as you probably guessed.

    Returns
    -------
    None
        plots and shows the surface.
    '''

    if figsize != None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    z_min = np.min(z)
    z_max = np.max(z)

    # we want to see a little above and a little below the max and min of the z axis
    if z_min < 0:
        #push a little further down if it's already below 0
        z_min *= 1.01
    else:
        # if it's positive we need to reduce the minimum closer to 0
        z_min *= 0.99

    # similar thinking for the maximum
    if z_max > 0:
        z_max *= 1.01
    else:
        z_max *= 0.99

    ax.set_zlim(z_min, z_max)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

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


def standard_normalizer(x):
    '''
    Unsure if used!
    '''
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 0)[np.newaxis, :]   
    x_stds = np.std(x,axis = 0)[np.newaxis, :]   

    # check to make sure thta x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind]
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust

    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds

    # return normalizer 
    return normalizer