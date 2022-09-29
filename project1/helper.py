# This a file with functions that can be usefull in other purposes than a single code or hierarchy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def triangular_number(n):
    '''this is to figure out how many permutations we have with x and y given the maximum order of the polynomial fit'''
    return int(n*(n+1)/2)

def show_me_the_money(x, y, z, figsize = None):
    '''plot surface given x, y, and z (meshgrids already)
    most of this is taken from the code given in the project description'''

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
    '''computes the franke function :)'''

    one = (3/4) * np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4)
    two = (3/4) * np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10)
    three = (1/2) * np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4)
    four = -(1/5) * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)

    return one + two + three + four

def our_tt_split(X, y, test_size=0.33, train_size = None, random_state=None):
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
    """
    inspired by Machine Learning Refined from Watt et. all
    """
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