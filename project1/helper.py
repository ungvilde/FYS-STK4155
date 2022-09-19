# This a file with functions that can be usefull in other purposes than a single code or hierarchy


def triangular_number(n):
    '''this is to figure out how many permutations we have with x and y given the maximum order of the polynomial fit'''
    return int(n*(n+1)/2)

def show_me_the_money(x, y, z, figsize = None):
    '''plot surface given x, y, and z (meshgrids already)
    most of this is taken from the code given in the project description'''

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

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
    import numpy as np

    one = (3/4) * np.exp(-(9*x - 2)**2/4 - (9*y - 2)**2/4)
    two = (3/4) * np.exp(-(9*x + 1)**2/49 - (9*y + 1)/10)
    three = (1/2) * np.exp(-(9*x - 7)**2/4 - (9*y - 3)**2/4)
    four = -(1/2) * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)

    return one + two + three + four