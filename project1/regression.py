class Regression():
    '''
    Superclass for the regression methods, the idea is to gather the most common methods here so they're easily accessible from the subclasses.
    MSE and R2 and estimators are unique for each regression method though plotting is common.
    This can be developped as we go along, splitting and scaling can be done in this superclass for instance :)

    The names of the fits for each subclass is no _fit, it might be smart to change that to _fitOLS, _fit_Ridge and _fitLASSO so they can be distinguished and used in the superclass.
    '''

    # the 'private' variables for the superclass
    _x = None
    _y = None
    _known = None

    def __init__(self):
        '''we just create the object without having to choose what to feed it'''
        pass

    def show(self, figsize=None):
        '''Plotting!
        this is mostly copied from the project description'''
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
        surf = ax.plot_surface(self._x, self._y, self._fit, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        z_min = np.min(self._fit)
        z_max = np.max(self._fit)

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

        # setting limits before plotting
        ax.set_zlim(z_min, z_max)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    