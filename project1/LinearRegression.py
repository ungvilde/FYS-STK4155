import numpy as np
from OLS import OLS
from LASSO import LASSO
from Ridge import Ridge
from Errors import Errors
from Outofbounds import OutOfBounds
from helper import triangular_number
from helper import our_tt_split
from sklearn.preprocessing import normalize

class LinearRegression(OLS, LASSO, Ridge):
    '''
    Superclass for the regression methods, the idea is to gather the most common methods 
    here so they're easily accessible.

    The creation of this objects decides which method is going to be used: OLS, Ridge or LASSO.

    Methods
    -------
    predict_resample(design_train, z_train, design_test)\n
    predict(design_test, z_test)\n
    design(scale=False)\n
    beta()\n
    mse(own=None)\n
    r2(own=None)\n
    var_beta() Note: only for OLS\n
    conf_int() Note: only for OLS\n
    split_predict_eval(test_size=0.2, fit=False, train=False, random_state=None)\n
    set_beta(design, z)\n
    set_known(z)\n
    set_order(order)\n
    set_design(design)\n
    get_beta()\n
    get_design()\n
    get_known()\n
    get_x()\n
    get_y()

    Private variables
    -----------------
    Beta\n
    Method\n
    Design matrix\n
    x, y and z\n
    Fit\n
    Lambda
    '''

    # the 'private' variables for the superclass
    _beta = None    # contains the estimator
    _method = None  # keeping track of the method used in the instance (ols, ridge or lasso)
    _design = None  # conatins the design matrix
    _order = None   # keeping track of the order of the polynomial fit
    _x = None   # self explanatory
    _y = None   # self explanatory
    _z = None   # self explanatory
    _fit = None # contains the most recent fit done
    _lambda = None  # contains the lambdas used for ridge or lasso


    def __init__(self, order, x, y, z, method=1, lmbd=None, scale=False):
        '''Creates an instance of LinearRegression which will do either OLS, Ridge or LASSO depending on the method number chosen.
        
        Parameters
        ----------
        order : int
            order of the polynomial fit. The initialization of the object will create a design matrix for the chosen order.

        x, y, z : numpy arrays
            x and y should be be meshgrids. That is a consequence of hoe the class was built.
        
        method : int (1, 2 or 3)
            Default 1\n
            Chosen method for linear regression. 1 is OLS, 2 is Ridge and 3 is LASSO.
        
        lmbd : float
            Default None\n
            Lambda to which we will make a fit for either Ridge or LASSO
        
        scale : boolean
            Default False\n
            If True the designmatrix will be normalized.
        
        Raises
        ------
        OutOfBounds(Exception)
            if method chosen is not 1, 2 or 3
        '''

        # setting the parameters that are chosen for this instance adn storing them in the object
        self._order = order
        self._z = z
        self._x = x
        self._y = y
        self._lambda = lmbd

        # checking if chosen method is valid
        if method not in [1, 2, 3]:
            raise OutOfBounds()

        self._method = method

        self.design(scale=scale)   # calculating the design matrix immediately

    def __call__(self):
        '''
        Makes a fit of chosen order.
        
        Calculates the beta with the chosen method and makes a fit

        Returns
        -------
        The polynomial fit.
        '''
        self.beta() # calculates the beta

        self._fit = self._design @ self._beta   # makes the fit
        self._fit = np.reshape(self._fit, np.shape(self._z))    # reshapes to match the z

        return self._fit
    
    def predict_resample(self, design_train, z_train, design_test):
        '''
        Prediction method for the resampling method (see Resampling class). Takes the training data and the test data. Makes a design matrix and an estimator using the training data and then a fit using the test data. Returns that fit.

        Parameters
        ----------
        design_train : numpy array (matrix)
            design matrix for the training data. The size and shape already decides the polynomial degree.
        
        z_train : numpy array
            known training output used to create the beta.
        
        design_test : numpy array (matrix)
            design matrix from the test data which will be used to calculate the fit.
        
        Returns
        -------
        The fit (1D) of the design_test @ beta, where the beta is calculated from the training parameters.
        '''
        self._design = design_train # sets the instance designmatrix to be the training design matrix
        self._z = z_train   # sets the known output for the training data which is used to calculate the estimtor based off of the training set
        self.beta() # calculates the beta using the training data-set

        self._fit = design_test @ self._beta    # makes a fit on the test-data

        return self._fit    # returns the fit

    def predict(self, design_test, z_test):
        '''
        Basic predicting of test data with an already computed beta, evaluates the model. The instance's stored z is not replaced for consistency (the testing z and design are used but not stored).

        Needs
        -----
        An already calculated beta.

        Parameters
        ----------
        design_test : numpy array (matrix)
            design matrix of the test data which will be used to make a fit.
        
        z_test : numpy array
            the fit which we will test against
        
        Returns
        -------
        The MSE and R2-score of the predicted fit.
        '''
        temp = self._z  # saves the z of the instance

        self._z = z_test    # sets the testing z
        self._design = design_test  # sets the testing design matrix

        shape = np.shape(z_test)    # so that we are comparing the same shapes after 
        self._fit = self._design @ self._beta   # makes the fit with an already calculated beta
        self._fit = np.reshape(self._fit, shape)    # reshapes the fit to be the same shape as z_test

        out_mse = self.mse()    # evaluates the fit
        out_r2 = self.r2()  # evaluates the fit
        
        self._z = temp  # resetting the z for consistency

        return out_mse, out_r2

    def design(self, scale=False):
        '''
        Makes the design matrix based off of the chosen order and the x and y data. Stores the design matrix within the object, to get it use get_design().

        Needs
        -----
        x and y data. A chosen order for the polynomial fit which can be updated through set_order(int(order)).
        '''

        # for the franke function we see that x and y are part of a meshgrid
        # to work with this we need to ravel them
        x = np.ravel(self._x)
        y = np.ravel(self._y)

        # x and y might not be the same length
        # we find the longest which will be the n dimention of the design matrix X
        N = max(len(x), len(y))

        # initializes the designmatrix with order 0
        X = np.ones((N, triangular_number(self._order+1)))

        # keeping track of where to fill values
        place = 1

        # filling in the design matrix with the different permutations of x and y (binomial style)
        for i in range(1, self._order+1):
            for ii in range(i+1):
                X[:, place] = x**ii * y**(i-ii)
                place += 1

        self._design = X

        if scale:
            self._design = normalize(self._design)

    def beta(self):
        '''
        Calculates the estimator with the chosen method. Calls upon the inherited beta calculator from the OLS, Ridge or LASSO classes. Returns the beta though that might not be necessary with a get_beta() method.

        Needs
        -----
        A design matrix and a known z stored in the instance. Meaning you must have set them when the instance was created.
        '''

        if self._method == 1:
            self._beta = self.beta_ols(self._design, self._z)   # calculates the beta with the chosen method

        elif self._method == 2:
            self._beta = self.beta_ridge(self._design, self._z, self._lambda)   # calculates the beta with the chosen method

        else:
            self._beta = self.beta_lasso(self._design, self._z, self._lambda)   # calculates the beta with the chosen method

        return self._beta

    # VARIANCE AND SUCH

    def mse(self, own=None):
        '''
        Calculates the MSE using the Errors class.

        Needs
        -----
        Needs to have made a fit and have it stored in the instance as well as a knwon z to ompare to.

        Parameters
        ----------
        own : numpy array
            Default None\n
            a chosen fit to which to compare to.

        Returns
        -------
        The calculated MSE as a float
        '''

        error = Errors(self._z) # creates the Error object and sets the knwon z we will compare to

        if type(own) == np.ndarray: # compares to the chosen fit
            error.set_fit(own)
            return error.mse()
        else:
            error.set_fit(self._fit)    # compares to the fit stored in the instance
            return error.mse()

    def r2(self, own=None):
        '''
        Calculates the R2 score using the Errors class.

        Needs
        -----
        Needs to have made a fit and have it stored in the instance as well as a knwon z to ompare to.

        Parameters
        ----------
        own : numpy array
            Default None\n
            a chosen fit to which to compare to.

        Returns
        -------
        The calculated R2 score as a float
        '''
        error = Errors(self._z) # creates the Error object and sets the knwon z we will compare to
    
        if type(own) == np.ndarray: # compares to the chosen fit
            error.set_fit(own)
            return error.r2()
        else:
            error.set_fit(self._fit)    # compares to the fit stored in the instance
            return error.r2()

    def conf_int(self):
        '''
        This method is only valid if the chosen method for the fitting is OLS! Calculates the confidence intervals of the estimator through the properties given by the linalg method of find the estimator.

        Needs
        -----
        to be an instance that has method == 1\n
        to have made a fit.

        Returns
        -------
        The confidence intervals of the estimator : numpy array

        Raises
        ------
        OutOfBounds(Exception) if the chosen method is not OLS
        '''

        if self._method != 1:
            raise OutOfBounds(conf_int=True)
        
        else:
            conf_intervals = Errors(self._z, self._fit)

            return conf_intervals.conf_int(self._beta, self._design)    # finds and returns the confidence intervals of the estimator

    def split_predict_eval(self, test_size=0.2, fit=False, train=False, random_state=None):
        '''
        Splits the contained data into training and testing and either returns the testing data and stores the training data or makes a fit and evaluates it.

        Parameters
        ---------
        test_size : float (between 0 and 1)
            decides the fraction which will make up the test data from the original data.
        
        fit : boolean
            Default False\n
            Flag for choosing if it fits and evaluates (True) or not (False)
        
        train : boolean
            Default False\n
            If True the returned mse and r2 will be evaluated from the training data and not the test.
        
        random_state : int
            Default None\n
            Seed from which to randomize the splitting of the data.
        '''

        X_train, X_test, z_train, z_test = our_tt_split(self._design, self._z, test_size=test_size, random_state=random_state)

        # setting the training data to be stored in the instance and calculates the beta from it
        self._design = X_train
        self._z = z_train
        self._beta = self.beta()

        if fit:
            if train: # evaluates the training data
                # this returns mse and r2
                return self.predict(X_train, z_train)

            else: # evaluates the model on the testing data
                # this returns mse and r2
                return self.predict(X_test, z_test)

        else:
            return X_test, z_test

# GET and SET

    def set_beta(self, design, z):
        '''
        Sets a design matrix and the known output and calculates beta. All of the parameters will be stored in the object.

        Needs
        -----
        It doesn't need anything but remember that it will calulate a beta using the chosen method and that the order of the polynomial fit is now decided by the design matrix.
        
        Parameters
        ----------
        design : numpy array (matrix)
            sets the new design matrix which the object will use
        
        z : numpy array
            sets the new z which the object will use
        
        Returns
        -------
        Nothing
        '''
        self._design = design
        self._z = z
        self.beta() # calculates and stores the new estimator in the object

    def set_known(self, z):
        '''
        Sets a new z.

        Parameters
        ----------
        z : numpy array
        '''
        self._z = z

    def set_order(self, order):
        '''
        Unsure if used!\n
        Sets a new order for the polynomial, this will not do anything unless you calculate a new design matrix.

        Parameters
        ----------
        z : numpy array
        '''
        self._order = order
    
    def set_design(self, design):
        '''
        Unsure if used!\n
        Sets a new design matrix.

        Parameters
        ----------
        design : numpy array (matrix)
        '''
        self._design = design

    def get_beta(self):
        '''
        Returns
        -------
        The stored estimator : numpy array
        '''
        return self._beta

    def get_design(self):
        '''
        Returns
        -------
        The stored design matrix : numpy array (matrix)
        '''
        return self._design
    
    def get_known(self):
        '''
        Returns
        -------
        The stored z : numpy array
        '''
        return self._z
    
    def get_x(self):
        '''
        Unsure if used!\n
        Returns
        -------
        The stored x values : numpy array
        '''
        return self._x
    
    def get_y(self):
        '''
        Unsure if used!\n
        Returns
        -------
        The stored y values : numpy array
        '''
        return self._y