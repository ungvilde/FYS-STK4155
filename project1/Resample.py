import numpy as np
from Errors import Errors
from LinearRegression import LinearRegression
from sklearn.preprocessing import normalize


class Resample():
    '''
    Class for the resampling methods that uses an instance of LinearRegression. Can perform bootstrap and Cross-Validation.

    Methods
    -------
    bootstrap(N, random_state)\n
    k_folds(k)\n
    cross_validation(k)

    Private variables
    -----------------
    LinearRegression object\n
    Estimator of the model

    '''

    _reg = None
    _betas = None

    def __init__(self, regression):
        '''
        Initalizes the instance and stores a LinearRegression object from which it will do the resampling.

        Parameters
        ----------
        regression : LinearRegression instance
        '''
        self._reg = regression

    # The resampling methods have to store the beta to then find the bias and the variance

    def bootstrap(self, N, random_state=None):
        '''
        Performs the bootstrap resampling method N times by randomly shuffling the data and checking it against a test data.

        Parameters
        ----------
        N : int
            Number of times it will perform the bootstrap.
        
        random_state : int
            Default None\n
            The seed for the randomness.
        
        Returns
        -------
        r2 (numpy array), mse (numpy array), bias (float), variance (float)
            Final R2 score and MSE after bootstrap has been done. Bias and variance of the model are also given.
        '''
        import helper
        from sklearn.utils import resample
        
        """
        does bootstrap sampling on the training data for N samples and returns bias, variance and the mean of R2 and mse.
        Also stores the orginial stuff.
        """

        original_X = self._reg.get_design()
        original_z = self._reg.get_known()

        #Fetch design matrix (design) and known  (known) values from LinearRegression object and X_train, X_test, z_train, z_test from helper our_tt_split
        X_train, X_test, z_train, z_test = helper.our_tt_split(self._reg.get_design(), self._reg.get_known(), test_size=0.2, random_state=random_state)

        #Create empty arrays to store predictions, R2 score and mse

        predictions = np.zeros((N,len(z_test)))
        R2 = np.zeros(N)
        mse = np.zeros(N)

        #Loop over N samples
        for i in range(N):
            #resample using sklearn.utils.resample
            X_resampled, y_resampled = resample(X_train, z_train)

            #Evaluate model on test data
            # self._reg.set_design(X_resampled,y_resampled)
            predictions[i,:] = self._reg.predict_resample(X_resampled, y_resampled, X_test)
            self._reg.set_known(z_test)
            R2[i] = self._reg.r2()
            mse[i] = self._reg.mse()
        
        #Calculate bias and variance
        bias = np.mean((z_test-np.mean(predictions,axis=0, keepdims=True))**2)
        variance = np.mean(np.var(predictions,axis=0, keepdims=True))

        # restores to the orginal to be able to do other resampling methodds in a fair way
        self._reg.set_design(original_X)
        self._reg.set_known(original_z)

        # return mean of R2 and mse, bias and variance
        return np.mean(R2), np.mean(mse), bias, variance, np.std(mse)

    def k_folds(self, k):
        '''
        Takes the design matrix and z from the LinearRegression object stored in the instance. Shuffles the indexes of these correspondingly and returns a set of k folds.

        Needs
        ------
        the length of the design matrix HAS to be the amount of elements in z which is len(np.ravel(z)).

        Parameters
        ---------
        k : int
            The number of folds to be made of the data
        
        Returns
        -------
        folds_design, folds_z : numpy array, numpy array

        Important note: the folded z are not reshaped! They are 1D arrays
        '''

        # we extract what we need
        design = self._reg.get_design()
        z = self._reg.get_known()
        z = np.ravel(z)

        # we shuffle the indexes
        new_ind = np.random.permutation(len(design))
        mix_design = design[new_ind]
        mix_z = z[new_ind]

        # now we need to split into folds and store these
        folds_design = np.array_split(mix_design, k)
        folds_z = np.array_split(mix_z, k)

        return folds_design, folds_z

    def cross_validation(self, k=5):
        '''
        Performs the Cross-validation with the chosen number of folds, default is 5. The folding is done in the k_folds() method. The original design matrix and z are the same as when the method was called as when it is done as they are stored before replaced and then they replace the temporary ones again before returning the desired values.

        Parameters
        ----------
        k : int
            number of folds to be made
        
        Returns
        -------
        mse, r2 : numpy array
            the mse of the cross validation for each polynomial degree with k folds.
        '''

        # stores the original desing matrix and z
        original_X = self._reg.get_design()
        original_z = self._reg.get_known()

        # makes k number of folds
        folds_design, folds_z = self.k_folds(k)

        # storage places for the MSE and R2 scores.
        R2 = np.zeros(k)
        mse = np.zeros(k)

        # we make an array which will be usefull to loop through to set up the training design matrix and the training z
        train_folds = np.linspace(0, k-1, k, dtype=int)

        # once for each fold we train the model
        for i in range(k):
            # takes out this round's testing data
            X_test = folds_design[i]
            z_test = folds_z[i]

            # indices for where the training data is
            temp_train_folds = np.delete(train_folds, i)

            # creates this round's training data by concatenating what is not the current test fold previously extracted
            X_train = np.concatenate([folds_design[i] for i in temp_train_folds])
            z_train = np.concatenate([folds_z[i] for i in temp_train_folds])

            # makes a model with the training data and makes a prediction
            self._reg.predict_resample(X_train, z_train, X_test)

            # evaluates the fit and stores the MSE and R2 score
            self._reg.set_known(z_test)
            R2[i] = self._reg.r2()
            mse[i] = self._reg.mse()
        
        # we cannot calculate bias-variance trade-off with cross validation as the test sample changes every time

        # restores to the orginal to be able to do other resampling methodds in a fair way
        self._reg.set_design(original_X)
        self._reg.set_known(original_z)

        return np.mean(R2), np.mean(mse)    # these are the mean MSE and R2 scores for the model degree with k folds

'''
        Here is some code to test the results the bootstrap method gives:

        from LinearRegression import LinearRegression
        from Resample import Resample
        from helper import *
        import numpy as np
        import matplotlib.pyplot as plt

        N = 100

        x = np.sort(np.random.rand(N)).reshape((-1, 1))
        y = np.sort(np.random.rand(N)).reshape((-1, 1))
        x, y = np.meshgrid(x, y)

        z = franke(x, y)# + np.random.rand(N, N)

        stop = 20
        start = 1

        r2 = np.zeros(stop - start)
        mse = np.zeros(stop - start)
        bias = np.zeros(stop - start)
        var = np.zeros(stop - start)
        orders = np.linspace(1, stop-1, stop-1)

        for i in range(start, stop):
            ols = LinearRegression(i, x, y, z)
            resampler = Resample(ols)
            r2[i-1], mse[i-1], bias[i-1], var[i-1] = resampler.bootstrap(N)

        # For Ridge
        # for i in range(start, stop):
        #     ols = LinearRegression(i, x, y, z, 2, 0.1)
        #     resampler = Resample(ols)
        #     r2[i-1], mse[i-1], bias[i-1], var[i-1] = resampler.bootstrap(N)

        # For Lasso
        # for i in range(start, stop):
        #     ols = LinearRegression(i, x, y, z, 3, 0.01)
        #     resampler = Resample(ols)
        #     r2[i-1], mse[i-1], bias[i-1], var[i-1] = resampler.bootstrap(N)

        plt.plot(orders, mse)
        plt.title("MSE")
        plt.show()

        plt.plot(orders, r2)
        plt.title("R2")
        plt.show()

        plt.plot(orders, bias)
        plt.plot(orders, var)
        plt.title("B-V")
        plt.show()

        ###################################################################################

        Extra code for testing the full functionality of the hierarchy:

from LinearRegression import LinearRegression
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
np.random.seed(123)
N = 20

x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)

z = franke(x, y) + np.random.rand(N, N)*0.25

stop = 12
start = 1

r2_cross = np.zeros(stop - start)
mse_cross = np.zeros(stop - start)

r2_boot = np.zeros(stop - start)
mse_boot = np.zeros(stop - start)
bias_boot = np.zeros(stop - start)
var_boot = np.zeros(stop - start)

orders = np.linspace(1, stop-1, stop-1)

for i in orders:
    print("At order: %d" %i, end='\r')

    i = int(i)  # gee okay

    ols = LinearRegression(i, x, y, z)
    resampler = Resample(ols)

    r2, mse = resampler.cross_validation()
    r2_cross[i-1] = r2
    mse_cross[i-1] = mse

    r2, mse, bias, var = resampler.bootstrap(50)

    r2_boot[i-1] = r2
    mse_boot[i-1] = mse
    bias_boot[i-1] = bias
    var_boot[i-1] = var

plt.figure(figsize=(15, 10))
plt.plot(orders, mse_cross, label="Cross-val")
plt.plot(orders, mse_boot, label="Bootstrap")
plt.title("MSE of both resampling methods.", fontsize=16)
plt.xlabel("Polynomial complexity", fontsize=16)
plt.ylabel("MSE", fontsize=16)
plt.legend()
plt.show()


plt.figure(figsize=(15, 10))
plt.plot(orders, r2_cross, label="Cross-val")
plt.plot(orders, r2_boot, label="Bootstrap")
plt.title("R2 score of both resampling methods", fontsize=16)
plt.xlabel("Polynomial complexity", fontsize=16)
plt.ylabel("R2", fontsize=16)
plt.legend()
plt.show()


plt.figure(figsize=(15, 10))
plt.plot(orders, bias_boot, label="Bias - Bootstrap")
plt.plot(orders, var_boot, label="Variantion - Bootstrap")
plt.title("Bias-Variance tradeoff of both methods.", fontsize=16)
plt.xlabel("Polynomial complexity", fontsize=16)
plt.ylabel("Bias-Variance", fontsize=16)
plt.legend()
plt.show()
        '''