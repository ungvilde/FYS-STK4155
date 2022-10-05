import numpy as np
from Errors import Errors
from LinearRegression import LinearRegression
from sklearn.preprocessing import normalize


class Resample():

    _reg = None
    _betas = None
    _fits = None

    def __init__(self, regression):
        self._reg = regression

    # The resampling methods have to store the beta to then find the bias and the variance

    def bootstrap(self, N, random_state=None):
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
        return np.mean(R2), np.mean(mse), bias, variance


    def k_folds(self, k):
        '''splits available data into chosen number of folds for cross validation. Folds are made from the design matrix!
        The design matrix is taken and the indices are shuffled and given as a an arrya of indeices so we have correspondence between the test and train data results.
        k=5 is the default of sklearn so we will use it as default aswell.
        NOTICE THERE IS NO TRAIN TEST SPLIT
        '''

        # we start by shuffling
        design = self._reg.get_design()

        z = self._reg.get_known()
        z = np.ravel(z)

        new_ind = np.random.permutation(len(design))
        mix_design = design[new_ind]
        mix_z = z[new_ind]

        # now we need to split into folds and store these
        folds_design = np.array_split(mix_design, k)
        folds_z = np.array_split(mix_z, k)

        return folds_design, folds_z

    def cross_validation(self, k=5):
        "does cross validation with k folds, stores original X and z."
        original_X = self._reg.get_design()
        original_z = self._reg.get_known()

        folds_design, folds_z = self.k_folds(k)

        predictions = np.zeros((k, len(folds_z[0])))
        R2 = np.zeros(k)
        mse = np.zeros(k)

        for i in range(k):
            X_test = folds_design[i]
            X_train = np.delete(folds_design, i, 0)
            X_train = np.array(X_train).reshape(-1,np.shape(X_test)[-1])

            z_test = folds_z[i]
            z_train = np.delete(folds_z, i, 0)
            z_train = np.array(z_train).reshape(-1, np.shape(folds_z)[-1])

            predictions[i,:] = self._reg.predict_resample(X_train, z_train, X_test)
            self._reg.set_known(z_test)
            R2[i] = self._reg.r2()
            mse[i] = self._reg.mse()
        
        # we cannot calculate bias-variance trade-off with cross validation as the test sample changes every time

        # restores to the orginal to be able to do other resampling methodds in a fair way
        self._reg.set_design(original_X)
        self._reg.set_known(original_z)

        return np.mean(R2), np.mean(mse)

    def var_beta(self):
        "finds the variance of beta set"

    def mse(self):
        "returns the mse of the performed crossvalidation"

    
    def var_model(self):
        '''Calculates the variance of the model once the resampling has been done. Returns a number!'''

        return np.var(self._fits, axis=1)

    def bias(self):
        '''Calculates the bias once the resampling has been done. Returns a number!'''

        known = self._reg.get_known()
        N = len(known)
        mean = self.estimate_fit()

        return np.sum((known - mean)**2) / N

    def estimate_fit(self):
        '''Calculates the mean of the different fits from the betas. Returns an array.'''

        return np.mean(self._fits, axis=1)



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