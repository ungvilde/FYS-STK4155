import numpy as np
from Errors import Errors
from LinearRegression import LinearRegression

class Resample():

    _reg = None
    _betas = None
    _fits = None

    def __init__(self, regression):
        self._reg = regression

    # The resampling methods have to store the beta to then find the bias and the variance

    def bootstrap(self, N):
        import helper
        from sklearn.utils import resample
        
        
        """
        does bootstrap sampling on the training data for N samples and returns bias, variance and the mean of R2 and mse 
        """
        
        #Fetch design matrix (design) and known  (known) values from LinearRegression object and X_train, X_test, y_train, y_test from helper our_tt_split
        X_train, X_test, y_train, y_test = helper.our_tt_split(self._reg.get_design(), self._reg.get_known(), test_size=0.2)

        #Create empty arrays to store predictions, R2 score and mse

        predictions = np.zeros((N,len(y_test)))
        R2 = np.zeros(N)
        mse = np.zeros(N)

        #Loop over N samples
        for i in range(N):
            #resample using sklearn.utils.resample
            X_resampled, y_resampled = resample(X_train, y_train)

            #Evaluate model on test data
            # self._reg.set_design(X_resampled,y_resampled)
            predictions[i,:] = self._reg.predict_resample(X_resampled, y_resampled, X_test)
            self._reg.set_known(y_test)
            R2[i] = self._reg.r2()
            mse[i] = self._reg.mse()
        
        #Calculate bias and variance
        bias = np.mean((y_test-np.mean(predictions,axis=0, keepdims=True))**2)
        variance = np.mean(np.var(predictions,axis=0, keepdims=True))

        # return mean of R2 and mse, bias and variance
        return np.mean(R2), np.mean(mse), bias, variance




    def cross_validation(self, k):
        "does cross validation with k folds"

        design_fold, z_fold = self.folds(k)

        predictions = []
        R2 = np.zeros(k)
        mse = np.zeros(k)

        for i in range(k):
            temp_design = design_fold
            temp_z = z_fold

            test_design = temp_design.pop(i)
            test_z = temp_z.pop(i)

            temp_design = np.concatenate(temp_design)
            temp_z = np.concatenate(temp_z)

            predictions.append(self._reg.predict_resample(temp_design, temp_z, test_design))
            self._reg.set_known(test_z)
            R2[i] = self._reg.r2()
            mse[i] = self._reg.mse()
        
        #Calculate bias and variance
        predictions = np.concatenate(predictions)

        bias = np.mean((test_z - np.mean(predictions,axis=0, keepdims=True))**2)
        variance = np.mean(np.var(predictions,axis=0, keepdims=True))

        # return mean of R2 and mse, bias and variance
        return np.mean(R2), np.mean(mse), bias, variance
    
    def k_folds(self, k):
        '''splits available data into chosen number of folds for cross validation. Folds are made from the design matrix!
        The design matrix is taken and the indices are shuffled and given as a an arrya of indeices so we have correspondence between the test and train data results.'''
        
        # we start by shuffling
        design = self._reg.get_design()
        z = self._reg.get_known()
        new_ind = np.random.permutation(len(design))
        mix_design = design[new_ind]
        mix_z = z[new_ind]

        # now we need to split into folds and store these
        fold_design = []
        fold_z = []
        start = 0
        step = np.floor(len(z)/k)

        for i in range(k-1):
            fold = np.asarray(mix_design[start:start+step])
            fold_design.append(fold)
            fold = np.asarray(mix_z[start:start+step])
            fold_z.append(fold)
            start += step
        
        fold_design.append(np.asarray(mix_design[start::]))
        fold_z.append(np.asarray(mix_z[start::]))

        return fold_design, fold_z
        

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
        '''