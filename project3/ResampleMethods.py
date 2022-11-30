import numpy as np

from sklearn.model_selection import train_test_split
from metrics import mean_squared_error, r2_score

def Bootstrap(model, X, y, B=100):

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)    
    n_train = Xtrain.shape[0]
    indeces = np.arange(n_train)

    MSE_values = []
    R2_values = []

    for _ in range(B):
        resampled_inds = np.random.choice(indeces, size = n_train, replace=True)
        X = Xtrain[resampled_inds]
        y = ytrain[resampled_inds]
        model.fit(X, y) # is a little slow with sgd

        predicted = model.predict(Xtest)
        MSE_values.append( mean_squared_error(y_true=predicted, y_pred=ytest) )
        R2_values.append( r2_score(y_true=ytest, y_pred=predicted) )

    return np.mean(MSE_values), np.mean(R2_values)

def CrossValidation(model, X, y, k=10):

    # make folds for training
    n_train = X.shape[0]  
    indeces = np.arange(n_train)
    shuffled_indeces = np.random.choice(indeces, replace=False, size=n_train)
    folds = np.array_split(shuffled_indeces, k)

    MSE_values = []
    MSE_values = np.zeros(k)
    R2_values = []
    R2_values = np.zeros(k)

    for i, test_fold in enumerate(folds):
        train_fold = folds.copy()
        train_fold.pop(i) #remove test fold 
        train_fold = np.concatenate(train_fold) # combines indeces into a single array
        Xtrain = X[train_fold]
        ytrain = y[train_fold]
        Xtest = X[test_fold]
        ytest = y[test_fold]
        model.fit(Xtrain, ytrain)
        predicted = model.predict(Xtest)

        #MSE_values.append( mean_squared_error(y_true=predicted, y_pred=ytest) )
        #R2_values.append( r2_score(y_true=ytest, y_pred=predicted) )
        MSE_values[i] = mean_squared_error(y_true=predicted, y_pred=ytest)
        R2_values[i] = r2_score(y_true=ytest, y_pred=predicted)

    return np.mean(MSE_values), np.mean(R2_values) #should include error of estimates too

