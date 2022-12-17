import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from preprocessing import *
from ResampleMethods import *

with open('datasets/neural_data.pickle','rb') as f:
    neural_data,vels_binned=pickle.load(f)

bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0

X=get_spikes_with_history(neural_data, bins_before, bins_after,bins_current)
y=vels_binned

X = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
X = X[(bins_before+1):, :]
y = y[(bins_before+1):]
print(X.shape)
print(y.shape)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
scaler.fit(y)
y_scaled = scaler.transform(y)
print(X[:10,:])
print(y[:10,:])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=123)

print("gamma = ", 1/(X_train.shape[1] * X_train.var()))

print("Regularisation params to search:")
C_vals=np.logspace(0, 5, 10)
print(C_vals)

print("Kernel coefficients to search:")
scale= 1/(X_train.shape[1] * X_train.var())
gamma_vals =  [0.1*scale, 0.2*scale, 0.3*scale, 0.5*scale, scale]
print(gamma_vals)

hyperparameters_to_search = {
    'estimator__C': C_vals,
    'estimator__gamma': gamma_vals
    }
svr = MultiOutputRegressor(SVR(max_iter=2000, cache_size=1000))

# regression = GridSearchCV(
#     svr, 
#     cv = 5, 
#     param_grid = hyperparameters_to_search, 
#     refit=True,
#     scoring="r2", 
#     n_jobs=-1, 
#     verbose=3
#     )

# print("Do the search (can take some time!)")
# search = regression.fit(X_train, y_train)

# print("Best parameters from gridsearch:")
# print(search.best_params_)

# print("Best CV R2 score from gridsearch:")
# print(search.best_score_)

# print("R2 score on test data:")
# r2 = search.score(X_test, y_test)
# print("R2 = ", r2)

# with open('datasets/cv_svr_results.pickle','wb') as f:
#     pickle.dump(search.cv_results_, f)

svr = MultiOutputRegressor(SVR(max_iter=2000, cache_size=1000, C=3.5938136638046276, gamma=0.00021767389160393826))
svr.fit(X_train, y_train)
with open('models/svr.pickle', 'wb') as f:
    pickle.dump(svr, f)
