import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from preprocessing import *
from ResampleMethods import CrossValidation
#from linear_model import LinearRegression

with open('datasets/neural_data.pickle','rb') as f:
    neural_data, vels_binned = pickle.load(f)

bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0

X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y = vels_binned

X = X.reshape(X.shape[0] , (X.shape[1]*X.shape[2]))
X = X[(bins_before+1):, :]
y = y[(bins_before+1):]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=123)

# linreg = Ridge() #lmbda
# lambda_vals = np.logspace(-8, 1, 100)
# lambda_vals = np.logspace(3, 5, 100)

# lambda_vals = np.concatenate(([0], lambda_vals))
# print(lambda_vals)
# hyperparameters_to_search = {
#     'alpha': lambda_vals,
#     }

# regression = GridSearchCV(linreg, param_grid=hyperparameters_to_search, scoring='r2', cv=5, refit=True, verbose=3, n_jobs=-1)

# print("Do the search (can take some time!)")
# search = regression.fit(X_train, y_train)

# print("Best parameters from gridsearch:")
# print(search.best_params_)

# print("Best CV R2 score from gridsearch:")
# print(search.best_score_)

# print("R2 score on test data:")
# r2 = search.score(X_test, y_test)
# print("R2 = ", r2)

# with open('datasets/cv_linreg_results3.pickle','wb') as f:
#     pickle.dump(search.cv_results_, f)

linreg = Ridge(alpha=7390.722033525775)
linreg.fit(X_train, y_train)
with open('models/linreg.pickle', 'wb') as f:
    pickle.dump(linreg, f)