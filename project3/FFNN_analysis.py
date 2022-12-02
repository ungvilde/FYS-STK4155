import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from preprocessing import *

with open('datasets/neural_data.pickle','rb') as f:
    neural_data, vels_binned=pickle.load(f)

bins_before=13 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0

X=get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
y=vels_binned

X = X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
X = X[(bins_before+1):, :] #the first 13 bins will not be used, because they lack covariates
y = y[(bins_before+1):]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

network = MLPRegressor(max_iter=1000)

print("Learning rates to search:")
eta_vals=np.logspace(-4, -1, 4) 
print(eta_vals)

print("Regularization params to search:")
alpha_vals = np.logspace(1, 3, 5)
print(alpha_vals)

hyperparameters_to_search = {
    "hidden_layer_sizes" : [ [100, 100], [100, 100, 100] ],
    "alpha" : alpha_vals,
    "learning_rate_init" : eta_vals
}

regression = GridSearchCV(
    network, 
    param_grid=hyperparameters_to_search, 
    scoring="r2", 
    n_jobs=-1, # use all cores in parallel
    refit=True,
    cv=5, # number of folds
    verbose=3
    )

print("Do the search (can take some time!)")
search = regression.fit(X_train, y_train)

print("Best parameters from gridsearch:")
print(search.best_params_)

print("Best CV R2 score from gridsearch:")
print(search.best_score_)

print("R2 score on test data:")
r2 = search.score(X_test, y_test)
print("R2 = ", r2)

with open('datasets/cv_ffnn_results2.pickle','wb') as f:
    pickle.dump(search.cv_results_, f)

