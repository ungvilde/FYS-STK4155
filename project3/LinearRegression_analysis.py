import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from preprocessing import *
from ResampleMethods import CrossValidation
from linear_model import LinearRegression

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

scaler = RobustScaler()
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

linreg = LinearRegression() #lmbda
mse, r2 = CrossValidation(linreg, X_train, y_train)
print("mse = ", mse)
print("r2 = ", r2)