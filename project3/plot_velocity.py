import seaborn as sns
from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme("notebook", "whitegrid", palette="colorblind")

cm=1/2.54
params = {
    'legend.fontsize': 9,
    'font.size': 9,
    'figure.figsize': (8.647*cm, 12.0*cm), # ideal figsize for two-column latex doc
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.markersize': 3.0,
    'lines.linewidth': 1.5,
    }

plt.rcParams.update(params)

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
dt=0.05
print(
np.mean(X) / dt,
np.var(X) / dt
)

print(sum(X[:,0] == 0))
plt.plot(X[:,0], '.')
plt.show()
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
scaler.fit(y)
y_scaled = scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=123)

# with open('models/ffnn.pickle','rb') as f:
#     network=pickle.load(f)
# with open('models/svr.pickle','rb') as f:
#     svr=pickle.load(f)
# with open('models/linreg.pickle','rb') as f:
#     linreg=pickle.load(f)

# y_pred_ffnn = network.predict(X_test)
# y_pred_svr = svr.predict(X_test)
# y_pred_svr_unscaled = scaler.inverse_transform(y_pred_svr)
# y_pred_linreg = linreg.predict(X_test)

# r2_score_ffnn = r2_score(y_true=y_test, y_pred=y_pred_ffnn)
# r2_score_svr = r2_score(y_true=y_test, y_pred=y_pred_svr_unscaled)
# r2_score_linreg = r2_score(y_true=y_test, y_pred=y_pred_linreg)
# print("Test R2 scores")
# print(r2_score_ffnn, r2_score_svr, r2_score_linreg)
# mse_ffnn = mean_squared_error(y_true=y_test, y_pred=y_pred_ffnn)
# mse_svr = mean_squared_error(y_true=y_test, y_pred=y_pred_svr_unscaled)
# mse_linreg = mean_squared_error(y_true=y_test, y_pred=y_pred_linreg)
# print("RMSE score:")
# print(np.sqrt(mse_ffnn), np.sqrt(mse_svr), np.sqrt(mse_linreg))

## ---------------------------

# K = 200
# dt = 50/1000 #s
# time = np.arange(0, K*dt, dt)
# y_pred_ffnn = network.predict(X_scaled[:K])
# y_pred_svr = svr.predict(X_scaled[:K])
# y_pred_svr_unscaled = scaler.inverse_transform(y_pred_svr)
# y_pred_linreg = linreg.predict(X_scaled[:K])

# y_true = y[:K]

# r2_score_ffnn = r2_score(y_true=y_true, y_pred=y_pred_ffnn)
# r2_score_svr = r2_score(y_true=y_true, y_pred=y_pred_svr_unscaled)
# r2_score_linreg = r2_score(y_true=y_true, y_pred=y_pred_linreg)

# print("R2 score:")
# print(r2_score_ffnn, r2_score_svr, r2_score_linreg)

# mse_ffnn = mean_squared_error(y_true=y_true, y_pred=y_pred_ffnn)
# mse_svr = mean_squared_error(y_true=y_true, y_pred=y_pred_svr_unscaled)
# mse_linreg = mean_squared_error(y_true=y_true, y_pred=y_pred_linreg)

# print("RMSE score:")
# print(np.sqrt(mse_ffnn), np.sqrt(mse_svr), np.sqrt(mse_linreg))

# fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)

# ax[0].plot(time, y_true[:,1], 'k-', label = "Observed")
# ax[0].plot(time,y_pred_ffnn[:,1],'g-', label ="Predicted")
# ax[0].set_title("FFNN")

# ax[1].set_ylabel("$y$ velocity [cm/s]")

# ax[1].plot(time, y_true[:,1], 'k-', label = "Observed")
# ax[1].plot(time, y_pred_svr_unscaled[:,1],'r-', label ="Predicted")
# ax[1].set_title("SVR")

# ax[2].plot(time, y_true[:,1], 'k-', label = "Observed")
# ax[2].plot(time, y_pred_linreg[:,1],'b-', label = "Predicted")
# ax[2].set_title("Linear model")
# ax[2].set_xlabel("Time [s]")

# # handles, labels = ax[0].get_legend_handles_labels()
# # fig.legend(handles, labels)
# plt.tight_layout()
# plt.savefig("figs/predicted_y_vel.pdf")