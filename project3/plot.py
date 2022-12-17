import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

sns.set_theme("notebook", "whitegrid", palette="colorblind")

cm=1/2.54
params = {
    'legend.fontsize': 9,
    'font.size': 9,
    'figure.figsize': (8.647*cm, 8.0*cm), # ideal figsize for two-column latex doc
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.markersize': 3.0,
    'lines.linewidth': 1.0,
    }

plt.rcParams.update(params)

with open('datasets/cv_ffnn_results.pickle','rb') as f:
    ffnn_res=pickle.load(f)

with open('datasets/cv_ffnn_results2.pickle','rb') as f:
    ffnn_res2=pickle.load(f)

with open('datasets/cv_svr_results.pickle','rb') as f:
    svr_res=pickle.load(f)

with open('datasets/cv_linreg_results.pickle','rb') as f:
    linreg_res=pickle.load(f)




colnames_ffnn = ['param_alpha', 'param_hidden_layer_sizes', 'param_learning_rate_init',
    'split0_test_score', 'split1_test_score', 'split2_test_score',
    'split3_test_score', 'split4_test_score', 'mean_test_score',
    'std_test_score', 'rank_test_score']
colnames_svr = ['param_estimator__C', 'param_estimator__gamma', 
       'split0_test_score', 'split1_test_score', 'split2_test_score',
       'split3_test_score', 'split4_test_score', 'mean_test_score',
       'std_test_score', 'rank_test_score']
colnames_linreg = ['param_alpha',
    'split0_test_score', 'split1_test_score', 'split2_test_score',
    'split3_test_score', 'split4_test_score', 'mean_test_score',
    'std_test_score', 'rank_test_score']

ffnn_res = pd.DataFrame(ffnn_res)
ffnn_res = ffnn_res.sort_values('rank_test_score')

print(ffnn_res[['mean_test_score', 'std_test_score']])

ffnn_res = ffnn_res[colnames_ffnn]

ffnn_res2 = pd.DataFrame(ffnn_res2)
ffnn_res2 = ffnn_res2.sort_values('rank_test_score')

print(ffnn_res2[['mean_test_score', 'std_test_score']])

ffnn_res2 = ffnn_res2[colnames_ffnn]

best = ffnn_res2.iloc[0,:]
print(best)

svr_res = pd.DataFrame(svr_res)
svr_res = svr_res[colnames_svr]
svr_res = svr_res.sort_values('rank_test_score')
print(svr_res[['mean_test_score', 'std_test_score']])

best = svr_res.iloc[0,:]
print(best)

linreg_res = pd.DataFrame(linreg_res)
linreg_res = linreg_res[colnames_linreg]
linreg_res = linreg_res.sort_values('rank_test_score')
best = linreg_res.iloc[0,:]
print(best)

