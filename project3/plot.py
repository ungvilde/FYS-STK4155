import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open('datasets/cv_ffnn_results.pickle','rb') as f:
    ffnn_res=pickle.load(f)

print(pd.DataFrame(ffnn_res).T
)