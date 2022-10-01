from LinearRegression import LinearRegression
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_theme()
np.random.seed(123)
N = 100

x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)

z = franke(x, y) + np.random.rand(N, N)*0.25

stop = 20
start = 0

#Split into train test split from helper our_tt_split

#Create a linear regression object
orders=np.linspace(1, stop, stop)

mse_list = []
r2_list = []

for i in orders:
    print("At order: %d" %i, end='\r')
    i = int(i)
    ols = LinearRegression(i, x, y, z)
    mse, r2 = ols.split(fit=True)
    mse_list.append(mse)
    r2_list.append(r2)

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

plt.plot(orders, mse_list)
plt.title("MSE")
plt.show()

plt.plot(orders, r2_list)
plt.title("R2")
plt.show()

'''plt.plot(orders, bias)
plt.plot(orders, var)
plt.title("B-V")
plt.show()'''

    ###################################################################################

    #Extra code for testing the full functionality of the hierarchy:


