from LinearRegression import LinearRegression
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_theme()
np.random.seed(42)
N = 9

x = np.sort(np.random.rand(N)).reshape((-1, 1))
y = np.sort(np.random.rand(N)).reshape((-1, 1))
x, y = np.meshgrid(x, y)

z = franke(x, y) + 0.05* np.random.rand(N, N)
stop = 8
start = 0

## create design matrix using biggest poly_degree
Linreg = LinearRegression(stop, x, y, z)
design = Linreg.get_design()


# now we need to do the train test split OUTSIDE THE LOOP
X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.3)

mse_list = []
r2_list = []

orders=np.linspace(1, stop, stop)
for i in orders:
    print("At order: %d" %i, end='\r')
    i = int(i)
    slice = int((i+1)*(i+2)/2)

    #Linreg.set_order(i)
    Linreg.set_beta(X_train[:, :slice], z_train)
    
    mse, r2 = Linreg.predict(X_test[:, :slice], z_test)
    beta = Linreg.get_beta()
    plt.plot(i,beta[0],'or',label=r'$\beta_0$'+f' d={i}')
    plt.plot(i,beta[1],'ob',label=r'$\beta_1$'+f' d={i}')
    plt.plot(i,beta[2],'og',label=r'$\beta_2$'+f' d={i}')
    if i>3:

        plt.plot(i,beta[3],'oy',label=r'$\beta_3$'+f' d={i}')

    if i>4:

        plt.plot(i,beta[4],'ok',label=r'$\beta_4$'+f' d={i}')

    if i>5:

        plt.plot(i,beta[5],'oc',label=r'$\beta_5$'+f' d={i}')

    if i>6:

        plt.plot(i,beta[6],'om',label=r'$\beta_6$'+f' d={i}')

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

#plt.plot(orders, mse_list)
#plt.title("MSE")
plt.show()

#plt.plot(orders, r2_list)
#plt.title("R2")
#plt.show()

'''plt.plot(orders, bias)
plt.plot(orders, var)
plt.title("B-V")
plt.show()'''

    ###################################################################################

    #Extra code for testing the full functionality of the hierarchy:


