from cProfile import label
from re import L
from imageio import imread
from LinearRegression import LinearRegression
from sklearn.preprocessing import normalize
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme()

project_data = input("Which data do you want to plot? (Franke or Terrain) ")
project_section = input('Which part of project 1 you want to plot? (b, c, d, e, f)')

def get_data_franke(N,noise):
    """
    Get the data for the Franke function.
    """
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    x, y = np.meshgrid(x,y)
    z = franke(x,y) + noise*np.random.randn(N,N)

    return x, y, z

def get_data_terrain(N):
    """
    Get the data for the terrain data.
    """
    terrain = imread('SRTM_data_Norway_1.tif')
    terrain = terrain[0:N,0:N]
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    x, y = np.meshgrid(x,y)
    z = terrain

    return x, y, z


########## PART B ####################
def part_b_request1(show_betas=False):
    """
    Perform a standard ordinary least square regression analysis using polynomials in x and y up to fifth order.
    Evaluate the mean Squared error (MSE) and the R2 score function.

    Notice this uses the test data to plot and that is why the mse starts to go up.
    NOTICE THIS HAS SCALING ALREADY!
    """
    np.random.seed(42)

    if project_data == "F":
        N = 12 ## number of points will be N x N

        x, y, z = get_data_franke(N, noise=0.1)
    if project_data == "T":
        N = 10
        x, y, z = get_data_terrain(N)

    max_degree = 5

    mse_list = []
    r2_list = []
    orders=np.linspace(1, max_degree, max_degree)
    for i in orders:
        print("At order: %d" %i, end='\r')

        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=False)
        mse, r2 = ols.split_predict_eval(test_size=0.2, fit=True, train=False, random_state=42)
        mse_list.append(mse)
        r2_list.append(r2)

        if show_betas == True:
            beta = ols.get_beta()
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


    if show_betas == True:

        plt.title("Betas x Model Complexity - TEST")
        plt.xlabel('Polynomial Degree', fontsize=12)
        plt.ylabel('Betas', fontsize=12)
        plt.show()
        exit(1)

    #(Adam) - Plotting Errors together and saving figs
    # Plotting MSE and R2 in same figure with two y-axis
    print(f"Sum:{mse_list+r2_list} ")
    fig, ax1 = plt.subplots()
    ax1.plot(orders, mse_list, 'b-')
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('MSE', color='b', fontsize=12)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, 'r-')
    ax2.set_ylabel('R2', color='r', fontsize=12)
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig(f'Figs/Errors_x_order_{max_degree}_TEST_'+str(project_data)+'.pdf')
    plt.show()




def part_b_request1_extra():
    """
    Here I plot the same thing as previously but with the test data as well AND with a degree larger than 5
    """
    np.random.seed(42)

    if project_data == "F":
        N = 12 ## number of points will be N x N

        x, y, z = get_data_franke(N,noise=0.1)
    if project_data == "T":
        N = 10
        x, y, z = get_data_terrain(N)

    
    max_degree = 15

    mse_list = []
    r2_list = []
    orders=np.linspace(1, max_degree, max_degree)
    for i in orders:
        print("At order: %d" %i, end='\r')

        i = int(i)
        ols = LinearRegression(i, x, y, z)
        mse, r2 = ols.split_predict_eval(test_size=0.2, fit=True, train=False, random_state=42)
        mse_list.append(mse)
        r2_list.append(r2)


    
    # Plotting MSE and R2 in same figure with two y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(orders, mse_list, 'b-')
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('MSE', color='b', fontsize=12)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, 'r-')
    ax2.set_ylabel('R2', color='r', fontsize=12)
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig(f'Figs/Errors_x_order_{max_degree}_TEST_'+str(project_data)+'.pdf')
    plt.show()




    mse_list = []
    r2_list = []
    ## The only difference is that this is the TRAIN plot of mse
    for i in orders:
        print("At order: %d" %i, end='\r')

        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        mse, r2 = ols.split_predict_eval(test_size=0.2, fit=True, train=True, random_state=42)
        mse_list.append(mse)
        r2_list.append(r2)



    
    
    # Plotting MSE and R2 in same figure with two y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(orders, mse_list, 'b-')
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('MSE', color='b', fontsize=12)
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, 'r-')
    ax2.set_ylabel('R2', color='r', fontsize=12)
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig(f'Figs/Errors_x_order_{max_degree}_TRAIN_'+str(project_data)+'.pdf')
    plt.show()



def part_b_request2():
    """
    Plot also the parameters beta as you increse the order of the polynomials!
    """
    part_b_request1(show_betas=True)


########## PART C ####################
def part_c_request1():
    """
    This is super similar to the part b request 1 extra plot, but now showing together the train and test and also
    NOTICE: this is without bootstrap!!!
    """
    np.random.seed(41)

    max_degree = 12
    if project_data == "F":
        N = 15 ## number of points will be N x N

        x, y, z = get_data_franke(N,noise=0.15)
    if project_data == "T":
        N = 10
        x, y, z = get_data_terrain(N)

    
    mse_list_train = []
    mse_list_test = []
    orders=np.linspace(1, max_degree, max_degree)
    for i in orders:
        print("At order: %d" %i, end='\r')

        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        mse_test, r2_test = ols.split_predict_eval(test_size=0.2, fit=True, train=False, random_state=42)
        ols = LinearRegression(i, x, y, z, scale=True)
        mse_train, r2_train = ols.split_predict_eval(test_size=0.2, fit=True, train=True, random_state=42)

        mse_list_train.append(mse_train)
        mse_list_test.append(mse_test)

    plt.plot(orders, mse_list_train,label='Train')
    plt.plot(orders, mse_list_test,label='Test')
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('prediction Error', fontsize=12)
    plt.legend()
    plt.savefig(f'Fig_2_11_'+str(project_data)+'.pdf')
    plt.show()


def part_c_request2():
    """
    This is super similar to the part c request 1 but now using bootstrap and showing the bv tradeoff
    """
    np.random.seed(41)

    N = 40 ## number of points will be N x N

    if project_data == "F":
        x, y, z = get_data_franke(N,noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    
    stop = 15
    start = 1

    r2 = np.zeros(stop - start)
    mse = np.zeros(stop - start)
    bias = np.zeros(stop - start)
    var = np.zeros(stop - start)
    orders = np.linspace(1, stop-1, stop-1)

    for i in range(start, stop):
        ols = LinearRegression(i, x, y, z)
        resampler = Resample(ols)
        r2[i-1], mse[i-1], bias[i-1], var[i-1] = resampler.bootstrap(N, random_state=42) ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    #print(f"Z avg:{np.mean(z)} ")
    plt.plot(orders, mse, label="MSE")

    plt.plot(orders, bias, label="Bias")
    plt.plot(orders, var, label="Variance")
    plt.legend()

    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('prediction Error', fontsize=12)
    plt.savefig('B-V_Tradeoff_Bootstrap_'+str(project_data)+'.pdf')
    plt.show()

########## PART D ####################

def part_d_request1():
    """
    Here the request is simple: compare the MSE from boostrap to the MSE from cross-validation with k from 5 to 10.
    Notice we will assess only for test MSE.

    SHOULD WE USE THE SCALING HERE? IF YES, IT SHOULD BE ADDED TO THE LinearRegression as an option
    """
    np.random.seed(41)

    N = 40 ## number of points will be N x N

    if project_data == "F":
        x, y, z = get_data_franke(N,noise=0.15)
    if project_data == "T":

        x, y, z = get_data_terrain(N)
    stop =20
    start = 1

    k=10
    r2_cross = np.zeros(stop - start)
    mse_cross = np.zeros(stop - start)

    orders = np.linspace(1, stop-1, stop-1)

    r2_boostrap = np.zeros(stop - start)
    mse_boostrap = np.zeros(stop - start)
    bias_boostrap = np.zeros(stop - start)
    var_boostrap = np.zeros(stop - start)

    for i in orders:
        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(ols)
        r2_cross[i-1] , mse_cross[i-1] = resampler.cross_validation(k=k)

        ols = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(ols)
        r2_boostrap[i-1], mse_boostrap[i-1], bias_boostrap[i-1], var_boostrap[i-1] = resampler.bootstrap(10, random_state=42)

    plt.plot(orders, mse_cross, label=f"MSE Crossvalidation k = {k}")
    plt.plot(orders, mse_boostrap, label="MSE Boostrap")

    plt.legend()
    plt.show()

########## PART E ####################

def part_e_request1():
    """
    Perform the same bootstrap analysis as in the part c for the same plynomials but now for RIDGE
    """
    np.random.seed(41)

    N = 10 ## number of points will be N x N

    if project_data == "F":
        x, y, z = get_data_franke(N,noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    stop = 15
    start = 1

    n_lambdas = 50
    lambdas = np.logspace(-5, 4, n_lambdas)
    orders = np.linspace(1, stop-1, stop-1)

    r2 = np.zeros((stop - start, n_lambdas))
    mse = np.zeros((stop - start, n_lambdas))
    bias = np.zeros((stop - start, n_lambdas))
    var = np.zeros((stop - start, n_lambdas))

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            ridge = LinearRegression(i, x, y, z, method=2, lmbd=lmbd, scale=True)
            resampler = Resample(ridge)
            r2[i-1,j], mse[i-1,j], bias[i-1,j], var[i-1,j] = resampler.bootstrap(N, random_state=42) ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    mse_min = np.min(mse)

    i_min, j_min = np.where(mse == mse_min)
    lambdas = np.log10(lambdas)
    lambdas, orders = np.meshgrid(lambdas, orders)

    plt.contourf(lambdas, orders, mse)

    plt.plot(lambdas[j_min, i_min], orders[i_min, j_min], '+', c='r')
    plt.colorbar()

    #plt.plot(orders, bias, label="Bias")
    #plt.plot(orders, var, label="Variance")
    #plt.legend()

    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('prediction Error', fontsize=12)
    plt.savefig('B-V_Tradeoff_Bootstrap_'+str(project_data)+'.pdf')
    plt.show()


############################################


if project_section == "b":
    part_b_request1()

    part_b_request1_extra()
    part_b_request2()


if project_section == "c":
    part_c_request1()
    part_c_request2()


if project_section == "d":
    part_d_request1()


if project_section == "e":
    part_e_request1()











## lambdas = np.logspace(-8, 8, 100)
## mse_list_ridge = []
## mse_list_lasso = []
## 
## for lmbd in lambdas:
##     ## create design matrix using biggest poly_degree
##     Linreg = LinearRegression(stop, x, y, z, 2, lmbd)
##     design = Linreg.get_design()
##     # now we need to do the train test split OUTSIDE THE LOOP
##     X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.3)
## 
##     #print("At order: %d" %i, end='\r')
##     i = int(8)
##     slice = int((i+1)*(i+2)/2)
## 
##     #Linreg.set_order(i)
##     Linreg.set_beta(X_train[:, :slice], z_train)
##     mse_ridge, r2_ridge = Linreg.predict(X_test[:, :slice], z_test)
## 
##     Linreg = LinearRegression(stop, x, y, z, 3, lmbd)
##     design = Linreg.get_design()
##     # now we need to do the train test split OUTSIDE THE LOOP
##     X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.3)
## 
##     #print("At order: %d" %i, end='\r')
##     i = int(4)
##     slice = int((i+1)*(i+2)/2)
## 
##     #Linreg.set_order(i)
##     Linreg.set_beta(X_train[:, :slice], z_train)
##     mse_lasso, r2_lasso = Linreg.predict(X_test[:, :slice], z_test)
## 
## 
##     mse_list_ridge.append(mse_ridge)
##     mse_list_lasso.append(mse_lasso)
## 
## 
## plt.plot(np.log10(lambdas), mse_list_ridge)
## plt.plot(np.log10(lambdas), mse_list_lasso)
## 
## plt.title("MSE")
## plt.show()
## 
## #plt.plot(orders, r2_list)
## #plt.title("R2")
## #plt.show()
## 
## '''plt.plot(orders, bias)
## plt.plot(orders, var)
## plt.title("B-V")
## plt.show()'''
## 
##     ###################################################################################
## 
##     #Extra code for testing the full functionality of the hierarchy:
## 
## 
## 