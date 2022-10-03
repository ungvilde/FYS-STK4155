from LinearRegression import LinearRegression
from sklearn.preprocessing import normalize
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_theme()

project_section = input('Which part of project 1 you want to plot? (b,c,d,e)')



########## PART B ####################
def part_b_request1(show_betas=False):
    """
    Perform a standard ordinary least square regression analysis using polynomials in x and y up to fifth order.
    Evaluate the mean Squared error (MSE) and the R2 score function.

    Notice this uses the test data to plot and that is why the mse starts to go up.
    NOTICE THIS HAS SCALING!
    """
    np.random.seed(42)
    N = 13

    x = np.sort(np.random.rand(N)).reshape((-1, 1))
    y = np.sort(np.random.rand(N)).reshape((-1, 1))
    x, y = np.meshgrid(x, y)

    z = franke(x, y) + 0.3* np.random.rand(N, N)
    stop = 6
    start = 0

    ## create design matrix using biggest poly_degree
    Linreg = LinearRegression(stop, x, y, z)
    design = Linreg.get_design()

    # We need here to scale/ center the data
    design = normalize(design)

    # now we need to do the train test split OUTSIDE THE LOOP
    X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.2, random_state=42)

    mse_list = []
    r2_list = []
    orders=np.linspace(1, stop, stop)
    for i in orders:
        i = int(i)
        slice = int((i+1)*(i+2)/2)

        Linreg.set_beta(X_train[:, :slice], z_train)
        mse, r2 = Linreg.predict(X_test[:, :slice], z_test)
        mse_list.append(mse)
        r2_list.append(r2)

        if show_betas == True:
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


    if show_betas == True:

        plt.title("Betas x Model Complexity - TEST")
        plt.xlabel('Polynomial Degree', fontsize=12)
        plt.ylabel('Betas', fontsize=12)
        plt.show()
        exit(1)

    plt.plot(orders, mse_list)
    plt.title("MSE x Model Complexity")
    plt.xlabel('MSE', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
    plt.show()

    plt.plot(orders, r2_list)
    plt.title("R2 x Model Complexity")
    plt.xlabel('R2', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
    plt.show()



def part_b_request1_extra():
    """
    Here I plot the same thing as previously but with the test data as well and with a degree larger than 5
    """
    np.random.seed(42)
    N = 11

    x = np.sort(np.random.rand(N)).reshape((-1, 1))
    y = np.sort(np.random.rand(N)).reshape((-1, 1))
    x, y = np.meshgrid(x, y)

    z = franke(x, y) + 0.2* np.random.rand(N, N)
    stop = 15
    start = 0

    ## create design matrix using biggest poly_degree
    Linreg = LinearRegression(stop, x, y, z)
    design = Linreg.get_design()

    # We need here to scale/ center the data
    design = normalize(design)

    # now we need to do the train test split OUTSIDE THE LOOP
    X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.2, random_state=42)

    mse_list = []
    r2_list = []
    orders=np.linspace(1, stop, stop)
    for i in orders:
        i = int(i)
        slice = int((i+1)*(i+2)/2)

        Linreg.set_beta(X_train[:, :slice], z_train)
        mse, r2 = Linreg.predict(X_test[:, :slice], z_test)
        mse_list.append(mse)
        r2_list.append(r2)


    plt.plot(orders, mse_list)
    plt.title("MSE x Model Complexity - TEST")
    plt.xlabel('MSE', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
    plt.show()

    plt.plot(orders, r2_list)
    plt.title("R2 x Model Complexity - TEST")
    plt.xlabel('R2', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
    plt.show()


    mse_list = []
    r2_list = []
    orders=np.linspace(1, stop, stop)
    for i in orders:
        i = int(i)
        slice = int((i+1)*(i+2)/2)

        Linreg.set_beta(X_train[:, :slice], z_train)
        mse, r2 = Linreg.predict(X_train[:, :slice], z_train)
        mse_list.append(mse)
        r2_list.append(r2)


    plt.plot(orders, mse_list)
    plt.title("MSE x Model Complexity - TRAIN")
    plt.xlabel('MSE', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
    plt.show()

    plt.plot(orders, r2_list)
    plt.title("R2 x Model Complexity - TRAIN")
    plt.xlabel('R2', fontsize=12)
    plt.ylabel('Polynomial Degree', fontsize=12)
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
    np.random.seed(42)
    N = 11

    x = np.sort(np.random.rand(N)).reshape((-1, 1))
    y = np.sort(np.random.rand(N)).reshape((-1, 1))
    x, y = np.meshgrid(x, y)

    z = franke(x, y) + 0.15* np.random.rand(N, N)
    stop = 15

    ## create design matrix using biggest poly_degree
    Linreg = LinearRegression(stop, x, y, z)

    design = Linreg.get_design()

    # We need here to scale/ center the data
    design = normalize(design)

    # now we need to do the train test split OUTSIDE THE LOOP
    X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.2, random_state=42)

    mse_list_train = []
    mse_list_test = []
    orders=np.linspace(1, stop, stop)
    for i in orders:
        i = int(i)
        slice = int((i+1)*(i+2)/2)

        Linreg.set_beta(X_train[:, :slice], z_train)

        mse_train, r2 = Linreg.predict(X_train[:, :slice], z_train)
        mse_test, r2 = Linreg.predict(X_test[:, :slice], z_test)

        mse_list_train.append(mse_test)
        mse_list_test.append(mse_train)



    plt.plot(orders, mse_list_train)
    plt.plot(orders, mse_list_test)

    plt.title("Fig 11 of Hastie")
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('prediction Error', fontsize=12)
    plt.show()


def part_c_request1_with_bootstrap():
    """
    This is super similar to the part b request 1 extra plot, but now showing together the train and test and also
    NOTICE: this is with boostrap!!!
    """
    np.random.seed(42)
    N = 11

    x = np.sort(np.random.rand(N)).reshape((-1, 1))
    y = np.sort(np.random.rand(N)).reshape((-1, 1))
    x, y = np.meshgrid(x, y)

    z = franke(x, y) + 0.15* np.random.rand(N, N)
    stop = 7

    ## create design matrix using biggest poly_degree
    Linreg = LinearRegression(stop, x, y, z)

    design = Linreg.get_design()

    # We need here to scale/ center the data
    design = normalize(design)

    # now we need to do the train test split OUTSIDE THE LOOP
    X_train, X_test, z_train, z_test = train_test_split(design, np.ravel(z), test_size = 0.2, random_state=42)

    # now we use the bootstrap
    resampler = Resample(Linreg)

    mse_list = []
    r2_list = []
    orders=np.linspace(1, stop, stop)
    for i in orders:
        i = int(i)
        slice = int((i+1)*(i+2)/2)

        Linreg.set_beta(X_train[:, :slice], z_train)

        r2, mse, bias, var = resampler.daniel_bootstrap(30, original_X=design[:, :slice], original_z=z, X_train=X_train[:, :slice], X_test=X_test[:, :slice], z_train=z_train, z_test=z_test)

        mse_list.append(mse)
        r2_list.append(r2)


    plt.plot(orders, mse_list)
    plt.plot(orders, r2_list)

    plt.title("Fig 11 of Hastie DANIEL boostrap")
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('prediction Error', fontsize=12)
    plt.show()




if project_section == "b":
    part_b_request1()
    part_b_request1_extra()
    part_b_request2()


if project_section == "c":
    part_c_request1()

    ## this next part is super wrong and I don't know why. Notice we have to correct the boostrap Adam Did here
    ## in the same manner we had to correct the previous evaluations 
    part_c_request1_with_bootstrap()












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