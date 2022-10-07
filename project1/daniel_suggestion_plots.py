from cProfile import label
from re import L
from imageio import imread
from LinearRegression import LinearRegression
from Resample import Resample
from helper import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_theme("notebook", "whitegrid")

project_data = input("Which data do you want to plot? (Franke=F or Terrain=T) ")
N = int(input("How many data points do you want to use? "))
stop = int(input("Max order:    "))
project_section = input("Which part of project 1 you want to plot? (b, c, d, e, f)")


def get_data_franke(N, noise):
    """
    Get the data for the Franke function.
    """
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    z = franke(x, y) + noise * np.random.randn(N, N)

    return x, y, z


def get_data_terrain(N):
    """
    Get the data for the terrain data.
    """
    S = 1000
    terrain = imread("SRTM_data_Norway_1.tif")
    terrain = terrain[S : S + N, S : S + N]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    
    z = terrain
    epsilon =  10**(-2)
    denominator = [x if x > epsilon else 1 for x in np.std(z, axis=0)]
    z = (z - np.mean(z, axis=0))/ denominator

    return x, y, z


########## PART B ####################
def part_b_request1(show_betas=False):
    """
    Perform a standard ordinary least square regression analysis using polynomials in x and y up to fifth order.
    Evaluate the mean Squared error (MSE) and the R2 score function.

    Notice this uses the test data to plot and that is why the mse starts to go up.
    NOTICE THIS HAS SCALING ALREADY!
    """
    np.random.seed(123)

    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.1)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    max_degree = stop
    mse_list = []
    r2_list = []
    orders = np.linspace(1, max_degree, max_degree)
    for i in orders:
        print("At order: %d" % i, end="\r")

        i = int(i)

        ols = LinearRegression(i, x, y, z, scale=True)
        mse, r2 = ols.split_predict_eval(
            test_size=0.2, fit=True, train=False, random_state=42
        )
        mse_list.append(mse)
        r2_list.append(r2)

        if show_betas == True:
            beta = ols.get_beta()
            plt.plot(i, beta[0], "or", label=r"$\beta_0$" + f" d={i}")
            plt.plot(i, beta[1], "ob", label=r"$\beta_1$" + f" d={i}")
            plt.plot(i, beta[2], "og", label=r"$\beta_2$" + f" d={i}")
            if i > 3:
                plt.plot(i, beta[3], "oy", label=r"$\beta_3$" + f" d={i}")
            if i > 4:
                plt.plot(i, beta[4], "ok", label=r"$\beta_4$" + f" d={i}")

            if i > 5:
                plt.plot(i, beta[5], "oc", label=r"$\beta_5$" + f" d={i}")

            if i > 6:
                plt.plot(i, beta[6], "om", label=r"$\beta_6$" + f" d={i}")

    if show_betas == True:

        plt.title("Betas x Model Complexity - TEST")
        plt.xlabel("Polynomial Degree", fontsize=12)
        plt.ylabel("Betas", fontsize=12)
        plt.show()
        exit(1)

    # (Adam) - Plotting Errors together and saving figs
    # Plotting MSE and R2 in same figure with two y-axis
    cm = 1 / 2.54
    fig, ax1 = plt.subplots(figsize=(12 * cm, 10 * cm))
    ax1.plot(orders, mse_list, "b-", label="MSE")
    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, "r-", label="$R^2$")

    ax1.set_xlabel("Polynomial Degree", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.tick_params("y")
    ax2.grid(False)  # to not plot grid on top of other graph

    ax2.set_ylabel("$R^2$", fontsize=12)
    ax2.tick_params("y")
    fig.legend(loc=7, bbox_to_anchor=(0.5, 0.0, 0.5, 0.5), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(f"Figs/Errors_x_order_{max_degree}_TEST_" + str(project_data) + ".pdf")

    ###########################################
    # code for reproducing fig 2.11
    mse_test = []
    mse_train = []
    d_max = max_degree

    for i in range(1, d_max + 1):
        i = int(i)
        Linreg = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(Linreg)
        # compute test error
        # mse, _ = Linreg.split_predict_eval(test_size=0.2, fit=True, train=False, random_state=42)
        _, mse, bias, var, _ = resampler.bootstrap(100)
        mse_test.append(mse)

        # compute training error
        Linreg = LinearRegression(i, x, y, z, scale=True)
        mse, _ = Linreg.split_predict_eval(
            test_size=0.2, fit=True, train=True, random_state=42
        )

        mse_train.append(mse)

    cm = 1 / 2.54
    plt.figure(figsize=(12 * cm, 10 * cm))
    d_values = np.arange(1, d_max + 1, step=1, dtype=int)
    plt.plot(d_values, mse_train, label="Training error")
    plt.plot(d_values, mse_test, label="Test error")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.xticks(np.arange(1, d_max + 1, step=2, dtype=int))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figs/train_v_test_error_plot_{project_data}.pdf")

    ###########################################
    # code for plotting Bias-Variance Trade-Off
    d_max = 10
    Linreg = LinearRegression(d_max, x, y, z)
    X = Linreg.get_design()
    X = X - np.mean(X, axis=0)  # / np.std(X, axis=0)
    X_train, X_test, z_train, z_test = train_test_split(
        X, np.ravel(z), test_size=0.2, random_state=42
    )

    # now we use the bootstrap
    mse_list = []
    bias_list = []
    var_list = []

    for i in range(1, d_max + 1):
        i = int(i)

        Linreg = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(Linreg)
        # X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size = 0.3, random_state=42)
        # Linreg.set_beta(X_train, z_train)

        _, mse, bias, var, _ = resampler.bootstrap(100)

        mse_list.append(mse)
        bias_list.append(bias)
        var_list.append(var)

    cm = 1 / 2.54
    plt.figure(figsize=(12 * cm, 10 * cm))
    d_values = np.arange(1, d_max + 1, step=1, dtype=int)
    plt.plot(d_values, mse_list, label="Test error")
    plt.plot(d_values, bias_list, "--", label="Bias")
    plt.plot(d_values, var_list, "--", label="Variance")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.xticks(np.arange(1, d_max + 1, step=2, dtype=int))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Figs/bias_variance_plot_{project_data}_N_{N}.pdf")

    ####################################################
    # code for plotting beta values with conf. intervals

    # x = np.sort(np.random.rand(N)).reshape((-1, 1))
    # y = np.sort(np.random.rand(N)).reshape((-1, 1))
    # x, y = np.meshgrid(x, y)
    # z = franke(x, y) + np.random.normal(loc=0, scale=0.1, size=(N,N))
    d_max = 5

    plt.figure(figsize=(12 * cm, 8 * cm))

    for i in range(d_max, 1, -1):
        i = int(i)
        slice = int((i + 1) * (i + 2) / 2)

        Linreg = LinearRegression(i, x, y, z, scale=True)

        Linreg()
        beta = Linreg.get_beta()
        conf_int = Linreg.conf_int()

        beta_inds = range(0, len(beta))
        # plt.plot(beta_inds, beta, 'o', label=f"Order {i}")
        plt.errorbar(x=beta_inds, y=beta, yerr=conf_int, fmt=".", label=f"$d=${i}")

    plt.legend()
    p = (d_max + 1) * (d_max + 2) / 2
    plt.xticks(np.arange(0, p, step=2, dtype=int))
    plt.xlabel(r"Index $j$")
    plt.ylabel(r"$\beta_j$")
    plt.tight_layout()
    plt.savefig(f"Figs/beta_coef_{project_data}.pdf")


def part_b_request1_extra():
    """
    Here I plot the same thing as previously but with the test data as well AND with a degree larger than 5
    """
    np.random.seed(42)

    if project_data == "F":

        x, y, z = get_data_franke(N, noise=0.1)

    if project_data == "T":
        x, y, z = get_data_terrain(N)

    max_degree = stop

    mse_list = []
    r2_list = []
    orders = np.linspace(1, max_degree, max_degree)

    for i in orders:
        print("At order: %d" % i, end="\r")

        i = int(i)
        ols = LinearRegression(i, x, y, z)
        mse, r2 = ols.split_predict_eval(
            test_size=0.2, fit=True, train=False, random_state=42
        )
        mse_list.append(mse)
        r2_list.append(r2)

    # Plotting MSE and R2 in same figure with two y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(orders, mse_list, "b-")
    ax1.set_xlabel("Polynomial Degree", fontsize=12)
    ax1.set_ylabel("MSE", color="b", fontsize=12)
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, "r-")
    ax2.set_ylabel("R2", color="r", fontsize=12)
    ax2.tick_params("y", colors="r")

    fig.tight_layout()
    plt.savefig(f"Figs/Errors_x_order_{max_degree}_TEST_" + str(project_data) + ".pdf")
    plt.show()

    mse_list = []
    r2_list = []
    ## The only difference is that this is the TRAIN plot of mse
    for i in orders:
        print("At order: %d" % i, end="\r")

        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        mse, r2 = ols.split_predict_eval(
            test_size=0.2, fit=True, train=True, random_state=42
        )
        mse_list.append(mse)
        r2_list.append(r2)

    # Plotting MSE and R2 in same figure with two y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(orders, mse_list, "b-")
    ax1.set_xlabel("Polynomial Degree", fontsize=12)
    ax1.set_ylabel("MSE", color="b", fontsize=12)
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(orders, r2_list, "r-")
    ax2.set_ylabel("R2", color="r", fontsize=12)
    ax2.tick_params("y", colors="r")

    fig.tight_layout()
    plt.savefig(f"Figs/Errors_x_order_{max_degree}_TRAIN_" + str(project_data) + ".pdf")
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

    max_degree = stop
    if project_data == "F":

        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    mse_list_train = []
    mse_list_test = []
    orders = np.linspace(1, max_degree, max_degree)
    for i in orders:
        print("At order: %d" % i, end="\r")

        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        mse_test, r2_test = ols.split_predict_eval(
            test_size=0.2, fit=True, train=False, random_state=42
        )
        ols = LinearRegression(i, x, y, z, scale=True)
        mse_train, r2_train = ols.split_predict_eval(
            test_size=0.2, fit=True, train=True, random_state=42
        )

        mse_list_train.append(mse_train)
        mse_list_test.append(mse_test)

    plt.plot(orders, mse_list_train, label="Train")
    plt.plot(orders, mse_list_test, label="Test")
    plt.xlabel("Polynomial Degree", fontsize=12)
    plt.ylabel("prediction Error", fontsize=12)
    plt.legend()
    plt.savefig(f"Figs/Fig_2_11_" + str(project_data) + ".pdf")
    plt.show()


def part_c_request2():
    """
    This is super similar to the part c request 1 but now using bootstrap and showing the bv tradeoff
    """
    np.random.seed(41)


    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    start = 1

    r2 = np.zeros(stop - start)
    mse = np.zeros(stop - start)
    bias = np.zeros(stop - start)
    var = np.zeros(stop - start)
    var_mse = np.zeros(stop - start)
    orders = np.linspace(1, stop - 1, stop - 1)

    for i in range(start, stop):
        ols = LinearRegression(i, x, y, z)
        resampler = Resample(ols)
        (
            r2[i - 1],
            mse[i - 1],
            bias[i - 1],
            var[i - 1],
            var_mse[i - 1],
        ) = resampler.bootstrap(
            N, random_state=42
        )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    # print(f"Z avg:{np.mean(z)} ")
    plt.plot(orders, mse, label="MSE")

    plt.plot(orders, bias, label="Bias")
    plt.plot(orders, var, label="Variance")
    plt.legend()

    plt.xlabel("Polynomial Degree", fontsize=12)
    plt.ylabel("prediction Error", fontsize=12)
    plt.savefig("Figs/B-V_Tradeoff_Bootstrap_" + str(project_data) + ".pdf")
    plt.show()


########## PART D ####################


def part_d_request1():
    """
    Here the request is simple: compare the MSE from boostrap to the MSE from cross-validation with k from 5 to 10.
    Notice we will assess only for test MSE.

    SHOULD WE USE THE SCALING HERE? IF YES, IT SHOULD BE ADDED TO THE LinearRegression as an option
    """
    np.random.seed(41)


    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":

        x, y, z = get_data_terrain(N)
    start = 1

    k = 10
    r2_cross = np.zeros(stop - start)
    mse_cross = np.zeros(stop - start)

    orders = np.linspace(1, stop - 1, stop - 1)

    r2_boostrap = np.zeros(stop - start)
    mse_boostrap = np.zeros(stop - start)
    bias_boostrap = np.zeros(stop - start)
    var_boostrap = np.zeros(stop - start)
    std_mse = np.zeros(stop - start)

    for i in orders:
        i = int(i)
        ols = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(ols)
        r2_cross[i - 1], mse_cross[i - 1] = resampler.cross_validation(k=k)

        ols = LinearRegression(i, x, y, z, scale=True)
        resampler = Resample(ols)
        (
            r2_boostrap[i - 1],
            mse_boostrap[i - 1],
            bias_boostrap[i - 1],
            var_boostrap[i - 1],
            std_mse[i - 1],
        ) = resampler.bootstrap(10, random_state=42)

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


    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    start = 1

    n_lambdas = 25
    lambdas = np.logspace(-5, 4, n_lambdas)
    orders = np.linspace(1, stop - 1, stop - 1)

    r2 = np.zeros((stop - start, n_lambdas))
    mse = np.zeros((stop - start, n_lambdas))
    bias = np.zeros((stop - start, n_lambdas))
    var = np.zeros((stop - start, n_lambdas))
    var_mse = np.zeros((stop - start, n_lambdas))

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            ridge = LinearRegression(i, x, y, z, method=2, lmbd=lmbd, scale=True)
            resampler = Resample(ridge)
            (
                r2[i - 1, j],
                mse[i - 1, j],
                bias[i - 1, j],
                var[i - 1, j],
                var_mse[i - 1, j],
            ) = resampler.bootstrap(
                N, random_state=42
            )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    mse_min = np.min(mse)
    print("BOOTS MIN MSE", mse_min)
    i_min, j_min = np.where(mse == mse_min)
    lambdas = np.log10(lambdas)
    lambdas, orders = np.meshgrid(lambdas, orders)

    plt.contourf(lambdas, orders, mse, levels=100)
    print("BOOTS MIN", lambdas[j_min, i_min])

    plt.plot(lambdas[j_min, i_min], orders[i_min, j_min], "+", c="r")
    plt.colorbar()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Polynomial Degree", fontsize=12)
    plt.savefig("Figs/Optimizing_Ridge_Bootstrap_" + str(project_data) + ".pdf")
    plt.show()

    ## NOW I WILL do the same thing but with crossval k= 10
    k = 10
    r2 = np.zeros((stop - start, n_lambdas))
    mse = np.zeros((stop - start, n_lambdas))
    bias = np.zeros((stop - start, n_lambdas))
    var = np.zeros((stop - start, n_lambdas))

    lambdas = np.logspace(-5, 4, n_lambdas)
    orders = np.linspace(1, stop - 1, stop - 1)

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            ridge = LinearRegression(i, x, y, z, method=2, lmbd=lmbd, scale=True)
            resampler = Resample(ridge)
            r2[i - 1, j], mse[i - 1, j] = resampler.cross_validation(
                k=k
            )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    mse_min = np.min(mse)
    print("CV MIN MSE", mse_min)

    i_min, j_min = np.where(mse == mse_min)
    lambdas = np.log10(lambdas)
    lambdas, orders = np.meshgrid(lambdas, orders)

    plt.contourf(lambdas, orders, mse, levels=50, cmap="plasma")
    print("CV MIN", lambdas[j_min, i_min])
    plt.plot(lambdas[j_min, i_min], orders[i_min, j_min], "+", c="r")
    plt.colorbar()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Polynomial Degree", fontsize=12)
    plt.savefig("Figs/Optimizing_Ridge_Croassvalk10" + str(project_data) + ".pdf")
    plt.show()

    ######### Finally I will select the poly degree with min mse and perform the B-V tradeoff with bootstrap
    lambdas = np.logspace(-5, 4, n_lambdas)

    r2 = np.zeros(n_lambdas)
    mse = np.zeros(n_lambdas)
    bias = np.zeros(n_lambdas)
    var = np.zeros(n_lambdas)
    var_mse = np.zeros(n_lambdas)

    chosen_poly_order = int(orders[i_min, j_min])

    for j in range(n_lambdas):
        lmbd = lambdas[j]
        ridge = LinearRegression(
            chosen_poly_order, x, y, z, method=2, lmbd=lmbd, scale=True
        )
        resampler = Resample(ridge)
        r2[j], mse[j], bias[j], var[j], var_mse[j] = resampler.bootstrap(
            N, random_state=42
        )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    lambdas = np.log10(lambdas)

    plt.plot(lambdas, mse, label="MSE")

    plt.plot(lambdas, bias, label="Bias", linestyle="dashed")
    plt.plot(lambdas, var, label="Variance", linestyle="dashed")
    plt.title(f"B-V Tradeoff for Ridge - With Bootstrap - d = {chosen_poly_order}")

    plt.legend()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Prediction Error", fontsize=12)
    plt.savefig("Figs/B-V_Tradeoff_Bootstrap_Ridge_" + str(project_data) + ".pdf")
    plt.show()


########## PART F ####################


def part_f_request1():
    """
    Perform the same bootstrap analysis as in the part c for the same plynomials but now for Lasso
    """
    np.random.seed(41)


    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    start = 1

    n_lambdas = 25
    lambdas = np.logspace(-4, 0, n_lambdas)
    orders = np.linspace(1, stop - 1, stop - 1)

    r2 = np.zeros((stop - start, n_lambdas))
    mse = np.zeros((stop - start, n_lambdas))
    bias = np.zeros((stop - start, n_lambdas))
    var = np.zeros((stop - start, n_lambdas))
    var_mse = np.zeros((stop - start, n_lambdas))

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            lasso = LinearRegression(i, x, y, z, method=3, lmbd=lmbd, scale=True)
            resampler = Resample(lasso)
            (
                r2[i - 1, j],
                mse[i - 1, j],
                bias[i - 1, j],
                var[i - 1, j],
                var_mse[i - 1, j],
            ) = resampler.bootstrap(
                N, random_state=42
            )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    mse_min = np.min(mse)
    print("BOOTS MIN MSE", mse_min)
    i_min, j_min = np.where(mse == mse_min)
    lambdas = np.log10(lambdas)
    lambdas_mesh, orders_mesh = np.meshgrid(lambdas, orders)

    plt.contourf(lambdas_mesh, orders_mesh, mse, levels=50,cmap="plasma")
    print("BOOTS MIN", lambdas[j_min])
    plt.plot(lambdas[j_min], orders[i_min], "+", c="r")

    plt.colorbar()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Polynomial Degree", fontsize=12)
    plt.savefig(
        "Figs/Optimizing_Lasso_Bootstrap_" + str(project_data) + f"N_{N*N}" + ".pdf"
    )
    plt.show()

    ## NOW I WILL do the same thing but with crossval k= 10
    k = 10
    r2 = np.zeros((stop - start, n_lambdas))
    mse = np.zeros((stop - start, n_lambdas))
    bias = np.zeros((stop - start, n_lambdas))
    var = np.zeros((stop - start, n_lambdas))

    lambdas = np.logspace(-4, 0, n_lambdas)
    orders = np.linspace(1, stop - 1, stop - 1)

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            lasso = LinearRegression(i, x, y, z, method=3, lmbd=lmbd, scale=True)
            resampler = Resample(lasso)
            r2[i - 1, j], mse[i - 1, j] = resampler.cross_validation(
                k=k
            )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    mse_min = np.min(mse)
    print("CV MIN MSE", mse_min)

    i_min, j_min = np.where(mse == mse_min)
    lambdas = np.log10(lambdas)
    lambdas_mesh, orders_mesh = np.meshgrid(lambdas, orders)

    plt.contourf(lambdas_mesh, orders_mesh, mse, levels=50, cmap="plasma")
    print("CV MIN", lambdas[j_min])
    plt.plot(lambdas[j_min], orders[i_min], "+", c="r")
    plt.colorbar()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Polynomial Degree", fontsize=12)
    plt.savefig(
        "Figs/Optimizing_lasso_Croassvalk10" + str(project_data) + f"N_{N*N}" + ".pdf"
    )
    plt.show()

    ######### Finally I will select the poly degree with min mse and perform the B-V tradeoff with bootstrap
    lambdas = np.logspace(-5, 4, n_lambdas)

    r2 = np.zeros(n_lambdas)
    mse = np.zeros(n_lambdas)
    bias = np.zeros(n_lambdas)
    var = np.zeros(n_lambdas)
    var_mse = np.zeros(n_lambdas)

    chosen_poly_order = int(orders[i_min])

    for j in range(n_lambdas):
        lmbd = lambdas[j]
        ridge = LinearRegression(
            chosen_poly_order, x, y, z, method=3, lmbd=lmbd, scale=True
        )
        resampler = Resample(ridge)
        r2[j], mse[j], bias[j], var[j], var_mse[j] = resampler.bootstrap(
            N, random_state=42
        )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    lambdas = np.log10(lambdas)

    plt.plot(lambdas, mse, label="MSE")

    plt.plot(lambdas, bias, label="Bias", linestyle="dashed")
    plt.plot(lambdas, var, label="Variance", linestyle="dashed")
    plt.title(f"B-V Tradeoff for Lasso - With Bootstrap - d = {chosen_poly_order}")

    plt.legend()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Prediction Error", fontsize=12)
    plt.savefig("Figs/B-V_Tradeoff_Bootstrap_Lasso_" + str(project_data) + ".pdf")
    plt.show()


################# Part F extra ###########################


def part_f_extra():
    np.random.seed(41)

    if project_data == "F":
        x, y, z = get_data_franke(N, noise=0.15)
    if project_data == "T":
        x, y, z = get_data_terrain(N)

    start = 1

    n_lambdas = 25
    lambdas = np.logspace(-5, 4, n_lambdas)
    orders = np.linspace(1, stop - 1, stop - 1)

    mse_ridge = np.zeros((stop - start, n_lambdas))

    mse_lasso = np.zeros((stop - start, n_lambdas))

    for i in range(start, stop):
        for j in range(n_lambdas):
            lmbd = lambdas[j]
            ridge = LinearRegression(i, x, y, z, method=2, lmbd=lmbd, scale=True)
            resampler = Resample(ridge)
            _, mse_ridge[i - 1, j], _, _, _ = resampler.bootstrap(
                N, random_state=42
            )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

            lasso = LinearRegression(i, x, y, z, method=3, lmbd=lmbd, scale=True)
            resampler = Resample(lasso)
            _, mse_lasso[i - 1, j], _, _, _ = resampler.bootstrap(N, random_state=42)

    mse_min_ridge = np.min(mse_ridge)
    i_min_ridge, j_min_ridge = np.where(mse_ridge == mse_min_ridge)

    mse_min_lasso = np.min(mse_lasso)
    i_min_lasso, j_min_lasso = np.where(mse_lasso == mse_min_lasso)

    chosen_poly_order_ridge = int(orders[i_min_ridge])
    chosen_poly_order_lasso = int(orders[i_min_lasso])

    r2 = np.zeros(n_lambdas)
    mse_ridge = np.zeros(n_lambdas)
    mse_lasso = np.zeros(n_lambdas)

    for j in range(n_lambdas):
        lmbd = lambdas[j]
        ridge = LinearRegression(
            chosen_poly_order_ridge, x, y, z, method=2, lmbd=lmbd, scale=True
        )
        resampler = Resample(ridge)
        _, mse_ridge[j], _, _, _ = resampler.bootstrap(
            N, random_state=42
        )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

        lasso = LinearRegression(
            chosen_poly_order_lasso, x, y, z, method=3, lmbd=lmbd, scale=True
        )
        resampler = Resample(lasso)
        _, mse_lasso[j], _, _, _ = resampler.bootstrap(
            N, random_state=42
        )  ## this random state is only for the train test split! This does not mean we are choosing the same sample on the bootstrap!

    lambdas = np.log10(lambdas)

    plt.plot(lambdas, mse_ridge, label="MSE Ridge")
    plt.plot(lambdas, mse_lasso, label="MSE Lasso")
    plt.title(
        f"MSE for Ridge and Lasso - W Boot - d_ridge = {chosen_poly_order_ridge} - d_lasso = {chosen_poly_order_lasso}"
    )

    plt.legend()

    plt.xlabel("$Log_{10}(\lambda)$", fontsize=12)
    plt.ylabel("Prediction Error", fontsize=12)
    plt.savefig("Figs/MSE_Bootstrap_Ridge__Lasso_" + str(project_data) + ".pdf")
    plt.show()


######################


if project_section == "b":
    betas = input("Do you want to plot the betas? (y/n) ")
    if betas == "y":
        part_b_request1(show_betas=True)
    else:
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

if project_section == "f":
    part_f_request1()

    part_f_extra()
