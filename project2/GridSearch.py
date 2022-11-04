import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FFNN import FFNN
from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression
from ResampleMethods import *

cm = 1/2.54

def GridSearch_FFNN_classifier(
    X,
    y, 
    lambda_values, 
    eta_values, 
    plot_grid=True,
    gamma=0.9,
    activation_hidden="reLU",
    n_epochs=200,
    batch_size=20,
    n_hidden_neurons = [100],
    k=5
    ):
    accuracy = np.zeros((len(eta_values), len(lambda_values)))

    for i, eta in enumerate(eta_values):
        for j, lmbda in enumerate(lambda_values):
            print(f"Computing eta={eta} and lambda={lmbda}.")
            network = FFNN(
                n_hidden_neurons=n_hidden_neurons, 
                task="classification", 
                n_epochs=n_epochs, 
                batch_size=batch_size, 
                eta=eta, lmbda=lmbda, 
                gamma=gamma, 
                activation_hidden=activation_hidden
                )

            accuracy_score = CrossValidation_classification(network, X, y, k=k)

            accuracy[i][j] = accuracy_score
    
    if plot_grid:
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            accuracy, 
            annot=True, 
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("Accuracy")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        info = f"_mom{gamma}" + "_activ" + activation_hidden + f"_epoch{n_epochs}_batch{batch_size}_layers{len(n_hidden_neurons)}_neuro{n_hidden_neurons[0]}"
        plt.savefig("figs/gridsearch_FFNN_class" + info + ".pdf")   

    return accuracy

def GridSearch_LogReg(
    X,
    y, 
    lambda_values, 
    eta_values, 
    solver="sgd",
    optimization="adam",
    plot_grid=True,
    gamma=0.9,
    max_iter=300,
    batch_size=20,
    k=5
    ):
    accuracy = np.zeros((len(eta_values), len(lambda_values)))

    for i, eta in enumerate(eta_values):
        for j, lmbda in enumerate(lambda_values):
            print(f"Computing eta={eta} and lambda={lmbda}.")
            logreg = LogisticRegression(
                solver=solver,
                optimization=optimization,
                batch_size=batch_size, 
                eta0=eta, 
                lmbda=lmbda, 
                gamma=gamma, 
                max_iter=max_iter
                )

            accuracy_score = CrossValidation_classification(logreg, X, y, k=k)

            accuracy[i][j] = accuracy_score
    
    if plot_grid:
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            accuracy, 
            annot=True, 
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("Accuracy")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        plt.tight_layout()
        info = f"_sol" + solver + "_opt" + optimization + f"_mom{gamma}_iter{max_iter}_batch{batch_size}"
        plt.savefig("figs/gridsearch_logreg" + info + ".pdf")   

    return accuracy

def GridSearch_LinReg(
    X,
    y, 
    lambda_values, 
    eta_values, 
    solver="analytic",
    optimization=None,
    plot_grid=True,
    gamma=0.9,
    max_iter=300,
    batch_size=20,
    k=5
    ):
    mse_values = np.zeros((len(eta_values), len(lambda_values)))
    r2_values = np.zeros((len(eta_values), len(lambda_values)))


    for i, eta in enumerate(eta_values):
        for j, lmbda in enumerate(lambda_values):
            print(f"Computing eta={eta} and lambda={lmbda}.")
            linreg = LinearRegression(
                solver=solver,
                optimization=optimization,
                batch_size=batch_size, 
                eta0=eta, 
                lmbda=lmbda, 
                gamma=gamma, 
                max_iter=max_iter
                )

            mse, r2 = CrossValidation_regression(linreg, X, y, k=k)

            mse_values[i][j] = mse
            r2_values[i][j] = r2

    if plot_grid:
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            mse_values, 
            cbar=False,
            annot=True, 
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("MSE")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        plt.tight_layout()
        info = "_sol" + solver 

        if solver != "analytic":
            info += f"_opt{optimization}_mom{gamma}_iter{max_iter}_batch{batch_size}"

        plt.savefig("figs/gridsearch_linreg_MSE" + info + ".pdf")
        
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            r2_values, 
            annot=True, 
            cbar=False,
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("$R^2$")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        plt.tight_layout()
        plt.savefig("figs/gridsearch_linreg_R2" + info + ".pdf")

    return mse_values, r2_values
       
def GridSearch_FFNN_reg(
    X,
    y, 
    lambda_values, 
    eta_values, 
    plot_grid=True,
    gamma=0.9,
    activation_hidden="reLU",
    n_epochs=200,
    batch_size=20,
    n_hidden_neurons = [100],
    k=5
    ):

    mse_values = np.zeros((len(eta_values), len(lambda_values)))
    r2_values = np.zeros((len(eta_values), len(lambda_values)))

    for i, eta in enumerate(eta_values):
        for j, lmbda in enumerate(lambda_values):
            print(f"Computing eta={eta} and lambda={lmbda}.")
            network = FFNN(
                n_hidden_neurons=n_hidden_neurons, 
                task="regression", 
                n_epochs=n_epochs, 
                batch_size=batch_size, 
                eta=eta, lmbda=lmbda, 
                gamma=gamma, 
                activation_hidden=activation_hidden
                )

            mse, r2 = CrossValidation_regression(network, X, y, k=k)

            mse_values[i][j] = mse
            r2_values[i][j] = r2
    
    if plot_grid:
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            mse_values, 
            annot=True, 
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("MSE")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        
        info = f"_mom{gamma}" + "_activ" + activation_hidden + f"_epoch{n_epochs}_batch{batch_size}_layers{len(n_hidden_neurons)}_neuro{n_hidden_neurons[0]}"
        
        plt.savefig("figs/gridsearch_FFNN_reg_MSE" + info + ".pdf")  

        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            r2_values, 
            annot=True, 
            ax=ax, 
            cmap="viridis", 
            yticklabels=np.round(np.log10(eta_values), 2), 
            xticklabels=np.round(np.log10(lambda_values), 2)
            )
        ax.set_title("$R^2$")
        ax.set_ylabel("$\log_{10}(\eta$)")
        ax.set_xlabel("$\log_{10}(\lambda$)")
        plt.savefig("figs/gridsearch_FFNN_reg_R2" + info + ".pdf")   

    return mse_values, r2_values

def GridSearch_LinReg_epochs_batchsize(
    X,
    y, 
    eta,
    batch_sizes,
    n_epochs,
    lmbda=0,  
    solver="sgd",
    optimization=None,
    plot_grid=True,
    gamma=0.0,
    k=5
    ):
    mse_values = np.zeros((len(batch_sizes), len(n_epochs)))
    r2_values = np.zeros((len(batch_sizes), len(n_epochs)))


    for i, M in enumerate(batch_sizes):
        for j, epochs in enumerate(n_epochs):
            print(f"Computing batch size={M} and num. epochs={epochs}.")
            linreg = LinearRegression(
                solver=solver,
                optimization=optimization,
                batch_size=M, 
                eta0=eta, 
                lmbda=lmbda, 
                gamma=gamma, 
                max_iter=epochs
                )

            mse, r2 = CrossValidation_regression(linreg, X, y, k=k)

            mse_values[i][j] = mse
            r2_values[i][j] = r2

    if plot_grid:
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            mse_values, 
            annot=True, 
            cbar=False,
            ax=ax, 
            cmap="viridis", 
            yticklabels=batch_sizes, 
            xticklabels=n_epochs
            )
        ax.set_title("MSE")
        ax.set_ylabel("Batch size")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        info = "_sol" + solver 

        if solver != "analytic":
            info += f"_opt{optimization}_mom{gamma}_eta{eta}_lmbda{lmbda}"

        plt.savefig("figs/gridsearch_linreg_MSE_epoch_batchsize" + info + ".pdf")
        
        fig, ax = plt.subplots(figsize = (12*cm, 12*cm))
        sns.heatmap(
            r2_values, 
            annot=True, 
            cbar=False,
            ax=ax, 
            cmap="viridis", 
            yticklabels=batch_sizes, 
            xticklabels=n_epochs
            )
        ax.set_title("$R^2$")
        ax.set_ylabel("Batch size")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        plt.savefig("figs/gridsearch_linreg_R2_epoch_batchsize" + info + ".pdf")

    return mse_values, r2_values