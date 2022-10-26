from random import seed
import numpy as np


def GD(X, y, n_iter, eta0, rate="constant", lmbda=0, gamma=0, eps=1e-10):
    
    """
    X: Design matrix
    y: Response data
    n_iter: number of iterations
    eta0: initial learning rate 
    rate: Choose an optimization method for the learning rate. Default is a constant learning rate. 
    Options are:
        - constant
        - adagrad
        - RMSprop
        - adam
    lmbda: Regularization term, if zero we do OLS regression, if lmbda > 0 we do Ridge regression
    gamma: Momentum term. If zero, we use gradient descent without momentum, if gamma > 0 we use momentum
    eps: Tolerance for gradient. If norm of gradient is very small, stop gradient descent algorithm.
    """
    
    p = X.shape[1] # num features
    N = X.shape[0]  # num data points
    beta = np.random.randn(p, 1) # randomly initiate beta
    
    eta = eta0 # learning rate 
    change = 0 # Initiate the value with which beta changes
    s = np.zeros_like(beta) # initiate vector used for optimizing learning rate
    delta = 1e-10 # to avoid division by zero
    decay = 0.9 # might make this tunable later

    Giter = np.zeros(shape=(p,p))
    for i in range(n_iter):

        gradient = 2.0 / N * X.T @ (X @ beta - y) + 2.0 * lmbda * beta # if lambda>0, we do Ridge regression
        
        if rate == "constant":
            scale = 1.0

        elif rate == "adagrad":
            Giter += gradient @ gradient.T
            scale = np.c_[1.0 / (delta + np.sqrt(np.diagonal(Giter)))]

        elif rate == "RMSprop":
            Previous = Giter
            Giter += gradient @ gradient.T
            Gnew = decay*Previous + (1-decay) * Giter
            scale = np.c_[1.0 / (delta + np.sqrt(np.diagonal(Gnew)))]

        change = -eta*np.multiply(scale, gradient) + gamma*change
        beta += change

        if(np.linalg.norm(gradient) <= eps): #stop iterating once it has converged sufficiently
            print(f"Stopped after {i} iterations.")
            break

    return beta

def SGD(X, y, n_epochs, eta0, rate, M=1, lmbda=0, gamma=0, eps=1e-8):
    """
    X: Design matrix
    y: Response data
    n_epochs: number of epochs
    t0, t1: scheduling option for learning rate
    rate: Choose an optimization method for the learning rate. Default is a constant learning rate. 
    Options are:
        - constant
        - adagrad
        - adam
        - RMSprop
    M: minibatch size
    lmbda: Regularization term, if zero we do OLS regression
    gamma: Momentum term, if zero we use SDG without momentum
    eps: Tolerance for gradient (if very small, stop gradient descent)
    """
    p = X.shape[1] # num. features
    N = X.shape[0] # num. data points

    m = int(N/M) # num. batches

    beta = np.random.randn(p, 1) #initialize beta
    delta = 1e-10 # to avoid division by zero
    decay = 0.9 # might make decay rate tunable
    change = np.zeros_like(beta) # initiate vector used for computing change in beta

    for epoch in range(1, n_epochs+1):
        Giter = np.zeros(shape=(p,p))

        for i in range(m):
            random_index = M*np.random.randint(m)
            Xk = X[random_index:random_index+M]
            yk = y[random_index:random_index+M]

            #Compute the gradient using the data in minibatch k
            gradient = 2.0/M * Xk.T @ (Xk @ beta - yk) + 2.0 * lmbda * beta
            
            if rate == "constant":
                scale = 1.0

            elif rate == "adagrad":
                Giter += gradient @ gradient.T
                scale = np.c_[1.0 / (delta + np.sqrt(np.diagonal(Giter)))]

            elif rate == "RMSprop":
                Previous = Giter
                Giter += gradient @ gradient.T
                Gnew = decay*Previous + (1-decay) * Giter
                scale = np.c_[ 1.0 / (delta + np.sqrt(np.diagonal(Gnew)))]
            
            elif rate == "adam":
                s = None

            change = -eta0*np.multiply(scale, gradient) + gamma*change
            beta += change

        if(np.linalg.norm(gradient) <= eps): #stop iterating once it has converged sufficiently
            print(f"Stopped after {m*epoch+i} iterations.")
            break
    
    return beta
