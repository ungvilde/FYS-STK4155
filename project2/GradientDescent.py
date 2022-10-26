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
        - adam
        - RMSprop
    lmbda: Regularization term, if zero we do OLS regression
    gamma: Momentum term. If zero, we use gradient descent without momentum
    eps: Tolerance for gradient (if very small, stop gradient descent)
    """
    p = X.shape[1] # num features
    N = X.shape[0]  # num data points
    beta = np.random.randn(p, 1) # randomly initiate beta
    
    eta = eta0 # learning rate 
    m = 0 # Initiate the value with which beta changes
    s = np.zeros_like(beta) # initiate vector used for optimizing learning rate
    delta = 1e-10 # to avoid division by zero
    decay = 0.9 # might make this tunable later

    for i in range(n_iter):

        gradient = 2.0 / N * X.T @ (X @ beta - y) + 2.0 * lmbda * beta # if lambda>0, we do Ridge regression
        
        if rate == "constant":
            scale = 1.0

        elif rate == "adagrad":
            s += np.multiply(gradient, gradient) 
            scale = 1.0/np.sqrt(s+delta) # for scaling gradient element-wise

        elif rate == "RMSprop":
            s = decay*s + (1-decay) * np.multiply(gradient, gradient)
            scale = 1.0/np.sqrt(s+delta)

        m = -eta*np.multiply(scale, gradient) + gamma*m
        beta += m

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
    lmbda: Regularization term, if zero we do OLS regression
    gamma: Momentum term, if zero we use SDG without momentum
    eps: Tolerance for gradient (if very small, stop gradient descent)
    """
    np.random.seed(123)
    p = X.shape[1]
    N = X.shape[0]

    m = int(N/M) # num. batches
    inds = np.arange(0,N)
    random_inds = np.random.choice(inds, N, replace=True) #randomize the indices

    # indeces for batch k
    k_inds = []
    for k in range(m):
        inds = random_inds[k*M:(k*M + M)]
        k_inds.append(inds)

    beta = np.random.randn(p, 1) #initialize beta
    beta = np.array([[-1.0856306 ],[ 0.99734545],[ 0.2829785 ]])
    print("beta = ", beta)
    s = np.zeros_like(beta) # initiate vector used for optimizing learning rate
    delta = 1e-10 # to avoid division by zero
    decay = 0.9 # might make this tunable later
    change = np.zeros_like(beta) # initiate vector used for computing change in beta

    for epoch in range(1, n_epochs+1):
        s = np.zeros_like(beta)
        
        for i in range(m):
            k = np.random.randint(m) #Pick the k-th minibatch at random
            inds = k_inds[k]
            Xk = X[inds, :]
            yk = y[inds]

            #Compute the gradient using the data in minibatch k
            gradient = 2.0/M * Xk.T @ (Xk @ beta - yk) + 2.0 * lmbda * beta
            if rate == "constant":
                scale = 1.0

            elif rate == "adagrad":
                s += np.multiply(gradient, gradient) 
                scale = 1.0/np.sqrt(s+delta) # for scaling gradient element-wise

            elif rate == "RMSprop":
                s = decay*s + (1-decay) * np.multiply(gradient, gradient)
                scale = 1.0/np.sqrt(s+delta)

            change = -eta0*np.multiply(scale, gradient) + gamma*change
            beta += change

        if(np.linalg.norm(gradient) <= eps): #stop iterating once it has converged sufficiently
            print(f"Stopped after {m*epoch+i} iterations.")
            break
    
    #print(f"Stopped after {m*(epoch-1)+i} iterations.")
    return beta


## OLD CODE

"""
def SDG(X, y, n_epochs, t0, t1, M=1, lmbda=0, gamma=0, eps=1e-8):
    
    # X: Design matrix
    # y: Response data
    # n_epochs: number of epochs
    # t0, t1: 
    # lmbda: Regularization term, if zero we do OLS regression
    # gamma: Momentum term, if zero we use SDG without momentum
    # eps: Tolerance for gradient (if very small, stop gradient descent)

    p = X.shape[1]
    N = X.shape[0]

    m = int(N/M) # num. batches
    inds = np.arange(0,N)
    random_inds = np.random.choice(inds, N, replace=True) #randomize the indices
    
    # indeces for batch k
    k_inds = []
    for k in range(m):
        inds = random_inds[k*M:(k*M + M)]
        k_inds.append(inds)

    def learning_schedule(t):
        return t0/(t+t1)

    beta = np.random.randn(p, 1) #initialize beta

    change = 0.0 
    for epoch in range(1, n_epochs+1):
        s = np.zeros_like(beta)

        for i in range(m):
            k = np.random.randint(m) #Pick the k-th minibatch at random
            inds = k_inds[k]
            Xk = X[inds, :]
            yk = y[inds]

            #Compute the gradient using the data in minibatch k
            gradient = 2.0/M * Xk.T @ (Xk @ beta - yk) + 2.0 * lmbda * beta
            eta = learning_schedule(epoch*m+i)
            new_change = eta*gradient+gamma*change #momentum 

            beta -= new_change
            change = new_change

            if(np.linalg.norm(gradient) <= eps): #stop iterating once it has converged sufficiently
                print(f"Stopped after {m*epoch+i} iterations.")
                break
    
    return beta

def AdaGrad(gradient):
    s = np.multiply(gradient, gradient) #elementwise multiplication
    return s

def RMSprop(gradient, decay):
    s = (1-decay)*np.multiply(gradient, gradient) #elementwise multiplication
    return s

"""