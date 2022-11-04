import numpy as np
from numba import njit

class LinearRegression:
    def __init__(self, 
    lmbda, 
    solver, 
    max_iter=200, 
    batch_size=20, 
    gamma=0.0, 
    optimization=None, 
    eta0=1e-3, 
    eps=1e-8,
    rho1=0.9,
    rho2=0.999
    ):
        
        self.lmbda = lmbda # regularization parameter
        self.solver = solver # method used for finding optimal parameters

        # Gradient descent parameters
        self.optimization = optimization
        self.gamma = gamma # moment parameter
        self.batch_size = batch_size 
        self.n_epochs = max_iter # epochs used in stochastic gradient descent
        self.n_iter = max_iter # iterations used in gradient descent
        self.eta0 = eta0 # initial learning rate
        self.eps = eps # tolerance for gradient descent
        self.rho1=rho1 # decay parameters for adam/RMSprop optimization
        self.rho2=rho2
    
    def fit(self, X, y):
        self.X_all = X
        self.y_all = y
        self.X = X
        self.y = y
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]

        if self.solver == "analytic":
            self.AnalyticSolution()
        elif self.solver == "gd":
            self.GD()
        elif self.solver == "sgd":
            self.SGD()
        else:
            raise Exception("Invalid solver.")
        
    def GD(self):
        beta = np.random.randn(self.n_features, 1) # randomly initiate beta
        self.y_all = np.c_[self.y_all]
        eta = self.eta0 # learning rate 
        change = np.zeros_like(beta) # Initiate the values with which beta changes
        
        delta = 1e-8 # to avoid division by zero
        r = np.zeros(shape=(self.n_features, self.n_features))

        for i in range(self.n_iter):

            gradient = 2.0 / self.n_inputs * self.X_all.T @ (self.X_all @ beta - self.y_all) + 2.0 * self.lmbda * beta # if lambda>0, we do Ridge regression
                        
            if self.optimization is None:
                change = -eta*gradient + self.gamma*change
                #print("No optimization Change = \n",change)
            elif self.optimization == "adagrad":
                r += gradient @ gradient.T
                scale = np.c_[1.0 / (delta + np.sqrt(np.diagonal(r)))]
                change = -eta*np.multiply(scale, gradient) + self.gamma*change
            else:
                raise Exception("Invalid optimization method.")
                
            beta += change

            if(np.linalg.norm(gradient) <= self.eps): #stop iterating once it has converged sufficiently
                print(f"Stopped after {i} iterations.")
                self.beta = beta
                break
        
        self.beta = beta
    
    def SGD(self):        
        m = int(self.n_inputs/self.batch_size) # num. batches

        beta = np.random.randn(self.n_features, 1) #initialize beta
        delta = 1e-8 # to avoid division by zero

        def learning_schedule(t):
            alpha = t / (self.n_epochs*m) # taken from Goodfellow
            return (1-alpha) * self.eta0 + alpha * self.eta0*0.01

        change = np.zeros_like(beta) # initiate vector used for computing change in beta
        eta = self.eta0

        indeces = np.arange(self.n_inputs)

        for epoch in range(1, self.n_epochs+1):
            s = np.zeros_like(beta) # for computing first and second moments
            r = np.zeros(shape=(self.n_features, self.n_features))
            random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
            batches = np.array_split(random_indeces, m) # split into batches

            for i in range(m):
                self.X = self.X_all[batches[i]]
                self.y = np.c_[self.y_all[batches[i]]]

                #Compute the gradient using the data in minibatch k
                gradient = 2.0/ self.batch_size * self.X.T @ (self.X @ beta - self.y) + 2.0 * self.lmbda * beta
                eta = learning_schedule(t = epoch * m + i)

                if self.optimization is None:
                    change = -eta*gradient + self.gamma*change

                elif self.optimization == "adagrad":
                    r = r + gradient @ gradient.T # sum of squared gradients
                    rr = np.diagonal(r) # we want g_i**2
                    scale = np.c_[1 / (delta + np.sqrt(rr))]
                    change = -eta*np.multiply(scale, gradient) + self.gamma*change # scale gradient element-wise

                elif self.optimization == "RMSprop":   
                    r = self.rho1 * r + (1 - self.rho1) * gradient @ gradient.T
                    rr = np.c_[np.diagonal(r)]
                    scale = np.c_[1.0 / (delta + np.sqrt(rr))]
                    change = -eta*np.multiply(scale, gradient) # scale gradient element-wise
                
                elif self.optimization == "adam":
                    t = i+1 # iteration number
                    #here we compute 1st and 2nd moments
                    r = self.rho2*r + (1 - self.rho2) * gradient @ gradient.T    
                    s = self.rho1*s + (1 - self.rho1) * gradient 
                    
                    ss = s/(1 - self.rho1**t) # here we correct the bias
                    rr = np.c_[np.diagonal(r)/(1 - self.rho2**t)]
                    
                    change = np.c_[ -eta * ss / (delta + np.sqrt(rr))] 

                else:
                    raise Exception("Invalid optimization method.")
                
                beta += change
        
        self.beta = beta

    def AnalyticSolution(self):
        # Optimal parameters found analytically. 
        # If lmbda > 0 then we do Ridge regression, otherwise OLS regression
        Id = np.eye((self.X_all.T @ self.X_all).shape[0])
        self.beta = np.linalg.inv((self.X_all.T @ self.X_all) + self.lmbda * Id) @ self.X_all.T @ self.y_all

    def predict(self, X):
        self.X = X
        
        self.prediction = self.X @ self.beta
        # make prediction based on trained parameters and test data X
        return self.prediction