import numpy as np

class LinearRegression:
    def __init__(self, 
    lmbda, 
    ):
        
        self.lmbda = lmbda # regularization parameter
    
    def fit(self, X, y):
        self.X_all = X
        self.y_all = y
        self.X = X
        self.y = y
        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]

        self.AnalyticSolution()

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