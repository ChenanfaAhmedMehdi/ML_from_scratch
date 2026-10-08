import numpy as np
import pandas as pd

class linear_regression():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    def fit(self,data,method="NE"):
        if (method =="NE"):
            self.normal_equation(data)
        elif (method == "GD"):
            self.gradient_descent(data,0.001)
    def normal_equation(self, data):
        X = np.array(data['Study Time'])
        y = np.array(data['Score'])
        Xb = np.c_[np.ones((X.shape[0],1)),X]
        A = np.linalg.inv(Xb.T@ Xb) @ Xb.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1:]
    def gradient_descent(self, data, lr):
        m, b = 0,0
        X = np.array(data['Study Time'])
        y = np.array(data['Score'])
        n = X.size
        avg, sigma = X.mean(), X.std()
        # standardization (z-score)
        Xs = (X - avg) / sigma
        epoch = 15000
        for _ in range(epoch):
            error = y - (m * Xs + b)   
            m -= lr * (-(2 / n) * (Xs * error).sum())
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = np.atleast_1d(m / sigma),b - m * avg / sigma
    def predict(self,Features,Result = False):
        Features = np.atleast_2d(Features)
        if (Result): print(Features @ self.coef_ + self.intercept_)
        return (Features @ self.coef_ + self.intercept_)
    def R2_score(self,Data,print_comp = False):
        prediction = self.predict(Data[['Study Time']])
        prediction = pd.Series(prediction)
        if (print_comp): print('prediction:', prediction,sep='\n',end='\n')
        ### Squared Sum Of Errors
        SSE = ((Data['Score']- prediction)**2).sum()
        if (print_comp): print('SSE=', SSE, end='\n')
        ### Total Squared Sum
        TSS = ((Data['Score'] - Data['Score'].mean())**2).sum()
        if (print_comp): print('TSS=', TSS, end='\n')
        ### Score
        return (1 - SSE/TSS)
    def print_parameters(self):
        print('slope:', self.coef_)
        print('intercept', self.intercept_)
