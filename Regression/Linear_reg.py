import numpy as np

class linear_regression():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self,X,y,method="NE",lr =0.01, epochs = 1000):
        if (method =="NE"):
            self.normal_equation(X,y)
        elif (method == "GD"):
            self.gradient_descent(X,y,lr,epochs)
    
    def normal_equation(self, X,y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.c_[np.ones(X.shape[0]), X]
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1]
    
    def gradient_descent(self, X, y, lr, epochs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.size
        scaling_method=1
        Xs = self._scaling_features(X, scaling_method)
        m, b = 0, 0
        for _ in range(epochs):
            error = y - (m * Xs + b)
            m -= lr * (-(2 / n) * (Xs * error).sum())
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = self._descaling_features(X, m, b, scaling_method)
        
    def predict(self, X, Result=False):
        X = np.asarray(X, dtype=float)
        pred = X * self.coef_ + self.intercept_
        if Result:
            print(pred)
        return pred

    def R2_score(self, X, y, print_comp=False):
        y = np.asarray(y, dtype=float)
        prediction = self.predict(X)
        if (print_comp): print('prediction:', prediction,sep='\n',end='\n')
        ### Squared Sum Of Errors
        SSE = ((y - prediction) ** 2).sum()
        if (print_comp): print('SSE=', SSE, end='\n')
        ### Total Squared Sum
        TSS = ((y - y.mean()) ** 2).sum()
        if (print_comp): print('TSS=', TSS, end='\n')
        ### Score
        return (1 - SSE/TSS)

    def print_parameters(self):
        print('slope:', self.coef_)
        print('intercept', self.intercept_)
    @staticmethod
    def _scaling_features(X,scaling_method):
        match (scaling_method):
            case 1: # standardization (z-score)
                avg, sigma = X.mean(), X.std()
                return (X - avg) / sigma
            case 2: # Min-Max Scaling
                max, min = X.max(), X.min()
                return (X - min) / (max - min)
            case 3: # Absolute Maximum
                abs_max = np.abs(X).max()
                return X / abs_max
    @staticmethod
    def _descaling_features(X, m, b, scaling_method):
        match (scaling_method):
            case 1: # standardization (z-score)
                avg, sigma = X.mean(), X.std()
                return m / sigma, b - m * avg / sigma
            case 2: # Min-Max Scaling
                maxX, minX = X.max(), X.min()
                return m / (maxX - minX), b - m * minX / (maxX - minX)
            case 3: # Absolute Maximum
                abs_max = np.abs(X).max()
                return m / abs_max, b
