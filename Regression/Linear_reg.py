import numpy as np
from itertools import product

class linear_regression():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self,X,y,method="NE"):
        if (method =="NE"):
            self.normal_equation(X,y)
        elif (method == "GD"):
            self.gradient_descent(X,y)
        else:
            raise ValueError(f"unknown fitting method: {method}")
    
    def normal_equation(self, X,y):
        X, y = self._prepare(X, y)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1:]
    
    def gradient_descent(self, X, y):
        X, y = self._prepare(X, y)
        n, k = X.shape
        scaling_method=1
        Xs = self._scaling_features(X, scaling_method)
        lr, epochs = self._determine_lr(Xs)
        w = np.zeros(k)
        b = 0
        for _ in range(epochs):
            error = y - (Xs @ w + b)
            w -= lr * (-(2 / n) * (Xs.T @ error))
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = self._descaling_features(X, w, b, scaling_method)
        
    def predict(self, X, Result=False):
        if (self.coef_ is None or self.intercept_ is None):
            raise ValueError("the model hasn't been fitted yet.")
        X = self._prepare(X)
        k = len(self.coef_)
        if (X.shape[1] != k):
            raise ValueError(f"expected {k} features, got {X.shape[1]}")
        pred = X @ self.coef_ + self.intercept_
        if Result:
            print(pred)
        return pred
    @staticmethod
    def _prepare(X,y=None):
        X = np.asarray(X, dtype=float)
        if (X.ndim == 1):
            X = X.reshape(-1,1)
        if (y is None):
            return X
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
        return X, y
    def R2_score(self, X, y, print_comp=False):
        X, y = self._prepare(X, y)
        prediction = self.predict(X)
        if (print_comp): print('prediction:', prediction,sep='\n',end='\n')
        ### Squared Sum Of Errors
        SSE = ((y - prediction) ** 2).sum()
        if (print_comp): print('SSE=', SSE, end='\n')
        ### Total Squared Sum
        TSS = ((y - y.mean()) ** 2).sum()
        if (TSS == 0):
            raise ValueError("R2 is undefined because y is constant")
        if (print_comp): print('TSS=', TSS, end='\n')
        ### Score
        return (1 - SSE/TSS)

    def poly_features(self, X, degree):
        X = self._prepare(X)
        number_col = X.shape[1]
        Xi = X
        X = np.empty((Xi.shape[0],0))
        for exponents in product(range(degree + 1), repeat = number_col):
            if sum(exponents) <= degree:
                term = np.prod(Xi ** exponents, axis=1)
                X = np.c_[X,term]
        return X[:,1:]
    def print_parameters(self):
        print('slope:', self.coef_)
        print('intercept', self.intercept_)
    def BIC(self,X,y):
        X, y = self._prepare(X,y)
        n = X.shape[0]
        pred = self.predict(X)
        RSS = ((y - pred)**2).sum()
        return n * np.log(RSS / n) + (X.shape[1] + 1) * np.log(n)
    @staticmethod
    def _scaling_features(X,scaling_method):
        match (scaling_method):
            case 1: # standardization (z-score)
                avg, sigma = X.mean(axis=0), X.std(axis=0)
                if ((sigma == 0).any()):
                    raise ValueError("X has at least one constant feature")
                return (X - avg) / sigma
            case 2: # Min-Max Scaling
                max, min = X.max(axis=0), X.min(axis=0)
                if ((max - min == 0).any()):
                    raise ValueError("X has at least one constant feature")
                return (X - min) / (max - min)
            case 3: # Absolute Maximum
                abs_max = np.abs(X).max(axis=0)
                if ((abs_max == 0).any()):
                    raise ValueError("X has at least one feature that is all zeros")
                return X / abs_max
            case _:
                raise ValueError(f"unknown scaling_method: {scaling_method}")
    @staticmethod
    def _descaling_features(X, w, b, scaling_method):
        match (scaling_method):
            case 1: # standardization (z-score)
                avg, sigma = X.mean(axis=0), X.std(axis=0)
                return w / sigma, b - (w * avg / sigma).sum()
            case 2: # Min-Max Scaling
                max, min = X.max(axis=0), X.min(axis=0)
                return w /(max - min), b - ((w * min) / (max - min)).sum()
            case 3: # Absolute Maximum
                abs_max = np.abs(X).max(axis=0)
                return w/abs_max, b
            case _:
                raise ValueError(f"unknown scaling_method: {scaling_method}")
    @staticmethod
    def _determine_lr(Xs):
        A = np.column_stack([np.ones(Xs.shape[0]), Xs])
        H = 2 * A.T @ A / Xs.shape[0]
        lr = 1 / np.linalg.eigvalsh(H).max()
        tol = 1e-9
        k = np.log(1 / tol) / (lr * np.linalg.eigvalsh(H).min())
        return lr, int(np.ceil(k))
