import numpy as np
from itertools import product
from Regression.Scaling import scaling_model
from Regression.Evaluation import BIC

class base_model():
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
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
    def poly_features(self, X,degree):
        X = self._prepare(X)
        k = X.shape[1]
        Xi = X
        X = np.empty((Xi.shape[0],0))
        for exponents in product(range(degree + 1), repeat=k):
            if sum(exponents) <= degree:
                term = np.prod(Xi ** exponents, axis=1)
                X = np.c_[X,term]
        return X[:,1:]
    def print_parameters(self):
        print('slope:', self.coef_)
        print('intercept', self.intercept_)
    def determine_degree(self, X, y, start_or_end, end = None):
        if (end is None): 
            if(start_or_end < 2 or start_or_end != int(start_or_end)):
                raise ValueError(f"degree must be an integer >= 2, got {start_or_end}")
            start = 2
            end = start_or_end
        else:
            if(start_or_end < 2 or start_or_end != int(start_or_end)):
                raise ValueError(f"start degree must be an integer >= 2, got {start_or_end}")
            elif (end < 2 or end != int(end)):
                raise ValueError(f"end degree must be an integer >= 2, got {end}")
            elif (end <= start_or_end):
                raise ValueError(f"end degree ({end}) must be > start degree ({start_or_end})")
            start = start_or_end
        results = []
        for i in range(start, end + 1):
            Xi = self.poly_features(X,i)
            results.append(BIC(self.predict(Xi), y, Xi.shape[1]))
        return results.index(min(results)) + start

class linear_regression(base_model):
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
        scaler = scaling_model()
        Xs = scaler.scale(X)
        lr, epochs = self._determine_lr(Xs)
        w = np.zeros(k)
        b = 0
        for _ in range(epochs):
            error = y - (Xs @ w + b)
            w -= lr * (-(2 / n) * (Xs.T @ error))
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = scaler.descale_param(w, b)
    @staticmethod
    def _determine_lr(Xs):
        n = Xs.shape
        A = np.column_stack([np.ones(n), Xs])
        H = 2 * A.T @ A / n
        eig = np.linalg.eigvalsh(H)
        lr = 1 / eig.max()
        tol = 1e-9
        k = np.log(1 / tol) / (lr * eig.min())
        return lr, int(np.ceil(k))
    
class elastic_net(base_model):
    def __init__(self):
        super().__init__()
        self.alpha2 = 1.0
    def fit(self,X,y,method="NE"):
        if (method =="NE"):
            self.normal_equation(X,y)
        elif (method == "GD"):
            self.gradient_descent(X,y)
        else:
            raise ValueError(f"unknown fitting method: {method}")
    def normal_equation(self, X, y, lam=0.0):
        X, y = self._prepare(X, y)
        n, k = X.shape
        Xb = np.c_[np.ones((n, 1)), X]
        penalty = np.diag([0.0] + [1.0] * k)
        A = np.linalg.inv(Xb.T @ Xb + 2 * lam * penalty) @ Xb.T @ y
        self.intercept_ = A[0]
        self.coef_ = A[1:]
    def gradient_descent(self, X, y, lam = 0.0):
        X, y = self._prepare(X, y)
        n, k = X.shape
        scaler = scaling_model()
        Xs = scaler.scale(X)
        lr, epochs = self._determine_lr(Xs, lam)
        w = np.zeros(k)
        b = 0
        for _ in range(epochs):
            error = y - (Xs @ w + b)
            w -= lr * (-(2 / n) * (Xs.T @ error) +  2 * lam * w)
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = scaler.descale_param(w, b)
    @staticmethod
    def _determine_lr(Xs, lam = 0.0):
        n, c = Xs.shape
        A = np.column_stack([np.ones(n), Xs])
        penalty = np.diag([0.0] + [1.0] * c)
        H = 2 * A.T @ A / n + 2 * lam * penalty
        eig = np.linalg.eigvalsh(H)
        lr = 1 / eig.max()
        tol = 1e-9
        k = np.log(1 / tol) / (lr * eig.min())
        return lr, int(np.ceil(k))
