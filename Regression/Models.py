import numpy as np
from itertools import product
from Scaling import scaling_model
from Evaluation import BIC
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
    def coordinate_descent(self, X, y):
        X, y = self._prepare(X,y)
        n, k = X.shape
        b_new = b = 0
        w = np.zeros((k, 1))
        w_new = w.copy()
        for _ in range(100_000):
            w_new = self._coordinate_update(X, y, b, w.copy(), n, k)
            b_new = (y - X @ w_new).sum() / n
            if (max(np.max(np.abs(w_new - w)), abs(b_new - b)) < 1e-9):
                break
            b = b_new
            w = w_new.copy()
        else:
            raise RuntimeError("Not fitted due to insufficient number of iterations")
        self.coef_, self.intercept_ = w, b_new
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
            self.fit(Xi,y)
            results.append(BIC(self.predict(Xi), y, Xi.shape[1]))
        return results.index(min(results)) + start

class ridge_regression(base_model):
    def __init__(self, a = 0):
        super().__init__()
        self.alpha = a
    def fit(self,X,y,method="NE"):
        if (method =="NE"):
            self.normal_equation(X,y)
        elif (method == "GD"):
            self.gradient_descent(X,y)
        elif(method == "CD"):
            self.coordinate_descent(X,y)
        else:
            raise ValueError(f"unknown fitting method: {method}")
    def normal_equation(self, X,y):
        X, y = self._prepare(X, y)
        n, k = X.shape
        Xb = np.c_[np.ones((n, 1)), X]
        penalty = np.diag([0.0] + [1.0] * k) 
        A = np.linalg.inv(Xb.T @ Xb + self.alpha * penalty) @ Xb.T @ y
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
            w -= lr * (-(2 / n) * (Xs.T @ error - self.alpha * w))
            b -= lr * (-(2 / n) * error.sum())
        self.coef_, self.intercept_ = scaler.descale_param(w, b)
    def _coordinate_update(self, X, y, b, w_new, n, k):
        for i in range(k):
            X_i = X[:,i:i+1]
            zi = X_i.T @ X_i 
            ri = X_i.T @ ((y - X @ w_new - b) + X_i * w_new[i,0])
            w_new[i] = ri / (zi + self.alpha)
        return w_new
    def _determine_lr(self, Xs):
        n, k = Xs.shape
        A = np.column_stack([np.ones(n), Xs])
        penalty = np.diag([0.0] + [1.0] * k)
        H = 2 * A.T @ A / n + 2 * self.alpha * penalty
        eig = np.linalg.eigvalsh(H)
        lr = 1 / eig.max()
        tol = 1e-9
        k = np.log(1 / tol) / (lr * eig.min())
        return lr, int(np.ceil(k))

class linear_regression(ridge_regression(0)):
    def __init__(self):
        super().__init__(0)

class elastic_net(base_model):
    def __init__(self,L1_ratio = 0.5):
        super().__init__()
        self.alpha = 1.0
        self.ratio = L1_ratio
    def fit(self,X,y):
        self.coordinate_descent(X,y)
    def _coordinate_update(self, X, y, b, w_new, n, k):
        for i in range(k):
            X_i = X[:,i:i+1]
            zi = ((X_i.T @ X_i) / n).item()
            ri = ((X_i.T @ ((y - X @ w_new - b) + X_i * w_new[i,0])) / n).item()
            soft_threshold = self.sign(ri) * max(abs(ri) - self.alpha * self.ratio, 0)
            w_new[i] = soft_threshold / (zi + self.alpha*(1 - self.ratio))
        return w_new
    @staticmethod
    def sign(num):
        if (num == 0):
            return 0
        elif (num > 0):
            return 1
        else:
            return -1
