import numpy as np
class scaling_model():
    def __init__(self, method=1):
        self.avg = self.sigma = self.max = self.min = self.abs_max = None
        if method not in (1, 2, 3):
            raise ValueError(f"unknown scaling method: {method}")
        self.method = method
        self.fitted = False
    def fit(self, X):
        if (self.method == 1):
            self.avg_, self.sigma_ = X.mean(axis=0), X.std(axis=0)
            if ((self.sigma_ == 0).any()):
                raise ValueError("X has at least one constant feature")
        elif (self.method == 2):
            self.max, self.min= X.max(axis=0), X.min(axis=0)
            if ((self.max - self.min == 0).any()):
                raise ValueError("X has at least one constant feature")
        elif (self.method == 3):
            self.abs_max = np.abs(X).max(axis=0)
            if ((self.abs_max == 0).any()):
                raise ValueError("X has at least one feature that is all zeros")
        self.fitted = True
        return self
    def scale(self,X):
        if (self.fitted == False):
            raise RuntimeError("Scaler has not been fitted. Call fit().")
        match (self.method):
            case 1: # standardization (z-score)
                return (X - self.avg) / self.sigma
            case 2: # Min-Max Scaling
                return (X - self.min) / (self.max - self.min)
            case 3: # Absolute Maximum
                return X / self.abs_max
    def descale_param(self, w, b):
        if (self.fitted == False):
            raise RuntimeError("Scaler has not been fitted. Call fit().")        
        match (self.method):
            case 1: # standardization (z-score)
                return w / self.sigma, b - (w * self.avg / self.sigma).sum()
            case 2: # Min-Max Scaling
                return w /(self.max - self.min), b - ((w * self.min) / (self.max - self.min)).sum()
            case 3: # Absolute Maximum
                return w/self.abs_max, b