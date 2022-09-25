
try: 
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

from numba import njit


class SGD():
    
    def __init__(self, alpha = 0.001):
        self.alpha = alpha

    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, gradient, weights, v, m, v_hat, m_hat):
        weights -= gradient * alpha

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, gradient, weights, v, m, v_hat, m_hat):
        weights -= gradient * alpha

        return weights, v, m, v_hat, m_hat

    


class Momentum():
    
    def __init__(self, alpha = 0.01, beta = 0.9):
        self.alpha = alpha
        self.beta = beta

    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, self.beta, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * gradient
        weights -= v * alpha

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * gradient
        weights -= v * alpha

        return weights, v, m, v_hat, m_hat
        

class RMSProp():
    
    def __init__(self, alpha = 0.01, beta = 0.9, epsilon = 0.000000001):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, self.beta, self.epsilon, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, epsilon, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * np.power(gradient, 2)
        weights -= alpha * gradient / (np.sqrt(v) + epsilon)

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, epsilon, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * np.power(gradient, 2)
        weights -= alpha * gradient / (np.sqrt(v) + epsilon)

        return weights, v, m, v_hat, m_hat

class Adam():

    def __init__(self, alpha = 0.001, beta = 0.9, beta2 = 0.999, epsilon = 0.000000001):
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon

    #Temporary solution for the issue with cupy and njit
    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, self.beta, self.beta2, self.epsilon, gradient, weights, v, m, v_hat, m_hat, t)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t))
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t))
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat

class Nadam():
    
    def __init__(self, alpha = 0.001, beta = 0.9, beta2 = 0.999, epsilon = 0.000000001):
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        

    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy
     
        return self._update(self.alpha, self.beta, self.beta2, self.epsilon, gradient, weights, v, m, v_hat, m_hat, t)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t)) + (1 - beta) * gradient / (
            1 - np.power(beta, t)
        )
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t)) + (1 - beta) * gradient / (
            1 - np.power(beta, t)
        )
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat



optimizers = {
    
    "sgd": SGD(),
    "momentum": Momentum(),
    "rmsprop": RMSProp(),
    "adam": Adam(),
    "nadam": Nadam(),
    
}