try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False

from diffusion.activations import Sigmoid, Softmax, ReLU, LogSoftmax


class MSE():

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.power(t - y, 2)

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -2 * (t - y) / np.prod(np.asarray(y.shape[1:]))


class BinaryCrossEntropy():

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -(t * np.log(y + 1e-8) + (1 - t) * np.log(1 - y + 1e-8))

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return -t / (y + 1e-8) + (1 - t) / (1 - (y + 1e-8))


class CategoricalCrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.where(t == self.ignore_index, 0, - t * np.log(y))

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        return np.where(t == self.ignore_index, 0, -t / y)


class CrossEntropy():
    def __init__(self, ignore_index = None) -> None:
        self.ignore_index = ignore_index
        self.log_softmax = LogSoftmax()

    def loss(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        log_softmax = self.log_softmax.forward(y)
        nll_loss = -log_softmax[np.arange(len(t)), t]
        
        return np.where(t == self.ignore_index, 0, nll_loss)

    def derivative(self, y, t):
        y = np.asarray(y)
        t = np.asarray(t)
        batch_size = y.shape[0]
        err = 1/batch_size
        nll_loss_der = -1 * np.where(np.isin(y, y[np.arange(len(t)), t]), err, 0).astype(y.dtype)
       
        output_err = self.log_softmax.jacobian_backward(nll_loss_der)
        
        return np.where(t.reshape(-1, 1) == self.ignore_index, 0, output_err)






loss_functions = {
    
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()

}