try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from diffusion.layers import Dense, Conv2D, BatchNormalization
from diffusion.activations import Sigmoid, Softmax, ReLU, LeakyReLU



class SimpleConvNet(): #Just example
    def __init__(self):
        
        self.layers = [
            Conv2D(input_shape = (1, 28, 28),   kernels_num = 32, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (32, 28, 28),  kernels_num = 64, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (64, 28, 28), kernels_num = 128, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (128, 28, 28), kernels_num = 256, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (256, 28, 28), kernels_num = 128, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (128, 28, 28), kernels_num = 64, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (64, 28, 28), kernels_num = 32, kernel_shape = (7, 7), padding = (3, 3, 3, 3)),
            LeakyReLU(),
            Conv2D(input_shape = (32, 28, 28),  kernels_num = 1, kernel_shape = (3, 3), padding = (1, 1, 1, 1))
        ]
    
    def forward(self, x, t = None):
        x = np.asarray(x)

        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, error):
        error = np.asarray(error)
        
        for layer in reversed(self.layers):
            error = layer.backward(error)

        return error

    def update_weights(self):
        for i, layer in enumerate(reversed(self.layers)):
            if hasattr(layer, 'update_weights'):
                layer.update_weights(layer_num = i + 1)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

        for layer in self.layers:
            if hasattr(layer, 'set_optimizer'):
                layer.set_optimizer(self.optimizer)