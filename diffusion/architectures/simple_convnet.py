try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from diffusion.layers import Dense, Conv2D, BatchNormalization
from diffusion.activations import Sigmoid, Softmax, ReLU, LeakyReLU



class SimpleConvNet(): #Just example
    def __init__(self, channels_num = 32):

        channels = [channels_num, channels_num * 2, channels_num * 4, channels_num * 8]
        
        self.layers = [
            Conv2D(channels_num = 1, kernels_num = channels[0], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[0],  kernels_num = channels[1], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[1], kernels_num = channels[2], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[2], kernels_num = channels[3], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[3], kernels_num = channels[2], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[2], kernels_num = channels[1], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[1], kernels_num = channels[0], kernel_shape = (7, 7), padding = (3, 3)),
            LeakyReLU(),
            Conv2D(channels_num = channels[0], kernels_num = 1, kernel_shape = (3, 3), padding = (1, 1))
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