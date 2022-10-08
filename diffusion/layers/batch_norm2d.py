try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class BatchNormalization2D():
    """
    Applies batch normalization to the input data
    ---------------------------------------------
        Args:
            `momentum` (float): the momentum parameter of the moving mean
            `epsilon` (float): the epsilon parameter of the algorithm
        Returns:
            output: the normalized input data with same shape
        References:
            https://kevinzakka.github.io/2016/09/14/batch_normalization/

            https://agustinus.kristia.de/techblog/2016/07/04/batchnorm/

            https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch
    """

    def __init__(self, features_num, momentum = 0.99, epsilon = 0.001, input_shape = None, data_type = np.float32):
        self.features_num = features_num

        self.momentum = momentum
        self.epsilon  = epsilon

        self.gamma = None
        self.beta = None

        self.mean = None
        self.var = None

        self.moving_mean = None
        self.moving_var = None

        self.optimizer = None

        self.input_shape = input_shape
        self.data_type = data_type

        self.build()
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    

    def build(self):
        self.gamma = np.ones(self.features_num).astype(self.data_type)
        self.beta = np.zeros(self.features_num).astype(self.data_type)


        self.vg, self.mg         = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)
        self.vg_hat, self.mg_hat = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)

        self.vb, self.mb         = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)
        self.vb_hat, self.mb_hat = np.zeros_like(self.gamma).astype(self.data_type), np.zeros_like(self.gamma).astype(self.data_type)

        self.output_shape = self.input_shape


    def forward(self, X, training = True):
        self.input_data = X
        self.batch_size = X.shape[0]

        if self.input_shape is None:
            self.input_shape = self.input_data.shape[1:]

            self.build()
        
        if self.moving_mean is None: self.moving_mean = np.mean(self.input_data, axis = (0, 2, 3))
        if self.moving_var is None: self.moving_var = np.var(self.input_data, axis = (0, 2, 3))
        
        if training == True:
            self.mean = np.mean(self.input_data, axis = (0, 2, 3))
            self.var = np.var(self.input_data, axis = (0, 2, 3))

            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * self.var
        else:
            self.mean = self.moving_mean
            self.var = self.moving_var

    
        self.X_centered = (self.input_data - self.mean[None, :, None, None])
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        X_hat = self.X_centered * self.stddev_inv[None, :, None, None]

        self.output_data = self.gamma[None, :, None, None] * X_hat + self.beta[None, :, None, None]
        
        return self.output_data


    def backward(self, error):

        B = self.input_data.shape[0] * self.input_data.shape[2] * self.input_data.shape[3]

        X_hat = self.X_centered * self.stddev_inv[None, :, None, None]
        
        dX_hat = error * self.gamma[None, :, None, None]
        dvar = (-0.5 * dX_hat * self.X_centered).sum((0, 2, 3), keepdims=True)  * (self.stddev_inv[None, :, None, None] ** 3.0)
        dmu = (- self.stddev_inv[None, :, None, None] * dX_hat).sum((0, 2, 3), keepdims = True) + (dvar * (-2.0 * self.X_centered).sum((0, 2, 3), keepdims = True) / B)
        
        output_error = (dX_hat * self.stddev_inv[None, :, None, None]) + (dvar * 2.0 * self.X_centered / B) + (dmu / B)
        
        self.grad_gamma = (error * X_hat).sum((0, 2, 3))
        self.grad_beta = (error).sum((0, 2, 3))

        return output_error


    def update_weights(self, layer_num):
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

    def get_grads(self):
        return self.grad_gamma, self.grad_beta

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads





