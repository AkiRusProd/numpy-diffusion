try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


class BatchNormalization():
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
    """

    def __init__(self, momentum = 0.99, epsilon = 0.001, input_shape = None, data_type = np.float32):

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
        self.gamma = np.ones(self.input_shape).astype(self.data_type)
        self.beta = np.zeros(self.input_shape).astype(self.data_type)


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
        
        if self.moving_mean is None: self.moving_mean = np.mean(self.input_data, axis = 0)
        if self.moving_var is None: self.moving_var = np.var(self.input_data, axis = 0)
        
        if training == True:
            self.mean = np.mean(self.input_data, axis = 0)
            self.var = np.var(self.input_data, axis = 0)

            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * self.var
        else:
            self.mean = self.moving_mean
            self.var = self.moving_var

    
        self.X_centered = (self.input_data - self.mean)
        self.stddev_inv = 1 / np.sqrt(self.var + self.epsilon)

        X_hat = self.X_centered * self.stddev_inv

        self.output_data = self.gamma * X_hat + self.beta
        
        return self.output_data

        

    def backward(self, error):
        
        X_hat = self.X_centered * self.stddev_inv

        #first variant
        output_error = (1 / self.batch_size) * self.gamma * self.stddev_inv * (
            self.batch_size * error
            - np.sum(error, axis = 0)
            - self.X_centered * np.power(self.stddev_inv, 2) * np.sum(error * self.X_centered, axis = 0)
            )

        #second variant
        # dX_hat = error * self.gamma
        # output_error = (1 / self.batch_size) * self.stddev_inv * (
        #     self.batch_size * dX_hat
        #     - np.sum(dX_hat, axis = 0)
        #     - X_hat * np.sum(dX_hat * X_hat, axis = 0)
        # )

        #third (naive slow) variant
        # dX_hat = error * self.gamma
        # dvar = np.sum(dX_hat * self.X_centered, axis=0) * -.5 * self.stddev_inv**3
        # dmu = np.sum(dX_hat * -self.stddev_inv, axis=0) + dvar * np.mean(-2. * self.X_centered, axis=0)

        # output_error = (dX_hat * self.stddev_inv) + (dvar * 2 * self.X_centered / self.batch_size) + (dmu / self.batch_size)

        self.grad_gamma = np.sum(error * X_hat, axis = 0)
        self.grad_beta = np.sum(error, axis = 0)

        
        return output_error

    def update_weights(self, layer_num):
        self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat  = self.optimizer.update(self.grad_gamma, self.gamma, self.vg, self.mg, self.vg_hat, self.mg_hat, layer_num)
        self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat = self.optimizer.update(self.grad_beta, self.beta, self.vb, self.mb, self.vb_hat, self.mb_hat, layer_num)

    def get_grads(self):
        return self.grad_gamma, self.grad_beta

    def set_grads(self, grads):
        self.grad_gamma, self.grad_beta = grads
