try:
    import cupy as np
    is_cupy_available = True
except:
    import numpy as np
    is_cupy_available = False


from diffusion.layers import Dense, Conv2D, Conv2DTranspose, PositionalEncoding, BatchNormalization2D
from diffusion.activations import ReLU, LeakyReLU



class ResBlock():
    def __init__(self, input_channels, output_channels, time_emb_dim, up = False):
        super().__init__()
        self.time_embedding =  Dense(time_emb_dim, output_channels)
        self.input_channels = input_channels
        self.output_channels = output_channels
        if up:
            self.conv1 = Conv2D(2 * input_channels, output_channels, kernel_shape = (3, 3), padding = (1, 1))
            self.transform = Conv2DTranspose(output_channels, output_channels, kernel_shape = (4, 4), stride = (2, 2), padding = (1, 1))
            
        else:
            self.conv1 = Conv2D(input_channels, output_channels, kernel_shape = (3, 3), padding=(1, 1))
            self.transform = Conv2D(output_channels, output_channels, kernel_shape = (4, 4), stride = (2, 2),  padding = (1, 1))

        self.conv2 = Conv2D(output_channels, output_channels, kernel_shape = (3, 3), padding=(1, 1))
        self.relu1  = LeakyReLU(alpha = 0.01)
        self.relu2  = LeakyReLU(alpha = 0.01)
        self.relu3  = LeakyReLU(alpha = 0.01)

        self.bnorm1 = BatchNormalization2D(output_channels, momentum = 0.1, epsilon = 1e-5)
        self.bnorm2 = BatchNormalization2D(output_channels, momentum = 0.1, epsilon = 1e-5)
        
        
    def forward(self, x, t, training):
    
        x = self.conv1.forward(x)
        h = self.relu1.forward(x)
        h = self.bnorm1.forward(h, training)

        t = self.time_embedding.forward(t)
        
        time_emb = self.relu2.forward(t)

        time_emb = time_emb[(..., ) + (None, ) * 2]
      
        h = h + time_emb
      
        
        h = self.conv2.forward(h)
        h = self.relu3.forward(h)
        h = self.bnorm2.forward(h, training)
 

        return self.transform.forward(h)

    def backward(self, error):
        
        h_error = self.transform.backward(error)

        h_error = self.bnorm2.backward(h_error)
        h_error = self.relu3.backward(h_error)
        h_error = self.conv2.backward(h_error)
      
       
        t_error = np.sum(h_error, axis = (2, 3))
        t_error = self.relu2.backward(t_error)
        t_error = self.time_embedding.backward(t_error)
       

        h_error = self.bnorm1.backward(h_error)
        h_error = self.relu1.backward(h_error)
        h_error = self.conv1.backward(h_error)
       
        
        return h_error, t_error

    def update_weights(self, layer_num):
        self.conv1.update_weights(layer_num)
        self.bnorm1.update_weights(layer_num + 1)
        self.time_embedding.update_weights(layer_num + 2)
        self.conv2.update_weights(layer_num + 3)
        self.bnorm2.update_weights(layer_num + 4)
        self.transform.update_weights(layer_num + 5)

        return layer_num + 5

    def set_optimizer(self, optimizer):
        self.transform.set_optimizer(optimizer)
        self.conv1.set_optimizer(optimizer)
        self.conv2.set_optimizer(optimizer)
        self.bnorm1.set_optimizer(optimizer)
        self.bnorm2.set_optimizer(optimizer)
        self.time_embedding.set_optimizer(optimizer)


class SimpleUNet():

    def __init__(self, image_channels,  image_size, down_channels = (32, 64, 128, 256, 512), up_channels = (512, 256, 128, 64, 32)):
      
        noise_channels = image_channels
        time_emb_dim = 32

       
        self.time_embedding = [
                PositionalEncoding(max_len = 1000, d_model = time_emb_dim),
                Dense(time_emb_dim, time_emb_dim),
                LeakyReLU()
            ]
        
       
        if image_size & (image_size - 1) != 0:
            self.input_conv = Conv2DTranspose(image_channels, down_channels[0], kernel_shape=(5, 5))
            self.output_conv = Conv2D(up_channels[-1], noise_channels,  kernel_shape=(5, 5))
        else:
            self.input_conv = Conv2D(image_channels, down_channels[0], kernel_shape=(3, 3), padding = (1, 1))
            self.output_conv = Conv2DTranspose(up_channels[-1], noise_channels,  kernel_shape=(3, 3), padding = (1, 1))

       
        self.down_layers  = [ResBlock(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)]
       
        self.up_layers = [ResBlock(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)]

        

    def forward(self, x, t, training):
        x = np.asarray(x)
        
        t = np.asarray(t[:, None, None], dtype = np.float32)

        
        for layer in self.time_embedding:
            
            # t = t.reshape(t.shape[0], -1)
            t = layer.forward(t)
        t = t.reshape(t.shape[0], -1)
       
        x = self.input_conv.forward(x)
       
        residual_inputs = []
        for down_layer in self.down_layers:
            x = down_layer.forward(x, t, training)
            residual_inputs.append(x)
          
        for up_layer in self.up_layers:
            residual_x = residual_inputs.pop()

            x = np.concatenate((x, residual_x), axis = 1)      
            x = up_layer.forward(x, t, training)
        return self.output_conv.forward(x)


    def backward(self, error):
        error = self.output_conv.backward(error) 

        residual_inputs = []
        t_errors = 0
        for up_layer in reversed(self.up_layers):
            error, t_error = up_layer.backward(error)
            error, residual_x = np.split(error, 2, axis = 1)
            residual_inputs.append(residual_x)
            t_errors += t_error
           
      
        for down_layer in reversed(self.down_layers):
            residual_x = residual_inputs.pop()
           
            error = error + residual_x         
            error, t_error = down_layer.backward(error)
            t_errors += t_error
        
        error = self.input_conv.backward(error)

       
        for layer in reversed(self.time_embedding):
            
            t_errors = layer.backward(t_errors)

        return error
        


    def update_weights(self):

        i = 1
        for layer in self.time_embedding:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(layer_num = i)
                
            i += 1

        self.input_conv.update_weights(layer_num = i)

        for layer in self.down_layers:
            i = layer.update_weights(layer_num = i)
            i += 1

        for layer in self.up_layers:
            i = layer.update_weights(layer_num = i)
            i += 1
        
        self.output_conv.update_weights(layer_num = i)




    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

        for layer in self.time_embedding:
            if hasattr(layer, 'set_optimizer'):
                layer.set_optimizer(optimizer)

        self.input_conv.set_optimizer(optimizer)

        for down in self.down_layers:
            if hasattr(down, 'set_optimizer'):
                down.set_optimizer(optimizer)

        for up in self.up_layers:
            if hasattr(up, 'set_optimizer'):
                up.set_optimizer(optimizer)

        self.output_conv.set_optimizer(optimizer)




