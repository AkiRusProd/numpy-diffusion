import numpy as np


try: 
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')



import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm
from PIL import Image
from typing import Type, Union, List, Tuple, Dict, Optional, Callable

from diffusion.layers import Dense, Conv2D, BatchNormalization
from diffusion.losses import MSE
from diffusion.optimizers import Adam, SGD, Momentum, Nadam
from diffusion.activations import LeakyReLU
from diffusion.models.simple_convnet import SimpleConvNet

# https://huggingface.co/blog/annotated-diffusion
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
# https://nn.labml.ai/diffusion/ddpm/index.html


training_data = open('dataset/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist_test.csv','r').readlines()





x_num, y_num = 4, 4
channels, image_size = 1, 28
margin = 10


def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:]).reshape(channels, image_size, image_size) / 127.5 - 1)#/ 255 [0; 1]  #/ 127.5-1 [-1; 1]
        targets.append(int(line[0]))

    return inputs, targets

training_inputs, training_targets = prepare_data(training_data)







def linear_schedule(start: float, end: float, timesteps: int):
    return np.linspace(start, end, timesteps)


class Diffusion():
    def __init__(self, model: Type[SimpleConvNet], timesteps: int, beta_start: float, beta_end: float, criterion, optimizer):
        self.model = model

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = linear_schedule(beta_start, beta_end, timesteps)
        self.sqrt_betas =  np.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.inv_sqrt_alphas =  1 / np.sqrt(self.alphas)
        
        self.alphas_cumprod =  np.cumprod(self.alphas, axis = 0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

        self.scaled_alphas =  (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod

        self.criterion = criterion
        self.optimizer = optimizer




    def forward(self, x: np.float32, t = None):

        timesteps_selection = np.random.randint(1, self.timesteps, (x.shape[0],))
        noise = np.random.normal(size = x.shape)
       
        x_t = self.sqrt_alphas_cumprod[timesteps_selection, None, None, None] * x + self.sqrt_one_minus_alphas_cumprod[timesteps_selection, None, None, None] * noise

        x = self.model.forward(x_t)

        return x, noise



    def denoise_sample(self, n_sample: int, image_size: Tuple[int, int, int]):

        x_t = np.random.normal(size = (n_sample, *image_size))
     
        for t in reversed(range(0, self.timesteps)):
            noise = np.random.normal(size = (n_sample, *image_size)) if t > 1 else 0
            output = cp.asnumpy(self.model.forward(x_t, t / self.timesteps))

            x_t = self.inv_sqrt_alphas[t] * (x_t - output.reshape(n_sample, *image_size) * self.scaled_alphas[t]) + self.sqrt_betas[t] * noise

        return x_t

    def get_images_set(self, x_num: int, y_num: int, margin: int, images: np.float32, image_size: Tuple[int, int, int]):

        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x)) * 255

        channels, H_size, W_size = image_size

        images_array = np.full((y_num * (margin + H_size), x_num * (margin + W_size), channels), 255, dtype = np.uint8)
        num = 0
        for i in range(y_num):
            for j in range(x_num):
                y = i * (margin + H_size)
                x = j * (margin + W_size)

                images_array[y :y + H_size, x: x + W_size] = normalize(images[num].reshape(H_size, W_size, channels))

                num += 1

        images_array = images_array[: (y_num - 1) * (H_size + margin) + H_size, : (x_num - 1) * (W_size + margin) + W_size]

        if channels == 1:
            return Image.fromarray(images_array.squeeze(axis = 2)).convert("L")
        else:
            return Image.fromarray(images_array)



    def train(self, dataset, epochs, batch_size = 128, save_every_epochs = 10):
        self.model.set_optimizer(self.optimizer)

        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        loss_history = []
        for epoch in range(epochs):
            tqdm_range = tqdm(enumerate(data_batches), total = len(data_batches))

            loss = []
            for batch_num, (batch) in tqdm_range:
                
                output, noise = self.forward(batch)
                loss.append(self.criterion.loss(output, noise).mean())
                error = self.criterion.derivative(output, noise)
                error = self.model.backward(error)
                self.model.update_weights()

                tqdm_range.set_description(
                    f"loss: {loss[-1]:.7f} | epoch {epoch + 1}/{epochs}"
                )


                if batch_num == (len(data_batches) - 1):
                    if is_cupy_available:
                        epoch_loss = cp.mean(cp.array(loss))
                    else:
                        epoch_loss = np.mean(loss)

                    tqdm_range.set_description(
                            f"loss: {epoch_loss:.7f} | epoch {epoch + 1}/{epochs}"
                    )


            if ((epoch + 1) % save_every_epochs == 0):

                samples = self.denoise_sample(x_num * y_num, (channels, image_size, image_size))
                images_grid = self.get_images_set(x_num, y_num, margin, samples, (channels, image_size, image_size))
                images_grid.save(f"saved images/np_ddpm_{epoch + 1}.png")

            loss_history.append(epoch_loss)

        return loss_history



diffusion = Diffusion(model = SimpleConvNet(), timesteps = 300, beta_start = 0.0001, beta_end = 0.02, criterion = MSE(), optimizer = Adam(alpha = 2e-4))
diffusion.train(training_inputs, epochs = 30, batch_size = 10, save_every_epochs = 1)

