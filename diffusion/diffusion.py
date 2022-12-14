import numpy as np


try: 
    import cupy as cp
    is_cupy_available = True
    print('CuPy is available. Using CuPy for all computations.')
except:
    is_cupy_available = False
    print('CuPy is not available. Switching to NumPy.')



import sys
import os
import pickle as pkl
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)

from tqdm import tqdm
from PIL import Image
from typing import Type, Union, List, Tuple, Dict, Optional, Callable
from diffusion.schedules import get_schedule

# https://arxiv.org/abs/2006.11239
# https://arxiv.org/abs/2102.09672

# https://huggingface.co/blog/annotated-diffusion
# https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
# https://nn.labml.ai/diffusion/ddpm/index.html





class Diffusion():
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, criterion, optimizer, model = None, schedule = "linear"):
        self.model = model

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = get_schedule(schedule, beta_start, beta_end, timesteps)
        self.sqrt_betas =  np.sqrt(self.betas)

        self.alphas = 1 - self.betas
        self.inv_sqrt_alphas =  1 / np.sqrt(self.alphas)
        
        self.alphas_cumprod =  np.cumprod(self.alphas, axis = 0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

        self.scaled_alphas =  (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod

        # self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        # self.alphas_cumprod_prev = np.concatenate([np.array([1]), self.alphas_cumprod[:-1]]) #np.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


        self.criterion = criterion
        self.optimizer = optimizer


    def load(self, path: str):
        pickle_model = open(f'{path}/model.pkl', 'rb')
        self.model = pkl.load(pickle_model)

        pickle_model.close()

        print(f'Loaded from "{path}"')

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_model = open(f'{path}/model.pkl', 'wb')
        pkl.dump(self.model, pickle_model)

        pickle_model.close()

        print(f'Saved to "{path}"')





    def forward(self, x: np.float32, t = None):
        """
        https://arxiv.org/abs/2006.11239

        Algorithm 1: Training; according to the paper
        """

        timesteps_selection = np.random.randint(1, self.timesteps, (x.shape[0],))
        noise = np.random.normal(size = x.shape)
       
        x_t = self.sqrt_alphas_cumprod[timesteps_selection, None, None, None] * x + self.sqrt_one_minus_alphas_cumprod[timesteps_selection, None, None, None] * noise

        x = self.model.forward(x_t, timesteps_selection / self.timesteps, training = True)

        return x, noise



    def ddpm_denoise_sample(self, sample_num = None, image_size = None, states_step_size = 1, x_t = None, orig_x = None, mask = None):
        """
        https://arxiv.org/abs/2006.11239

        Algorithm 2: Sampling; according to the paper
        """

        if mask is not None:
            assert orig_x is not None
            assert orig_x.shape == mask.shape

        if x_t is None:
            if orig_x is None:
                x_t = np.random.normal(size = (sample_num, *image_size))
            else:
                x_t = np.random.normal(size = orig_x.shape)

        x_ts = []
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'ddpm denoisinig samples', total = self.timesteps):
            noise = np.random.normal(size = x_t.shape) if t > 1 else 0
            epsilon = cp.asnumpy(self.model.forward(x_t, np.array([t]) / self.timesteps, training = False)).reshape(x_t.shape)

            x_t = self.inv_sqrt_alphas[t] * (x_t - epsilon * self.scaled_alphas[t]) + self.sqrt_betas[t] * noise
            # x_t = self.sqrt_recip_alphas[t] * (x_t - self.betas[t] * epsilon / self.sqrt_one_minus_alphas_cumprod[t]) + np.sqrt(self.posterior_variance[t]) * noise

            if mask is not None:
                orig_x_noise = np.random.normal(size = orig_x.shape)
                
                orig_x_t = self.sqrt_alphas_cumprod[t] * orig_x + self.sqrt_one_minus_alphas_cumprod[t] * orig_x_noise
                x_t = orig_x_t * mask + x_t * (1 - mask)

            if t % states_step_size == 0:
                x_ts.append(x_t)

        return x_t, x_ts

    def ddim_denoise_sample(self, sample_num = None, image_size = None, states_step_size = 1, eta = 1., perform_steps = 100, x_t = None, orig_x = None, mask = None):
        """
        https://arxiv.org/abs/2010.02502

        Denoising Diffusion Implicit Models (DDIM) sampling; according to the paper
        """

        if mask is not None:
            assert orig_x is not None
            assert orig_x.shape == mask.shape

        if x_t is None:
            if orig_x is None:
                x_t = np.random.normal(size = (sample_num, *image_size))
            else:
                x_t = np.random.normal(size = orig_x.shape)

        x_ts = []
        for t in tqdm(reversed(range(1, self.timesteps)[:perform_steps]), desc = 'ddim denoisinig samples', total = perform_steps):
            noise = np.random.normal(size = x_t.shape) if t > 1 else 0
            epsilon = cp.asnumpy(self.model.forward(x_t, np.array([t]) / self.timesteps, training = False)).reshape(x_t.shape)

            x0_t = (x_t - epsilon * np.sqrt(1 - self.alphas_cumprod[t])) / np.sqrt(self.alphas_cumprod[t])

            sigma = eta * np.sqrt((1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t]) * (1 - self.alphas_cumprod[t] / self.alphas_cumprod[t - 1]))
            c = np.sqrt((1 - self.alphas_cumprod[t - 1]) - sigma ** 2)

            x_t = np.sqrt(self.alphas_cumprod[t - 1]) * x0_t - c * epsilon + sigma * noise

            if mask is not None:
                orig_x_noise = np.random.normal(size = orig_x.shape)
                
                orig_x_t = self.sqrt_alphas_cumprod[t] * orig_x + self.sqrt_one_minus_alphas_cumprod[t] * orig_x_noise
                x_t = orig_x_t * mask + x_t * (1 - mask)
           
            if t % states_step_size == 0:
                x_ts.append(x_t)

        return x_t, x_ts



    def get_images_set(self, x_num: int, y_num: int, margin: int, images: np.float32, image_size: Tuple[int, int, int]):

        def denormalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
       

        channels, H_size, W_size = image_size

        images_array = np.full((y_num * (margin + H_size), x_num * (margin + W_size), channels), 255, dtype = np.uint8)
        num = 0
        for i in range(y_num):
            for j in range(x_num):
                y = i * (margin + H_size)
                x = j * (margin + W_size)

                images_array[y :y + H_size, x: x + W_size] = denormalize(images[num].transpose(1, 2, 0))

                num += 1

        images_array = images_array[: (y_num - 1) * (H_size + margin) + H_size, : (x_num - 1) * (W_size + margin) + W_size]

        if channels == 1:
            return Image.fromarray(images_array.squeeze(axis = 2)).convert("L")
        else:
            return Image.fromarray(images_array)


    def train(self, dataset, epochs, batch_size, save_every_epochs, image_path, save_path, image_size):
        channels, H_size, W_size = image_size

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
                self.save(f"{save_path}")

                margin = 10
                x_num, y_num = 5, 5

                samples, samples_in_time = self.ddpm_denoise_sample(x_num * y_num, (channels, H_size, W_size), step_size = 10)
                images_grid = self.get_images_set(x_num, y_num, margin, samples, (channels, H_size, W_size))
                images_grid.save(f"{image_path}/np_ddpm_{epoch + 1}.png")

                images_grid_in_time = []
                for sample in samples_in_time:
                    images_grid_in_time.append(self.get_images_set(x_num, y_num, margin, sample, (channels, H_size, W_size)))

                images_grid_in_time[0].save(f"{image_path}/np_ddpm_in_time_{epoch + 1}.gif", save_all = True, append_images = images_grid_in_time[1:], duration = 50, loop = 0)
                
                
               
                
            loss_history.append(epoch_loss)

        return loss_history





