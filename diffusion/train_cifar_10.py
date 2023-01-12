import numpy as np


try: 
    import cupy as cp
    is_cupy_available = True
except:
    is_cupy_available = False



import sys
import os
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


from tqdm import tqdm
from diffusion.diffusion import Diffusion
from diffusion.optimizers import Adam
from diffusion.losses import MSE
from diffusion.architectures import SimpleUNet

import matplotlib.pyplot as plt



image_size = (3, 32, 32)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

training_data = [unpickle(f'dataset/cifar-10/cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]

prepared_data = lambda data: (np.asfarray(data[b'data']).reshape(-1, *image_size) / 127.5 - 1, np.asfarray(data[b'labels']))

training_inputs = [prepared_data(data)[0] for data in training_data]
training_inputs = np.concatenate(np.asfarray(training_inputs), axis = 0)





diffusion = Diffusion(
    model = SimpleUNet(
        image_channels = 3, 
        image_size = image_size[1], 
        down_channels = (128, 256, 512, 1024), 
        up_channels = (1024, 512, 256, 128)
        ),
    timesteps = 300, 
    beta_start = 0.0001, 
    beta_end = 0.02, 
    criterion = MSE(), 
    optimizer = Adam(alpha = 2e-4)
    )

if not os.path.exists("diffusion/saved models/cifar10_model"):
    diffusion.train(training_inputs, epochs = 30, batch_size = 5, save_every_epochs = 1, image_path = f"images/cifar-10", save_path = f"diffusion/saved models/cifar10_model", image_size = image_size)
else:
    diffusion.load("diffusion/saved models/cifar10_model")



# Image generating example
x_num, y_num = 5, 5
n = x_num * y_num

generated_images, generated_images_in_time = diffusion.ddpm_denoise_sample(sample_num = n, image_size = image_size, states_step_size = 10)

generated_images_grid = diffusion.get_images_set(x_num, y_num, images = generated_images, margin = 10, image_size = image_size)


plt.imshow(generated_images_grid, cmap = 'gray')
plt.axis('off')
plt.title('Generated images')
plt.show()


# Image inpainting example
random_images = training_inputs[np.random.randint(0, len(training_inputs), n)].reshape(n, *image_size)

masks = np.ones(random_images.shape)
masks[:, :, :16, :] = 0

images_grid = diffusion.get_images_set(x_num, y_num, images = random_images, margin = 10, image_size = image_size)

masked_images_grid = diffusion.get_images_set(x_num, y_num, images = random_images * masks, margin = 10, image_size = image_size)
reconstructed_images = diffusion.ddpm_denoise_sample(orig_x = random_images, mask = masks)[0]
reconstructed_images_grid = diffusion.get_images_set(x_num, y_num, images = reconstructed_images, margin = 10, image_size = image_size)

plt.subplot(1, 3, 1)
plt.imshow(images_grid, cmap = 'gray')
plt.axis('off')
plt.title('Original images')

plt.subplot(1, 3, 2)
plt.imshow(masked_images_grid, cmap = 'gray')
plt.axis('off')
plt.title('Masked images')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_images_grid, cmap = 'gray')
plt.axis('off')
plt.title('Reconstructed images')

plt.show()