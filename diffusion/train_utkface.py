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

def prepare_data(path):
    
    from PIL import Image
    import numpy as np
    import random
    
    images = os.listdir(path)
    random.shuffle(images)
    
    training_inputs = []
    for image in tqdm(images, desc = 'preparing data'):
        image = Image.open(path + "/" + image)
        image = image.resize((32, 32))
        image = np.asarray(image)
        image = image.transpose(2, 0, 1)
        image = image / 127.5 - 1
        training_inputs.append(image)

    return np.array(training_inputs)

if not os.path.exists("dataset/utkface/UTKFace.npy"):
    training_inputs = prepare_data("dataset/utkface/UTKFace")
    np.save("dataset/utkface/UTKFace.npy", training_inputs)
else:
    training_inputs = np.load("dataset/utkface/UTKFace.npy")





image_size = (3, 32, 32)

diffusion = Diffusion(model = SimpleUNet(image_channels = 3, image_size = 32, down_channels = (128, 256, 512, 1024), up_channels = (1024, 512, 256, 128))
                    , timesteps = 300, beta_start = 0.0001, beta_end = 0.02, criterion = MSE(), optimizer = Adam(alpha = 2e-4)) #alpha = 2e-4

if not os.path.exists("diffusion/saved models/utkface_model"):
    diffusion.train(training_inputs, epochs = 30, batch_size = 5, save_every_epochs = 1, image_path = f"images/utkface", save_path = f"diffusion/saved models/utkface_model", image_size = image_size)
else: 
    diffusion.load("diffusion/saved models/utkface_model")



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