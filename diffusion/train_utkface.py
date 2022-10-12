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
# diffusion.load("diffusion/saved models/utkface_model")
diffusion.train(training_inputs, epochs = 30, batch_size = 5, save_every_epochs = 1, image_path = f"images/utkface", save_path = f"diffusion/saved models/utkface_model", image_size = image_size)