import numpy as np


try: 
    import cupy as cp
    is_cupy_available = True
except:
    is_cupy_available = False



import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent)


from tqdm import tqdm
from diffusion.diffusion import Diffusion
from diffusion.optimizers import Adam
from diffusion.losses import MSE
from diffusion.architectures import SimpleUNet





def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

training_data = [unpickle(f'dataset/cifar-10/cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]

prepared_data = lambda data: (np.asfarray(data[b'data']).reshape(-1, 3, 32, 32) / 127.5 - 1, np.asfarray(data[b'labels']))

training_inputs = [prepared_data(data)[0] for data in training_data]
training_inputs = np.concatenate(np.asfarray(training_inputs), axis = 0)




image_size = (3, 32, 32)

diffusion = Diffusion(model = SimpleUNet(image_channels = 3, image_size = 32, down_channels = (128, 256, 512, 1024), up_channels = (1024, 512, 256, 128))#, down_channels = (64, 128, 256, 512, 1024), up_channels = (1024, 512, 256, 128, 64)
                    , timesteps = 300, beta_start = 0.0001, beta_end = 0.02, criterion = MSE(), optimizer = Adam(alpha = 2e-4)) #alpha = 2e-4
# diffusion.load("diffusion/saved models/np_ddpm")
diffusion.train(training_inputs, epochs = 30, batch_size = 5, save_every_epochs = 1, image_path = f"images/cifar-10", save_path = f"diffusion/saved models/cifar10_model", image_size = image_size)