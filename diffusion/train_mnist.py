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
from diffusion.architectures.unet import SimpleUNet



training_data = open('dataset/mnist/mnist_train.csv','r').readlines()
test_data = open('dataset/mnist/mnist_test.csv','r').readlines()





image_size = (1, 28, 28)



def prepare_data(data):
    inputs, targets = [], []

    for raw_line in tqdm(data, desc = 'preparing data'):

        line = raw_line.split(',')
    
        inputs.append(np.asfarray(line[1:]).reshape(1, 28, 28) / 127.5 - 1)#/ 255 [0; 1]  #/ 127.5-1 [-1; 1]
        targets.append(int(line[0]))

    return inputs, targets

if not os.path.exists("dataset/mnist/mnist_train.npy"):
    training_inputs, training_targets = prepare_data(training_data)
    np.save("dataset/mnist/mnist_train.npy", training_inputs)
    np.save("dataset/mnist/mnist_train_targets.npy", training_targets)
else:
    training_inputs = np.load("dataset/mnist/mnist_train.npy")
    training_targets = np.load("dataset/mnist/mnist_train_targets.npy")



diffusion = Diffusion(model = SimpleUNet(image_channels = 1, image_size = 28, down_channels = (32, 64, 128), up_channels = (128, 64, 32)), timesteps = 300, beta_start = 0.0001, beta_end = 0.02, criterion = MSE(), optimizer = Adam(alpha = 2e-4))
diffusion.train(training_inputs, epochs = 30, batch_size = 10, save_every_epochs = 1, image_path = f"images/mnist", save_path = f"diffusion/saved models/mnist_model", image_size = image_size)