import numpy as np

# https://huggingface.co/blog/annotated-diffusion
# https://arxiv.org/pdf/2102.09672.pdf


def linear_schedule(start, end, timesteps):
    return np.linspace(start, end, timesteps)

def quadratic_schedule(start, end, timesteps):
    return np.linspace(start ** 0.5, end ** 0.5, timesteps) ** 2

def exponential_schedule(start, end, timesteps):
    return np.geomspace(start, end, timesteps)

def logarithmic_schedule(start, end, timesteps):
    return np.logspace(np.log10(start), np.log10(end), timesteps)

def sigmoid_schedule(start, end, timesteps):

    betas = np.linspace(-6, 6, timesteps)
    betas_sigmoid = 1 / (1 + np.exp(-betas))

    return betas_sigmoid * (end - start) + start

def cosine_schedule(start, end, timesteps):
    s = 8e-3
    x = np.linspace(0, timesteps, timesteps + 1)

    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return np.clip(betas, 0.01, 0.99)


def get_schedule(schedule, start, end, timesteps):
    if schedule == 'linear':
        return linear_schedule(start, end, timesteps)
    elif schedule == 'quadratic':
        return quadratic_schedule(start, end, timesteps)
    elif schedule == 'exponential':
        return exponential_schedule(start, end, timesteps)
    elif schedule == 'logarithmic':
        return logarithmic_schedule(start, end, timesteps)
    elif schedule == 'sigmoid':
        return sigmoid_schedule(start, end, timesteps)
    elif schedule == 'cosine':
        return cosine_schedule(start, end, timesteps)
    else:
        raise ValueError(f'Unknown schedule: {schedule}')



