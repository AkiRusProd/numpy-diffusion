import requests
from tqdm import tqdm

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def download(url, path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(path, 'wb') as file, tqdm(
            desc = f'downloading file to {path = }',
            total = total,
            unit = 'iB',
            unit_scale = True,
            unit_divisor = 1024,
    ) as bar:
        for data in resp.iter_content(chunk_size = 1024):
            size = file.write(data)
            bar.update(size)
    

download(url, 'dataset/cifar-10/cifar-10-python.tar.gz')


def extract(path):
    import tarfile
    tar = tarfile.open(path, 'r:gz')
    tar.extractall('dataset/cifar-10')
    tar.close()

extract('dataset/cifar-10/cifar-10-python.tar.gz')