import requests
from tqdm import tqdm

# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
url = 'https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing&resourcekey=0-dabpv_3J0C0cditpiAfhAw'



def download(id, path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    resp = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(resp)

    if token:
        params = { 'id' : id, 'confirm' : token }
        resp = session.get(URL, params = params, stream = True)

    save_resp_content(resp, path)    

def get_confirm_token(resp):
    for key, value in resp.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_resp_content(resp, path):
    CHUNK_SIZE = 32768
   
    total = int(resp.headers.get('content-length', 0))

    with open(path, "wb") as file, tqdm(
            desc = f'downloading file to {path = }',
            total = total,
            unit = 'iB',
            unit_scale = True,
            unit_divisor = 1024,
    ) as bar:
        for chunk in resp.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                size = file.write(chunk)
                bar.update(size)



download('0BxYys69jI14kYVM3aVhKS1VhRUk', 'dataset/utkface/utkface.tar.gz')

def extract(path):
    import tarfile
    # tar = tarfile.open(path, 'r:gz')
    # tar.extractall('dataset/utkface')
    # tar.close()
    extract_path = 'dataset/utkface'
    with tarfile.open(path, 'r:gz') as tar:
        for member in tqdm(iterable = tar.getmembers(), total = len(tar.getmembers()), desc = f'extracting file to {extract_path = }'):
            tar.extract(member = member, path = extract_path)

extract('dataset/utkface/utkface.tar.gz')