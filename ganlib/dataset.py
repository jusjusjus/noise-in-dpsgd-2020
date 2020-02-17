
from os import makedirs
from os.path import join
from torchvision import datasets


class Dataset(datasets.MNIST):

    def __init__(self, *args, labels=False, **kwargs):
        data_dir = join('cache', 'data')
        makedirs(data_dir, exist_ok=True)
        super().__init__(data_dir, *args, download=True, **kwargs)
        self.labels = labels

    def __getitem__(self, i):
        img, labels = super().__getitem__(i)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = np.array(img)[None, ...]
        img = img.astype(np.float32) / 255.0
        img = 2 * img - 1
        return (img, labels) if self.labels else img
