
from os import makedirs
from os.path import join

import numpy as np
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize


class MNIST(datasets.MNIST):

    def __init__(self, *args, transform=None, labels=False, **kwargs):
        data_dir = join('cache', 'data')
        makedirs(data_dir, exist_ok=True)
        super().__init__(data_dir, *args, download=True, **kwargs)
        self.labels = labels
        self.transform = transform or \
            Compose([ToTensor(), Normalize([0.5], [0.5], inplace=True)])

    def __getitem__(self, i):
        img, labels = super().__getitem__(i)
        return (img, labels) if self.labels else img


class CIFAR10(datasets.CIFAR10):

    def __init__(self, *args, transform=None, labels=False, **kwargs):
        data_dir = join('cache', 'data')
        makedirs(data_dir, exist_ok=True)
        super().__init__(data_dir, *args, download=True, **kwargs)
        self.labels = labels
        self.transform = transform or \
            Compose([ToTensor(), Normalize([0.5], [0.5], inplace=True)])

    def __getitem__(self, i):
        img, labels = super().__getitem__(i)
        return (img, labels) if self.labels else img


choices = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}
