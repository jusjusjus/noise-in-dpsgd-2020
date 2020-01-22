#!/usr/bin/env python

from os import makedirs
from os.path import join, dirname, exists
from argparse import ArgumentParser

parser = ArgumentParser(description="Compute inception score")
parser.add_argument("-p", "--params", type=str, default='MNIST',
                    help="'MNIST' or path to model checkpoint")
parser.add_argument('--cpu', action='store_true', help="use cpu only")
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets

from ganlib.generator import MNISTGenerator
from ganlib.classifier import Classifier

class Dataset(datasets.MNIST):

    def __init__(self, *args, **kwargs):
        data_dir = join('cache', 'data')
        makedirs(data_dir, exist_ok=True)
        super().__init__(data_dir, *args, download=True, **kwargs)

    def __getitem__(self, i):
        img, _ = super().__getitem__(i)
        img = img.resize((28, 28), Image.ANTIALIAS)
        img = np.array(img)[None, ...]
        img = img.astype(np.float32) / 255.0
        img = 2 * img - 1
        return img


splits = 10
epochs = 3
batch_size = 128
lr_per_example = 1e-4
eval_every = 1000
adapt_every = 100
weight_decay = 0.001
num_examples = 10000
best_model_filename = join("cache", "mnist_classifier.ckpt")
assert exists(best_model_filename), f"""
No MNIST classifier found in '{best_model_filename}'.  Please build one using:
`python build-classifier.py`"""

loc = 'cpu' if opt.cpu else None
clf = Classifier.from_checkpoint(best_model_filename, map_location=loc).eval()
device = next(clf.parameters()).device

if opt.params == 'MNIST':
    dset =  Dataset(train=False)
    assert num_examples == len(dset)
    dataloader = DataLoader(dset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
else:
    num_batches = int(10000 // batch_size)
    ckpt = torch.load(opt.params, map_location=loc)
    ckpt = ckpt['state_dict']['generator']
    generator = MNISTGenerator.from_state_dict(ckpt)
    dataloader = generator.dataloader(batch_size, num_batches)


# Compute the probabilities for all examples in the MNIST test set

truths, probs = [], []
with torch.no_grad():
    for examples in dataloader:
        examples = examples.to(device)
        probs.append(clf.probabilities(examples))

    probs = torch.cat(probs).cpu().numpy()

# Compute inception score with bootstrapped standard deviation

scores = []
for i in range(splits):
    P_yx = probs[i::splits]
    P_y = P_yx.mean(0, keepdims=True)
    P_yx = np.maximum(P_yx, 1e-12)
    P_y = np.maximum(P_y, 1e-12)
    KL_div = np.sum(P_yx * (np.log(P_yx) - np.log(P_y)), 1)
    s_G = np.exp(np.mean(KL_div))
    scores.append(s_G)

mean_score, std_score = np.mean(scores), np.std(scores)
print(f"Inception score: {mean_score:.3f} +/- {std_score:.3f}")
