#!/usr/bin/env python

from os import makedirs
from os.path import join, dirname, exists
from argparse import ArgumentParser

parser = ArgumentParser(description="Compute inception score")
parser.add_argument("-p", "--params", type=str, default='dataset',
                    help="'MNIST' or path to model checkpoint")
parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--cpu', action='store_true', help="use cpu only")
parser.add_argument('--splits', type=int, default=10,
                    help="boot-strapping splits")
parser.add_argument('--quiet', action='store_true',
                    help="only output inception score")
parser.add_argument('--train-set', action='store_true', help="Use the MNIST "
                    "training data set (only valid for '--params MNIST').")
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets

from ganlib import dataset
from ganlib import generator
from ganlib import classifier


epochs = 3
batch_size = 128
lr_per_example = 1e-4
eval_every = 1000
adapt_every = 100
weight_decay = 0.001
num_examples = 10000
best_model_filename = join("cache", opt.dataset + "_classifier.ckpt")
assert exists(best_model_filename), f"""
No MNIST classifier found in '{best_model_filename}'.  Please build one using:
`python build-classifier.py`"""

loc = 'cpu' if opt.cpu else None
clf = classifier.choices[opt.dataset].from_checkpoint(best_model_filename, map_location=loc)
clf = clf.eval()
device = next(clf.parameters()).device

if opt.params == 'dataset':
    dset =  dataset.choices[opt.dataset](train=opt.train_set)
    dataloader = DataLoader(dset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
else:
    num_batches = int(10000 // batch_size)
    ckpt = torch.load(opt.params, map_location=loc)
    ckpt = ckpt['state_dict']['generator']
    generator = generator.choices[opt.dataset].from_state_dict(ckpt)
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
for i in range(opt.splits):
    P_yx = probs[i::opt.splits]
    P_y = P_yx.mean(0, keepdims=True)
    P_yx = np.maximum(P_yx, 1e-12)
    P_y = np.maximum(P_y, 1e-12)
    KL_div = np.sum(P_yx * (np.log(P_yx) - np.log(P_y)), 1)
    s_G = np.exp(np.mean(KL_div))
    scores.append(s_G)

mean_score = np.mean(scores)
if opt.quiet:
    print(mean_score)
else:
    std_score = np.std(scores)
    print(f"Inception score: {mean_score:.3f} +/- {std_score:.3f}")
