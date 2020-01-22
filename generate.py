#!/usr/bin/env python

from sys import path
path.insert(0, '.')
from os.path import splitext, basename
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate sample images from parameter checkpoint")
parser.add_argument('params', type=str, help="model parameters")
parser.add_argument('-o', type=str, default=None, help="output file name (.png/.jpg)")
parser.add_argument('--num-rows', type=int, default=4, help="number of rows to plot")
parser.add_argument('--cpu', action='store_true', help="use cpu only")
opt = parser.parse_args()

import torch
import numpy as np
import matplotlib.pyplot as plt

from ganlib.generator import MNISTGenerator, join_image_batch

def plot_images(images):
    """plot compound image"""
    labels = np.arange(1, opt.num_rows ** 2 + 1, dtype=int)
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111, xticks=[], yticks=[])
    plt.title(f"'{opt.params}'")

    plt.imshow(images, cmap='Greys_r')
    dx = 1/opt.num_rows - 0.01
    dy = 1/opt.num_rows - 0.01
    for i in range(opt.num_rows):
        x = 0.04 + dx * i
        for j in range(opt.num_rows):
            y = 0.93-dy * j
            t = labels[i + opt.num_rows * j]
            if isinstance(t, np.int64):
                color = 'w'
            else:
                color = 'g' if t is 'Real' else 'r'
            plt.figtext(x, y, t, fontsize=15, color=color)
    plt.tight_layout()


# process command-line parameters
output_filename = opt.o or splitext(basename(opt.params))[0] + '.png'
cuda = torch.cuda.is_available() and not opt.cpu
num_images = opt.num_rows ** 2
# load generator model
ckpt = torch.load(opt.params, map_location='cpu' if opt.cpu else None)
generator = MNISTGenerator.from_state_dict(ckpt['state_dict']['generator'])
generator = generator.cuda() if cuda else generator
num_fake = num_images
# generate images
images = generator.compute_sample_images(num_fake, cuda=cuda)
# optionally mix and join images
perm = np.random.permutation(images.shape[0]) if opt.mix \
       else np.arange(images.shape[0])
images = join_image_batch(images[perm], opt.num_rows)
# plot images (with solution if mixed)
plot_images(images)
plt.savefig(output_filename)
