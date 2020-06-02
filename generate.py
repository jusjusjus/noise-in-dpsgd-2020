#!/usr/bin/env python

from sys import path
path.insert(0, '.')
from os.path import splitext, basename
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate sample images from parameter checkpoint")
parser.add_argument('params', type=str, help="model parameters")
parser.add_argument('--dataset', type=str, choices=["mnist", "cifar10"], default='mnist')
parser.add_argument('-o', type=str, default=None, help="output file name (.png/.jpg)")
parser.add_argument('--num-rows', type=int, default=4, help="number of rows to plot")
parser.add_argument('--cpu', action='store_true', help="use cpu only")
parser.add_argument('--version', type=str, default=None, help="use cpu only")
opt = parser.parse_args()

import torch
import matplotlib.pyplot as plt

from ganlib import generator

def plot_images(images):
    """plot compound image"""
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111, xticks=[], yticks=[])
    plt.title(f"'{opt.params}'")
    if len(images.shape) == 2:
        plt.imshow(images, cmap='Greys_r')
    else:
        images = 0.5 * (images + 1.0)
        plt.imshow(images.transpose((1, 2, 0)))

    dx = 1/opt.num_rows - 0.01
    for i in range(opt.num_rows):
        x = 0.04 + dx * i
        for j in range(opt.num_rows):
            y = 0.93 - dx * j
            t = str(1 + j * opt.num_rows + i)
            plt.figtext(x, y, t, fontsize=15, color='w')

    plt.tight_layout()


# process command-line options
output_filename = opt.o or splitext(basename(opt.params))[0] + '.png'
cuda = torch.cuda.is_available() and not opt.cpu
num_images = opt.num_rows ** 2
# load generator and generate images
ckpt = torch.load(opt.params, map_location='cpu' if opt.cpu else None)
choice = opt.version or opt.dataset
G = generator.choices[choice].from_state_dict(
        ckpt['state_dict']['generator'])

G = G.cuda() if cuda else G
images = G.compute_sample_images(num_images)
# join and plot images
images = generator.join_image_batch(images, opt.num_rows)
plot_images(images)
print(f"saving sample images to '{output_filename}'")
plt.savefig(output_filename)
