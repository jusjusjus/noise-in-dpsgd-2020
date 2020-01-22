#! /usr/bin/env python

from os import makedirs
from os.path import join, dirname
from sys import path
path.insert(0, '.')
from argparse import ArgumentParser

parser = ArgumentParser(description="Train MNIST generator with DP-WGAN-GP")
parser.add_argument('--nodp', action='store_true',
                    help="Train without differential privacy")
parser.add_argument('--sigma', type=float, default=1.0,
                    help="Ratio of noise std. dev. and mechanism L2-norm")
parser.add_argument('--grad-clip', type=float, default=4.0,
                    help="L2-norm clipping parameter")
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

torch.manual_seed(42 * 42)

from ganlib import scripts
from ganlib.gan import GenerativeAdversarialNet
from ganlib.logger import Logger
from ganlib.privacy import compute_renyi_privacy
from ganlib.trainer import DPWGANGPTrainer, WGANGPTrainer
from ganlib.generator import MNISTGenerator, Optimizable

cuda = torch.cuda.is_available()

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


class MNISTCritic(Optimizable):

    def __init__(self):
        super().__init__()
        kw = {'padding': 2, 'stride': 2, 'kernel_size': 5}
        C = capacity = 64
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(1,     1 * C, **kw)
        self.conv2 = nn.Conv2d(1 * C, 2 * C, **kw)
        self.conv3 = nn.Conv2d(2 * C, 4 * C, **kw)
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(4 * 4 * 4 * C, 1)

    def forward(self, images):
        images = self.activation(self.conv1(images))
        images = self.activation(self.conv2(images))
        images = self.activation(self.conv3(images))
        images = self.flatten(images)
        images = self.projection(images)
        criticism =  images.squeeze(-1)
        return criticism


def log(logger, info, tag, network, global_step):
    """print every 25, and plot every 250 steps network output"""
    if global_step % 25 == 0:
        logger.add_scalars(tag, info, global_step)
        s = f"[Step {global_step}] "
        s += ' '.join(f"{tag}/{k} = {v:.3g}" for k, v in info.items())
        print(s)

    if (global_step + 1) % 250 == 0:
        ckpt = logger.add_checkpoint(network, global_step)
        scripts.generate(logger=logger, params=ckpt,
                         step=global_step)
        network.train()

# Set optional parameters

sigma = opt.sigma
clip = opt.grad_clip

# Set default parameters

batch_size = 128
lr_per_example = 3.125e-6
delta = 1e-5
critic_steps = 4

# Process parameters

logdir = join('cache', 'logs')
logdir = join(logdir, 'nodp' if opt.nodp else f"sigma_{sigma}-clip_{clip}")
learning_rate = batch_size * lr_per_example

# Initialize generator and critic.  We wrap generator and critic into
# `GenerativeAdversarialNet` and provide methods `cuda` and `state_dict`

generator = MNISTGenerator()
critic = MNISTCritic()
gan = GenerativeAdversarialNet(generator, critic)
gan = gan.cuda() if cuda else gan

dset = Dataset()
dataloader = DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

# Initialize optimization.  We make optimizers part of the network and provide
# methods `.zero_grad` and `.step` to simplify the code.

generator.init_optimizer(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.9))
critic.init_optimizer(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.9))

if opt.nodp:
    trainer = WGANGPTrainer(batch_size=batch_size)
else:
    print("training with differential privacy")
    print(f"> delta = {delta}")
    print(f"> sigma = {sigma}")
    print(f"> L2-clip = {clip}")
    trainer = DPWGANGPTrainer(sigma=sigma, l2_clip=clip, batch_size=batch_size)

print(f"> learning rate = {learning_rate} (at {batch_size}-minibatches)")

logs = {}
global_step = 0
logger = Logger(logdir=logdir)
for epoch in range(100):
    for imgs in dataloader:

        if (global_step + 1) % critic_steps == 0:
            genlog = trainer.generator_step(gan)
            logs.update(**genlog)

        critlog = trainer.critic_step(gan, imgs)
        logs.update(**critlog)
        
        if not opt.nodp:
            spent = compute_renyi_privacy(
                len(dset), batch_size, global_step + 1, sigma, delta)
            logs['epsilon'] = spent.eps

        log(logger, logs, 'train', gan, global_step)

        global_step += 1
