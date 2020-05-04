#! /usr/bin/env python

from os import makedirs
from os.path import join, dirname, exists
from sys import path
path.insert(0, '.')
from time import time
from argparse import ArgumentParser

parser = ArgumentParser(description="Train MNIST generator with DP-WGAN-GP")
parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--capacity', type=int, default=64,
                    help="number-of-filters factor in GAN")
parser.add_argument('--critic-steps', type=int, default=4,
                    help="number of critic steps per generator step")
parser.add_argument('--nodp', action='store_true',
                    help="Train without differential privacy")
parser.add_argument('--sigma', type=float, default=0.5,
                    help="Ratio of noise std. dev. and mechanism L2-norm")
parser.add_argument('--grad-clip', type=float, default=1.0,
                    help="L2-norm clipping parameter")
parser.add_argument('--epochs', type=int, default=100,
                    help="number of epochs to train the GAN")
parser.add_argument('--batch-size', type=int, default=128,
                    help="set mini-batch size for training")
parser.add_argument('--print-every', type=int, default=25,
                    help="print every x steps")
parser.add_argument('--eval-every', type=int, default=500,
                    help="evaluate every x steps")
parser.add_argument('--seed', type=int, default=42 * 42, help="pytorch seed")
parser.add_argument('--continue-from', type=str, default=None,
                    help="continue training from a checkpoint")
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

torch.manual_seed(opt.seed)

from ganlib import scripts
from ganlib.gan import GenerativeAdversarialNet
from ganlib.logger import Logger
from ganlib import dataset
from ganlib.privacy import compute_renyi_privacy
from ganlib.trainer import DPWGANGPTrainer, WGANGPTrainer
from ganlib import generator

cuda = torch.cuda.is_available()


class Critic(generator.Optimizable):

    def __init__(self, colors, capacity):
        super().__init__()
        C = capacity
        kw = {'padding': 2, 'stride': 2, 'kernel_size': 5}
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(colors, 1 * C, **kw)
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
    if global_step % opt.print_every == 0:
        logger.add_scalars(tag, info, global_step)
        s = f"[Step {global_step}] "
        s += ' '.join(f"{tag}/{k} = {v:.3g}" for k, v in info.items())
        print(s)

    if global_step % opt.eval_every == 0:
        ckpt = logger.add_checkpoint(network, global_step)
        scripts.generate(logger=logger, params=ckpt,
                         step=global_step, dataset=opt.dataset)
        if exists(join("cache", opt.dataset + "_classifier.ckpt")):
            scripts.inception(logger=logger, params=ckpt,
                              step=global_step, dataset=opt.dataset)
        network.train()

# Set default parameters

delta = 1e-5
lr_per_example = 3.125e-6

# Process parameters

learning_rate = opt.batch_size * lr_per_example
logdir = join('cache', 'logs')
logdir = join(logdir, f"dset_{opt.dataset}-cap_{opt.capacity}-steps_{opt.critic_steps}-batchsize_{opt.batch_size}")
if not opt.nodp:
    logdir += f"-sig_{opt.sigma}-clip_{opt.grad_clip}"

# Initialize generator and critic.  We wrap generator and critic into
# `GenerativeAdversarialNet` and provide methods `cuda` and `state_dict`

generator = generator.choices[opt.dataset](opt.capacity)
critic = Critic(generator.colors, opt.capacity)
gan = GenerativeAdversarialNet(generator, critic)
gan = gan.cuda() if cuda else gan

dset = dataset.choices[opt.dataset]()
dataloader = DataLoader(dset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=4)

# Initialize optimization.  We make optimizers part of the network and provide
# methods `.zero_grad` and `.step` to simplify the code.

generator.init_optimizer(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.9))
critic.init_optimizer(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.9))

if opt.nodp:
    trainer = WGANGPTrainer(opt.batch_size)
else:
    print("training with differential privacy")
    print(f"> delta = {delta}")
    print(f"> sigma = {opt.sigma}")
    print(f"> L2-clip = {opt.grad_clip}")
    trainer = DPWGANGPTrainer(opt.sigma, opt.grad_clip, batch_size=opt.batch_size)

print(f"> learning rate = {learning_rate} (at {opt.batch_size}-minibatches)")

if opt.continue_from:
    ckpt = torch.load(opt.continue_from)
    global_step = ckpt['global_step'] + 1
    ckpt = ckpt['state_dict']
    gan.generator.load_state_dict(ckpt['generator']['params'])
    gan.critic.load_state_dict(ckpt['critic']['params'])
    print(f"Continuing from step {global_step} ..")
else:
    global_step = 0

logs = {}
logger = Logger(logdir=logdir)
for epoch in range(opt.epochs):
    t0 = time()
    for imgs in dataloader:
        if global_step % opt.critic_steps == 0:
            genlog = trainer.generator_step(gan)
            logs.update(**genlog)

        critlog = trainer.critic_step(gan, imgs)
        t1 = time()
        logs.update(**critlog)
        logs['sampling_rate'] = imgs.shape[0] / (t1 - t0)
        if not opt.nodp and global_step % opt.print_every == 0:
            spent = compute_renyi_privacy(
                len(dset), opt.batch_size, global_step + 1, opt.sigma, delta)
            logs['epsilon'] = spent.eps

        log(logger, logs, 'train', gan, global_step)
        global_step += 1
        t0 = t1
