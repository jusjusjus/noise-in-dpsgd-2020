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
parser.add_argument('--critic-capacity', type=int, default=None)
parser.add_argument('--lr-per-example', type=float, default=3.125e-6)
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
parser.add_argument('--eval-every', type=int, default=1000,
                    help="evaluate every x steps")
parser.add_argument('--seed', type=int, default=42 * 42, help="pytorch seed")
parser.add_argument('--continue-from', type=str, default=None,
                    help="continue training from a checkpoint")
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--momentum', type=float, default=0.5)

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
from ganlib import critic
from ganlib import generator

cuda = torch.cuda.is_available()


def log(logger, info, tag, network, global_step):
    if global_step % opt.print_every == 0:
        logger.add_scalars(tag, info, global_step)
        s = f"[Step {global_step}] "
        s += ' '.join(f"{tag}/{k} = {v:.3g}" for k, v in info.items())
        print(s)

    if global_step % opt.eval_every == 0:
        ckpt = logger.add_checkpoint(network, global_step)
        scripts.generate(logger=logger, params=ckpt, step=global_step,
                         dataset=opt.dataset, version=opt.version)
        if exists(join("cache", opt.dataset + "_classifier.ckpt")):
            scripts.inception(logger=logger, params=ckpt, step=global_step,
                              dataset=opt.dataset, version=opt.version)
        network.train()

# Set default parameters

delta = 1e-5

# Process parameters

learning_rate = opt.batch_size * opt.lr_per_example
logdir = join('cache', 'logs')
logdir = join(logdir, f"dset_{opt.dataset}-cap_{opt.capacity}-steps_{opt.critic_steps}-bsz_{opt.batch_size}-lr_{opt.lr_per_example}")
logdir += f"-opt_{opt.optimizer}-mom_{opt.momentum}"
if not opt.nodp:
    logdir += f"-sig_{opt.sigma}-clip_{opt.grad_clip}"
if opt.version:
    logdir += f"-{opt.version}"
if opt.critic_capacity:
    logdir += f"-ccap_{opt.critic_capacity}"

# Initialize generator and critic.  We wrap generator and critic into
# `GenerativeAdversarialNet` and provide methods `cuda` and `state_dict`

choice = opt.version or opt.dataset
generator = generator.choices[choice](opt.capacity)
critic_capacity = opt.critic_capacity or opt.capacity
critic = critic.choices[choice](critic_capacity)

gan = GenerativeAdversarialNet(generator, critic)
gan = gan.cuda() if cuda else gan

dset = dataset.choices[opt.dataset]()
dataloader = DataLoader(dset, batch_size=opt.batch_size,
                        shuffle=True, num_workers=4)

# Initialize optimization.  We make optimizers part of the network and provide
# methods `.zero_grad` and `.step` to simplify the code.

optimizer = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}[opt.optimizer]

optimizer_args = {
    'adam': {'betas': (opt.momentum, 0.9)},
    'sgd': {'nesterov': True, 'momentum': opt.momentum}
}[opt.optimizer]

optimizer_args['lr'] = learning_rate
generator.init_optimizer(optimizer, **optimizer_args)
critic.init_optimizer(optimizer, **optimizer_args)

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
