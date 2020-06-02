#!/usr/bin/env python

from os import makedirs
from os.path import join, dirname
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'],
                    default='mnist')
parser.add_argument("--best-model-filename", type=str, default=None)
parser.add_argument("--nesterov", action="store_true")
<<<<<<< HEAD
# Cifar10 specific parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
=======
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, ToTensor, Normalize,
<<<<<<< HEAD
                                    RandomAffine, RandomHorizontalFlip,
                                    RandomCrop)
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR

from ganlib import dataset
from ganlib import classifier
from ganlib.cutout import Cutout
=======
                                    RandomAffine, RandomHorizontalFlip)

from ganlib import dataset
from ganlib import classifier
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba

torch.manual_seed(42 * 42)

def evaluate(model, dataloader):
    model.eval()
    acc, examples_seen = 0.0, 0
    with torch.no_grad():
        for i, (examples, labels) in enumerate(dataloader):
            batch_size = labels.shape[0]
            examples = examples.to(device)
            labels = labels.to(device)
            logits = model(examples)
            y_pred = torch.argmax(logits, dim=-1)

            acc_i = (y_pred == labels).sum().item()
            acc = (examples_seen * acc + acc_i) / (examples_seen + batch_size)
            examples_seen += batch_size
        
    model.train()
    return 100 * acc

<<<<<<< HEAD
=======

def schedule(lr, loss):
    return lr if loss > 1.0 else loss * lr


if 'mnist' == opt.dataset:
    epochs = 20
    batch_size = 128
    lr_per_example = 1e-4
elif 'cifar10' == opt.dataset:
    epochs = 200
    batch_size = 512
    lr_per_example = 0.001 / 64

eval_every = 1000
adapt_every = 100
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
best_model_filename = opt.best_model_filename or join(
        "cache", opt.dataset + "_classifier.ckpt")
makedirs(dirname(best_model_filename), exist_ok=True)

dset_class = dataset.choices[opt.dataset]

<<<<<<< HEAD
if 'mnist' == opt.dataset:
    def schedule(lr, loss):
        return lr if loss > 1.0 else loss * lr

    epochs = 20
    batch_size = 128
    lr_per_example = 1e-4
    learning_rate = batch_size * lr_per_example
    eval_every = 1000
    adapt_every = 100
    print(f"learning rate: {learning_rate} (at {batch_size}-minibatches)")

    # Data augmentation
    train_transform = Compose([
=======
learning_rate = batch_size * lr_per_example

print(f"learning rate: {learning_rate} (at {batch_size}-minibatches)")

# Data augmentation
if 'mnist' == opt.dataset:
    trafo = Compose([
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
        RandomAffine(degrees=10, shear=10, scale=(0.95, 1.15)),
        ToTensor(),
        Normalize([0.5], [0.5], inplace=True)
    ])
<<<<<<< HEAD

    clf = classifier.choices[opt.dataset]()

    loss_op = nn.NLLLoss(reduction='mean')
    optimizer = optim.Adam(clf.parameters(), lr=learning_rate, weight_decay=0.001)

elif 'cifar10' == opt.dataset:
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    num_classes = 10
    print(f"learning rate: {learning_rate} (at {batch_size}-minibatches)")

    # Data pre-processing and  augmentation
    normalize = Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = Compose([])
    if opt.data_augmentation:
        train_transform.transforms.append(RandomCrop(32, padding=4))
        train_transform.transforms.append(RandomHorizontalFlip())
    train_transform.transforms.append(ToTensor())
    train_transform.transforms.append(normalize)
    if opt.cutout:
        train_transform.transforms.append(Cutout(n_holes=opt.n_holes, length=opt.length)) 
    test_transform = transforms.Compose([
        ToTensor(),
        normalize])

    clf = classifier.WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
            dropRate=0.3)

    loss_op = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(clf.parameters(), lr=learning_rate, momentum=0.9,
            nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

trainset = dset_class(transform=train_transform, train=True, labels=True)
testset = dset_class(transform=test_transform, train=False, labels=True)
=======
if 'cifar10' == opt.dataset:
    trafo = Compose([
        RandomAffine(degrees=10, shear=5, scale=(0.95, 1.15), translate=(0.1, 0.1)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.5], inplace=True)
    ])

trainset = dset_class(transform=trafo, train=True, labels=True)
testset = dset_class(train=False, labels=True)
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
trainloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, pin_memory=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=4)

<<<<<<< HEAD
=======
clf = classifier.choices[opt.dataset]()
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
clf = clf.cuda() if torch.cuda.is_available() else clf
clf.train()
device = next(clf.parameters()).device
print(f"Training on device '{device}'")
<<<<<<< HEAD
=======

loss_op = nn.NLLLoss(reduction='mean')

if 'mnist' == opt.dataset:
    optimizer = optim.Adam(clf.parameters(), lr=learning_rate, weight_decay=0.001)
if 'cifar10' == opt.dataset:
    optimizer = optim.SGD(clf.parameters(), lr=learning_rate, momentum=0.9,
                          nesterov=opt.nesterov)
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba

global_step, running_loss = 0, 1.0
best_acc = 2.0
for epoch in range(epochs):
    for i, (examples, labels) in enumerate(trainloader):
        batch_size = labels.shape[0]
        examples = examples.to(device)
        labels = labels.to(device)
        logits = clf(examples)
        loss = loss_op(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = 0.99 * running_loss + 0.01 * loss.item()
<<<<<<< HEAD
        
        if opt.dataset == 'mnist':
            if global_step % adapt_every == 0:
                lr = 0.9 ** epoch * schedule(learning_rate, running_loss)
                print(f"[{global_step}, epoch {epoch+1}] "
                      f"train loss = {running_loss:.3f}, "
                      f"new learning rate = {lr:.5f}")
                for g in optimizer.param_groups:
                    g.update(lr=lr)
            if global_step % eval_every == 0:
                acc = evaluate(clf, testloader)
                print(f"[{global_step}, epoch {epoch+1}] "
                      f"train loss = {running_loss:.3f}, "
                      f"test acc = {acc:.2f} %")
                if acc > best_acc:
                    clf.to_checkpoint(best_model_filename)
                    best_acc = acc
=======

        if global_step % adapt_every == 0:
            if opt.dataset == 'mnist':
                lr = 0.9 ** epoch * schedule(learning_rate, running_loss)
            elif opt.dataset == 'cifar10':
                lr = 0.99 ** epoch * learning_rate
            print(f"[{global_step}, epoch {epoch+1}] "
                  f"train loss = {running_loss:.3f}, "
                  f"new learning rate = {lr:.5f}")
            for g in optimizer.param_groups:
                g.update(lr=lr)

        if global_step % eval_every == 0:
            acc = evaluate(clf, testloader)
            print(f"[{global_step}, epoch {epoch+1}] "
                  f"train loss = {running_loss:.3f}, "
                  f"test acc = {acc:.2f} %")

            if acc > best_acc:
                clf.to_checkpoint(best_model_filename)
                best_acc = acc

>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
        global_step += 1

    if opt.dataset == 'cifar10':
        acc = evaluate(clf, testloader)
        print(f"[{global_step}, epoch {epoch+1}] "
              f"train loss = {running_loss:.3f}, "
              f"test acc = {acc:.2f} %")
        scheduler.step(epoch)

print("Running final evaluation")
acc = evaluate(clf, testloader)
print(f"[{global_step}, final evaluation] "
      f"train loss = {running_loss:.3f}, "
      f"test acc = {acc:.1f}")

if acc > best_acc:
    clf.to_checkpoint(best_model_filename)
    best_acc = acc
