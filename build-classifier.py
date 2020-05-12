#!/usr/bin/env python

from os import makedirs
from os.path import join, dirname
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'],
                    default='mnist')
opt = parser.parse_args()

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, ToTensor, Normalize,
                                    RandomAffine, RandomHorizontalFlip)

from ganlib import dataset
from ganlib import classifier

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


def schedule(lr, loss):
    return lr if loss > 1.0 else loss * lr


epochs = {
    'mnist': 20,
    'cifar10': 200
}[opt.dataset]
batch_size = {
    'mnist': 128,
    'cifar10': 512
}[opt.dataset]
lr_per_example = {
    'mnist': 1e-4,
    'cifar10': 0.001 / 64
}[opt.dataset]
eval_every = 1000
adapt_every = 100
weight_decay = {
    'mnist': 0.001,
    'cifar10': 0.0,
}[opt.dataset]
best_model_filename = join("cache", opt.dataset + "_classifier.ckpt")
makedirs(dirname(best_model_filename), exist_ok=True)

dset_class = dataset.choices[opt.dataset]

learning_rate = batch_size * lr_per_example

print(f"learning rate: {learning_rate} (at {batch_size}-minibatches)")

# Data augmentation
if opt.dataset == 'mnist':
    trafo = Compose([RandomAffine(degrees=10, shear=10, scale=(0.95, 1.15)),
                     ToTensor(), Normalize([0.5], [0.5], inplace=True)])
elif opt.dataset == 'cifar10':
    trafo = Compose([
        RandomAffine(degrees=10, shear=5, scale=(0.95, 1.15), translate=(0.1, 0.1)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.5], inplace=True)
    ])

trainset = dset_class(transform=trafo, train=True, labels=True)
testset = dset_class(train=False, labels=True)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

clf = classifier.choices[opt.dataset]()
clf = clf.cuda() if torch.cuda.is_available() else clf
clf.train()
device = next(clf.parameters()).device
print(f"Training on device '{device}'")

loss_op = nn.NLLLoss(reduction='mean')

if opt.dataset == 'mnist':
    optimizer = optim.Adam(clf.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif opt.dataset == 'cifar10':
    optimizer = optim.SGD(clf.parameters(), lr=learning_rate, momentum=0.9)
else:
    raise ValueError(f"Unknown dataset {opt.dataset}")

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

        global_step += 1

print("Running final evaluation")
acc = evaluate(clf, testloader)
print(f"[{global_step}, final evaluation] "
      f"train loss = {running_loss:.3f}, "
      f"test acc = {acc:.1f}")

if acc > best_acc:
    clf.to_checkpoint(best_model_filename)
    best_acc = acc
