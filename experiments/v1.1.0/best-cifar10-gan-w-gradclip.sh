#!/bin/bash

CLIP=1.0
SIGMA=0.01
EPOCHS=10000
MOMENTUM=0.9
DEVICEID=0
CAPACITY=128
BATCHSIZE=256
EVALEVERY=10000
PRINTEVERY=1000
CRITICSTEPS=10
LRPEREXAMPLE=1.6e-06

CUDA_VISIBLE_DEVICES=$DEVICEID ./train.py --dataset cifar10 \
	--capacity $CAPACITY \
	--sigma $SIGMA --grad-clip $CLIP \
	--batch-size $BATCHSIZE --epochs $EPOCHS \
	--eval-every $EVALEVERY --print-every $PRINTEVERY \
	--momentum $MOMENTUM \
	--critic-steps $CRITICSTEPS --lr-per-example $LRPEREXAMPLE
