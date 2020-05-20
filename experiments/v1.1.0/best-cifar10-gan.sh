#!/bin/bash

EPOCHS=10000
CAPACITY=128
BATCHSIZE=256
EVALEVERY=4000
PRINTEVERY=500
CRITICSTEPS=5
LRPEREXAMPLE=1.6e-06

CUDA_VISIBLE_DEVICES=0 ./train.py --nodp --dataset cifar10 \
	--capacity $CAPACITY \
	--batch-size $BATCHSIZE --epochs $EPOCHS \
	--eval-every $EVALEVERY --print-every $PRINTEVERY \
	--critic-steps $CRITICSTEPS --lr-per-example $LRPEREXAMPLE
