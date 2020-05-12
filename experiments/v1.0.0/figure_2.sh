#!/bin/bash

CLIP=1.0
EPOCHS=40
CAPACITY=32
CRITICSTEPS=1
EVALEVERY=1000
PRINTEVERY=500

SIGMA=1.0
CUDA_VISIBLE_DEVICES=0 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY \
	&

SIGMA=0.9
CUDA_VISIBLE_DEVICES=1 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY \
	&

SIGMA=0.8
CUDA_VISIBLE_DEVICES=2 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY

# SIGMA=0.5
# CUDA_VISIBLE_DEVICES=3 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY
