#!/bin/bash

SIGMA=0.5
EPOCHS=40
CAPACITY=32
CRITICSTEPS=1
EVALEVERY=1000
PRINTEVERY=500

CLIP=0.1
CUDA_VISIBLE_DEVICES=1 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY &

CLIP=1.0
CUDA_VISIBLE_DEVICES=2 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY &

CLIP=100.0
CUDA_VISIBLE_DEVICES=3 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY
