#!/bin/bash

CLIP=1.0
SIGMA=0.5
EPOCHS=40
CRITICSTEPS=1
EVALEVERY=1000
PRINTEVERY=500

CAPACITY=16
CUDA_VISIBLE_DEVICES=0 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY &

# CAPACITY=32
# CUDA_VISIBLE_DEVICES=1 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY &

CAPACITY=96
CUDA_VISIBLE_DEVICES=2 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY &

CAPACITY=128
CUDA_VISIBLE_DEVICES=3 ./train.py --grad-clip $CLIP --capacity $CAPACITY --sigma $SIGMA --epochs $EPOCHS --critic-steps $CRITICSTEPS --print-every $PRINTEVERY --eval-every $EVALEVERY
