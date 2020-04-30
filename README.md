
# Repository to "Scaling clipping and noise variance in differentially private stochastic gradient descent"

With this repository, you can train with differential privacy Wasserstein
generative adversarial networks with gradient penalty.  

## Notes on the code

The minimal implimentation of the DP-SGD algorithm for GANs is based on pytorch
and has been tuned for clarity.  We use the MNIST dataset included in pytorch.
The generator ("/ganlib/generator.py") and critic ("train.py") has been
designed to mimic the one used in Zhang et al as close as necessary.

Training is done using stochastic gradient descent (SGD) with minibatches using
Adam in both generator and discriminator.  SGD is modified as suggested by
Zhang et al to preserve differential privacy.  The core algorithm of DP-SGD can
be found in method `DPWGANGPTrainer.critic_step` ("/ganlib/trainer.py:108").

## Running the example

```bash
conda create -n dp python=3.6
conda activate dp
pip install -r requirements.txt
```

## Training

- `python train.py` to train with parameters from Zhang et al
- `python train.py` --nodp` to train without differential privacy
- `python train.py --sigma 0.5 --grad-clip 1.0` to train successfully

Tensorboard logs are automatically written during training.  See them by
running `tensorboard --logdir cache/logs` and navigating your browser to
"http://localhost:6006".

Note, if you want to compute the inception score during training from each
model checkpoint, you need to first train a classifier (see below).

## Generating samples

To generate samples from a checkpoint us the script "/generate.py".  For
example after training with argument `--nodp` you can generate a sample image
like so:

```bash
python generate.py cache/logs/nodp/checkpoint-3749.pth -o nodp.png
```

## Compute inception score

After training a GAN you can estimate the inception score from created
checkpoints.  For this you first need to train an MNIST classifier by `python
build-classifier.py`.  Classifier-inferred probabilities are used to compute
the inceptions core.  This is done in the script by running:

```bash
python inception.py -p <path-to-checkpoint/'MNIST'>
```

By not prividing a checkpoint or 'MNIST', you will compute the inception score
of the MNIST testing set.

## Reproduce Results from the Paper

In "/experiments" we provide a number of bash scripts that will run the
experiments to reproduce the data we show in our paper.  The results may
slightly differ because we use different seeds.  That is intended because we
believe that the essence of the result should not rely on the precise numerical
outcome.

## Authors

* Justus Schwabedal
* Pascal Michel
* Mario Riontino
