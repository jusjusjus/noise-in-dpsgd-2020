
# Repository to "Scaling clipping and noise variance in differentially private stochastic gradient descent"

This repository accompanies our discovery that tensorflow-privacy had a pivotal
bug in randomizing gradients.

## Running the example

```bash
conda create -n dp python=3.6
conda activate dp
pip install -r requirements.txt
jupyter notebook dp-mnist.ipynb
```

After running the above commands, open "localhost:8888" in a browser (dependent
on your jupyter configurations) and follow the code.

In short, we load the MNIST train and test set, define a convolutional network
that is equivalent to the one in the [tensorflow-privacy
example](https://github.com/tensorflow/privacy/tree/master/tutorials/Classification_Privacy.ipynb).
In the last cell, we compute the corrected privacy bound which gives about
`eps=2e5`.

## Authors

* Justus Schwabedal
* Pascal Michel
* Mario Riontino
