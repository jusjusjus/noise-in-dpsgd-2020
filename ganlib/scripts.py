#!/usr/bin/env python

from os import makedirs
from os.path import join, basename, splitext, dirname, exists
from warnings import warn
from threading import Thread
from subprocess import check_output


base_dir = join(dirname(__file__), '..')
cache_dir = join(base_dir, "cache")
makedirs(cache_dir, exist_ok=True)

def threaded(fn):

    def wrapped(**kwargs):
        thread = Thread(target=fn, kwargs=kwargs)
        thread.start()

    return wrapped

@threaded
def generate(logger, params: str, step: int, dataset: str = 'mnist') -> None:
    script = join(base_dir, 'generate.py')
    temp = join(cache_dir, splitext(basename(params))[0] + '.png')
    cmd = [script, params, "-o", temp, "--cpu", "--dataset", dataset]
    check_output(cmd)
    logger.add_image_file('generated', temp, step)
    check_output(["rm", temp])

@threaded
def inception(logger, params: str, step: int, dataset: str = 'mnist') -> None:
    assert exists(join(base_dir, "cache", "mnist_classifier.ckpt")), """
    Inception during training failed: missing classifier."""
    cmd = [
        join(base_dir, 'inception.py'),
        "-p", params,
        "--cpu",
        "--splits", "1",
        "--quiet",
        "--dataset", dataset
    ]
    output = check_output(cmd)
    score = float(output.decode('utf-8'))
    logger.add_scalar('inception_score', score, step)
