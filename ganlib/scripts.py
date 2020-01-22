#!/usr/bin/env python

from os import makedirs
from os.path import join, basename, splitext, dirname
from threading import Thread
from subprocess import check_output

base_dir = join(dirname(__file__), '..')
cache_dir = join(base_dir, "cache")
makedirs(cache_dir, exist_ok=True)

def generate(**kwargs):
    thread = Thread(target=_generate, kwargs=kwargs)
    thread.start()

def _generate(logger, params: str, step: int) -> None:
    script = join(base_dir, 'generate.py')
    temp = join(cache_dir, splitext(basename(params))[0] + '.png')
    cmd = [script, params, "-o", temp, "--cpu"]
    check_output(cmd)
    logger.add_image_file('generated', temp, step)
    check_output(["rm", temp])
