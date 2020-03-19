
from os.path import join, abspath, exists, basename
from glob import glob
from typing import List

import pandas as pd
import tensorflow as tf


class LogReader:

    def __init__(self, logdir: str):
        self.logdir = abspath(logdir)

    @property
    def logdir(self) -> str:
        return self._logdir

    @logdir.setter
    def logdir(self, l: str):
        self._logdir = abspath(l)
        assert exists(self.logdir)

    @property
    def files(self) -> List[str]:
        files = glob(join(self.logdir, '*'))
        return list(filter(lambda f: 'tfevents' in f, files))

    @property
    def scalars(self) -> pd.DataFrame:
        scalars = {}
        for f in self.files:
            try:
                for summary in tf.compat.v1.train.summary_iterator(f):
                    if len(summary.summary.value) == 1:
                        t = summary.summary.value[0]
                        scalars[(t.tag, summary.step)] = t.simple_value
            except BaseException as err:
                print(f"err reading {f}", err)

        scalars = pd.Series(scalars)
        scalars.index.names = ['tag', 'step']
        scalars = scalars.unstack('tag')
        for k, v in self.info_from_path.items():
            scalars[k] = v

        return scalars

    @property
    def info_from_path(self):
        info = {}
        if "cache/logs" in self.logdir:
            for prop in basename(self.logdir).split('-'):
                k, v = prop.split('_')
                info[k] = float(v)

        return info
