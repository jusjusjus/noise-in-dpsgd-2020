
from os.path import join
from time import time
from typing import Dict

import torch
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):

    def __init__(self, *args, logdir='log', **kwargs):
        """Create a summary writer"""
        super().__init__(*args, logdir, **kwargs)
        self.logdir = logdir
        self.timer = {}

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    global_step: int) -> None:
        for k, v in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{k}", v, global_step)

    def add_checkpoint(self, network, step: int, filename=None):
        filename = join(self.logdir, filename or f"checkpoint-{step}.pth")
        state = {'global_step': step, 'state_dict': network.state_dict()}
        torch.save(state, filename)
        return filename

    def add_image_file(self, tag: str, filename, step: int) -> None:
        image = np.asarray(Image.open(filename))
        image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)
        super().add_image(tag, torch.from_numpy(image), step)
