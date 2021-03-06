
import torch
from torch import nn

class Classifier(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 4 ** 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )

    def probabilities(self, images):
        logits = self(images)
        return torch.exp(logits)

    @classmethod
    def from_checkpoint(cls, ckpt, *args, **kwargs):
        instance = cls()
        ckpt = torch.load(ckpt, *args, **kwargs)
        instance.load_state_dict(ckpt)
        return instance

    def to_checkpoint(self, ckpt):
        torch.save(self.state_dict(), ckpt)
