
import torch.nn as nn

from .optimizable import Optimizable


class MNIST(Optimizable):

    colors = 1

    def __init__(self, capacity, *args, **kwargs):
        super().__init__()
        C = capacity

        def ConvBlock(cin, cout):
            return [
                nn.Conv2d(cin, cout, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2)
            ]

        layers = ConvBlock(self.colors, C)
        layers += ConvBlock(C, 2 * C)
        layers += ConvBlock(2 * C, 4 * C)
        layers += [
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * C, 1)
        ]
        self.sequential = nn.Sequential(*layers)

    def forward(self, images):
        images = self.sequential(images)
        criticism = images.squeeze(-1)
        return criticism


class CIFAR10(Optimizable):

    colors = 3

    def __init__(self, capacity, *args, **kwargs):
        super().__init__()
        C = capacity
        kernel_size = 3
        pad = kernel_size // 2

        def ConvBlock(cin, cout):
            return [
                nn.Conv2d(cin, cout, kernel_size, 2, pad),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cout, cout, kernel_size, 1, pad),
                nn.LeakyReLU(0.2),
            ]
        
        layers = ConvBlock(self.colors, C)
        layers += ConvBlock(C, 2 * C)
        layers += ConvBlock(2 * C, 4 * C)
        layers += [
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * C, 1)
        ]
        self.sequential = nn.Sequential(*layers)


    def forward(self, images):
        images = self.sequential(images)
        criticism = images.squeeze(-1)
        return criticism


choices = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
}
