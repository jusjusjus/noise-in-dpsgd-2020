import math 
import torch
from torch import nn
import torch.nn.functional as F

class MNIST(nn.Sequential):

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


class CIFAR10(nn.Sequential):

    def __init__(self):

        C = 32

        super().__init__(
            nn.Conv2d(3, C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(C),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(C),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),

            nn.Conv2d(    C, 2 * C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(2 * C),
            nn.Conv2d(2 * C, 2 * C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(2 * C),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.3),

            nn.Conv2d(2 * C, 4 * C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(4 * C),
            nn.Conv2d(4 * C, 4 * C, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(4 * C),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.4),
            nn.Flatten(),

            nn.Linear(156800, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10), nn.LogSoftmax(dim=1)
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

<<<<<<< HEAD
choices = {
    'mnist': MNIST,
    'wide_resnet':WideResNet,
    'cifar10': CIFAR10
}



""" Copyright 2020 Vithursan Thangarasa Licensed under the Educational Community License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.osedu.org/licenses /ECL-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."""

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)        
        
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
        
    def to_checkpoint(self, ckpt):
        torch.save(self.state_dict(), ckpt)

    @classmethod
    def from_checkpoint(cls, ckpt, *args, **kwargs):
        instance = cls()
        ckpt = torch.load(ckpt, *args, **kwargs)
        instance.load_state_dict(ckpt)
        return instance

=======

choices = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}
>>>>>>> 7d1284d8a0d58f88bc0bf3a42b6f0afcf484c6ba
