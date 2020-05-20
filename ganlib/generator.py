
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .optimizable import Optimizable


def join_image_batch(images, num_rows):
    images = images.squeeze()
    if len(images.shape) == 3:
        num_batch, lenx, leny = images.shape
        colors = 1
    else:
        num_batch, colors, lenx, leny = images.shape
        assert colors == 3

    num_cols = int(np.ceil(num_batch/num_rows))
    collection = np.zeros((colors, num_rows*lenx, num_cols*leny), dtype=images.dtype)
    for idx, image in enumerate(images):
        i, j = idx // num_cols, idx % num_cols
        image = image[None, ...] if len(image.shape) == 2 else image
        collection[:, i*lenx:(i+1)*lenx, j*leny:(j+1)*leny] = image

    return collection.squeeze()


class Generator(Optimizable):

    latent_dim = 128

    def compute_sample_images(self, n: int):
        """return `n` images"""
        self.eval()
        with torch.no_grad():
            z = self.get_latent_variable(n)
            z = z.to(self.device)
            z = Variable(z)  # type: ignore
            imgs = self(z)  # type: ignore
            imgs = imgs.data.cpu().numpy()

        return imgs

    def compute_joined_sample_images(self, num_rows=3):
        """return joined image batch with num_rows x num_rows images"""
        imgs = self.compute_sample_images(num_rows ** 2)
        return join_image_batch(imgs, num_rows)

    def dataloader(self, batch_size, num_batches=256):

        class _dataloader:

            def __iter__(this):
                self.eval()
                for b in range(num_batches):
                    with torch.no_grad():
                        latent = self.get_latent_variable(batch_size)
                        latent = Variable(latent)
                        imgs = self(latent)  # type: ignore

                    yield imgs

        return _dataloader()

    def get_latent_variable(self, batch_size):
        shp = (batch_size, self.latent_dim)
        return torch.randn(*shp, dtype=torch.float32, device=self.device)


class MNIST(Generator):

    colors = 1

    def __init__(self, capacity):
        super().__init__(capacity=capacity)
        C = self.capacity = capacity
        
        lin_out_features = 4 * 4 * 4 * C
        self.activation = nn.ReLU()
        self.projection = nn.Linear(self.latent_dim, lin_out_features)
        self.bn_proj = nn.BatchNorm1d(lin_out_features)

        pad = (2, 2, 2)
        outpad = (0, 1, 1)
        kw = {'kernel_size': 5, 'stride': 2}
        self.deconv1 = nn.ConvTranspose2d(
            4 * C, 2 * C, padding=pad[0], output_padding=outpad[0], **kw)
        self.deconv2 = nn.ConvTranspose2d(
            2 * C, 1 * C, padding=pad[1], output_padding=outpad[1], **kw)
        self.deconv3 = nn.ConvTranspose2d(
            1 * C, self.colors, padding=pad[2], output_padding=outpad[2], **kw)
        self.output = nn.Tanh()

    def forward(self, state):
        state = self.projection(state.contiguous())
        state = self.bn_proj(state)
        state = self.activation(state)
        state = state.view(-1, 4 * self.capacity, 4, 4)
        state = self.activation(self.deconv1(state))
        state = self.activation(self.deconv2(state))
        return self.output(self.deconv3(state))


class CIFAR10(Generator):

    colors = 3

    def __init__(self, capacity, *args, **kwargs):
        super().__init__(capacity=capacity)
        C = self.capacity = capacity

        lin_out_features = 4 * 4 * 4 * C
        self.zspace_layers = nn.Sequential(
            nn.Linear(self.latent_dim, lin_out_features),
            nn.BatchNorm1d(lin_out_features),
            nn.ReLU()
        )
        kernel_size = 3
        stride = 2
        pad = kernel_size // 2
        outpad = 1
        def DeconvBlock(cin, cout):
            return [
                nn.ConvTranspose2d(cin, cout, kernel_size, stride, pad, outpad),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cout, cout, kernel_size, 1, pad),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2),
            ]

        layers = DeconvBlock(4 * C, 2 * C)
        layers += DeconvBlock(2 * C, C)
        layers.append(
            nn.ConvTranspose2d(C, self.colors, kernel_size, stride, pad, outpad))
        layers.append(nn.Tanh())
        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, state):
        state = self.zspace_layers(state.contiguous())
        state = state.view(-1, 4 * self.capacity, 4, 4)
        state = self.deconv_layers(state)
        return state


choices = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
}
