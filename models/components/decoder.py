import torch.nn as nn

from blocks import fc_block, conv_transpose_block
from ..utils.ops import Reshape

class Decoder(nn.Module):
    def __init__(self, filters, fc, out_channels):
        super(Decoder, self).__init__()

        modules = []
        for x_in, x_out in zip(fc[:-1], fc[1:]):
            modules.append(fc_block(x_in, x_out))
        modules.append(Reshape(-1, filters[0], 8, 8))

        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_transpose_block(x_in, x_out))
        modules.append(nn.Conv2d(filters[-1], out_channels, 5, padding=2))
        modules.append(nn.Tanh())
        modules.append(Reshape(-1,out_channels,64,64))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net.forward(x)
