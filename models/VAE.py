from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

from components.blocks import fc_block
from components.decoder import Decoder

from utils.distributions import Normal
from utils.ops import Reshape

class VAE(nn.Module):
    def __init__(self, opts):
        super(VAE, self).__init__()
        self.alpha = opts['model_opts']['alpha']


        if opts['model_opts']['arch'] == 'vanilla':
            self.encoder = Encoder([opts['data']['x_dim'][0]] + opts['model_opts']['enc_filters'],
                                   opts['model_opts']['enc_fc'])
            self.decoder = Decoder(opts['model_opts']['dec_filters'],
                                   opts['model_opts']['dec_fc'],
                                   opts['data']['x_dim'][0])
            self.z_dim = opts['model_opts']['z_dim']
        elif opts['model_opts']['arch'] == 'fullconv':
            self.encoder = FullConvEncoder([opts['data']['x_dim'][0]] + opts['model_opts']['enc_filters'])
            self.decoder = FullConvDecoder(opts['model_opts']['dec_filters'] + [opts['data']['x_dim'][0]])
            self.z_dim = opts['model_opts']['enc_filters'][-1]

        self.k = 6  # What is this used for?

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, z):
        return self.decoder.forward(z)

    def sample(self, batch_size):
        z = Variable(torch.randn(batch_size, self.z_dim))
        if next(self.parameters()).is_cuda:
            z = z.cuda()

        return self.decode(z)

    def forward(self, x):
        pzIx = self.encode(x)
        if self.training:
            z = pzIx.sample()
        else:
            z = pzIx.mu
        return self.decode(z), pzIx, z

    def loss(self, data):
        x = data[0]
        recon_x, pzIx, samples = self.forward(x)
        x = x.view(x.size(0), -1)
        recon_x = recon_x.view(x.size(0), -1)

        pxIzw = torch.sum((recon_x - x) ** 2, -1)
        pxIzw = torch.mean(pxIzw)

        klz = pzIx.kl_div().mean()

        lt = self.alpha[0] * pxIzw + self.alpha[1] * klz

        return {
            'mu': pzIx.mu,
            'logvar': pzIx.logvar,
            'recon': recon_x
        }, {
            'lt': lt,
            'klz': klz,
            'pxIzw': pxIzw
        }

    def change(self, target_subspace, target_val, x, y):
        # decodes by changing the ground truth label y
        # target_subspace: integer indicating which subspace to manipulate
        # target_val: the target value in the target subspace
        # x: input image
        # y: onehot ground truth labels
        pzIx = self.encode(x)
        z_mu = pzIx.mu

        # target_val is of shape 1 x (K+1) x D and represents the means
        # so we need to select and get the difference vector to apply

        y_vec = torch.max(y, 1)[1]

        z_src = torch.index_select(target_val[0], 0, y_vec)
        z_target = target_val[0,target_subspace].unsqueeze(0)

        z_changed = z_mu - z_src + z_target

        return self.decode(z_changed)

def fc_conv_block(in_channels, out_channels, use_bn=True):
    modules = [torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]

    if use_bn:
        modules.append(torch.nn.BatchNorm2d(out_channels))

    modules.append(torch.nn.LeakyReLU(0.2))

    return torch.nn.Sequential(*modules)

class FullConvEncoder(torch.nn.Module):
    def __init__(self, filters):
        super(FullConvEncoder, self).__init__()

        modules = []
        for i, (in_dim, out_dim) in enumerate(zip(filters[:-1], filters[1:])):
            if i == 0:
                use_bn = True
            else:
                use_bn = False

            if i == len(filters) - 2:
                modules.append(fc_conv_block(in_dim, 2*out_dim))
            else:
                modules.append(fc_conv_block(in_dim, out_dim))

        modules.append(Reshape(-1, 2*filters[-1]))

        self.net1 = torch.nn.Sequential(*modules)
        self.net2 = torch.nn.Sequential()
        self.net3 = torch.nn.Sequential()

    def forward(self, x):
        z = self.net1.forward(x)
        split = int(z.size(-1)/2)
        return Normal(z[:,:split], z[:,split:])

def fc_deconv_block(in_channels, out_channels, use_bn=True, activation=torch.nn.ReLU()):
    modules = [torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

    if use_bn:
        modules.append(torch.nn.BatchNorm2d(out_channels))

    modules.append(activation)

    return torch.nn.Sequential(*modules)

class FullConvDecoder(torch.nn.Module):
    def __init__(self, filters):
        super(FullConvDecoder, self).__init__()

        modules = [Reshape(-1, filters[0], 1, 1)]
        for i, (in_dim, out_dim) in enumerate(zip(filters[:-1], filters[1:])):
            if i == len(filters) - 2:
                use_bn = False
                activation = torch.nn.Tanh()
            else:
                use_bn = True
                activation = torch.nn.ReLU()

            modules.append(fc_deconv_block(in_dim, out_dim, use_bn=use_bn, activation=activation))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net.forward(x)

def conv_block(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 5, padding=2),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2)
    )

class Encoder(nn.Module):
    def __init__(self, filters, fc):
        super(Encoder, self).__init__()

        modules = []
        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_block(x_in, x_out))
        modules.append(Reshape(-1, filters[-1] * 8 * 8))

        for x_in, x_out in zip(fc[:-1], fc[1:]):
            modules.append(fc_block(x_in, x_out))

        self.net1 = torch.nn.Sequential(*modules)

        self.net2 = torch.nn.Sequential()
        self.net3 = torch.nn.Sequential()

    def forward(self, x):
        z = self.net1.forward(x)
        split = int(z.size(-1)/2)
        return Normal(z[:,:split], z[:,split:])
