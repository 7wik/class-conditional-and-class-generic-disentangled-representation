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
import math

class CONDVAE2(nn.Module):
    def __init__(self, opts):
        super(CONDVAE2, self).__init__()
        self.k = 6
        self.encoder = Encoder(opts['enc_filters'], opts['enc_fc'], opts['w_dim'], self.k)
        self.decoder = Decoder(opts['dec_filters'], opts['dec_fc'], opts['w_dim'], self.k)
        self.x_dim = opts['x_dim']
        self.alpha = opts['alpha']
        self.w_dim = opts['w_dim']

    def encode(self, x, y):
        return self.encoder.forward(x, y)

    def sample(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return mu + eps * std
        else:
          return mu

    def decode(self, z, y):
        return self.decoder.forward(z, y)

    def forward(self, x, y):
        mu, logvar, y_0 = self.encode(x.unsqueeze(1), y)
        z = self.sample(mu, logvar)
        return self.decode(z, y), mu, logvar, z, y_0

    def loss(self, data):
        x = data[0]
        y = data[2]
        # y = self.process_labels(y, 3)
        y = y[:,:-1]

        recon_x, mu, logvar, samples, y_0 = self.forward(x, y)
        x = x.view(x.size(0), -1)
        recon_x = recon_x.view(x.size(0), -1)
        w = samples[:, -self.k * self.w_dim:]

        '''loss 1'''
        pxIzw = torch.sum((recon_x - x) ** 2, -1)
        pxIzw = torch.mean(pxIzw)

        # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        # kld = torch.mean(kld)

        klz = 0.5 * torch.sum(-1 - logvar[:, :-self.k * self.w_dim] + (mu[:, :-self.k * self.w_dim]**2) + logvar[:, :-self.k * self.w_dim].exp(), -1)
        klz = torch.mean(klz)

        klw = 0.5 * torch.sum(-1 - logvar[:, -self.k * self.w_dim:] + (mu[:, -self.k * self.w_dim:]**2) + logvar[:, -self.k * self.w_dim:].exp(), -1)
        klw = torch.mean(klw) 

        # p = -(torch.sum((w) ** 2, -1) / (2 * 0.01)) - 0.5 * math.log(math.pi * 2 * 0.1)
        pyIw = 0
        sigma = 2
        for i in range(self.k):
            w_i = w[:, self.w_dim*i:self.w_dim*(i+1)]
            p = -(torch.sum((w_i) ** 2, -1) / (2 * sigma**2))
            pyIw += (1 - y[:,i]) * 5 * p + y[:,i] * torch.log(1 - torch.exp(p) + 1e-12)

        pyIw = -torch.mean(pyIw)

        pyIz = -torch.sum(y_0, 1)
        pyIz = torch.mean(pyIz)

        # l1 = 20 * (pxIzw + 0.1 * kld) + 10 * pyIw + 100 * pyIz
        l1 = self.alpha[0] * pxIzw + self.alpha[1] * (klw + klz) + self.alpha[2] * pyIw
        lt = l1 + self.alpha[3] * pyIz
        # l1 = self.alpha[0] * pxIzw + self.alpha[1] * (klz + klw) + self.alpha[2] * pyIw + self.alpha[3] * pyIz


        '''predict y from z'''
        # l_2 = - y_0 * torch.cat([1-y, y], 1)
        l2 = - y_0 * torch.cat([1-y, y], 1)
        l2 = self.alpha[4] * torch.mean(l2)

        '''compute classification loss'''
        argmax = torch.max(y_0, 1)[1]
        ind = torch.arange(y_0.size(0)).type(torch.cuda.LongTensor)
        y = torch.cat([1-y, y], 1)[ind, argmax]
        acc = y.mean()

        return {'mu': mu, 'logvar': logvar, 'recon': recon_x}, {
                'l1': l1, 
                'lt': lt, 
                'l2': l2, 
                'pxIzw': pxIzw, 
                'klz': klz,
                'klw': klw,
                'pyIw': pyIw, 
                'pyIz': pyIz,
                'acc': acc
                }

    def process_labels(self, y, e):
        m1 = y.eq(e)
        m2 = y.ne(e)
        return y.masked_fill(m1, 1).masked_fill(m2, 0)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def conv_block(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 5, padding=2),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2)
    )

def conv_transpose_block(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2,  padding=2, output_padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )

def fc_block(in_dim, out_dim, activation=True, dropout=False):
    modules = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim)
    ]

    if dropout:
        modules.append(torch.nn.Dropout())

    if activation:
        modules += [
            nn.ReLU()
    ]

    return nn.Sequential(*modules)


class Encoder(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder, self).__init__()
        modules = []
        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_block(x_in, x_out))
        modules.append(Reshape(-1, filters[-1] * 8 * 8))
        self.net1 = torch.nn.Sequential(*modules)

        modules = []
        for x_in, x_out in zip(fc[:-2], fc[1:-1]):
            modules.append(fc_block(x_in, x_out))
        modules.append(fc_block(fc[-2], fc[-1], activation=False))
        self.net2 = torch.nn.Sequential(*modules)

        self.net3 = torch.nn.Sequential(
            fc_block(fc[0] + k, 2048),
            fc_block(2048, 1024),
            fc_block(1024, 128),
            fc_block(128, 2 * k * w_dim, activation=False)
            )

        self.net4 = torch.nn.Sequential(
            fc_block(int(fc[-1]/2), 1024),
            fc_block(1024, 512),
            fc_block(512, 128),
            fc_block(128, k * 2, activation=False),
            nn.LogSoftmax(dim=1)
            )

    def forward(self, x, y):
        z1 = self.net1.forward(x)

        z = self.net2.forward(z1)
        w = self.net3.forward(torch.cat([z1,y], 1))

        z_split = int(z.size(-1)/2)
        w_split = int(w.size(-1)/2)

        y_0 = self.net4.forward(z[:,:z_split])
        return torch.cat([z[:,:z_split], w[:,:w_split]], 1), torch.cat([z[:,z_split:], w[:,w_split:]], 1), y_0

class Decoder(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Decoder, self).__init__()
        fc[0] += w_dim * k

        modules = []
        for x_in, x_out in zip(fc[:-1], fc[1:]):
            modules.append(fc_block(x_in, x_out))
        modules.append(Reshape(-1, filters[0], 8, 8))

        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_transpose_block(x_in, x_out))
        modules.append(torch.nn.Conv2d(filters[-1], 1, 5, padding=2))
        modules.append(torch.nn.Tanh())
        modules.append(Reshape(-1,64,64))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, x, y):
        return self.net.forward(x)