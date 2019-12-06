from __future__ import print_function
import argparse
import copy
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import math

from components.blocks import fc_block, conv_block
from components.decoder import Decoder
from utils.distributions import Normal
from utils.ops import Reshape

class CONTROL2(nn.Module):
    def __init__(self, opts):
        super(CONTROL2, self).__init__()
        self.w_dim = opts['model_opts']['w_dim']
        self.z_dim = opts['model_opts']['z_dim']
        self.alpha = opts['model_opts']['alpha']
        self.beta = opts['model_opts']['beta']
        self.k = opts['model_opts']['k']

        self.encoder = Encoder([opts['data']['x_dim'][0]] + opts['model_opts']['enc_filters'],
                               opts['model_opts']['enc_fc'],
                               self.w_dim,
                               self.k)

        self.encoder2 = Encoder2(opts['model_opts']['enc_filters'], 
                                opts['model_opts']['enc_fc'], 
                                self.w_dim, 
                                self.k)

        dec_fc = copy.copy(opts['model_opts']['dec_fc'])
        dec_fc[0] += self.k
        self.decoder = Decoder(opts['model_opts']['dec_filters'],
                               dec_fc,
                               opts['data']['x_dim'][0])

    def encode(self, x, y):
        return self.encoder.forward(x, y)

    def encode2(self, z):
        return self.encoder2.forward(z)

    def decode(self, z):
        return self.decoder.forward(z)

    def sample(self, batch_size):
        z = Variable(torch.randn(batch_size, self.z_dim))
        ind = torch.LongTensor(batch_size, 1).random_() % self.k
        y = Variable(torch.FloatTensor(batch_size, self.k).scatter_(1, ind, 1))

        zy = torch.cat([z, y], 1)

        if next(self.parameters()).is_cuda:
            zy = zy.cuda()

        return self.decode(zy)

    def forward(self, x, y):
        pzIx = self.encode(x, y)
        if self.training:
            z = pzIx.sample()
        else:
            z = pzIx.mu

        y_0 = self.encode2(z)

        return self.decode(torch.cat([z, y], 1)), pzIx, z, y_0

    def loss(self, data):
        x = data[0]
        y = data[2]
        # y = self.process_labels(y, 3)
        # y = y[:, :-1]
        recon_x, pzIx, samples, y_0 = self.forward(x, y[:, :-1])
        x = x.view(x.size(0), -1)
        recon_x = recon_x.view(x.size(0), -1)

        # loss
        pxIzw = torch.sum((recon_x - x) ** 2, -1)
        pxIzw = torch.mean(pxIzw)

        pyIz = - 1./(self.k + 1) * torch.sum(y_0, 1)
        pyIz = torch.mean(pyIz)
        
        klz = pzIx.kl_div().mean()

        lt = self.alpha[0] * pxIzw + self.alpha[1] * klz

        '''loss 2, cross entropy (y | y_0)'''
        l2 = - torch.sum(y_0 * y, 1)
        l2 = self.alpha[5] * torch.mean(l2)

        '''compute classification loss'''
        argmax = torch.max(y_0, 1)[1]
        acc = y[torch.arange(y_0.size(0)).type(torch.cuda.LongTensor), argmax].mean()

        return {
            'mu': pzIx.mu,
            'logvar': pzIx.logvar,
            'recon': recon_x
        }, {
            'lt': lt,
            'l2': l2, 
            'klz': klz,
            'pxIzw': pxIzw,
            'pyIz': pyIz,
            'acc': acc,
        }

    def change(self, target_subspace, target_val, x, y):
        # decodes by changing the ground truth label y
        # target_subspace: integer indicating which subspace to manipulate
        # target_val: the target value in the target subspace
        # x: input image
        # y: onehot ground truth labels
        pzIx = self.encode(x, y[:,:-1])
        z_mu = pzIx.mu

        z_r = torch.zeros(z_mu.size(0), self.k)

        if target_subspace < self.k:
            z_r[:,target_subspace] = target_val.data.expand(z_r.size(0), 1)

        if next(self.parameters()).is_cuda:
            z_r = z_r.cuda()
        z_r = Variable(z_r, volatile=True)

        return self.decode(torch.cat([z_mu, z_r], 1))

class Encoder(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder, self).__init__()
        modules = []
        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_block(x_in, x_out))
        modules.append(Reshape(-1, filters[-1] * 8 * 8))
        self.net1 = torch.nn.Sequential(*modules)

        modules = []
        modules.append(fc_block(fc[0] + k, fc[1], activation=False))
        self.net2 = torch.nn.Sequential(*modules)

        self.net3 = torch.nn.Sequential()
        self.net4 = torch.nn.Sequential()

    def forward(self, x, y):
        z1 = self.net1.forward(x)
        z = self.net2.forward(torch.cat([z1, y], 1))

        z_split = int(z.size(-1)/2)

        return Normal(z[:,:z_split], z[:,z_split:])

class Encoder2(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder2, self).__init__()
        self.net = torch.nn.Sequential(
            fc_block(int(fc[-1]/2), 1024),
            fc_block(1024, 512),
            fc_block(512, 128),
            fc_block(128, k + 1, activation=False),
            nn.LogSoftmax(dim=1)
            )

    def forward(self, z):
        return self.net.forward(z)
