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

from components.blocks import conv_block, fc_block
from components.decoder import Decoder
from utils.distributions import Normal
from utils.ops import Reshape

class CONDVAE5(nn.Module):
    def __init__(self, opts):
        super(CONDVAE5, self).__init__()
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
        dec_fc[0] += self.w_dim * self.k
        self.decoder = Decoder(opts['model_opts']['dec_filters'],
                               dec_fc,
                               opts['data']['x_dim'][0])

    def encode(self, x, y):
        return self.encoder.forward(x, y)

    def encode2(self, z):
        return self.encoder2.forward(z)

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x, y):
        pzIx, pwIxy = self.encode(x, y)

        if self.training:
            z = pzIx.sample()
            w = pwIxy.sample()
        else:
            z = pzIx.mu
            w = pwIxy.mu

        y_0 = self.encode2(z)

        return self.decode(torch.cat([z, w], 1)), pzIx, pwIxy, z, w, y_0

    def sample(self, batch_size):
        z = Variable(torch.randn(batch_size, self.z_dim))
        w = Variable(torch.randn(batch_size, self.w_dim * self.k))
        ind = torch.sort(torch.norm(w, 2, 1))[1]
        z = z[ind]
        w = w[ind]

        if next(self.parameters()).is_cuda:
            z = z.cuda()
            w = w.cuda()

        return self.decode(torch.cat([z, w], 1))

    def loss(self, data):
        x = data[0]
        y = data[2]

        recon_x, pzIx, pwIxy, z, w, y_0 = self.forward(x, y)
        x = x.view(x.size(0), -1)
        recon_x = recon_x.view(x.size(0), -1)

        '''loss 1'''
        pxIzw = -0.5 * ( ((recon_x - x) ** 2).sum(-1) + np.log(2 * np.pi) ) 
        pxIzw = -torch.mean(pxIzw)

        klz = pzIx.kl_div().mean()

        # computing kl div with pwIy
        pyIw = 0

        target1 = Normal(Variable(torch.cuda.FloatTensor([0.0])),
                         Variable(torch.cuda.FloatTensor([np.log(0.01)])))

        target2 = Normal(Variable(torch.cuda.FloatTensor([3.0])),
                         Variable(torch.cuda.FloatTensor([np.log(1.0)])))

        for i in range(self.k):
            pw_i = pwIxy[self.w_dim*i:self.w_dim*(i+1)]

            klw_i1 = pw_i.kl_div_from(target1)
            klw_i2 = pw_i.kl_div_from(target2)

            pyIw += (1 - y[:,i]) * self.beta[0] * klw_i1 + y[:,i] * self.beta[1] * klw_i2

        pyIw = torch.mean(pyIw)

        pyIz = - 1./(self.k + 1) * torch.sum(y_0, 1)
        pyIz = torch.mean(pyIz)

        l1 = self.alpha[0] * pxIzw + self.alpha[1] * klz + self.alpha[3] * pyIw
        lt = l1 + self.alpha[4] * pyIz

        '''loss 2, cross entropy (y | y_0)'''
        l2 = - torch.sum(y_0 * y, 1)
        l2 = self.alpha[5] * torch.mean(l2)

        '''compute classification loss'''
        argmax = torch.max(y_0, 1)[1]
        acc = y[torch.arange(y_0.size(0)).type(torch.cuda.LongTensor), argmax].mean()

        return {
            'mu': torch.cat([pzIx.mu, pwIxy.mu], 1),
            'logvar': torch.cat([pzIx.logvar, pwIxy.logvar], 1), 
            'recon': recon_x
        }, {
            'lt': lt,
            'l1': l1, 
            'l2': l2, 
            'pxIzw': pxIzw, 
            'klz': klz,
            'pyIw': pyIw, 
            'pyIz': pyIz,
            'acc': acc
            }

    def change(self, target_subspace, target_val, x, y):
        # decodes by changing the ground truth label y
        # target_subspace: integer indicating which subspace to manipulate
        # target_val: the target value in the target subspace
        # x: input image
        # y: onehot ground truth labels
        pzIx, pwIxy = self.encode(x, y)
        z_mu = pzIx.mu
        w_mu = pwIxy.mu.data # note use of data

        y_vec = torch.max(y, 1)[1].data

        # zero out ground truth subspaces
        col_inds = torch.arange(0, w_mu.size(1)).unsqueeze(0).long()
        if next(self.parameters()).is_cuda:
            col_inds = col_inds.cuda()

        # mask which elements of w correspond to ground truth subspaces
        mask = (col_inds >= y_vec.unsqueeze(1) * self.w_dim) * (col_inds < (y_vec.unsqueeze(1) + 1) * self.w_dim)

        w_mu.masked_fill_(mask, 0.0)  # perform the zeroing

        if target_subspace < self.k:
            # set the target_subspace to be the target_val
            w_mu[:,target_subspace*self.w_dim:(target_subspace+1)*self.w_dim] = target_val.data.expand(x.size(0), self.w_dim)

        w_mu = Variable(w_mu, volatile=True)

        return self.decode(torch.cat([z_mu, w_mu], 1))

    def process_labels(self, y, e):
        m1 = y.eq(e)
        m2 = y.ne(e)
        return y.masked_fill(m1, 1).masked_fill(m2, 0)

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
            fc_block(fc[0] + k + 1, 2048),
            fc_block(2048, 1024),
            fc_block(1024, 128),
            fc_block(128, 2 * k * w_dim, activation=False)
            )

    def forward(self, x, y):
        z1 = self.net1.forward(x)

        z = self.net2.forward(z1)
        w = self.net3.forward(torch.cat([z1,y], 1))

        z_split = int(z.size(-1)/2)
        w_split = int(w.size(-1)/2)

        return Normal(z[:,:z_split], z[:,z_split:]), Normal(w[:,:w_split], w[:,w_split:])

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
