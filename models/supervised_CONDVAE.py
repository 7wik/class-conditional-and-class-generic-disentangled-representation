from __future__ import print_function
import math
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import cPickle as pkl
import copy
import socket
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
from models.components.blocks import conv_block, fc_block
from models.components.decoder import Decoder
from models.utils.distributions import Normal
from models.utils.ops import Reshape
import os
import numpy as np


class CONDVAE6(nn.Module):
    def __init__(self):
        super(CONDVAE6, self).__init__()
        self.w_dim = 10
        self.alpha = [1,1,1,1,1,1]
        self.k = 10
        self.priors = "no"
        self.encoder = Encoder(
            [1] + [64, 128, 256],
            [256 * 8 * 8, 2048 * 2],
            10,10,)

        self.encoder2 = classifier()
        # self.encoder2 = torch.load('/home/satwik/xentropy/classifier_model/classifier/model')

        self.encoder3 = Encoder3(10)

        dec_fc = copy.copy([2048, 8 * 8 * 256])
        dec_fc.insert(0, 20)
        self.decoder = Decoder([256, 256, 128, 32], dec_fc, 1)

    def encode(self, x):
        return self.encoder.forward(x)

    def encode2(self, z):
        return self.encoder2.forward(z)

    def decode(self, z):
        return self.decoder.forward(z)

    def encode3(self,z):
        return self.encoder3.forward(z)
    def forward(self, x):
        pzIx, pwIxy = self.encode(x)

        if self.training:
            z = pzIx.sample()
            w = torch.stack([pwiIxy.sample() for pwiIxy in pwIxy],1)
        else:
            z = pzIx.mu
            w = torch.stack([pwiIxy.mu for pwiIxy in pwIxy],1)

        with torch.no_grad():
	        y_0 = self.encode2(x.view(-1,64*64))
	        pyIx = F.softmax(y_0,dim=1)

        pxIwz = [] 
        for i in range(self.k):
            pxIwz.append(self.decode(torch.cat([z, (w[:,i,:])], 1)))
        
        return pxIwz, pzIx, pwIxy,pyIx, z, w, y_0

    def sample(self, batch_size):
        z = Variable(torch.randn(batch_size, 10))
        w = Variable(torch.randn(batch_size, 10 ))
        ind = torch.sort(torch.norm(w, 2, 1))[1]
        z = z[ind]
        w = w[ind]
        if next(self.parameters()).is_cuda:
            z = z.cuda()
            w = w.cuda()
        return self.decode(torch.cat([z, w], 1))
    def loss(self, data):
        x = data[0]
        recon_x, pzIx, pwIxy,pyIx, z, w, y_0 = self.forward(x)
        x = x.view(x.size(0), -1)
        a=[]
        pxIzw = 0 
        for i in range(len(recon_x)):
            pxIzw+= -0.5*(((recon_x[i].view(x.size(0), -1)-x)**2).sum(-1) + np.log(2*np.pi))*pyIx[:,i].view(-1,1)
        pxIzw = -torch.mean(pxIzw)
        klz = pzIx.kl_div().mean()
        klw = 0
        target =[ Normal(
            Variable(torch.cuda.FloatTensor([0.5 * i])),
            Variable(torch.cuda.FloatTensor([np.log(np.sqrt(0.4))])),
        ) for i in range(self.k)]
        for i in range(len(pwIxy)):
            pwiIxy = pwIxy[i]
            pyiIx  = pyIx[:,i]
            klw += (pwiIxy.kl_div())*pyiIx
        klw = (klw).mean()
        labs=torch.eye(self.k)
        yIz = self.encode3(z)
        pyIz=  F.softmax(yIz,dim=-1)        
        log_pyIz = F.log_softmax(yIz,dim=-1)
        HyIz =  ((pyIz*log_pyIz).sum(-1)).mean(0)             
        y_onehot = torch.zeros((data[1].size()[0],self.k))
        y_onehot[torch.arange(data[1].size()[0]),data[1]]=1
        y_onehot=Variable(y_onehot).cuda()
        l1 = self.alpha[0] * pxIzw + self.alpha[1] * klz + self.alpha[3] * klw
        lt = l1 + self.alpha[5] * HyIz
        l2 = ((y_onehot*(log_pyIz)+(1-y_onehot)*torch.log(1-F.softmax(yIz,dim=-1)+1e-12)).sum(-1)).mean(0)
        l2 = self.alpha[5] * torch.mean(l2)       
        indices = torch.max(pyIx,1)[1].cpu().numpy()
        all_mu = []
        all_var= []
        mu=[]
        log_var=[]
        return_x=[]
        for i,pwiIxy in enumerate(pwIxy):
            all_mu.append(pwiIxy.mu)
            all_var.append(pwiIxy.logvar)
        all_mu = torch.stack(all_mu,1)
        all_var= torch.stack(all_var,1) 
        recon_x= torch.stack(recon_x,1)
        return_x=[]
        for i in range(x.size()[0]):
            mu.append(all_mu[i][indices[i]])
            return_x.append(recon_x[i][indices[i]])            
            log_var.append(all_var[i][indices[i]])

        return ({   "mu":torch.stack(mu,0),
                "logvar": torch.stack(log_var,0),
                "recon": torch.stack(return_x,1),
                "all_recon":recon_x,
                "pzIx":pzIx,
                "w_sample":w,
                "w_mu":all_mu,
                "w_var":all_var
            },
            {
                "lt": lt,
                "l1": l1,
                "l2": l2,
                "pxIzw": pxIzw,
                "klz": klz,
                "klw": klw,
                "HyIz": HyIz,
            },
        )

class Encoder(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder, self).__init__()
        modules = []
        self.k =k
        for x_in, x_out in zip(filters[:-1], filters[1:]):
            modules.append(conv_block(x_in, x_out))
        modules.append(Reshape(-1, filters[-1] * 8 * 8))
        self.net1 = torch.nn.Sequential(*modules)

        modules = []
        for x_in, x_out in zip(fc[:-2], fc[1:-1]):
            modules.append(fc_block(x_in, x_out))
        modules.append(fc_block(fc[-2], fc[-1]))
        modules.append(fc_block(fc[-1], 2*10,activation=False))
        self.net2 = torch.nn.Sequential(*modules)

        self.net3 = torch.nn.Sequential(
        fc_block(fc[0], 2048), 
        fc_block(2048, 1024),
        fc_block(1024, 128),
        )

        self.net4 = torch.nn.Sequential(
        fc_block(128+k,2*10,activation=False),
        )

    def forward(self, x):
        k=self.k
        z1  = self.net1.forward(x)
        z   = self.net2.forward(z1)
        w1  = self.net3.forward(z1)
        labs= torch.eye(k)
        W   = torch.stack([self.net4(torch.cat((w1, Variable(labs[i, :].repeat(w1.size(0), 1)).cuda()), 1))
                                    for i in range(k)],1)

        z_split = int(z.size(-1) / 2)
        w_split = int(W.size(-1) / 2)
        return (
            Normal(z[:, :z_split], z[:, z_split:]),
            [Normal(W[:, i, :w_split], W[:,i, w_split:]) for i in range(k)],
        )

class Encoder3(nn.Module):
    #For q(y|z)
    def __init__(self,z_size):
        super(Encoder3, self).__init__()
        self.net = torch.nn.Sequential(
            fc_block(z_size, 1024),
            fc_block(1024, 512),
            fc_block(512, 128),
            fc_block(128, z_size, activation=False),
        )


    def forward(self, z):
        return self.net.forward(z)

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        modules = []
        modules.append(fc_block(64*64, 256))
        modules.append(fc_block(256, 512))
        modules.append(fc_block(512, 1024))
        modules.append(fc_block(1024, 10,activation=False))
        self.net1 = torch.nn.Sequential(*modules)

    def forward(self,x):

        return (self.net1.forward(x.view(-1,64*64)))
