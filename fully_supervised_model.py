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


class fully_supervised_CONDVAE(nn.Module):
    def __init__(self):
        super(fully_supervised_CONDVAE, self).__init__()
        self.w_dim = 10
        self.alpha = [1,2,1,2,1,1]
        self.k = 10
        self.priors = "no"
        self.encoder = Encoder(
            [1] + [64, 128, 256],
            [256 * 8 * 8, 2048 * 2],
            10,10,)
        self.encoder3 = Encoder3(10)
        dec_fc = copy.copy([2048, 8 * 8 * 256])
        dec_fc.insert(0, 20)
        self.decoder = Decoder([256, 256, 128, 32], dec_fc, 1)

    def encode(self, x,y):
        return self.encoder.forward(x,y)

    def decode(self, z):
        return self.decoder.forward(z)

    def encode3(self,z):
        return self.encoder3.forward(z)
    def forward(self, x,y):
        pzIx, pwIxy = self.encode(x,y)

        if self.training:
            z = pzIx.sample()
            w = (pwIxy.sample())
        else:
            z = pzIx.mu
            w = (pwIxy.mu)

        pxIwz = [] 
        pxIwz = (self.decode(torch.cat([z, (w)],dim=-1)))
        
        return pxIwz, pzIx, pwIxy, z, w

    def loss(self, data):
        x = data[0]
        y_onehot = torch.zeros((data[1].size()[0],self.k))
        y_onehot[torch.arange(data[1].size()[0]),data[1]]=1
        y_onehot=Variable(y_onehot).cuda()
        recon_x, pzIx, pwIxy, z, w = self.forward(x,y_onehot)
        x = x.view(x.size(0), -1)
        pxIzw= -0.5*(((recon_x.view(x.size(0), -1)-x)**2).sum(-1) + np.log(2*np.pi))
        pxIzw= -torch.mean(pxIzw)
        klz = (pzIx.kl_div()).mean()
        klw = (pwIxy.kl_div()).mean()
        yIz = self.encode3(z)
        yIw = self.encode3(w)
        HyIz= ((y_onehot*F.log_softmax(yIz,dim=-1)+(1-y_onehot)*torch.log(1-F.softmax(yIz,dim=-1)+1e-20)).sum(-1)).mean()
        HyIw= ((y_onehot*F.log_softmax(yIw,dim=-1)+(1-y_onehot)*torch.log(1-F.softmax(yIw,dim=-1)+1e-20)).sum(-1)).mean()
        # pyIz=  F.softmax(yIz,dim=-1)        
        # log_pyIz = F.log_softmax(yIz,dim=-1)

        '''

        #############################################################
        Keep caution about the sign of HyIz
        #############################################################
        
        '''
        # HyIz =  ((1.0/(self.k)*log_pyIz).sum(-1)).mean(0)             
        l1 = self.alpha[0] * pxIzw + self.alpha[1] * klz + self.alpha[3] * klw
        lt = l1 + self.alpha[5] * (HyIw-HyIz)
        # l2 = ((y_onehot*(log_pyIz)+(1-y_onehot)*torch.log(1-F.softmax(yIz,dim=-1)+1e-12)).sum(-1)).mean(0)
        l2 = self.alpha[5]*(HyIw+HyIz) 
        mu=pwIxy.mu
        log_var=pwIxy.logvar
        return ({"mu":mu,
                "logvar": log_var,
                "recon": recon_x,
                "pzIx":pzIx,
                "w_sample":w},
                # {"lt": lt,
                # "l1": l1,
                # "l2": l2,
                # "pxIzw": pxIzw,
                # "klz": klz,
                # "klw": klw,
                # "HyIz": HyIz,},
                {"lt": lt,
                "l1": l1,
                "l2": l2,
                "pxIzw": pxIzw,
                "klz": klz,
                "klw": klw,
                "HyIz":HyIz,
                "HyIw":HyIw})

class Encoder(nn.Module):
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder, self).__init__()
        modules = []
        self.k =k
        modules.append(fc_block(64*64+self.k,64*64))
        modules.append(Reshape(-1,1,64,64))
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
        self.net3 = torch.nn.Sequential(fc_block(fc[0], 2048),fc_block(2048, 1024),fc_block(1024, 128),)
        self.net4 = torch.nn.Sequential(fc_block(128+k,2*10,activation=False),)

    def forward(self, x,y):
        k=self.k
        x_  = torch.cat((x.view(-1,64*64),y),dim=-1)
        z1  = self.net1.forward(x_)
        z   = self.net2.forward(z1)
        w1  = self.net3.forward(z1)
        labs= torch.eye(k)
        W   = self.net4(torch.cat((w1, y), 1))
        z_split = int(z.size(-1) / 2)
        w_split = int(W.size(-1) / 2)
        return (Normal(z[:, :z_split], z[:, z_split:]),Normal(W[:,:w_split], W[:,w_split:]))

class Encoder3(nn.Module):
    #For q(y|z)
    def __init__(self,z_size):
        super(Encoder3, self).__init__()
        self.net = torch.nn.Sequential(fc_block(z_size, 1024),fc_block(1024, 512),fc_block(512, 128),fc_block(128, z_size, activation=False))
    def forward(self, z):
        return self.net.forward(z)

def print_fields(out):
    s = ''
    for key, val in out.items():
        s += '{} : {}\t'.format(key, val)
    print(s)

def print_avg_fields(out,l):
    s = ''
    for key, val in out.items():
        s += '{} : {}\t'.format(key, val/l)
    print(s)

def accumulate_loss(out_fields, out):
    for key in out.keys():
        out_fields[key] += out[key].cpu().detach().numpy()
def train():
    prob = fully_supervised_CONDVAE()
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/mnist/', train=False, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.5,), (0.5,))])),batch_size=64, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
                            torchvision.datasets.MNIST('./data/mnist/', train=True, download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])),
        batch_size=64, shuffle=True)
    l1_params = list(prob.encoder.net1.parameters()) \
        + list(prob.encoder.net2.parameters()) \
        + list(prob.encoder.net3.parameters()) \
        +list(prob.encoder.net4.parameters())\
        + list(prob.decoder.net.parameters())
    optimizer_1 = optim.Adam(l1_params, lr=0.0005)
    l2_params = list(prob.encoder3.net.parameters())
    optimizer_2 = optim.Adam(l2_params, lr=0.0005)
    prob.cuda()
    k = int(math.log(10)/math.log(3))#here the log should be on the number of epochs
    milestones = [3**(i+1) for i in range(k)]
    gamma = 0.1 ** (1./k)
    scheduler_1 = optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=milestones, gamma=gamma)
    scheduler_2 = optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=milestones, gamma=gamma)
    for epoch in range(10):
        out_fields={key:0 for key in ["lt","l1","l2","pxIzw","klz","klw","HyIz","HyIw"]}#["lt","l1","l2","pxIzw","klz","klw","HyIz"]}
        scheduler_1.step()
        scheduler_2.step()
        i=0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            for j in data:
                data[0] = Variable(data[0].cuda())
                data[1] = Variable(data[1].cuda())

            out_vec, out = prob.loss(data)

            loss_1 = out['lt']
            optimizer_1.zero_grad()
            loss_1.backward(retain_graph=True)
            optimizer_1.step()

            loss_2 = out['l2']
            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            print_fields(out)
            accumulate_loss(out_fields,out)
            i+=1
            # break
        print ("avg loss for epoch {} ==>".format(epoch))
        print_avg_fields(out_fields,float(i))
        if not os.path.exists("./classifier_model/wz_supervised/"):
            os.makedirs("./classifier_model/wz_supervised/")
            print ("created the file--------------------------------------")
        torch.save(prob, "./classifier_model/wz_supervised/model")
        # break
if __name__=="__main__":
    train()
