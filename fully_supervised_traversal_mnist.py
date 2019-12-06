from __future__ import print_function
import argparse
from tqdm import tqdm
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image,make_grid
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import torch
import cPickle as pkl
import subprocess
import copy
import socket
from fully_supervised_model import* 
from data import *
st=-10
en= 10
ga= 2
def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)

def dim_traversal(opts,z_mu,w_mu,start=-1,end=1,gap=1.0/6,variable='z'):
    stack=[]
    c=start
    if variable=='z':
        for j in range(z_mu.size()[-1]):
            substack=[]
            c=start
            for k in range(1,(int((end-start)/gap)+1)):
                adder = (torch.zeros(z_mu.size()).cuda())
                adder[:,j]+=c
                substack.append(opts['model'].decode(torch.cat([z_mu+adder, (w_mu)], 1))[0])
                c+=gap
            substack = torch.stack(substack)
            stack.append(substack)
    elif variable=='w':
        for j in range(w_mu.size()[-1]):
            substack=[]
            c=start
            for k in range(1,(int((end-start)/gap)+1)):
                adder = (torch.zeros(w_mu.size()).cuda())
                adder[:,j]+=c
                substack.append(opts['model'].decode(torch.cat([z_mu,adder+(w_mu)], 1))[0])
                c+=gap
            substack = torch.stack(substack,dim=0)
            stack.append(substack)
    return (torch.stack(stack,dim=0).transpose(0,1))

def traversal_visualize(opts):
    # opts['model']=torch.load('/home/satwik/xentropy/results/'+opts['data']['data']+'/'+opts['data']['model']+'/2/model0')
    # opts['model']=torch.load('/home/satwik/xentropy/classifier_model/fully_supervised_model/model')
    opts['model']=torch.load('/home/satwik/xentropy/classifier_model/wz_supervised/model')
    opts['model'].eval()
    s='/home/satwik/xentropy/classifier_model/classifier' +'/z_traversal_supervised'+str(st)+'+'+str(en)+'/'
    r='/home/satwik/xentropy/classifier_model/classifier' +'/w_traversal_supervised'+str(st)+'+'+str(en)+'/'
    # s='/home/satwik/xentropy/results/fmnist/CONDVAE6/2' +'/z_traversal/'
    # r='/home/satwik/xentropy/results/fmnist/CONDVAE6/2' +'/w_traversal/'
    if not os.path.exists(s):
        os.makedirs(s)
    if not os.path.exists(r):
        os.makedirs(r)
    a=[]
    for i, data in enumerate(opts['test_loader']):
        actual_lab=data[1][0].cpu().numpy()
        if actual_lab in a:
            continue
        elif len(a)==int(opts['model_opts']['k']):
            break
        a.append(actual_lab)
        labs = torch.eye(opts['model_opts']['k']).float()
        for j, d in enumerate(data):
            data[j] = Variable(d.cuda())

        z,w = opts['model'].encode(data[0],Variable(labs[data[1]].cuda()))
        z_mu = z.mu
        z_mu = z.mu
        w_mu = w.mu
        w_var= w.logvar
        # z traversal
        z_stack=dim_traversal(opts,z_mu,w_mu,start=st,end=en,gap=ga,variable='z')
        for j,substack in enumerate(z_stack):
            save_image(make_grid(substack),s + 'z_'+str(i)+'_'+str(actual_lab)+'_'+str(j)+'.jpg')
        grid2gif(str(os.path.join(s,'z_'+str(i)+'_'+str(actual_lab)+"*.jpg")),\
        str(os.path.join(s,'z_'+str(i)+'_'+str(actual_lab)+".gif")),delay=10)

        for j in range((opts['model_opts']['k'])):
            data[1] = Variable(labs[j].view(1,-1)).cuda()

            z,w = opts['model'].encode(data[0],data[1])
            z_mu = z.mu
            w_mu = w.mu
            w_var= w.logvar

            # w traversal
            save_image((opts['model'].decode(torch.cat([z_mu,(w_mu)], 1))[0]),r+'w_label_'+str(i)+'_'+str(j)+'.jpg')
            w_stack=dim_traversal(opts,z_mu,w_mu,start=st,end=en,gap=en,variable='w')
            for k,substack in enumerate(w_stack):
                save_image(make_grid(substack),r+'w_'+str(i)+'_'+str(actual_lab)+'_'+str(j)+'_'+str(k)+'.jpg')
            grid2gif(str(os.path.join(r,'w_'+str(i)+'_'+str(actual_lab)+'_'+str(j)+'*.jpg')),\
            str(os.path.join(r,'w_'+str(i)+'_'+str(actual_lab)+'_'+str(j)+'.gif')),delay=10)

        grid2gif(str(os.path.join(r,'w_label_'+str(i)+'*.jpg')),str(os.path.join(r,'w_label_'+str(i)+'.gif')),delay=10)
        print(str(i)+"===>> done")


parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='CONDVAE6')
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-data', type=str, default='fmnist')
parser.add_argument('-arch', type=str, default='vanilla')
parser.add_argument('-arg', type=int, default=8)
parser.add_argument('-diff_priors', type=str, default='no')
args = parser.parse_args()

opts = {
    'data': {
        'batch_size': 1,
        'data': args.data,
        'cuda': True,
        'shuffle': False,
        'arg': args.arg,
        'diff_priors':args.diff_priors,
        'model':args.model,
    },
    'train': {
        'epochs': args.epochs,
    },
    'log': {
        'log_interval': 100,
        'out_fields': [ "lt", "l1","l2","pxIzw","kly","klz","klw","Iyz"],
    }
}

if args.arch == 'vanilla':
    opts['model_opts'] = {
        'alpha': [1, 0.1 * 10, 0.1 * 10, 1, 10, 1],
        'arch': 'vanilla',
        'enc_filters': [64, 128, 256],
        'dec_filters': [256, 256, 128, 32],
        'enc_fc': [256 * 8 * 8, 2048 * 2],
        'dec_fc': [2048, 8 * 8 * 256],
        'w_dim': 10,
        'beta': [1, 1],
        'learning_rate1': (1e-3)/2,
        'learning_rate2': (1e-3)/2
    }
elif args.arch == 'fullconv':
    opts['model_opts'] = {
        'alpha': [20, 0.1 * 20, 0.1 * 20, 1, 10, 1],
        'arch': 'fullconv',
        'enc_filters': [32, 64, 128, 256, 512, 512],
        'dec_filters': [512, 512, 256, 128, 64, 32],
        'w_dim': 2,
        'beta': [1, 1],
        'learning_rate1': (1e-3)/2,
        'learning_rate2': (1e-3)/2
    }

opts['train_losses'] = {key: [] for key in opts['log']['out_fields']}
opts['test_losses'] = {key: [] for key in opts['log']['out_fields']}

loaders = load_data(args.data, opts)
opts['model_opts']['k'] = int(10)



opts_to_save = copy.copy(opts)

opts['train_loader'] = torch.utils.data.DataLoader(
                            torchvision.datasets.FashionMNIST('./data/fmnist/', train=True, download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])),
        batch_size=1, shuffle=True)
opts['test_loader']  = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data/fmnist/', train=False, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.5,), (0.5,))])),batch_size=1, shuffle=False)
torch.manual_seed(1)
if opts['data']['cuda']: 
    torch.cuda.manual_seed(1)

opts['model'] = fully_supervised_CONDVAE()
traversal_visualize(opts)





