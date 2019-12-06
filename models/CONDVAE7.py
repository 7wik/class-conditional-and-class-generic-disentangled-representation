from __future__ import print_function
import copy
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.components.blocks import conv_block, fc_block
from models.components.decoder import Decoder
from models.utils.distributions import Normal
from models.utils.ops import Reshape


class CONDVAE7(nn.Module):
    def __init__(self, opts):
        super(CONDVAE7, self).__init__()
        self.w_dim = opts["model_opts"]["w_dim"]
        self.alpha = opts["model_opts"]["alpha"]
        self.beta = opts["model_opts"]["beta"]
        self.k = opts["model_opts"]["k"]
        self.priors = opts["data"]['diff_priors']

        self.encoder = Encoder(
            [opts["data"]["x_dim"][0]] + opts["model_opts"]["enc_filters"],
            opts["model_opts"]["enc_fc"],
            self.w_dim,
            self.k,
        )

        self.encoder2 = Encoder2(
            opts["model_opts"]["enc_filters"],
            opts["model_opts"]["enc_fc"],
            self.w_dim,
            self.k,
        )

        self.encoder3 = Encoder3(
            z_size=self.k,
            )


        dec_fc = copy.copy(opts["model_opts"]["dec_fc"])
        dec_fc.insert(0, self.w_dim +self.k)
        self.decoder = Decoder(
            opts["model_opts"]["dec_filters"], dec_fc, opts["data"]["x_dim"][0]
        )

        


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
        y_0 = self.encode2(x.view(-1,64*64))

        pyIx = F.softmax(y_0,dim=1)

        pxIwz = [] 
        for i in range(self.k):
            pxIwz.append(self.decode(torch.cat([z, (w[:,i,:])], 1)))
        

        return pxIwz, pzIx, pwIxy,pyIx, z, w, y_0

    def sample(self, batch_size):
        z = Variable(torch.randn(batch_size, self.k))
        w = Variable(torch.randn(batch_size, self.w_dim ))
        ind = torch.sort(torch.norm(w, 2, 1))[1]
        z = z[ind]
        w = w[ind]

        if next(self.parameters()).is_cuda:
            z = z.cuda()
            w = w.cuda()

        return self.decode(torch.cat([z, w], 1))
    
    def gumbel_softmax(self, logits, temperature=1.0, eps=1e-9):

        noise = torch.rand(logits.size())
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        noise = Variable(noise)
        noise = noise.cuda()

        x = (logits + noise) / temperature
        x = F.softmax(x, dim=1)
        return x


    def loss(self, data):
        """Computes the loss of a data sample.

        Args:
            data (list): The input data sample.
                The data sample consists of four elements:
                    1. (float, BCHW) Input image.
                    2. (float, B1) Categorical image label)

        Returns:
            tuple: Statistics pertaining to the data sample.
        """
        x = data[0]

        # ones and zeros for discriminator
        ones = torch.ones(int(x.size()[0]),dtype=torch.long).cuda()
        zeros = torch.zeros(int(x.size()[0]),dtype=torch.long).cuda()

        # recon_x (List): size (k, (B, C, H, W))
        # pzIx (Tensor): size (B, Z)
        # pwIxy (List): size ( k ,(B, 2* w_dim))
        # y_0  (Tensor) : size(B,k)--> logits
        recon_x, pzIx, pwIxy,pyIx, z, w, y_0 = self.forward(x)
        x = x.view(x.size(0), -1)
        a=[]
        pxIzw = 0 
        """loss 1"""
        for i in range(len(recon_x)):
            pxIzw+= -0.5*(((recon_x[i].view(x.size(0), -1)-x)**2).sum(-1) + np.log(2*np.pi))*pyIx[:,i].view(-1,1)
        pxIzw = -torch.mean(pxIzw)

        klz = pzIx.kl_div().mean()

        # computing kl div with pwIy
        klw = 0

        target =[ Normal(
            Variable(torch.cuda.FloatTensor([0.5 * i])),
            Variable(torch.cuda.FloatTensor([np.log(np.sqrt(0.4))])),
        ) for i in range(self.k)]


        for i in range(len(pwIxy)):
            pwiIxy = pwIxy[i]
            pyiIx  = pyIx[:,i]
            if self.priors=="yes":
                klw += (pwiIxy.kl_div_from(target[i]))*pyiIx
            else:
                klw += (pwiIxy.kl_div())*pyiIx
        klw = (klw).mean()

        # computing kl div for y
        kly = ((pyIx*(torch.log(pyIx)-np.log(1/float(self.k)))).sum(1)).mean()
        labs=torch.eye(self.k)
        yIz = self.encode3(z)
        pyIz=  F.softmax(yIz,dim=-1)        #(torch.stack([self.discriminate(((Variable(labs[i, :].repeat(x.size(0), 1)).cuda()), 1)) for i in range(self.k)],1)[:,:,self.k:])
        log_pyIz = F.log_softmax(yIz,dim=-1)
        Hy  =  ((F.softmax(y_0)*F.log_softmax(y_0)).sum(-1)).mean()
        HyIz=  ((pyIz*log_pyIz).sum(-1)).mean(0)             #(((pzIy.sum(-1))*pyIx).sum(-1)).mean(0)
        Iyz =  HyIz-Hy
        y_onehot = (self.gumbel_softmax(y_0))

        l1 = self.alpha[0] * pxIzw + self.alpha[1] * klz + self.alpha[3] * klw + self.alpha[4]*kly 
        lt = l1 + self.alpha[5] * Iyz

        """loss 2, cross entropy (y | y_0)"""
        l2 = ((y_onehot*(log_pyIz)+(1-y_onehot)*torch.log(1-F.softmax(yIz,dim=-1)+1e-12)).sum(-1)).mean(0)
        l2 = self.alpha[5] * torch.mean(l2)       
        indices = torch.max(pyIx,1)[1].cpu().numpy()
        all_mu = []
        all_var= []
        mu=[]
        log_var=[]
        for i,pwiIxy in enumerate(pwIxy):
            all_mu.append(pwiIxy.mu)
            all_var.append(pwiIxy.logvar)
        all_mu = torch.stack(all_mu,1)
        all_var= torch.stack(all_var,1) 
        recon_x= torch.stack(recon_x,1)
        return_x= []
        for i in range(x.size()[0]):
            mu.append(all_mu[i][indices[i]])
            return_x.append(recon_x[i][indices[i]])
            log_var.append(all_var[i][indices[i]])
        
        return (
            {
                "mu":torch.stack(mu,0),
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
                "kly":kly,
                "klz": klz,
                "klw": klw,
                "HyIz": HyIz,
            },
        )

    def change(self, target_subspace, target_val, x, y):
        # decodes by changing the ground truth label y
        # target_subspace: integer indicating which subspace to manipulate
        # target_val: the target value in the target subspace
        # x: input image
        # y: onehot ground truth labels
        pzIx, pwIxy = self.encode(x)
        z_mu = pzIx.mu
        w_mu = pwIxy.mu.data  # note use of data

        y_vec = torch.max(y, 1)[1].data

        # zero out ground truth subspaces
        col_inds = torch.arange(0, w_mu.size(1)).unsqueeze(0).long()
        if next(self.parameters()).is_cuda:
            col_inds = col_inds.cuda()

        # mask which elements of w correspond to ground truth subspaces
        mask = (col_inds >= y_vec.unsqueeze(1) * self.w_dim) * (
            col_inds < (y_vec.unsqueeze(1) + 1) * self.w_dim
        )

        w_mu.masked_fill_(mask, 0.0)  # perform the zeroing

        if target_subspace < self.k:
            # set the target_subspace to be the target_val
            w_mu[
                :, target_subspace * self.w_dim : (target_subspace + 1) * self.w_dim
            ] = target_val.data.expand(x.size(0), self.w_dim)

        w_mu = Variable(w_mu, volatile=True)

        return self.decode(torch.cat([z_mu, w_mu], 1))


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
        modules.append(fc_block(fc[-1], 2*self.k,activation=False))
        self.net2 = torch.nn.Sequential(*modules)

        self.net3 = torch.nn.Sequential(
            fc_block(fc[0], 2048), 
            fc_block(2048, 1024),
            fc_block(1024, 128),
        )

        self.net4 = torch.nn.Sequential(
            fc_block(128+k,2*w_dim,activation=False),
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



class Encoder2(nn.Module):
    #For q(y|x)
    def __init__(self, filters, fc, w_dim, k):
        super(Encoder2, self).__init__()
        self.net = torch.nn.Sequential(
            fc_block(int(fc[-1]), 1024),
            fc_block(1024, 512),
            fc_block(512, 128),
            fc_block(128, k, activation=False),
        )

    def forward(self, z):
        return self.net.forward(z)

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