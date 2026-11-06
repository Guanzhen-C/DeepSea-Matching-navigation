"""Our domain-conditioned wavelet residual model for public sonar data.

Not a reproduction of WTCRNet/TIDY and not pretrained sonar weights. Display
RGB, envelope log amplitude, and complex-field log amplitude use distinct
output heads. There is no global image pooling or geometric resampling.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def haar(x):
    if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar input must have even native dimensions")
    a,b,c,d=x[...,0::2,0::2],x[...,0::2,1::2],x[...,1::2,0::2],x[...,1::2,1::2]
    return torch.cat(((a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),dim=1)


def inverse_haar(x):
    if x.ndim != 4 or x.shape[1] % 4:
        raise ValueError("four corresponding Haar subbands required")
    ll,lh,hl,hh=x.chunk(4,dim=1)
    out=x.new_empty((x.shape[0],x.shape[1]//4,2*x.shape[-2],2*x.shape[-1]))
    out[...,0::2,0::2]=(ll+lh+hl+hh)/2
    out[...,0::2,1::2]=(ll-lh+hl-hh)/2
    out[...,1::2,0::2]=(ll+lh-hl-hh)/2
    out[...,1::2,1::2]=(ll-lh-hl+hh)/2
    return out


class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):
        return x+self.body(x)


class PublicSonarWavelet(nn.Module):
    domains=("display_luminance","envelope","complex_field")
    def __init__(self,width=16):
        super().__init__()
        self.width=width
        self.input=nn.Conv2d(3,width,3,padding=1)
        self.condition=nn.Embedding(3,width*2)
        nn.init.zeros_(self.condition.weight)
        self.level0=ResidualBlock(width)
        self.down1=nn.Conv2d(width*4,width*2,1)
        self.level1=nn.Sequential(ResidualBlock(width*2),ResidualBlock(width*2))
        self.down2=nn.Conv2d(width*8,width*4,1)
        self.middle=nn.Sequential(ResidualBlock(width*4),ResidualBlock(width*4))
        self.up1=nn.Conv2d(width*4,width*8,1)
        self.decode1=ResidualBlock(width*2)
        self.up0=nn.Conv2d(width*2,width*4,1)
        self.decode0=ResidualBlock(width)
        self.heads=nn.ModuleList((nn.Conv2d(width,3,3,padding=1),nn.Conv2d(width,1,3,padding=1),nn.Conv2d(width,1,3,padding=1)))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self,features,domain,*,base=None):
        if domain not in range(3) or features.ndim != 4 or features.shape[1] != 3:
            raise ValueError("reviewed scalar domain and native 3CHW features required")
        if base is None:
            if domain == 2:
                raise ValueError("complex features require an explicit log-amplitude base")
            base=features
        if base.shape != features.shape:
            raise ValueError("feature and residual-base native grids differ")
        h,w=features.shape[-2:]
        # Boundary padding only. Every measured output pixel is retained.
        features=F.pad(features,(0,(-w)%4,0,(-h)%4),mode="replicate")
        gain,bias=self.condition.weight[domain].chunk(2)
        x=self.input(features)*(1+gain[None,:,None,None])+bias[None,:,None,None]
        e0=self.level0(x)
        e1=self.level1(self.down1(haar(e0)))
        middle=self.middle(self.down2(haar(e1)))
        d1=self.decode1(inverse_haar(self.up1(middle))+e1)
        d0=self.decode0(inverse_haar(self.up0(d1))+e0)
        residual=self.heads[domain](d0)[...,:h,:w]
        if domain:
            residual=residual.repeat(1,3,1,1)
        return base+residual
