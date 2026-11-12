"""Finite-support directional consensus of a learned wavelet noise field.

Only the PREDICTED log-gain field is aggregated, not measured amplitudes,
phase or seabed coordinates. This is our sonar adaptation experiment, not
an ASCNet reproduction. ASCNet motivates directional long-range context:
https://arxiv.org/abs/2401.15578 . No upstream code or weights are copied.
"""
from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import functional as F

from PublicSonarEnvelopeModel import PublicSonarEnvelopeModel


class DirectionalNoiseConsensus(nn.Module):
    """Mask-normalized learned mixture of finite oriented noise-field means.

    A sparse kernel samples seven positions across 193 native range bins.
    Fractional record offsets interpolate only the estimated NOISE field.
    Orientation bins cover our augmentation family, not a sensor calibration.
    No whole-image pooling or unbounded inference dependence is introduced.
    """
    slopes=(-.04,-.02,0.,.02,.04)
    offsets=(-96,-64,-32,0,32,64,96)
    radius=96

    def __init__(self):
        super().__init__()
        self.row_radius=math.ceil(max(abs(s*x) for s in self.slopes for x in self.offsets))
        kernels=torch.zeros((len(self.slopes),1,2*self.row_radius+1,2*self.radius+1))
        for direction,slope in enumerate(self.slopes):
            for x in self.offsets:
                shift=slope*x;lower=math.floor(shift);fraction=shift-lower
                kernels[direction,0,self.row_radius+lower,self.radius+x]+=(1-fraction)/len(self.offsets)
                if fraction:
                    kernels[direction,0,self.row_radius+lower+1,self.radius+x]+=fraction/len(self.offsets)
        self.register_buffer('kernels',kernels)
        count=len(self.slopes)
        self.selector=nn.Sequential(nn.Conv2d(2*count,8,1),nn.GELU(),nn.Conv2d(8,count,1))
        nn.init.zeros_(self.selector[-1].weight);nn.init.zeros_(self.selector[-1].bias)

    def forward(self,estimated_delta,support):
        if estimated_delta.ndim!=4 or estimated_delta.shape[1]!=1 or estimated_delta.shape!=support.shape:
            raise ValueError('aligned native B1HW estimated noise and observation support required')
        mask=support.to(estimated_delta.dtype)
        padding=(self.row_radius,self.radius)
        # Group the moments in one convolution call. The kernels have unit
        # sum; dividing by observed weight avoids boundary/zero attenuation.
        fields=torch.cat((estimated_delta*mask,estimated_delta.square()*mask,mask),dim=1)
        moments=F.conv2d(fields,self.kernels.repeat(3,1,1,1),padding=padding,groups=3)
        first,second,weight=moments.chunk(3,dim=1)
        denominator=weight.clamp_min(torch.finfo(estimated_delta.dtype).eps)
        mean=first/denominator
        variance=(second/denominator-mean.square()).clamp_min(0)
        mixture=torch.softmax(self.selector(torch.cat((mean,variance),dim=1)),dim=1)
        return (mean*mixture).sum(dim=1,keepdim=True)*mask


class PublicEnvelopeCoherentModel(PublicSonarEnvelopeModel):
    family='global_origin_wavelet_directional_noise_consensus_v1'
    required_halo=160

    def __init__(self,width=16):
        super().__init__(width=width)
        self.consensus=DirectionalNoiseConsensus()

    def residual(self,log_amplitude,support,*,origin=None):
        estimated=super().residual(log_amplitude,support,origin=origin)
        return self.consensus(estimated,support)


def envelope_model_from_state(state):
    """Strictly load an explicitly tagged model; never reinterpret tensors."""
    family=state.get('model_family')
    if family==PublicSonarEnvelopeModel.family:
        model=PublicSonarEnvelopeModel(width=state['width'])
    elif family==PublicEnvelopeCoherentModel.family:
        model=PublicEnvelopeCoherentModel(width=state['width'])
    else:
        raise ValueError('unreviewed envelope model family')
    model.load_state_dict(state['model'],strict=True)
    return model
