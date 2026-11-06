"""A separate envelope-only, global-unit-equivariant wavelet experiment.

A single scalar log-amplitude origin is removed from features, not from the
residual base. No per-ping or per-range normalization changes striping. Full
native inference must supply one whole-beam origin unchanged for every tile.
"""
from __future__ import annotations
import torch
from torch import nn
from PublicSonarWavelet import PublicSonarWavelet


class PublicSonarEnvelopeModel(nn.Module):
    family = "global_log_origin_equivariant_public_envelope_v1"

    def __init__(self, width=16):
        super().__init__()
        self.network = PublicSonarWavelet(width=width)

    def residual(self, log_amplitude, support, *, origin=None):
        if log_amplitude.ndim != 4 or log_amplitude.shape[1] != 1 or support.shape != log_amplitude.shape:
            raise ValueError("native B1HW log amplitudes and matching positive support required")
        if not torch.isfinite(log_amplitude).all():
            raise ValueError("nonfinite log envelope")
        mask = support.to(log_amplitude.dtype)
        if origin is None:
            origin = (log_amplitude*mask).sum((-2,-1),keepdim=True)/mask.sum((-2,-1),keepdim=True).clamp_min(1)
        if origin.shape != (len(log_amplitude),1,1,1) or not torch.isfinite(origin).all():
            raise ValueError("one explicit finite global log origin per observation required")
        centered = (log_amplitude-origin)*mask
        features = centered.repeat(1,3,1,1)
        prediction = self.network(features,1,base=torch.zeros_like(features))[:,0:1]
        return prediction*mask

    def forward(self, log_amplitude, support, *, origin=None):
        return log_amplitude+self.residual(log_amplitude,support,origin=origin)
