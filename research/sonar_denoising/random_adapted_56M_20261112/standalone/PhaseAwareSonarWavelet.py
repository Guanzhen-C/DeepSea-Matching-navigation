"""Scalar amplitude restoration with optional gauge-invariant phase context.

No phase is predicted or called clean. A former envelope model is exactly
reparameterized at initialization: input channel zero receives the sum of
the three old amplitude kernels; phase input kernels start at zero. Thus
phase and amplitude-only conditions have the same initial function.

Compute phase features on the COMPLETE native source before taking tiles.
The left neighbour of an interior tile must never be replaced by zero.
"""
from __future__ import annotations

import torch

from PublicEnvelopeCoherentModel import PublicEnvelopeCoherentModel


class PhaseAwareSonarWavelet(PublicEnvelopeCoherentModel):
    family='native_complex_phase_gradient_scalar_wavelet_consensus_v1'

    @classmethod
    def from_envelope(cls,state):
        if state.get('model_family')!=PublicEnvelopeCoherentModel.family:
            raise ValueError('explicit reviewed scalar directional checkpoint required')
        model=cls(width=state['width'])
        model.load_state_dict(state['model'],strict=True)
        with torch.no_grad():
            weight=model.network.input.weight
            amplitude_weight=weight.sum(dim=1,keepdim=True)
            weight.zero_();weight[:,0:1].copy_(amplitude_weight)
        for index in (0,2):model.network.heads[index].requires_grad_(False)
        return model

    def feature_residual(self,features,support,*,use_phase):
        if features.ndim!=4 or features.shape[1]!=3 or support.shape!=features[:,0:1].shape:
            raise ValueError('full-source-derived native three-channel features and support required')
        if not torch.isfinite(features).all():raise ValueError('nonfinite phase/amplitude features')
        if not isinstance(use_phase,bool):raise ValueError('explicit phase condition required')
        if not use_phase:
            features=torch.cat((features[:,0:1],torch.zeros_like(features[:,1:])),dim=1)
        prediction=self.network(features,1,base=torch.zeros_like(features))[:,0:1]
        return self.consensus(prediction*support,support)

    def residual(self,*args,**kwargs):
        raise ValueError('complex-feature model requires full-source feature_residual, not scalar envelope adapter')

