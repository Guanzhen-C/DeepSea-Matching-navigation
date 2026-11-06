"""Streaming native complex-field descriptors, not noise estimators.

Unit-gradient coherence and energy-weighted range covariance are different
observed statistics. Neither is a calibrated SNR or a cleanliness label.
Each input pixel contributes, without image resizing or selection by output.
"""
from __future__ import annotations
import numpy as np


class ComplexSourceStatistics:
    def __init__(self,beams):
        if beams<1:raise ValueError('positive native beam count required')
        self.beams=beams;self.records=0;self.bins=None
        self.count=np.zeros(beams,np.int64);self.positive=np.zeros(beams,np.int64)
        self.near_zero=np.zeros(beams,np.int64);self.imag_nonzero=np.zeros(beams,np.int64);self.imag_available=False
        self.scale=np.ones(beams);self.power=np.zeros(beams);self.fourth=np.zeros(beams)
        self.log_sum=np.zeros(beams);self.log_square_sum=np.zeros(beams)
        self.pairs=np.zeros(beams,np.int64);self.triplets=np.zeros(beams,np.int64)
        self.phase_sum=np.zeros(beams,np.complex128);self.curvature_sum=np.zeros(beams,np.complex128)
        self.weighted_pair_sum=np.zeros(beams,np.complex128)
        self.left_power=np.zeros(beams);self.right_power=np.zeros(beams)

    def update_complex(self,values):
        z=np.asarray(values)
        if z.ndim==2:z=z[:,None,:]
        if z.ndim!=3 or z.shape[0]!=self.beams or not np.iscomplexobj(z) or not np.isfinite(z).all():
            raise ValueError('finite native beam/record/range complex observations required')
        a=np.abs(z);unit=np.divide(z,np.where(a>0,a,1.))
        gradient=unit[...,1:]*unit[...,:-1].conj()
        self.update_amplitude_gradient(a,gradient,imaginary_nonzero=(z.imag!=0).sum((1,2)))

    def update_amplitude_gradient(self,amplitudes,gradient,*,imaginary_nonzero=None):
        a=np.asarray(amplitudes);g=np.asarray(gradient)
        if a.ndim!=3 or a.shape[0]!=self.beams or g.shape!=(*a.shape[:-1],a.shape[-1]-1) or not np.iscomplexobj(g):
            raise ValueError('aligned native amplitude and adjacent unit-phase gradients required')
        if a.shape[-1]<3 or not np.isfinite(a).all() or (a<0).any() or not np.isfinite(g).all():
            raise ValueError('finite complete nonnegative native observations required')
        if self.bins is None:self.bins=a.shape[-1]
        if self.bins!=a.shape[-1]:raise ValueError('range grid changed inside a source')
        self.records+=a.shape[1];self.count+=a.shape[1]*a.shape[2]
        support=a>0;valid=support[...,1:]&support[...,:-1]
        triplet=support[...,2:]&support[...,1:-1]&support[...,:-2]
        self.positive+=support.sum((1,2));self.near_zero+=(a<=1e-12).sum((1,2))
        self.pairs+=valid.sum((1,2));self.triplets+=triplet.sum((1,2))
        self.phase_sum+=np.where(valid,g,0).sum((1,2),dtype=np.complex128)
        self.curvature_sum+=np.where(triplet,g[...,1:]*g[...,:-1].conj(),0).sum((1,2),dtype=np.complex128)
        if imaginary_nonzero is not None:
            self.imag_nonzero+=imaginary_nonzero;self.imag_available=True
        logs=np.zeros(a.shape,np.float64);np.log(a,out=logs,where=support)
        self.log_sum+=logs.sum((1,2));self.log_square_sum+=np.square(logs).sum((1,2))
        scale=np.maximum(self.scale,a.max((1,2)))
        ratio=self.scale/scale
        self.power*=ratio**2;self.fourth*=ratio**4
        self.weighted_pair_sum*=ratio**2;self.left_power*=ratio**2;self.right_power*=ratio**2
        normalized=a/scale[:,None,None];power=normalized**2
        self.power+=power.sum((1,2));self.fourth+=(power**2).sum((1,2))
        self.weighted_pair_sum+=(normalized[...,1:]*normalized[...,:-1]*g).sum((1,2),dtype=np.complex128)
        self.left_power+=power[...,:-1].sum((1,2));self.right_power+=power[...,1:].sum((1,2));self.scale=scale

    def result(self):
        rows=[]
        for b in range(self.beams):
            count=int(self.count[b]);positive=int(self.positive[b])
            if not count or not positive:raise ValueError('no original positive observations in source')
            mean=self.phase_sum[b]/max(int(self.pairs[b]),1)
            curvature=self.curvature_sum[b]/max(int(self.triplets[b]),1)
            denominator=np.sqrt(self.left_power[b]*self.right_power[b])
            covariance=self.weighted_pair_sum[b]/denominator if denominator else 0j
            log_mean=self.log_sum[b]/positive
            rows.append(dict(beam_1based=b+1,native_values=count,positive_values=positive,
                numeric_near_zero_fraction=float(self.near_zero[b]/count),
                imaginary_nonzero_fraction=float(self.imag_nonzero[b]/count) if self.imag_available else None,
                mean_positive_log_amplitude=float(log_mean),
                std_positive_log_amplitude=float(np.sqrt(max(0.,self.log_square_sum[b]/positive-log_mean**2))),
                RMS_amplitude=float(self.scale[b]*np.sqrt(self.power[b]/count)),
                amplitude_normalized_fourth_moment=float(count*self.fourth[b]/self.power[b]**2),
                adjacent_phase_pairs=int(self.pairs[b]),adjacent_unit_phase_mean=[float(mean.real),float(mean.imag)],
                adjacent_unit_phase_coherence=float(abs(mean)),
                adjacent_energy_weighted_covariance=[float(covariance.real),float(covariance.imag)],
                adjacent_energy_weighted_covariance_magnitude=float(abs(covariance)),
                phase_curvature_coherence=float(abs(curvature))))
        return dict(native_records=self.records,native_range_bins=self.bins,beams=rows,
            native_values=sum(r['native_values'] for r in rows),
            descriptor_is_noise_SNR=False,clean_reference_claimed=False)

