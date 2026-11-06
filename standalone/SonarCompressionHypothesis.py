"""Hypothesis-only post-beamforming LFM correlation, never a production fix.

MatchFilterCorrelationFlag semantics are not supplied. A different flag could
select different algorithms, not disable filtering. Near-field beamforming
may not commute with pulse compression; raw ADC and upstream code are absent.
"""
import numpy as np
from scipy import signal


def lfm_template(fs=37650.,duration=.005,slope=-6000000.,center_frequency=-1800.):
    n=int(round(fs*duration))
    if n<2 or abs(center_frequency)+abs(slope)*duration/2>=fs/2:raise ValueError("invalid Nyquist-contained experimental LFM template")
    t=(np.arange(n)-(n-1)/2)/fs
    return np.exp(2j*np.pi*(center_frequency*t+.5*slope*t*t))


def correlate_range(complex_field,template):
    z=np.asarray(complex_field);s=np.asarray(template)
    if z.ndim!=3 or s.ndim!=1 or not np.isfinite(z).all():raise ValueError("native complex cube and template required")
    kernel=np.conj(s[::-1])/np.linalg.norm(s)
    result=signal.fftconvolve(z,kernel[None,None,:],mode="same",axes=-1)
    energy=signal.fftconvolve(abs(z)**2,np.ones((1,1,len(s))),mode="same",axes=-1)
    coherence=np.divide(abs(result)**2,energy,out=np.zeros_like(energy),where=energy>np.finfo(float).tiny)
    return result,coherence
