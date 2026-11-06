"""Full-source carrier-reference-invariant observed phase features.

Let g be a native adjacent unit phase gradient, with valid mean mu over the
WHOLE observed source. We use g*conj(mu), not g in a fixed complex frame.
Under z[x] -> z[x]*exp(i*alpha*x), g and mu rotate equally; their product is
unchanged. No division by small coherence, threshold, clean phase estimate,
range resampling or original observation modification is involved.

The feature radius is weighted by OBSERVED source coherence. Coherence is
not a calibrated noise confidence/SNR, and zero mean loses this feature's
phase information. A tile must receive the same source mean as every other
tile; estimating it separately per tile changes the estimator.
"""
from __future__ import annotations

import math
import numpy as np


PROTOCOL='whole_source_coherence_weighted_carrier_relative_gradient_v1'


def carrier_relative_features(features,support,*,source_mean=None,complete_source=False):
    f=np.asarray(features);mask=np.asarray(support)
    if f.ndim!=3 or f.shape[0]!=3 or mask.shape!=(1,*f.shape[-2:]) or mask.dtype!=np.bool_:
        raise ValueError('native three-channel features and explicit boolean observation support required')
    if f.dtype!=np.float32 or not np.isfinite(f).all():raise ValueError('finite native float32 features required')
    if source_mean is None:
        if not complete_source:raise ValueError('complete-source mean or explicit complete-source declaration required')
        valid=np.zeros(mask.shape[-2:],bool)
        valid[:,1:]=mask[0,:,1:]&mask[0,:,:-1]
        count=int(valid.sum())
        source_mean=(complex(float(f[1][valid].sum(dtype=np.float64)),float(f[2][valid].sum(dtype=np.float64)))/count if count else 0j)
    else:
        if complete_source:raise ValueError('do not mix a supplied mean with a new whole-source estimate')
        source_mean=complex(source_mean);count=None
    if not math.isfinite(source_mean.real) or not math.isfinite(source_mean.imag) or abs(source_mean)>1.00001:
        raise ValueError('finite observed mean of unit phase gradients required')
    result=f.copy()
    result[1]=f[1]*source_mean.real+f[2]*source_mean.imag
    result[2]=f[2]*source_mean.real-f[1]*source_mean.imag
    # Numerical masks were already formed with actual left neighbours in
    # the full source. Never overwrite a valid first column of an inner tile.
    result[1:]*=mask
    if not np.array_equal(result[0],f[0]):raise ValueError('amplitude feature changed under carrier-reference transform')
    return result,dict(protocol=PROTOCOL,full_source_phase_gradient_mean=[source_mean.real,source_mean.imag],
        observed_source_coherence=abs(source_mean),valid_full_source_phase_pairs=count,
        original_amplitude_features_unchanged=True,source_coherence_is_not_noise_SNR=True,
        small_coherence_division=False,source_pixels_resized=False)

