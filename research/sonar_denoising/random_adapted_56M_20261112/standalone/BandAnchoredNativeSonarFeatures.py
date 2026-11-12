"""Observed phase increments relative to a FIXED recorded frequency reference.

Keeps the previous finite amplitude-contrast channel exactly. Unlike the
local-carrier-invariant family, a linear frequency shift is informative here,
not removed using an observed (possibly interference-dominated) local mean.
The reference comes from a recorded processing band, not a clean-phase label
or an asserted calibration of the unknown upstream processing chain.
"""
import numpy as np

from LocalRelativeSonarFeatures import local_features


PROTOCOL='finite_log_contrast_fixed_recorded_band_phase_increment_v1'
REFERENCE_FS=37650.
REFERENCE_BAND=(-15000.,-3000.)


def validate_reference(fs,band):
    b=np.asarray(band,np.float64)
    if (not np.isfinite(fs) or fs<=0 or b.shape!=(2,) or not np.isfinite(b).all()
            or not -fs/2<b[0]<b[1]<fs/2):
        raise ValueError('recorded finite subband within the native Nyquist interval required')
    return float(np.mean(b))


def band_features(values,*,first_column=0,sampling_frequency=REFERENCE_FS,reference_subband=REFERENCE_BAND):
    center=validate_reference(sampling_frequency,reference_subband)
    z=np.asarray(values)
    # This intentionally uses the frozen amplitude builder, including its
    # exact log/float precision and support. Only channels one and two change.
    features,logs,mask=local_features(z,first_column=first_column)
    amplitude=np.abs(z);positive=amplitude>0
    unit=np.divide(z,amplitude,out=np.zeros_like(z),where=positive)
    gradient=unit[:,1:]*unit[:,:-1].conj()
    valid=positive[:,1:]&positive[:,:-1]
    if not first_column:
        gradient=np.pad(gradient,((0,0),(1,0)))
        valid=np.pad(valid,((0,0),(1,0)))
    relative=np.where(valid,gradient,0)*np.exp(-2j*np.pi*center/sampling_frequency)
    features[1]=relative.real.astype(np.float32)
    features[2]=relative.imag.astype(np.float32)
    return features,logs,mask
