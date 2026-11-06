"""Exact original mosaic layout and explicitly separated observed-water mask."""
import numpy as np


def supported_total_delta(amplitude,residual,limits,columns,sound_speed,sampling_frequency):
    """Same final-only observed-water support as previous real diagnostics.

    This is NOT a clean training label, learned bottom estimator or gain floor.
    The network itself remains the original trained raw/front composition.
    """
    a,residual = np.asarray(amplitude),np.asarray(residual)
    bounds = np.asarray(limits,np.float64)
    if a.shape != residual.shape or bounds.shape != (len(a),2):raise ValueError('aligned native bounds required')
    if not np.isfinite(bounds).all() or (bounds[:,0]<=0).any() or (bounds[:,1]<=bounds[:,0]).any():
        raise ValueError('reviewed original observed range limits required')
    ranges = (np.asarray(columns)+1)*sound_speed/(2*sampling_frequency)
    u = np.clip((ranges[None]-bounds[:,:1])/(bounds[:,1:]-bounds[:,:1]),0,1)
    support = u*u*(3-2*u)
    result = (residual*support).astype(np.float32);result[a<=1e-12] = 0
    if not np.isfinite(result).all():raise ValueError('nonfinite supported total log gain')
    return result


def overlap_block(amplitudes,previous=None):
    """Both sides × physical record × eight simultaneous beams × range."""
    a = np.asarray(amplitudes)
    if a.ndim!=4 or a.shape[0]!=2 or a.shape[2]!=8 or a.shape[1]<1:raise ValueError('complete native layout required')
    previous = np.zeros((2,4,a.shape[-1]),a.dtype) if previous is None else np.asarray(previous)
    if previous.shape!=(2,4,a.shape[-1]):raise ValueError('actual previous four-beam tail required')
    rows = a[:,:,:4].copy();rows[:,0] += previous;rows[:,1:] += a[:,:-1,4:]
    sides = rows.reshape(2,4*a.shape[1],a.shape[-1])
    return np.concatenate((sides[0,:,::-1],sides[1]),axis=1),a[:,-1,4:].copy()


def overlap_tail(previous):
    a = np.asarray(previous)
    if a.ndim!=3 or a.shape[:2]!=(2,4):raise ValueError('native four-beam tail required')
    return np.concatenate((a[0,:,::-1],a[1]),axis=1)
