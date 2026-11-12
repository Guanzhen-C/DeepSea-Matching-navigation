"""Finite observed amplitude/phase context, never a filtered target image.

The local means define FEATURES only. The predicted log-amplitude residual is
added to the original log amplitude. No clean phase is predicted, no coherence
is interpreted as SNR, and zero local coherence explicitly loses phase context.
Unlike the frozen v8 feature family, this protocol has no voyage-global origin.
"""
import numpy as np
from scipy.ndimage import uniform_filter


PROTOCOL = 'finite_5x65_observed_log_contrast_and_relative_phase_v1'
WINDOW = (5, 65)
FEATURE_RADIUS = (2, 32)


def local_mean(values, support):
    values = np.asarray(values, np.float64)
    mask = np.asarray(support, bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError('aligned native 2-D values and support required')
    weight = uniform_filter(mask.astype(np.float64), WINDOW, mode='constant', cval=0.)
    numerator = uniform_filter(np.where(mask, values, 0.), WINDOW, mode='constant', cval=0.)
    result = np.zeros_like(numerator)
    np.divide(numerator, weight, out=result, where=weight > 0)
    return result


def local_features(values, *, first_column=0):
    """Interior inputs include their actual predecessor as one extra column.

    The supplied context's outer means have truncated support. Inference must
    either compute features on the full source, or discard the model's declared
    real halo around a cropped context. There is no synthetic neighbor padding.
    """
    z = np.asarray(values)
    if (z.ndim != 2 or not np.iscomplexobj(z) or not np.isfinite(z).all()
            or first_column < 0 or min(z.shape) < 2):
        raise ValueError('finite native complex observations and real predecessor required')
    a = np.abs(z)
    positive = a > 0
    unit = np.divide(z, a, out=np.zeros_like(z), where=positive)
    g = unit[:, 1:] * unit[:, :-1].conj()
    pairs = positive[:, 1:] & positive[:, :-1]
    if first_column:
        a = a[:, 1:]; mask = positive[:, 1:]
    else:
        mask = positive
        g = np.pad(g, ((0, 0), (1, 0)))
        pairs = np.pad(pairs, ((0, 0), (1, 0)))
    g = np.where(pairs, g, 0)
    logged = np.zeros(a.shape, np.float64)
    np.log(a, out=logged, where=mask)
    mean = local_mean(g.real, pairs) + 1j * local_mean(g.imag, pairs)
    relative = g * mean.conj()
    features = np.stack(((logged - local_mean(logged, mask)) * mask,
                         relative.real, relative.imag)).astype(np.float32)
    return features, logged.astype(np.float32), mask
