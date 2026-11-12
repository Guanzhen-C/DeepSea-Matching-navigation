"""Explicit acquisition-band hypothesis and the SAME train/inference front end.

Only simulated additive noise receives the acquisition lowpass. Targets are
never filtered to make an easier label. The recorded negative-band front end
is a separate factorial condition, not an identification of pure real noise.
Full range coordinates keep artificial crop edges out of protection masks.
"""
import numpy as np

from SonarRecordedSubbandHypothesis import complex_bandpass,centered_filter


FS=37650.
ACQUISITION_BAND=(-15000.,15000.)
REFERENCE_BAND=(-15000.,-3000.)
NOISE_KERNEL=complex_bandpass(FS,ACQUISITION_BAND,taps=129)
FRONT_KERNEL=complex_bandpass(FS,REFERENCE_BAND,taps=257)
CONDITIONS={
    'mandatory_wide_raw':dict(noise='wide',frontend=False,aggregation='mandatory'),
    'fusion_wide_raw':dict(noise='wide',frontend=False,aggregation='learned_fusion'),
    'fusion_limited_raw':dict(noise='limited',frontend=False,aggregation='learned_fusion'),
    'fusion_wide_front':dict(noise='wide',frontend=True,aggregation='learned_fusion'),
    'fusion_limited_front':dict(noise='limited',frontend=True,aggregation='learned_fusion'),
}
HALO=(80,416)
WEAK_TO_STRONG=(.03,.08,.20,.50,1.0,1.25)


def noise_variants(reference,base_delta,scale):
    if not np.iscomplexobj(reference) or not np.iscomplexobj(base_delta):raise ValueError('actual complex observations and perturbations required')
    z=np.asarray(reference,np.complex64);delta=np.asarray(base_delta,np.complex128)
    if (z.ndim!=2 or z.shape!=delta.shape or not np.isfinite(z).all()
            or not np.isfinite(delta).all() or not np.isfinite(scale) or scale<0):
        raise ValueError('aligned native complex observations/delta and finite nonnegative RMS scale required')
    support=abs(z)>0
    if not support.any():raise ValueError('observed native support required')
    source_RMS=float(np.sqrt(np.mean(np.abs(z[support]).astype(np.float64)**2)))
    fields={};diagnostic={}
    for name,field in (('wide',delta),('limited',centered_filter(delta,NOISE_KERNEL))):
        field=np.where(support,field,0)
        RMS=float(np.sqrt(np.mean(np.abs(field[support])**2)))
        if RMS<=0 and scale>0:raise ValueError('nonempty known simulated perturbation required')
        field=field*(scale*source_RMS/RMS) if scale else np.zeros_like(field)
        result=np.asarray(z.astype(np.complex128)+field,np.complex64)
        result=np.where(support,result,z)
        if not np.isfinite(result).all():raise ValueError('known native perturbation overflowed complex64')
        realized=result-z
        fields[name]=result
        diagnostic[name]=dict(actual_added_complex_noise_RMS_over_source_RMS=float(
            np.sqrt(np.mean(np.abs(realized[support]).astype(np.float64)**2))/source_RMS),
            missing_original_support_preserved=bool(np.array_equal(result[~support],z[~support])))
    return fields,diagnostic


def negative_frontend(values,*,first_range_bin=0,full_range_bins=None):
    """Recorded complex band, true-global FIR-edge/zero protection, no depth GT.

    A partial context must include the finite FIR halo. Protection uses global
    range indices, NOT the partial array's artificial edges. This function has
    no invented bottom-depth labels and no amplitude/peak renormalization.
    """
    if not np.iscomplexobj(values):raise ValueError('actual complex observations required, not invented zero phases')
    z=np.asarray(values,np.complex64)
    n=z.shape[-1];full=n if full_range_bins is None else full_range_bins
    if (z.ndim<2 or not np.isfinite(z).all() or first_range_bin<0
            or first_range_bin+n>full or full<=512):
        raise ValueError('actual complete range size and finite complex context required')
    x=np.arange(first_range_bin,first_range_bin+n);radius=len(FRONT_KERNEL)//2
    edge=np.minimum(np.clip((x-radius)/radius,0,1),np.clip((full-1-radius-x)/radius,0,1))
    blend=edge*edge*(3-2*edge)
    # scipy.signal.fftconvolve can transform its complex64 operand in single
    # precision even with a complex128 kernel. Explicit promotion BEFORE the
    # FFT keeps full-trace and real-context convolutions numerically aligned;
    # casting only the convolution's output is too late.
    filtered=centered_filter(z.astype(np.complex128),FRONT_KERNEL)
    result=z+(filtered-z)*blend
    result=np.where((abs(z)<=1e-12)|(blend==0),z,result)
    return np.asarray(result,np.complex64)


def source_logs(values):
    a=abs(np.asarray(values));mask=a>0;logs=np.zeros(a.shape,np.float64)
    np.log(a,out=logs,where=mask)
    return logs.astype(np.float32),mask
