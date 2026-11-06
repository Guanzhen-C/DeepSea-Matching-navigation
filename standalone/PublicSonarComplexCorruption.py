"""Experimental cyclostationary complex-additive stripe hypothesis.

Uses actual public complex reference phases, not invented clean phases.
Periodically varying variance and coloured range noise act BEFORE abs().
This is not an identified device law, raw-ADC simulator or production filter.
The target-task coordinates are native pixels, not calibrated SAS ping rates.
"""
from __future__ import annotations
import math

import torch
from torch.nn import functional as F


def complex_phase_features(values,*,origin=None):
    """Log magnitude plus global-phase-invariant native range-phase gradient.

    Supply one full-source origin for every inference tile. The first native
    column has no preceding observation; no artificial neighbour is invented.
    The original complex tensor is not modified.
    """
    if values.ndim!=4 or values.shape[1]!=1 or not values.is_complex() or not torch.isfinite(values).all():
        raise ValueError('finite native B1HW complex observations required')
    amplitude=values.abs();support=amplitude>0
    safe=torch.where(support,amplitude,torch.ones_like(amplitude))
    logs=torch.where(support,safe.log(),torch.zeros_like(amplitude))
    if origin is None:
        origin=logs.sum((-2,-1),keepdim=True)/support.sum((-2,-1),keepdim=True).clamp_min(1)
    if origin.shape!=(len(values),1,1,1) or not torch.isfinite(origin).all():raise ValueError('one finite full-source log origin required')
    unit=values/safe
    gradient=unit[...,1:]*unit[...,:-1].conj()
    valid=support[...,1:]&support[...,:-1]
    gradient=torch.where(valid,gradient,torch.zeros_like(gradient))
    gradient=F.pad(gradient,(1,0,0,0))
    features=torch.cat(((logs-origin)*support,gradient.real,gradient.imag),dim=1)
    return features,logs,support,origin


@torch.no_grad()
def complex_additive_stripes(reference,generator,profile,identity_probability=.2,*,noise_scale=None):
    if reference.ndim!=4 or reference.shape[:2]!=(1,1) or not reference.is_complex() or not torch.isfinite(reference).all():
        raise ValueError('one complete finite native complex source required')
    if min(reference.shape[-2:])<4 or profile.get('style')!='acoustic_institute_random_bright_stripes':
        raise ValueError('native source and reviewed aggregate appearance profile required')
    if noise_scale is not None and (not math.isfinite(noise_scale) or noise_scale<=0):raise ValueError('positive finite hypothetical noise scale required')
    real_dtype=reference.real.dtype
    def rand(lo,hi):return lo+(hi-lo)*float(torch.rand((),device=reference.device,generator=generator))
    if rand(0,1)<identity_probability:
        return reference.clone(),dict(identity=True,underlying_phase_from_actual_public_complex_source=True,
            noise_law_calibrated=False,source_modified=False)
    amplitude=reference.abs();support=amplitude>0
    if not support.any():raise ValueError('nonempty public complex observations required')
    h,w=reference.shape[-2:];choice=int(rand(0,len(profile['reference_period_records'])))
    period=profile['reference_period_records'][choice]*rand(*profile['period_random_multiplier'])
    slope=profile['reference_abs_native_slopes'][choice]/4*rand(*profile['slope_random_multiplier'])
    if rand(0,1)<.5:slope=-slope
    y=torch.arange(h,device=reference.device,dtype=real_dtype)[:,None]
    x=torch.arange(w,device=reference.device,dtype=real_dtype)[None,:]-(w-1)/2
    curvature=rand(*profile['normalized_range_curvature_fraction'])
    phase0=rand(-math.pi,math.pi)
    phase=2*math.pi*(y-slope*(x+curvature*x.square()/max(w/2,1)))/period+phase0
    knots=torch.rand((1,1,4,5),device=reference.device,dtype=real_dtype,generator=generator)-.5
    # Smooth parameter drift, NOT an independent-spacing random walk. Only
    # noise parameters are interpolated, never measured seafloor observations.
    drift=rand(.05,.5)
    phase=phase+drift*F.interpolate(knots,size=(h,w),mode='bilinear',align_corners=True)[0,0]
    crest=torch.exp(rand(2.,6.)*(torch.cos(phase)-1))
    envelope=.3+.7*F.interpolate(torch.rand((1,1,3,5),device=reference.device,dtype=real_dtype,generator=generator),size=(h,w),mode='bilinear',align_corners=True)[0,0]
    white=torch.randn((1,2,h,w),device=reference.device,dtype=real_dtype,generator=generator)
    kernel=reference.real.new_tensor([1.,2.,1.])/math.sqrt(6.)
    coloured=F.conv2d(F.pad(white,(1,1,0,0),mode='reflect'),kernel.reshape(1,1,1,3).repeat(2,1,1,1),groups=2)
    white_fraction=rand(0.,1.)
    # Correlated mixture; normalize its variance analytically rather than
    # pretending white and filtered samples are independent.
    covariance=float(kernel[1]);variance=white_fraction**2+(1-white_fraction)**2+2*white_fraction*(1-white_fraction)*covariance
    field=(white_fraction*white+(1-white_fraction)*coloured)/math.sqrt(variance)
    carrier=rand(.08,.2)
    if rand(0,1)<.25:carrier=-carrier
    base=torch.complex(field[:,0:1],field[:,1:2])/math.sqrt(2.)
    carrier_phase=2*math.pi*carrier*x
    rotation=torch.complex(torch.cos(carrier_phase),torch.sin(carrier_phase))[None,None]
    maximum=amplitude[support].max()
    rms=maximum*torch.sqrt(((amplitude/maximum)[support]**2).mean())
    sampled_severity=rand(.15,.65)
    severity=sampled_severity if noise_scale is None else float(noise_scale)
    sigma=rms*severity*envelope*(.05+.95*crest)
    # A circular Gaussian field has no preferred absolute phase. A unit
    # reference anchor makes deterministic replays gauge-equivariant too;
    # it does not alter the circular conditional noise distribution.
    first=torch.nonzero(support.reshape(-1),as_tuple=False)[0,0]
    anchor=reference.reshape(-1)[first]/amplitude.reshape(-1)[first]
    noise=base*rotation*anchor*sigma[None,None]
    noise=torch.where(support,noise,torch.zeros_like(noise))
    observed=reference+noise
    if not torch.isfinite(observed).all():raise ValueError('nonfinite hypothetical complex corruption')
    return observed,dict(identity=False,recipe='cyclostationary_coloured_complex_additive_v1',
        native_shape=[h,w],stripe_period_native_pixel_rows=period,slope_pixel_rows_per_column=slope,
        stripe_phase_parameters=dict(curvature=curvature,phase0=phase0,drift=drift,knots=knots[0,0].cpu().tolist()),
        carrier_cycles_per_native_column=carrier,white_mixture_fraction=white_fraction,
        source_RMS_relative_noise_scale=severity,range_FIR=[1.,2.,1.],
        stripe_spacing_independent_random_walk=False,source_pixels_resized=False,source_modified=False,
        underlying_phase_from_actual_public_complex_source=True,raw_transducer_ADC_claimed=False,
        periods_are_physical_ping_intervals=False,noise_law_calibrated=False,
        carrier_prior_is_descriptive_observed_band_not_identified_noise=True,
        range_FIR_variance_normalization_applies_to_interior=True,
        reflected_parameter_boundary_is_not_a_sensor_model=True,
        zeros_are_preserved_missing_support_not_clean_water_labels=True)


def known_stripe_phase(metadata,*,device='cpu',dtype=torch.float32):
    """Known synthetic parameter field for diagnostics/labels, NOT model input."""
    h,w=metadata['native_shape'];parameters=metadata['stripe_phase_parameters']
    y=torch.arange(h,device=device,dtype=dtype)[:,None]
    x=torch.arange(w,device=device,dtype=dtype)[None,:]-(w-1)/2
    phase=2*math.pi*(y-metadata['slope_pixel_rows_per_column']*(x+parameters['curvature']*x.square()/max(w/2,1)))/metadata['stripe_period_native_pixel_rows']+parameters['phase0']
    knots=torch.tensor(parameters['knots'],device=device,dtype=dtype)[None,None]
    return phase+parameters['drift']*F.interpolate(knots,size=(h,w),mode='bilinear',align_corners=True)[0,0]
