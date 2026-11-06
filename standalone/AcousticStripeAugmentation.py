"""Random source-shaped stripe appearance, not a physical sonar simulator.

Statistics are aggregate measurements from source stripe regions. No source
image, terrain patch, complex phase or waveform is copied into public data.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F


def acoustic_stripes(target,domain,generator,profile,identity_probability=.2):
    if target.ndim!=4 or target.shape[0]!=1 or target.shape[1]!=3 or domain not in (0,1,2):
        raise ValueError("one complete native 3CHW observation and reviewed domain required")
    if profile.get("style")!="acoustic_institute_random_bright_stripes":raise ValueError("reviewed appearance profile required")
    def rand(lo,hi):
        return lo+(hi-lo)*float(torch.rand((),device=target.device,generator=generator))
    if rand(0,1)<identity_probability:
        return target.clone(),dict(identity=True,source_pixels_copied=False)
    h,w=target.shape[-2:]
    choice=int(rand(0,len(profile["reference_period_mosaic_rows"])))
    periods=profile["reference_period_mosaic_rows"] if domain==0 else profile["reference_period_records"]
    period=periods[choice]*rand(*profile["period_random_multiplier"])
    slope=profile["reference_abs_native_slopes"][choice]*rand(*profile["slope_random_multiplier"])
    if domain!=0:slope/=4
    if rand(0,1)<.5:slope=-slope
    curvature=rand(*profile["normalized_range_curvature_fraction"])
    jitter=rand(*profile["local_spacing_jitter_cycle_range"])
    crest_power=rand(*profile["crest_power_range"])
    phase0=rand(-math.pi,math.pi)
    mirrored=rand(0,1)<.5
    y=torch.arange(h,device=target.device,dtype=target.dtype)[:,None]
    x=torch.arange(w,device=target.device,dtype=target.dtype)[None,:]-(w-1)/2
    range_x=x.abs() if mirrored else x
    # Dividing curvature by frame width keeps local slope variation bounded
    # on a 5000-column image as well as on a training crop. Only the random
    # parameter fields are interpolated; measured pixels are never resized.
    warped_x=range_x+curvature*range_x.square()/max(w/2,1)
    # Independent band positions/widths/strengths avoid the regular laser-line
    # appearance of a single sinusoid. Coordinate limits include tilted bands
    # entering the measured frame from outside its top/bottom boundaries.
    margin=abs(slope)*w*(1+abs(curvature))+2*period
    center=-margin+phase0/(2*math.pi)*period
    crest=torch.zeros((h,w),device=target.device,dtype=target.dtype)
    bands=[]
    while center<h+margin:
        spacing=period*rand(.78,1.22)
        center+=spacing
        width_fraction=rand(.12,.30)
        sigma=period*width_fraction/2.354820045
        local_slope=slope*rand(.8,1.2)
        wobble=jitter*period*torch.sin(2*math.pi*x/max(w,1)*rand(.6,2.3)+rand(-math.pi,math.pi))
        distance=(y-center-local_slope*warped_x-wobble)/max(sigma,.2)
        band=.75*torch.exp(-.5*distance.square())+.25*torch.exp(-.5*(distance/1.8).square())
        local_knots=.2+.8*torch.rand((1,1,1,8),device=target.device,generator=generator)
        along_band=F.interpolate(local_knots,size=(1,w),mode="bilinear",align_corners=True)[0,0]
        amplitude=rand(.35,1.8)
        crest+=amplitude*band*along_band
        bands.append((center,period*width_fraction,amplitude))
    wave=crest-crest.mean()
    envelope_knots=.25+.75*torch.rand((1,1,4,5),device=target.device,generator=generator)
    envelope=F.interpolate(envelope_knots,size=(h,w),mode="bilinear",align_corners=True)[0,0]
    # A slow range envelope and intermittent bright grains approximate the
    # visible appearance, not a calibrated electronic-noise distribution.
    away=x.abs()/max(w/2,1)
    range_envelope=.35+.65*away.pow(rand(.4,1.5))
    if rand(0,1)<.5:range_envelope=range_envelope.flip(-1)
    envelope=envelope*range_envelope
    u=torch.rand((h,w),device=target.device,generator=generator).clamp_min(torch.finfo(target.dtype).tiny)
    rayleigh=(-u.log()).sqrt()/(math.sqrt(math.pi)/2)
    grain_fraction=rand(*profile["grain_mixture_fraction"])
    grain=(1-grain_fraction)+grain_fraction*rayleigh
    row_modulated=domain==0 and rand(0,1)<.5
    if row_modulated:
        gains=torch.tensor(profile["four_row_modulation_profiles"][choice],device=target.device,dtype=target.dtype)
        rows=gains[torch.arange(h,device=target.device)%4][:,None]
    else:rows=1.
    depth=rand(profile["log_modulation_depth_quantiles"][0],profile["log_modulation_depth_quantiles"][-1])
    severity=rand(.75,3.)
    style_field=(wave+.6*crest*(grain-1))*envelope*rows
    if domain==0:
        # Display-luminance appearance only. The additive severity is an
        # augmentation design, not a measurement of absolute acoustic noise.
        additive=depth*severity*crest*grain*envelope*rows
        damaged=target*torch.exp(depth*severity*style_field[None,None])+additive[None,None]
        damaged=damaged.clamp(0,1)
        valid=target.mean(1,keepdim=True)>0
        damaged=torch.where(valid,damaged,target)
    else:
        # Log-amplitude residual: later decoding is a strictly positive
        # amplitude gain and does not overwrite an original complex phase.
        damaged=target+depth*severity*style_field[None,None]
    return damaged,dict(identity=False,recipe_version=2,reference_profile_index=choice,
        period_native_rows=period,slope_rows_per_column=slope,normalized_curvature=curvature,
        phase_jitter_cycles=jitter,crest_power=crest_power,grain_fraction=grain_fraction,
        severity=severity,estimated_log_depth=depth,mirrored_range=mirrored,
        independently_randomized_band_count=len(bands),
        band_center_width_strength=bands,
        weak_four_row_appearance_modulation=row_modulated,
        four_row_modulation_is_not_public_sensor_beam_count=True,
        display_strength_is_appearance_design_not_radiometric_calibration=domain==0,
        source_pixels_copied=False,noise_truth_claimed=False)
