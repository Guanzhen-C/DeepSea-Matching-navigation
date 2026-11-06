"""Irregular complex-additive stripes shaped by observed acoustic-institute bands.

This is a new augmentation hypothesis, NOT an identified electronic-noise law.
The old trained corruption functions are deliberately left unchanged. No private
pixels are sampled, and a public SAS image axis is not a calibrated ping axis.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


RECIPE = "irregular_per_band_complex_additive_v2"


def _validate_profile(profile):
    if profile.get("style") != "acoustic_institute_random_bright_stripes":
        raise ValueError("reviewed aggregate stripe appearance profile required")
    records = profile.get("reference_period_records", [])
    mosaic = profile.get("reference_period_mosaic_rows", [])
    slopes = profile.get("reference_abs_native_slopes", [])
    if not records or not (len(records) == len(mosaic) == len(slopes)):
        raise ValueError("aligned native-record/mosaic period and slope profiles required")
    if any(not math.isfinite(v) or v <= 0 for v in records + mosaic):
        raise ValueError("positive finite descriptive periods required")
    if any(not math.isfinite(v) or v < 0 for v in slopes):
        raise ValueError("finite nonnegative descriptive slopes required")


@torch.no_grad()
def irregular_complex_stripes(reference, generator, profile,
                              identity_probability=.2, *, noise_scale=None,
                              row_layout="native_record"):
    """Add complex noise before abs(), with independently randomized bands.

    ``noise_scale`` is the *actual* added-noise RMS / original-observation RMS
    on positive support, not a fitted SNR. The original observation is not clean
    ground truth. No input pixels are resized, clipped, recoloured or overwritten.
    ``four_row_mosaic`` changes only the appearance coordinates, not beam count.
    """
    if (reference.ndim != 4 or reference.shape[:2] != (1, 1)
            or not reference.is_complex() or not torch.isfinite(reference).all()
            or min(reference.shape[-2:]) < 4):
        raise ValueError("one complete finite native complex B1HW source required")
    if not math.isfinite(identity_probability) or not 0 <= identity_probability <= 1:
        raise ValueError("identity probability must be between zero and one")
    if noise_scale is not None and (not math.isfinite(noise_scale) or noise_scale <= 0):
        raise ValueError("positive finite relative RMS noise scale required")
    if row_layout not in ("native_record", "four_row_mosaic"):
        raise ValueError("explicit descriptive row layout required")
    _validate_profile(profile)
    device = reference.device
    dtype = reference.real.dtype

    def rand(lo, hi):
        return lo + (hi - lo) * float(torch.rand((), device=device,
                                                generator=generator))

    if rand(0., 1.) < identity_probability:
        return reference.clone(), dict(identity=True, recipe=RECIPE,
                                       noise_law_calibrated=False,
                                       source_modified=False,
                                       source_pixels_resized=False)
    amplitude = reference.abs()
    support = amplitude > 0
    if not support.any():
        raise ValueError("positive original complex observations required")
    h, w = reference.shape[-2:]
    index = min(int(rand(0, len(profile["reference_period_records"]))),
                len(profile["reference_period_records"]) - 1)
    periods = profile["reference_period_records"] if row_layout == "native_record" else profile["reference_period_mosaic_rows"]
    period = periods[index] * rand(*profile["period_random_multiplier"])
    slope = profile["reference_abs_native_slopes"][index] * rand(*profile["slope_random_multiplier"])
    if row_layout == "native_record":
        slope /= 4
    if rand(0., 1.) < .5:
        slope = -slope
    x = torch.arange(w, device=device, dtype=dtype)[None, :] - (w - 1) / 2
    y = torch.arange(h, device=device, dtype=dtype)[:, None]
    curvature = rand(*profile["normalized_range_curvature_fraction"])
    warped_x = x + curvature * x.square() / max(w / 2, 1)
    # Include bands entering from outside the frame. The margin bounds local
    # slope, curvature, sinusoidal bending and the finite Gaussian tail.
    margin = abs(slope) * 1.2 * w / 2 * (1 + abs(curvature)) + 5 * period
    center = -margin - rand(0., period)
    crest = torch.zeros((h, w), device=device, dtype=dtype)
    bands = []
    while center < h + margin:
        spacing = period * rand(.78, 1.22)
        center += spacing
        # Broad uncertain-duty bands, rather than thin periodic laser lines.
        # Bounds are design coverage, NOT measured pure-noise FWHM quantiles.
        fwhm = period * rand(.35, .65)
        sigma = fwhm / 2.354820045
        strength = rand(.35, 1.8)
        local_slope = slope * rand(.8, 1.2)
        wobble = period * rand(*profile["local_spacing_jitter_cycle_range"])
        wobble_cycles = rand(.6, 2.3)
        wobble_phase = rand(-math.pi, math.pi)
        knots = .2 + .8 * torch.rand((1, 1, 1, 8), device=device,
                                     dtype=dtype, generator=generator)
        along = F.interpolate(knots, size=(1, w), mode="bilinear",
                              align_corners=True)[0, 0]
        distance = (y - center - local_slope * warped_x
                    - wobble * torch.sin(2 * math.pi * x / w * wobble_cycles
                                         + wobble_phase)) / sigma
        crest += strength * torch.exp(-.5 * distance.square()) * along
        bands.append(dict(center=center, preceding_spacing=spacing,
                          FWHM=fwhm, relative_strength=strength,
                          slope_rows_per_column=local_slope,
                          wobble_amplitude=wobble, wobble_cycles=wobble_cycles,
                          wobble_phase=wobble_phase,
                          along_band_knots=knots[0, 0, 0].cpu().tolist()))
    envelope_knots = .25 + .75 * torch.rand((1, 1, 4, 5), device=device,
                                            dtype=dtype, generator=generator)
    envelope = F.interpolate(envelope_knots, size=(h, w), mode="bilinear",
                             align_corners=True)[0, 0]
    floor = rand(.02, .15)
    modulation = (floor + crest) * envelope

    white = torch.randn((1, 2, h, w), device=device, dtype=dtype,
                        generator=generator)
    kernel = reference.real.new_tensor([1., 2., 1.]) / math.sqrt(6.)
    coloured = F.conv2d(F.pad(white, (1, 1, 0, 0), mode="reflect"),
                        kernel.reshape(1, 1, 1, 3).repeat(2, 1, 1, 1), groups=2)
    white_fraction = rand(0., 1.)
    field = white_fraction * white + (1 - white_fraction) * coloured
    carrier = rand(.08, .2)
    if rand(0., 1.) < .25:
        carrier = -carrier
    carrier_phase = 2 * math.pi * carrier * x
    rotation = torch.complex(carrier_phase.cos(), carrier_phase.sin())[None, None]
    base = torch.complex(field[:, :1], field[:, 1:]) * rotation * modulation
    base = torch.where(support, base, torch.zeros_like(base))
    first = torch.nonzero(support.reshape(-1), as_tuple=False)[0, 0]
    anchor = reference.reshape(-1)[first] / amplitude.reshape(-1)[first]
    # Unit-RMS normalization is on this synthetic field, not on the input or
    # output image. It neither rescales source peaks nor creates clean labels.
    base = base * anchor / torch.sqrt(base.abs()[support].square().mean())
    maximum = amplitude[support].max()
    source_rms = maximum * torch.sqrt((amplitude[support] / maximum).square().mean())
    sampled_scale = rand(.15, 1.25)
    scale = sampled_scale if noise_scale is None else float(noise_scale)
    observed = reference + base * (source_rms * scale)
    if not torch.isfinite(observed).all():
        raise ValueError("nonfinite hypothetical complex corruption")
    return observed, dict(identity=False, recipe=RECIPE, native_shape=[h, w],
        row_layout=row_layout, profile_index=index, period_native_rows=period,
        slope_rows_per_column=slope, normalized_range_curvature=curvature,
        band_count=len(bands), bands=bands, background_modulation_floor=floor,
        envelope_knots=envelope_knots[0, 0].cpu().tolist(),
        carrier_cycles_per_native_column=carrier,
        white_mixture_fraction=white_fraction, range_FIR=[1., 2., 1.],
        source_RMS_relative_actual_added_noise_scale=scale,
        independently_randomized_band_spacing_width_strength=True,
        actual_noise_RMS_normalized_on_positive_support=True,
        geometry_bounds_are_design_coverage_not_fitted_noise_quantiles=True,
        source_pixels_resized=False, source_modified=False,
        underlying_phase_from_actual_public_complex_source=True,
        raw_transducer_ADC_claimed=False, noise_law_calibrated=False,
        public_axes_are_calibrated_physical_ping_range=False,
        zeros_preserved_as_missing_support_not_clean_water_labels=True,
        whole_source_statistics_make_this_an_offline_augmentation=True)
