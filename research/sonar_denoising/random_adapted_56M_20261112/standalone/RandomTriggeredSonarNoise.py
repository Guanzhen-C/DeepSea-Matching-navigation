"""Seeded, crop-consistent random lateral ripple augmentation.

Effective image-source geometry is a hypothesis, not a calibrated acoustic
emitter. Only added noise is spectrally coloured: received observations are
never FIR filtered. Display/envelope domains never acquire invented IQ.
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch
from torch.nn import functional as F

from MeasuredStripeNoiseModel import covariance_factor

PROTOCOL = 'random_lateral_multisource_global_coordinate_ripples_v1'


def draw_scenario(profile, shape, domain, seed, *, side=None):
    if domain not in range(4) or len(shape) != 2 or min(shape) < 2:
        raise ValueError('Explicit native shape and observed domain required.')
    rng = random.Random(int(seed))
    h, w = map(int, shape)
    model = profile['triggered_source_model']
    sides = [side] if side in (0, 1) else [0, 1]
    events = []
    for which in sides:
        base = model['sides'][str(which)]
        spacing = float(np.median([e['wave_spacing_scaled_samples'] for e in base['events']]))
        if domain == 0:
            spacing *= 4  # display waterfall rows, not physical pings
        count = rng.choice((1, 1, 2, 2, 3))
        for index in range(count):
            # Known native single-side arrays have increasing range from the
            # nadir. Both port and starboard outer edges are their last bin.
            cx = rng.uniform(.88, 1.10) if side is not None or which else rng.uniform(-.10, .12)
            events.append(dict(side=which, source_index=index,
                centre_record=rng.uniform(-.15, 1.15) * (h - 1), centre_range=cx * (w - 1),
                range_axis_scale=rng.uniform(.18, .85) * h / w,
                spacing=spacing * rng.uniform(.65, 1.65), phase=rng.random(),
                shear=rng.uniform(-.10, .10) * h / w,
                spacing_wobble=rng.uniform(0, .22), wobble_period=rng.uniform(5., 22.),
                crest_width_scale=rng.uniform(.65, 1.45),
                relative_strength=1. if index == 0 else rng.uniform(.12, .70),
                carrier_phase=rng.uniform(-np.pi, np.pi), seed=rng.randrange(1, 2**31)))
    return dict(protocol=PROTOCOL, seed=int(seed), shape=[h, w], domain=domain, side=side,
        relative_noise_RMS=rng.choice((.04, .08, .15, .30, .50, .80)), events=events,
        synthetic_source_geometry_NOT_identified_device_noise=True,
        separate_illumination_field=False, no_noncrest_noise_floor=True,
        original_scene_warped=False, source_number_positions_phases_randomized=True)


def scenario_sha(scenario):
    return hashlib.sha256(json.dumps(scenario, sort_keys=True, allow_nan=False).encode()).hexdigest()


def counter_complex(seed, rows, columns, component=0):
    """Coordinate-keyed Gaussian field: the same pixels for any crop/chunk."""
    y = np.asarray(rows, np.int64).astype(np.uint64)
    x = np.asarray(columns, np.int64).astype(np.uint64)
    with np.errstate(over='ignore'):
        key = (y * np.uint64(0xD2B74407B1CE6E93)
               ^ x * np.uint64(0xCA5A826395121157)
               ^ np.uint64(int(seed)) ^ np.uint64(component + 1) * np.uint64(0x9E3779B97F4A7C15))
        def uniform(value):
            value = value + np.uint64(0x9E3779B97F4A7C15)
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            value = value ^ (value >> np.uint64(31))
            return ((value >> np.uint64(11)).astype(np.float64) + .5) / 2**53
        u = uniform(key)
        v = uniform(key ^ np.uint64(0xDB4F0B9175AE2165))
    radius = np.sqrt(-np.log(u))
    return (radius * np.exp(2j * np.pi * v)).astype(np.complex64)


def colour_kernel(specification, taps=33):
    psd = np.asarray(specification['complex_range_PSD'], np.float64)
    if (psd.ndim != 1 or not np.isfinite(psd).all() or (psd < 0).any() or psd.sum() <= 0
            or taps < 1 or taps % 2 != 1 or taps > len(psd)):
        raise ValueError('Positive candidate PSD and odd bounded kernel required.')
    impulse = np.fft.fftshift(np.fft.ifft(np.sqrt(psd / psd.mean())))
    middle = len(impulse) // 2
    kernel = impulse[middle - taps // 2:middle + taps // 2 + 1]
    return (kernel / np.sqrt(np.square(np.abs(kernel)).sum())).astype(np.complex64)


def _carrier(event, profile, *, beam, rows, columns, device, complex_observation):
    spec = profile['triggered_source_model']['sides'][str(event['side'])]
    kernel = colour_kernel(spec)
    margin = len(kernel) // 2
    xx = np.arange(int(columns[0]) - margin, int(columns[-1]) + margin + 1)[None]
    yy = np.asarray(rows)[:, None]
    if complex_observation and beam is not None:
        factor = covariance_factor(spec)[beam]
        white = sum((factor[k] * counter_complex(event['seed'], yy, xx, k) for k in range(8)),
                    np.zeros((len(rows), xx.size), np.complex64))
    else:
        white = counter_complex(event['seed'], yy, xx)
    # Convolution is of newly sampled noise only, never of reference samples.
    # Reversing the impulse accounts for conv1d's cross-correlation convention.
    kr, ki = (torch.from_numpy(a.copy()).to(device) for a in (kernel.real[::-1], kernel.imag[::-1]))
    weights = torch.stack((torch.stack((kr, -ki)), torch.stack((ki, kr))))
    packed = torch.from_numpy(np.stack((white.real, white.imag), 1)).to(device)
    coloured = F.conv1d(packed, weights)
    result = torch.complex(coloured[:, 0], coloured[:, 1])
    return result * complex(np.cos(event['carrier_phase']), np.sin(event['carrier_phase']))


@torch.no_grad()
def apply_scenario(values, profile, scenario, *, origin=(0, 0), reference_RMS,
                   support=None, beam=None, identity=False):
    domain = scenario['domain']
    if (values.ndim != 2 or domain not in range(4) or (domain >= 2) != values.is_complex()
            or not torch.isfinite(values).all() or min(origin) < 0 or reference_RMS <= 0
            or not np.isfinite(reference_RMS)):
        raise ValueError('Finite typed reference and global reference RMS required.')
    h, w = values.shape
    if origin[0] + h > scenario['shape'][0] or origin[1] + w > scenario['shape'][1]:
        raise ValueError('Crop outside native scenario geometry.')
    gate = torch.ones_like(values.real) if support is None else torch.as_tensor(support, device=values.device)
    if gate.shape != values.shape or not torch.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
        raise ValueError('Explicit finite range support in [0, 1] required.')
    valid = (values.abs() > 0) & (gate > 0)
    if identity:
        return values.clone(), dict(identity=True, frequency_FIR=False, protocol=PROTOCOL,
                                   scenario_SHA256=scenario_sha(scenario), actual_changed_values=0)
    rows = np.arange(origin[0], origin[0] + h)
    columns = np.arange(origin[1], origin[1] + w)
    y = torch.tensor(rows, dtype=torch.float64, device=values.device)[:, None]
    x = torch.tensor(columns, dtype=torch.float64, device=values.device)[None]
    delta = torch.zeros_like(values)
    normalizer = max(sum(e['relative_strength']**2 for e in scenario['events'])**.5, 1.)
    active = torch.zeros_like(valid)
    for event in scenario['events']:
        dy = y - event['centre_record'] + event['shear'] * (x - event['centre_range'])
        radius = torch.hypot(dy, event['range_axis_scale'] * (x - event['centre_range']))
        cycles = radius / event['spacing']
        phase = cycles + event['phase'] + event['spacing_wobble'] * torch.sin(2 * torch.pi * cycles / event['wobble_period'])
        phase = ((phase + .5) % 1 - .5) / event['crest_width_scale']
        spec = profile['triggered_source_model']['sides'][str(event['side'])]
        wave = np.asarray(spec['crest_waveform'], np.float32)
        wave = wave[beam] if beam is not None else np.mean(wave, axis=0)
        # Each waveform is aligned to beam0 in the measured descriptor. Width
        # modulation affects the noise shape only; zero off-crest is retained.
        wave_t = torch.tensor(wave, device=values.device)
        bins = (phase % 1) * len(wave)
        lower = bins.floor().long() % len(wave)
        fraction = (bins - bins.floor()).float()
        crest = wave_t[lower] * (1 - fraction) + wave_t[(lower + 1) % len(wave)] * fraction
        crest = torch.where(phase.abs() < .5, crest, 0)
        active |= crest > 0
        carrier = _carrier(event, profile, beam=beam, rows=rows, columns=columns,
                           device=values.device, complex_observation=domain >= 2)
        base_rms = max(float(np.sqrt(np.mean(wave.astype(np.float64)**2))), .05)
        scale = reference_RMS * scenario['relative_noise_RMS'] * event['relative_strength'] / (normalizer * base_rms)
        # Amplitude/display observations have no received phase; this branch
        # uses a nonnegative amplitude nuisance, not fabricated reference IQ.
        noise = carrier if domain >= 2 else carrier.abs()
        delta += (noise * crest * scale).to(delta.dtype)
    delta *= gate
    output = torch.where(valid, values + delta, values)
    if domain == 0:
        output = output.clamp(0, 1)  # explicit display saturation only
    quiet = ~active | ~valid
    if not torch.equal(output[quiet], values[quiet]):
        raise ValueError('Protected or noncrest observation changed.')
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite synthetic observation.')
    return output, dict(identity=False, protocol=PROTOCOL, scenario_SHA256=scenario_sha(scenario),
        source_count=len(scenario['events']), relative_noise_RMS=scenario['relative_noise_RMS'],
        seed=scenario['seed'], global_coordinate_origin=list(origin),
        reference_RMS_once_for_complete_source=float(reference_RMS), frequency_FIR=False,
        only_added_noise_spectrally_coloured=True, reference_filtered=False,
        separate_illumination_field=False, patch_and_full_use_identical_noise_generator=True,
        protected_values_unchanged=int((~valid).sum()), actual_changed_values=int((output != values).sum()),
        unmeasured_phase_invented=False)
