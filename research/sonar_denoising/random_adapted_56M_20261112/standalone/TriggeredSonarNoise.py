"""Effective multi-source ripple events without an added illumination field.

Source locations below are reviewable IMAGE-SOURCE hypotheses, not identified
physical emitters. Only wave crests carry additive complex interference. There
is no noise floor, reference gain map, range-zone blend or blockwise RMS gain.
"""
from __future__ import annotations

import numpy as np

from MeasuredStripeNoiseModel import covariance_factor, FFT_SIZE, PHASE_BINS


PROTOCOL = 'multiple_effective_trigger_sources_crest_only_complex_noise_v1'


def build_source_model(profile, records, range_bins):
    if records < 2 or range_bins < 2:
        raise ValueError('At least two native records and range bins required.')
    sides = {}
    for side in (0, 1):
        tiles = [t for t in profile['tiles'] if t['side'] == side]
        if not tiles:
            raise ValueError('Candidate statistics for both sides required.')
        # Prefer coherent stripe candidates; not denoising-error selection.
        cutoff = float(np.quantile([t['candidate_peak_to_background_power'] for t in tiles], .75))
        coherent = [t for t in tiles if t['candidate_peak_to_background_power'] >= cutoff]
        aligned = []
        for tile in coherent:
            wave = np.array(tile['folded_log_amplitude_templates'])
            aligned.append(np.roll(wave, -int(wave[0].argmax()), axis=-1))
        waveform = np.mean(aligned, axis=0)
        amplitude = np.exp(waveform - waveform.max(axis=1, keepdims=True))
        # Subtract the phase baseline, NOT any observed seabed amplitude.
        level = (1 + np.median(amplitude, axis=1, keepdims=True)) / 2
        crest = np.maximum(amplitude - level, 0)
        crest /= np.maximum(crest.max(axis=1, keepdims=True), 1e-12)
        psd = np.mean([t['candidate_complex_range_PSD'] for t in coherent], axis=0)
        psd /= psd.sum()
        real = np.mean([t['beam_covariance_real'] for t in coherent], axis=0)
        imag = np.mean([t['beam_covariance_imag'] for t in coherent], axis=0)
        fitted = (profile.get('measured_fitted_cubic_phase_geometry_retained_for_audit')
                  or profile['curved_phase_geometry'])[str(side)]
        fy = float(fitted['fitted_record_frequency_q05_q50_q95'][1])
        fx = abs(float(fitted['fitted_range_frequency_q05_q50_q95'][1]))
        if not np.isfinite([fy, fx]).all() or min(fy, fx) <= 0:
            raise ValueError('Nonzero finite measured candidate frequencies required.')
        # Initial hypotheses from the user's "both lateral sides, more than
        # one source" constraint; never mislabel them as detected emitters.
        events = []
        for index, fraction in enumerate((.35, .70)):
            cy, cx = fraction * (records - 1), .96 * (range_bins - 1)
            dy = max(records / 4, 1.)
            dx = max(range_bins / 2, 1.)
            aspect = float(np.sqrt((fx / fy) * dy / dx))
            radial_frequency = fy * np.sqrt(dy**2 + (aspect * dx)**2) / dy
            maximum_radius = max(np.hypot(y - cy, aspect * (x - cx))
                                 for y in (0, records - 1) for x in (0, range_bins - 1))
            events.append(dict(id=f'{side}_{index}', centre_native_record=float(cy),
                centre_native_range_bin=float(cx), trigger_record_proxy=float(cy),
                range_axis_scale=aspect, wave_spacing_scaled_samples=float(1 / radial_frequency),
                initial_radius=0., wave_count=int(np.ceil(maximum_radius * radial_frequency)) + 1,
                relative_source_strength=1.,
                position_and_trigger_are_initial_hypotheses_NOT_detected_physical_sources=True))
        sides[str(side)] = dict(events=events, crest_waveform=crest.tolist(),
            complex_range_PSD=psd.tolist(), beam_covariance_real=real.tolist(), beam_covariance_imag=imag.tolist(),
            relative_global_noise_RMS=float(np.median([t['candidate_RMS_over_FIR_anchor_RMS'] for t in coherent])),
            coherent_candidate_tiles=len(coherent))
    return dict(protocol=PROTOCOL, sides=sides, native_shape=[records, range_bins],
        source_count_is_a_reviewable_hypothesis=True,
        separate_illumination_field=False, range_zone_gain_field=False,
        blockwise_reference_RMS_gain=False, noncrest_noise_floor=False,
        original_scene_pixels_warped=False, source_is_image_geometry_NOT_calibrated_3D_emitter=True)


def source_phase(event, rows, columns):
    parameters = [event[k] for k in ('centre_native_record', 'centre_native_range_bin',
        'range_axis_scale', 'initial_radius', 'wave_spacing_scaled_samples', 'wave_count')]
    if (not np.isfinite(parameters).all() or event['range_axis_scale'] <= 0
            or event['wave_spacing_scaled_samples'] <= 0 or event['initial_radius'] < 0
            or event['wave_count'] < 0):
        raise ValueError('Finite source geometry and positive wave spacing required.')
    y, x = np.broadcast_arrays(np.asarray(rows, np.float64), np.asarray(columns, np.float64))
    radius = np.hypot(y - event['centre_native_record'],
                      event['range_axis_scale'] * (x - event['centre_native_range_bin']))
    cycles = (radius - event['initial_radius']) / event['wave_spacing_scaled_samples']
    return cycles, (cycles >= 0) & (cycles < event['wave_count'])


def _global_rms(values, support):
    energy, count = 0., 0
    for start in range(0, values.shape[1], 32):
        a = np.abs(np.asarray(values[:, start:start + 32], np.complex64)).astype(np.float64)
        mask = (a > 0) & (support[None, start:start + 32] > 0)
        energy += float(np.square(a[mask]).sum())
        count += int(mask.sum())
    return np.sqrt(energy / count) if count else 0.


def synthesize_triggered_side(reference, profile, side, seed, *, noise_support,
                              device='cpu', strength=1., record_chunk=32):
    import torch
    z = np.asarray(reference)
    gate_array = np.asarray(noise_support, np.float32)
    if (z.ndim != 3 or z.shape[0] != 8 or not np.iscomplexobj(z) or not np.isfinite(z).all()
            or gate_array.shape != z.shape[1:] or not np.isfinite(gate_array).all()
            or (gate_array < 0).any() or (gate_array > 1).any() or not np.isfinite(strength) or strength <= 0
            or not isinstance(record_chunk, int) or record_chunk < 1):
        raise ValueError('Eight actual complex beams and explicit valid noise support required.')
    if list(z.shape[1:]) != profile['triggered_source_model']['native_shape']:
        raise ValueError('Source hypotheses must match the original native record/range geometry.')
    specification = profile['triggered_source_model']['sides'][str(side)]
    events = specification['events']
    h, w = z.shape[1:]
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    covariance = covariance_factor(specification)
    factor = torch.from_numpy(covariance).to(device)
    psd = np.array(specification['complex_range_PSD'])
    xp = np.fft.fftfreq(FFT_SIZE)
    order = np.argsort(xp)
    colour = np.interp(np.fft.fftfreq(w), xp[order], psd[order], period=1.)
    colour = torch.tensor(np.sqrt(colour / colour.mean()), dtype=torch.float32, device=device)
    wave = torch.tensor(specification['crest_waveform'], dtype=torch.float32, device=device)
    if (wave.shape != (8, PHASE_BINS) or not torch.isfinite(wave).all()
            or (wave < 0).any() or (wave > 1).any()):
        raise ValueError('Finite eight-beam crest-only waveforms in [0, 1] required.')
    reference_rms = _global_rms(z, gate_array)
    crest_rms = float(np.sqrt(np.mean(np.square(specification['crest_waveform']))))
    amplitude = (strength * reference_rms * specification['relative_global_noise_RMS']
                 / max(np.sqrt(max(len(events), 1)) * crest_rms, 1e-12))
    offsets = [float(torch.rand((), generator=generator, device=device)) for _ in events]
    result = np.empty(z.shape, np.complex64)
    gains = np.zeros(z.shape, np.float32)
    protected_count = quiet_count = 0
    with torch.inference_mode():
        for start in range(0, h, record_chunk):
            stop = min(start + record_chunk, h)
            original = torch.from_numpy(np.array(z[:, start:stop], np.complex64, copy=True)).to(device)
            gate = torch.from_numpy(np.array(gate_array[start:stop], copy=True)).to(device)[None]
            eligible = (original.abs() > 0) & (gate > 0)
            delta = torch.zeros_like(original)
            active = torch.zeros_like(eligible)
            for event, offset in zip(events, offsets):
                phase, present = source_phase(event, np.arange(start, stop)[:, None], np.arange(w)[None, :])
                phase = torch.tensor(((phase + offset) % 1) * PHASE_BINS, dtype=torch.float32, device=device)
                present = torch.tensor(present, device=device)[None]
                lower = phase.floor().long() % PHASE_BINS
                fraction = phase - phase.floor()
                crest = (wave[:, lower] * (1 - fraction) + wave[:, (lower + 1) % PHASE_BINS] * fraction) * present
                active |= crest > 0
                white = torch.complex(torch.randn(original.shape, generator=generator, device=device),
                                      torch.randn(original.shape, generator=generator, device=device)) / 2**.5
                coloured = torch.einsum('ab,brx->arx', factor, white)
                coloured = torch.fft.ifft(torch.fft.fft(coloured, dim=-1) * colour, dim=-1)
                delta += coloured * crest * amplitude * event['relative_source_strength']
            delta *= gate
            actual = torch.where(eligible, original + delta, original)
            protected = ~eligible
            quiet = ~active | protected
            if not torch.equal(actual[quiet], original[quiet]):
                raise ValueError('A noncrest/protected sample changed; hidden illumination is forbidden.')
            protected_count += int(protected.sum())
            quiet_count += int(quiet.sum())
            result[:, start:stop] = actual.cpu().numpy()
            positive = original.abs() > 0
            change = torch.zeros_like(original.real)
            change[positive] = (actual.abs()[positive] / original.abs()[positive]).log()
            gains[:, start:stop] = change.cpu().numpy()
    if not np.isfinite(result).all() or not np.isfinite(gains).all():
        raise ValueError('Nonfinite source-driven candidate.')
    return result, gains, dict(recipe=PROTOCOL, side=side, seed=seed, trigger_sources=events,
        global_reference_RMS_once=float(reference_rms), source_amplitude_scalar=float(amplitude),
        separate_illumination_field=False, blockwise_RMS_gain=False, range_zone_gain=False,
        noncrest_added_noise_floor=False, quiet_or_protected_native_values_exactly_unchanged=quiet_count,
        total_native_complex_values=int(z.size),
        protected_water_column_and_missing_values_exactly_verified=protected_count,
        original_reference_not_filtered_or_gain_modified=True,
        positions_and_number_are_reviewable_hypotheses_NOT_physical_source_identification=True)
