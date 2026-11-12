"""Candidate-interference statistical bootstrap, NOT an identified noise law.

Fit only spectra, channel covariance and folded modulation statistics. Never
copy a measured residual patch into a training target. The FIR-rejected signal
also contains genuine echoes, so this model requires visual/held-out review.
"""
from __future__ import annotations

import numpy as np


RECIPE = "measured_candidate_PSD_beam_covariance_phase_fold_v1"
FFT_SIZE = 512
PHASE_BINS = 48


def noise_support_from_bounds(bounds, sound_speed, sampling_frequency, range_bins):
    """Conservatively protect through the observed first-target boundary.

    Compared with the legacy output-correction ramp (bottom -> first target),
    augmentation must NOT begin inside that possibly dark interval. Its ramp
    starts AFTER the first target and spans the observed detection gap. This
    is an augmentation support policy, not a measured clean-water label.
    """
    bounds = np.asarray(bounds, np.float64)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all()
            or (bounds[:, 0] <= 0).any() or (bounds[:, 1] <= bounds[:, 0]).any()
            or not np.isfinite([sound_speed, sampling_frequency]).all()
            or min(sound_speed, sampling_frequency) <= 0 or range_bins < 1):
        raise ValueError("Valid ordered per-ping observed bounds and positive acquisition units required.")
    spacing = sound_speed / (2 * sampling_frequency)
    ranges = (np.arange(range_bins) + 1) * spacing
    width = np.maximum(bounds[:, 1:2] - bounds[:, 0:1], spacing)
    u = np.clip((ranges[None] - bounds[:, 1:2]) / width, 0, 1)
    return (u * u * (3 - 2 * u)).astype(np.float32)


def fit_tile(original, candidate, *, side, row, column, phase_cycles=None):
    """Eight synchronous complex fields in actual record/range coordinates."""
    z, d = np.asarray(original), np.asarray(candidate)
    if (z.shape != d.shape or z.ndim != 3 or z.shape[0] != 8
            or not np.iscomplexobj(z) or not np.iscomplexobj(d)):
        raise ValueError("Eight aligned actual complex beams required.")
    if not np.isfinite(z).all() or not np.isfinite(d).all():
        raise ValueError("Finite observed and candidate fields required.")
    mask = np.abs(z) > 1e-12
    if mask.mean() < .95:
        raise ValueError("Tile lacks measured support; not a pure-noise sample.")
    h, w = z.shape[-2:]
    log = np.log(np.maximum(np.abs(d).astype(np.float64), 1e-12))
    log -= np.median(log, axis=1, keepdims=True)
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    transform = np.fft.fft2(log * window)
    power = np.mean(np.abs(transform) ** 2, axis=0)
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    eligible = ((fy[:, None] >= 1 / 24) & (fy[:, None] <= 1 / 4)
                & (np.abs(fx)[None, :] <= .04))
    yi, xi = np.unravel_index(np.where(eligible, power, -1).argmax(), power.shape)
    frequency_y, frequency_x = float(fy[yi]), float(fx[xi])
    y, x = np.ogrid[:h, :w]
    if phase_cycles is None:
        phase = (frequency_y * y + frequency_x * x) % 1
    else:
        phase = np.asarray(phase_cycles, np.float64)
        if phase.shape != (h, w) or not np.isfinite(phase).all():
            raise ValueError("Aligned finite continuous native phase cycles required.")
        phase = phase % 1
    bins = np.floor(phase * PHASE_BINS).astype(np.int64) % PHASE_BINS
    templates = np.zeros((8, PHASE_BINS), np.float64)
    for beam in range(8):
        for index in range(PHASE_BINS):
            values = log[beam][(bins == index) & mask[beam]]
            templates[beam, index] = np.median(values) if len(values) else 0.
        templates[beam] -= templates[beam].mean()
    # Keep low harmonics of a DESCRIPTIVE phase template, not image pixels.
    coefficients = np.fft.rfft(templates, axis=-1)
    coefficients[:, 9:] = 0
    templates = np.fft.irfft(coefficients, n=PHASE_BINS, axis=-1)
    pieces = d[..., :w // FFT_SIZE * FFT_SIZE].reshape(8, h, -1, FFT_SIZE)
    spectrum = np.fft.fft(pieces * np.hanning(FFT_SIZE), axis=-1)
    psd = np.mean(np.abs(spectrum) ** 2, axis=(0, 1, 2))
    psd /= psd.sum()
    flattened = d.reshape(8, -1).astype(np.complex128)
    covariance = flattened @ flattened.conj().T / flattened.shape[1]
    covariance /= max(float(np.trace(covariance).real / 8), 1e-30)
    # FIR rejection is a candidate nuisance, not a clean residual label.
    anchor = z - d
    denominator = float(np.mean(np.abs(anchor[mask]).astype(np.float64) ** 2))
    ratio = float(np.sqrt(np.mean(np.abs(d[mask]).astype(np.float64) ** 2) / max(denominator, 1e-30)))
    peak_ratio = float(power[yi, xi] / max(float(np.median(power[eligible])), 1e-30))
    return dict(side=int(side), native_row=int(row), native_column=int(column),
                native_shape=[h, w], record_frequency=frequency_y, range_modulation_frequency=frequency_x,
                native_record_period=1 / frequency_y,
                native_mosaic_slope=-4 * frequency_x / frequency_y,
                candidate_peak_to_background_power=peak_ratio,
                folded_log_amplitude_templates=templates.tolist(),
                candidate_complex_range_PSD=psd.tolist(),
                beam_covariance_real=covariance.real.tolist(), beam_covariance_imag=covariance.imag.tolist(),
                candidate_RMS_over_FIR_anchor_RMS=ratio,
                candidate_log_modulation_half_span=float(np.mean((templates.max(1) - templates.min(1)) / 2)),
                folded_on_continuous_curved_phase=phase_cycles is not None,
                rejected_component_is_NOT_noise_truth=True)


def covariance_factor(tile):
    covariance = (np.array(tile["beam_covariance_real"])
                  + 1j * np.array(tile["beam_covariance_imag"]))
    covariance = (covariance + covariance.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]).astype(np.complex64)


def choose_tiles(profile, side, count=5):
    records = [r for r in profile["tiles"] if r["side"] == side]
    if not records:
        raise ValueError("Both observed sides required.")
    # Fixed median-row profile in each range zone; not chosen by model quality.
    groups = {}
    for record in records:
        groups.setdefault(record["native_column"], []).append(record)
    selected = []
    for column in sorted(groups):
        group = sorted(groups[column], key=lambda r: r["native_row"])
        selected.append(group[len(group) // 2])
    return selected


def synthesize_side(reference, profile, side, seed, *, noise_support,
                    device="cpu", strength=1., record_chunk=32):
    """Add newly drawn complex noise; preserve reference phase and zero support.

    The nuisance has a measured-candidate PSD and cross-beam covariance. Its
    random spatial phase prevents literal copying of candidate terrain pixels.
    This operation never applies a FIR to the reference observation. Explicit
    per-ping range support is mandatory: protected water-column samples are
    copied from the complex64 working reference, NOT set to black or inferred
    by thresholding all dark seabed pixels.
    """
    if profile.get('triggered_source_model') is not None:
        from TriggeredSonarNoise import synthesize_triggered_side
        return synthesize_triggered_side(reference, profile, side, seed, noise_support=noise_support,
                                         device=device, strength=strength, record_chunk=record_chunk)
    import torch
    z = np.asarray(reference)
    if z.ndim != 3 or z.shape[0] != 8 or not np.iscomplexobj(z) or not np.isfinite(z).all():
        raise ValueError("Eight finite native complex reference beams required.")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("Finite positive candidate strength required.")
    support_weights = np.asarray(noise_support, dtype=np.float32)
    if (support_weights.shape != z.shape[1:] or not np.isfinite(support_weights).all()
            or (support_weights < 0).any() or (support_weights > 1).any()):
        raise ValueError("Explicit finite per-ping range noise support in [0, 1] required.")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    tiles = choose_tiles(profile, side)
    h, w = z.shape[-2:]
    frequency = np.fft.fftfreq(w)
    xp = np.fft.fftfreq(FFT_SIZE)
    order = np.argsort(xp)
    x = torch.arange(w, device=device, dtype=torch.float32)
    weights = []
    for tile in tiles:
        centre = tile["native_column"] + tile["native_shape"][1] / 2
        distance = (np.arange(w) - centre) / 1536
        weights.append(np.exp(-.5 * distance * distance))
    weights = np.array(weights)
    weights /= np.maximum(weights.sum(0, keepdims=True), 1e-30)
    result = np.empty(z.shape, np.complex64)
    amplitude_gain = np.empty(z.shape, np.float32)
    factors = [torch.from_numpy(covariance_factor(t)).to(device) for t in tiles]
    colours, templates, offsets = [], [], []
    for tile in tiles:
        psd = np.array(tile["candidate_complex_range_PSD"])
        values = np.interp(frequency, xp[order], psd[order], period=1.)
        values /= values.mean()
        colours.append(torch.from_numpy(np.sqrt(values).astype(np.float32)).to(device))
        templates.append(torch.tensor(tile["folded_log_amplitude_templates"], device=device))
        offsets.append(float(torch.rand((), generator=generator, device=device)))
    geometry = profile.get("curved_phase_geometry", {}).get(str(side))
    if geometry is not None:
        from CurvedSonarStripePhase import phase_values
        common_offset = offsets[0]
    protected_values_verified = 0
    with torch.inference_mode():
        for start in range(0, h, record_chunk):
            stop = min(start + record_chunk, h)
            y = torch.arange(start, stop, device=device, dtype=torch.float32)[:, None]
            original = torch.from_numpy(np.array(z[:, start:stop], np.complex64, copy=True)).to(device)
            support = original.abs() > 0
            gate = torch.from_numpy(np.array(support_weights[start:stop], copy=True)).to(device)
            eligible = support & (gate[None] > 0)
            accumulated = torch.zeros_like(original)
            if geometry is not None:
                global_phase = torch.from_numpy(phase_values(
                    geometry, np.arange(start, stop)[:, None], np.arange(w)[None, :]).astype(np.float32)).to(device)
                curved_phase = ((global_phase + common_offset) % 1) * PHASE_BINS
            for index, tile in enumerate(tiles):
                white = torch.complex(
                    torch.randn(original.shape, device=device, generator=generator),
                    torch.randn(original.shape, device=device, generator=generator)) / 2**.5
                correlated = torch.einsum("ab,brx->arx", factors[index], white)
                coloured = torch.fft.ifft(torch.fft.fft(correlated, dim=-1) * colours[index], dim=-1)
                phase = curved_phase if geometry is not None else (
                    (tile["record_frequency"] * y + tile["range_modulation_frequency"] * x[None]
                     + offsets[index]) % 1) * PHASE_BINS
                lower = phase.floor().long() % PHASE_BINS
                fraction = phase - phase.floor()
                template = templates[index]
                log_envelope = template[:, lower] * (1 - fraction) + template[:, (lower + 1) % PHASE_BINS] * fraction
                envelope = log_envelope.exp()
                weight = torch.tensor(weights[index], dtype=torch.float32, device=device)
                valid = eligible & (weight[None, None] > .1)
                if not bool(valid.any()):
                    continue
                source_rms = original.abs()[valid].square().mean().sqrt()
                nuisance = coloured * envelope
                noise_rms = nuisance.abs()[valid].square().mean().sqrt()
                nuisance = nuisance * (source_rms * tile["candidate_RMS_over_FIR_anchor_RMS"] * strength / noise_rms)
                accumulated += nuisance * weight[None, None]
            accumulated *= gate[None]
            observed = torch.where(eligible, original + accumulated, original)
            protected = ~eligible
            if not torch.equal(observed[protected], original[protected]):
                raise ValueError("Protected water-column or missing-source samples changed.")
            protected_values_verified += int(protected.sum())
            result[:, start:stop] = observed.cpu().numpy()
            logs = torch.zeros_like(original.real)
            logs[support] = (observed.abs()[support] / original.abs()[support]).log()
            amplitude_gain[:, start:stop] = logs.cpu().numpy()
    if not np.isfinite(result).all() or not np.isfinite(amplitude_gain).all():
        raise ValueError("Nonfinite candidate synthetic field.")
    return result, amplitude_gain, dict(recipe=RECIPE, side=side, seed=seed, strength=strength,
        selected_tile_coordinates=[[t["native_row"], t["native_column"]] for t in tiles],
        actual_reference_pixels_modified=False, reference_filtered=False,
        missing_original_support_preserved=bool(np.array_equal(result[np.abs(z) == 0], z[np.abs(z) == 0])),
        explicit_observed_range_support_required=True,
        continuous_curved_native_phase_used=geometry is not None,
        independent_range_zone_phase_offsets_used=geometry is None,
        protected_water_column_and_missing_values_exactly_verified=protected_values_verified,
        protected_samples_equal_complex64_working_reference=True,
        water_column_forced_to_black=False, dark_pixel_threshold_used=False,
        measured_spatial_pixels_copied=False, noise_law_identified=False,
        PSD_and_covariance_of_FIR_rejected_candidate_NOT_pure_noise=True)
