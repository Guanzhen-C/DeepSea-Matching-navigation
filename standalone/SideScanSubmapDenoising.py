"""Causal, amplitude-only denoising AFTER measured scan-row alignment.

This is robust multi-observation fusion, NOT the frozen complex-IQ neural
denoiser. It needs no DEM, reference image, invented phase, or future ping.
Unsupported pixels remain zero; one observation is preserved unchanged.
"""
from __future__ import annotations

import warnings
import numpy as np


def fuse_aligned_rows(observations, height, width):
    """Fuse (local raster row, received amplitudes) with log-domain Huber.

    The local raster is supplied by the motion estimator, never world truth.
    A stack includes every supported observation contributing to each cell.
    Temporal-median/MAD initialize robust consensus; no brightness re-scaling
    or interpolation across unknown terrain is performed here.
    """
    if height <= 0 or width <= 0:
        raise ValueError('Positive output dimensions required.')
    observations = [(int(row), np.asarray(values, float)) for row, values in observations]
    if any(not 0 <= row < height or values.shape != (width,) or not np.isfinite(values).all()
           or np.any(values < 0) for row, values in observations):
        raise ValueError('Finite nonnegative aligned amplitude rows required.')
    row_counts = np.bincount([row for row, _ in observations], minlength=height)
    stack = np.full((max(1, int(row_counts.max())), height, width), np.nan)
    row_slots = np.zeros(height, int)
    for row, values in observations:
        logs = np.full(values.shape, np.nan)
        np.log(values, out=logs, where=values > 0)
        stack[row_slots[row], row] = logs
        row_slots[row] += 1
    supported = np.isfinite(stack)
    counts = supported.sum(axis=0)
    support = counts > 0
    # All-unsupported cells are intentionally allowed, not filled terrain.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(abs(stack-median[None]), axis=0)
    # Numerical scale floor is solely for a zero-MAD repeated observation;
    # it is not a claimed calibrated sonar-noise standard deviation.
    robust_scale = np.maximum(1.4826*np.nan_to_num(mad), 1e-6)
    residual = abs(stack-median[None])
    weight = np.where(supported, np.minimum(1., 1.345*robust_scale[None]/np.maximum(residual, 1e-12)), 0.)
    estimate = np.sum(np.nan_to_num(stack)*weight, axis=0)/np.maximum(weight.sum(axis=0), 1e-30)
    denoised = np.zeros((height, width))
    denoised[support] = np.exp(estimate[support])
    arithmetic = np.zeros_like(denoised)
    amplitude_stack = np.where(supported, np.exp(np.nan_to_num(stack)), 0.)
    np.divide(amplitude_stack.sum(axis=0), counts, out=arithmetic, where=support)
    # No phase/network input is manufactured when only intensities exist.
    metadata = dict(method='aligned_overlapping_amplitude_log_Huber_fusion',
        denoising_order='observations_then_local_motion_alignment_then_robust_fusion_then_features',
        received_scan_rows=len(observations), supported_cells=int(support.sum()),
        multiply_observed_cells=int((counts > 1).sum()),
        changed_cells=int((abs(denoised-arithmetic) > 1e-12).sum()),
        frozen_complex_IQ_model_used=False, invented_phase_used=False,
        reference_map_read_by_denoiser=False, measured_heights_used=False,
        brightness_rescaling=False, unknown_terrain_filled=False,
        caveat='Robust overlap fusion; not proof of removing correlated interference or recovering clean truth.')
    return dict(before_denoising_amplitude=arithmetic, denoised_amplitude=denoised,
                support=support, observation_count=counts, denoising=metadata)
