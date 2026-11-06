"""Preserve native side-scan bins using SAME-ping measured bottom geometry.

Angles are linearly interpolated only between adjacent, physically valid DOA
nodes. Native amplitude is never downsampled before raster accumulation.
Interpolated angles are not extra independent geometric measurements.
"""
from __future__ import annotations

import numpy as np


def native_bottom_samples(angles, delays, magnitude, log_gain, *, side,
                          sound_speed, sampling_frequency, installation_deg,
                          beam_spacing_m, vehicle_along_m, platform_depth_m):
    angles, delays = np.asarray(angles, float), np.asarray(delays, float)
    magnitude, log_gain = np.asarray(magnitude), np.asarray(log_gain)
    if (angles.shape != delays.shape or angles.ndim != 2 or angles.shape[0] != 8
            or magnitude.shape != log_gain.shape or magnitude.ndim != 2 or magnitude.shape[0] != 8
            or side not in (0, 1)):
        raise ValueError('Aligned8-beam native magnitude/gain and same-ping angle/delay required.')
    if sound_speed <= 0 or sampling_frequency <= 0 or beam_spacing_m <= 0:
        raise ValueError('Physical positive sampling/geometry scales required.')
    native_ranges = (np.arange(magnitude.shape[1])+1)*sound_speed/(2*sampling_frequency)
    xyz_parts, scalar_parts, beam_parts, bin_parts = [], [], [], []
    for beam in range(8):
        nodes = delays[beam]*sound_speed/2
        index = np.flatnonzero(np.isfinite(nodes)&(nodes > 0))
        index = index[np.argsort(nodes[index], kind='stable')]
        if len(index) < 2:
            continue
        node_ranges, node_angles = nodes[index], angles[beam, index]
        local_z = -node_ranges*np.sin(np.deg2rad(installation_deg-node_angles))
        good = (np.isfinite(node_angles)&(node_angles >= -60)&(node_angles <= 20)
                &(local_z > -60)&(local_z < -10))
        hi = np.searchsorted(node_ranges, native_ranges, side='right')
        inside = (hi > 0)&(hi < len(index))
        bins = np.flatnonzero(inside)
        hi, lo = hi[inside], hi[inside]-1
        keep = good[lo]&good[hi]&(index[hi]-index[lo] == 1)&(node_ranges[hi] > node_ranges[lo])
        bins, lo, hi = bins[keep], lo[keep], hi[keep]
        if not len(bins):
            continue
        r = native_ranges[bins]
        alpha = (r-node_ranges[lo])/(node_ranges[hi]-node_ranges[lo])
        angle = node_angles[lo]*(1-alpha)+node_angles[hi]*alpha
        grazing = np.deg2rad(installation_deg-angle)
        z = -r*np.sin(grazing)
        values = np.asarray(magnitude[beam, bins], float)*np.exp(np.asarray(log_gain[beam, bins], float))
        keep = (z > -60)&(z < -10)&np.isfinite(values)&(values > 0)
        across = (1 if side == 0 else -1)*r*np.cos(grazing)
        along = np.full(len(r), vehicle_along_m+(beam-3.5)*beam_spacing_m)
        xyz_parts.append(np.column_stack([along, across, z-platform_depth_m])[keep])
        scalar_parts.append(values[keep])
        beam_parts.append(np.full(np.count_nonzero(keep), beam, np.int8))
        bin_parts.append(bins[keep])
    if not xyz_parts:
        return np.empty((0, 3)), np.empty(0), np.empty(0, np.int8), np.empty(0, np.int64)
    return np.concatenate(xyz_parts), np.concatenate(scalar_parts), np.concatenate(beam_parts), np.concatenate(bin_parts)
