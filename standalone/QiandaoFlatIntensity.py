"""Measured native side-scan amplitude on an explicitly assumed flat plane.

No measured bottom angle/delay/elevation is accepted by this module. Geometry
is a simulation assumption; it is not a georeferenced terrain reconstruction.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from QiandaoLocalScene import sample_native_profiles


def project_native_intensity(profiles, log_gain, across_m, *, source_altitude_m,
                             sound_speed, sampling_frequency):
    """r=sqrt(y^2+H^2), evaluated in native floating amplitude, not RGB."""
    profiles, log_gain = np.asarray(profiles), np.asarray(log_gain)
    across = np.asarray(across_m, float)
    if profiles.shape != log_gain.shape or profiles.ndim != 2 or profiles.shape[0] != 8:
        raise ValueError('Aligned native eight-beam amplitude and frozen gain required.')
    if not np.isfinite(source_altitude_m) or source_altitude_m <= 0:
        raise ValueError('Explicit positive assumed altitude required.')
    if across.ndim != 1 or not np.isfinite(across).all():
        raise ValueError('Finite cross-track coordinates required.')
    ranges = np.tile(np.hypot(across, source_altitude_m), 8)
    beams = np.repeat(np.arange(8), len(across))
    processed = profiles*np.exp(log_gain.astype(float))
    values, valid = sample_native_profiles(processed, beams, ranges,
        sound_speed=sound_speed, sampling_frequency=sampling_frequency)
    valid &= np.isfinite(values) & (values > 0)
    return values.reshape(8, -1), valid.reshape(8, -1)


def sample_mask(mask, x, y, resolution, *, all_four=False):
    """Reject the exterior; optionally require every bilinear neighbour."""
    mask = np.asarray(mask, bool)
    x, y = np.broadcast_arrays(x, y)
    col = x/resolution+(mask.shape[1]-1)/2
    row = y/resolution+(mask.shape[0]-1)/2
    if all_four:
        # Includes all four contributing cells, not a nearest-cell promise
        # that fails as soon as an actual float scalar is interpolated.
        ix, iy = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix+1 < mask.shape[1]) & (iy+1 < mask.shape[0])
        result = np.zeros(x.shape, bool)
        ci, ri = ix[inside], iy[inside]
        result[inside] = (mask[ri, ci] & mask[ri+1, ci] & mask[ri, ci+1] & mask[ri+1, ci+1])
        return result
    return map_coordinates(mask.astype(np.uint8), [row, col], order=0,
                           mode='constant', cval=0).astype(bool)


def flat_fan_support(xy, tangent, origins_sensor, directions_sensor, *, mask,
                     resolution, altitude_m, max_range_m):
    """Nominal footprint audit; NOT a navigation accuracy or dynamics test."""
    xy, tangent = np.asarray(xy), np.asarray(tangent)
    origins, directions = np.asarray(origins_sensor), np.asarray(directions_sensor)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    ranges = (-altitude_m-origins[:, 2])/directions[:, 2]
    ray_xy = origins[:, :2]+directions[:, :2]*ranges[:, None]
    world_x = xy[:, 0, None]+tangent[:, 0, None]*ray_xy[None, :, 0]+normal[:, 0, None]*ray_xy[None, :, 1]
    world_y = xy[:, 1, None]+tangent[:, 1, None]*ray_xy[None, :, 0]+normal[:, 1, None]*ray_xy[None, :, 1]
    valid = sample_mask(mask, world_x, world_y, resolution, all_four=True)
    valid &= ((ranges > 0) & (ranges <= max_range_m))[None, :]
    return valid, world_x, world_y
