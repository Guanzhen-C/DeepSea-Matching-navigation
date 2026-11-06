"""GPS-free measured-scene reconstruction primitives (not a real survey map).

Coordinates along track require the explicitly assumed nominal vessel motion.
Cross-track/depth use the supplied angle/delay/installation parameters. The
result is a controlled local simulation scene, NOT a calibrated geodetic map.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def reject_gps_file(event, arguments):
    """Process-wide audit guard: metadata/output 'nogps' names are allowed."""
    if event == 'open' and arguments and isinstance(arguments[0], (str, bytes)):
        from pathlib import Path
        name = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).name.lower()
        if name.startswith(('gps', 'gnss')) and name.endswith(('.mat', '.csv', '.npy', '.npz', '.json')):
            raise PermissionError('GPS/GNSS input is forbidden in this experiment: '+name)


def bottom_points(angles, delays, *, side, sound_speed, installation_deg,
                  beam_spacing_m, vehicle_along_m, platform_depth_m):
    angles, delays = np.asarray(angles, float), np.asarray(delays, float)
    if angles.shape != delays.shape or angles.ndim != 2 or angles.shape[0] != 8 or side not in (0, 1):
        raise ValueError('Two aligned eight-simultaneous-beam angle/delay fields required.')
    grazing = np.deg2rad(installation_deg-angles)
    ranges = .5*sound_speed*delays
    local_z = -ranges*np.sin(grazing)
    valid = (np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
             & (angles >= -60) & (angles <= 20) & (local_z > -60) & (local_z < -10))
    # Port is positive world/body Y, as required by the existing sonar API.
    across = (1 if side == 0 else -1)*ranges*np.cos(grazing)
    along = np.broadcast_to(vehicle_along_m+(np.arange(8)[:, None]-3.5)*beam_spacing_m, angles.shape)
    xyz = np.stack((along, across, local_z-platform_depth_m), axis=-1)
    beam, sample = np.nonzero(valid)
    return xyz[valid], ranges[valid], beam, sample


def sample_native_profiles(profiles, beam, ranges, *, sound_speed, sampling_frequency):
    """Linear interpolation of native scalar observations, not palette RGB."""
    a = np.asarray(profiles)
    r = np.asarray(ranges)
    b = np.asarray(beam)
    if a.ndim != 2 or a.shape[0] != 8 or b.shape != r.shape or not np.isfinite(a).all():
        raise ValueError('Eight native finite profiles and aligned requested ranges required.')
    position = r*2*sampling_frequency/sound_speed-1
    valid = (position >= 0) & (position <= a.shape[1]-1)
    out = np.full(r.shape, np.nan)
    low = np.floor(position[valid]).astype(int)
    high = np.minimum(low+1, a.shape[1]-1)
    alpha = position[valid]-low
    out[valid] = a[b[valid], low]*(1-alpha)+a[b[valid], high]*alpha
    return out, valid


def grid_average(xyz, fields, *, shape, resolution):
    """Bounded regular-cell binning; no huge Delaunay triangulation."""
    xyz = np.asarray(xyz)
    rows, cols = shape
    ix = np.rint(xyz[:, 0]/resolution+(cols-1)/2).astype(int)
    iy = np.rint(xyz[:, 1]/resolution+(rows-1)/2).astype(int)
    inside = ((ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
              & np.isfinite(xyz).all(axis=1))
    key = iy[inside]*cols+ix[inside]
    count = np.bincount(key, minlength=rows*cols).reshape(shape)
    output = {}
    for name, value in fields.items():
        v = np.asarray(value)
        if v.shape != (len(xyz),) or not np.isfinite(v[inside]).all():
            raise ValueError('One finite scalar for each retained bottom point required.')
        total = np.bincount(key, weights=v[inside], minlength=rows*cols).reshape(shape)
        output[name] = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
    return output, count


def bounded_interpolation(field, count, *, resolution, max_distance_m, smooth_sigma_m=0.):
    """Local nearest filling only inside the declared support distance.

    It is interpolation, not a new measurement, and missing exterior remains
    masked. This is deliberately not an unbounded gap/scene extrapolator.
    """
    field, count = np.asarray(field), np.asarray(count)
    valid = (count > 0) & np.isfinite(field)
    if not valid.any():
        raise ValueError('Observed cells required.')
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    mask = distance*resolution <= max_distance_m
    out = field[tuple(nearest)].copy()
    if smooth_sigma_m:
        weights = gaussian_filter(mask.astype(float), smooth_sigma_m/resolution)
        total = gaussian_filter(np.where(mask, out, 0.), smooth_sigma_m/resolution)
        out = np.divide(total, weights, out=out, where=weights > 0)
    out[~mask] = np.nan
    return out.astype(np.float32), mask


def physical_partition(record_ids):
    ids = np.asarray(record_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique physical record IDs required.')
    reference = ids[ids % 2 == 0]
    query = ids[ids % 2 == 1]
    if not len(reference) or not len(query) or np.intersect1d(reference, query).size:
        raise ValueError('Disjoint nonempty record partitions required.')
    return reference, query
