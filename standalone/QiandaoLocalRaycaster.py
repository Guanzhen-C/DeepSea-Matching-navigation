"""Bounded local-map ray bracketing, without extrapolating missing terrain.

The regional demo assumed the 250m ray endpoint itself lay on a valid raster.
That assumption fails for this small reconstructed scene. Find a finite local
surface crossing first, then apply the same bisection; reject missing brackets.
An eroded observation-support mask supplies margin for simulated range noise.
"""
from __future__ import annotations

import numpy as np


def intersect_local_grid(grid, origin_world, directions_world, *, min_range, max_range,
                         iterations=24, coarse_samples=65):
    origin = np.asarray(origin_world, float).reshape(3)
    direction = np.asarray(directions_world, float).reshape(-1, 3)
    direction = direction/np.linalg.norm(direction, axis=1)[:, None]
    distances = np.linspace(min_range, max_range, coarse_samples)
    points = origin[None, None]+direction[:, None]*distances[None, :, None]
    heights = grid._sample(grid.elevation_m, points[..., 0], points[..., 1],
                           valid_mask=grid.ray_support_mask)
    signed = points[..., 2]-heights
    crosses = (np.isfinite(signed[:, :-1]) & np.isfinite(signed[:, 1:])
               & (signed[:, :-1] >= 0) & (signed[:, 1:] <= 0))
    eligible = crosses.any(axis=1) & (direction[:, 2] < -1e-4)
    first = np.argmax(crosses, axis=1)
    low, high = distances[first].copy(), distances[first+1].copy()
    for _ in range(iterations):
        middle = (low+high)*.5
        xyz = origin+direction*middle[:, None]
        z = grid._sample(grid.elevation_m, xyz[:, 0], xyz[:, 1], valid_mask=grid.ray_support_mask)
        gap = xyz[:, 2]-z
        eligible &= np.isfinite(gap)
        low = np.where(eligible & (gap > 0), middle, low)
        high = np.where(eligible & (gap <= 0), middle, high)
    result = (low+high)*.5
    xyz = origin+direction*result[:, None]
    normal = np.zeros_like(xyz)
    normal[eligible] = grid.normal(xyz[eligible, 0], xyz[eligible, 1])
    eligible &= np.isfinite(normal).all(axis=1)
    incidence = np.zeros(len(direction))
    incidence[eligible] = np.clip(np.sum(normal[eligible]*-direction[eligible], axis=1), 0., 1.)
    xyz[~eligible] = np.nan
    result[~eligible] = np.nan
    return eligible, result, xyz, incidence
