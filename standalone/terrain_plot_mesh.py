"""Bounded-memory display mesh; never interpolate across unmeasured cells."""
from __future__ import annotations

import numpy as np


def measured_surface(elevation, valid, resolution_m, bounds, *, stride=8):
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("stride must be a positive integer")
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution must be finite and positive")
    rows, columns = elevation.shape
    if valid.shape != elevation.shape:
        raise ValueError("mask and elevation shapes differ")
    xmin, xmax, ymin, ymax = bounds
    c0 = max(0, int(np.floor(xmin/resolution_m + (columns-1)/2)))
    c1 = min(columns-1, int(np.ceil(xmax/resolution_m + (columns-1)/2)))
    r0 = max(0, int(np.floor(ymin/resolution_m + (rows-1)/2)))
    r1 = min(rows-1, int(np.ceil(ymax/resolution_m + (rows-1)/2)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("plot bounds do not overlap the measured raster")
    ri = np.unique(np.r_[np.arange(r0, r1+1, stride), r1])
    ci = np.unique(np.r_[np.arange(c0, c1+1, stride), c1])
    cells_valid = np.empty((len(ri)-1, len(ci)-1), dtype=bool)
    for i, (start, stop) in enumerate(zip(ri[:-1], ri[1:])):
        bad_columns = np.any(~valid[start:stop+1, c0:c1+1], axis=0)
        prefix = np.r_[0, np.cumsum(bad_columns, dtype=np.int64)]
        cells_valid[i] = (prefix[ci[1:]-c0+1] - prefix[ci[:-1]-c0]) == 0
    top_left = (np.arange(len(ri)-1)[:, None]*len(ci)
                + np.arange(len(ci)-1)[None, :])[cells_valid]
    if not len(top_left):
        raise ValueError("no fully measured plotting cells in this crop")
    triangles = np.concatenate((np.column_stack((top_left, top_left+1, top_left+len(ci)+1)),
        np.column_stack((top_left, top_left+len(ci)+1, top_left+len(ci)))))
    used, inverse = np.unique(triangles, return_inverse=True)
    sr, sc = ri[used//len(ci)], ci[used % len(ci)]
    points = np.column_stack(((sc-(columns-1)/2)*resolution_m,
                             (sr-(rows-1)/2)*resolution_m, elevation[sr, sc]))
    if not np.isfinite(points).all():
        raise ValueError("nonfinite elevation in a supposedly measured cell")
    return points, inverse.reshape(-1, 3)
