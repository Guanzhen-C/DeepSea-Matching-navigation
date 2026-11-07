"""Typed historical-image matching with source-coordinate restoration.

SE(2) seeds only arrange APPEARANCE for bounded native-resolution matching.
The image warp/affine consensus NEVER becomes a loop pose measurement.
Matched pixels are restored before true source range/plane verification.
"""
from __future__ import annotations

import numpy as np

from SideScanProjectedMatching import tile_bounds


def appearance_grid(query):
    sx = query['step']/query['scale']
    sy = query.get('grid_step_y_m', query['max_range_m']/query['bins'])
    top = query.get('grid_y_max_m', (query['bins']-.5)*query['max_range_m']/query['bins'])
    return np.array([[0., sx], [-sy, 0.]]), np.array([query['x0']+(.5/query['scale']-.5)*query['step'], top])


def appearance_seed_transform(old, new, seed_pose2):
    a, oa = appearance_grid(old); b, ob = appearance_grid(new)
    c, s = np.cos(seed_pose2[2]), np.sin(seed_pose2[2]); r = np.array([[c, -s], [s, c]])
    inverse = np.linalg.inv(a)
    return inverse@r@b, inverse@(r@ob+np.asarray(seed_pose2)[:2]-oa)


def native_supported(query, pixels):
    point = np.asarray(pixels)
    h, w = query['image'].shape
    x = np.rint(point[:, 0]).astype(int); y = np.rint(point[:, 1]).astype(int)
    valid = np.isfinite(point).all(axis=1)&(x >= 0)&(y >= 0)&(x < w)&(y < h)
    mask = np.repeat(query['support'], query['scale'], axis=0)
    valid &= mask[np.clip(y, 0, h-1), np.clip(x, 0, w-1)]
    return valid


def match_typed_historical(old, new, seed_pose2, matcher, *, margin_m=30., tile_shape=(256, 384), overlap=32):
    import cv2
    if old['scale'] != 1 or new['scale'] != 1:
        raise ValueError('Typed historical matching currently requires the native scale=1 submap grids.')
    f, d = appearance_seed_transform(old, new, seed_pose2); inverse = np.linalg.inv(f)
    old_mask = old['support'].astype(np.uint8)*255
    new_mask = new['support'].astype(np.uint8)*255
    pad_u = int(np.ceil(margin_m/old.get('grid_step_y_m', old['max_range_m']/old['bins'])))
    pad_v = int(np.ceil(margin_m/old['step']))
    oh, ow = old['image'].shape
    aa, bb, scores = [], [], []; tiles = attempted = 0
    for r0, r1, c0, c1 in tile_bounds(new['image'].shape, tile_shape, overlap):
        tiles += 1
        support = new_mask[r0:r1, c0:c1]
        if np.count_nonzero(support) < 24: continue
        corners = np.array([[u, v] for u in [c0, c1-1] for v in [r0, r1-1]])
        warped_corners = corners@f.T+d
        lo = np.floor(warped_corners.min(axis=0)).astype(int)
        hi = np.ceil(warped_corners.max(axis=0)).astype(int)+1
        u0, v0 = np.maximum(lo-[pad_u, pad_v], 0)
        u1, v1 = np.minimum(hi+[pad_u, pad_v], [ow, oh])
        if u1 <= u0 or v1 <= v0: continue
        reference_mask = old_mask[v0:v1, u0:u1]
        if np.count_nonzero(reference_mask) < 24: continue
        affine = np.column_stack([f, f@np.array([c0, r0])+d-lo])
        shape = tuple((hi-lo).tolist())
        warped = cv2.warpAffine(new['image'][r0:r1, c0:c1], affine, shape,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_mask = cv2.warpAffine(support, affine, shape,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        q, p, confidence = matcher(warped, old['image'][v0:v1, u0:u1], warped_mask, reference_mask)
        attempted += 1
        restored = (q+lo-d)@inverse.T
        prior_pixels = p+np.array([u0, v0])
        keep = native_supported(new, restored)&native_supported(old, prior_pixels)
        aa.append(prior_pixels[keep]); bb.append(restored[keep]); scores.append(confidence[keep])
    if aa:
        a, b, score = np.concatenate(aa), np.concatenate(bb), np.concatenate(scores)
        if len(a):
            order = np.argsort(-score, kind='stable')
            rounded = np.rint(np.column_stack([a[order], b[order]])).astype(np.int64)
            _, unique = np.unique(rounded, axis=0, return_index=True)
            keep = order[np.sort(unique)]; a, b, score = a[keep], b[keep], score[keep]
    else: a, b, score = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return a, b, score, dict(tiles=tiles, attempted_tiles=attempted, global_image_downsampling_used=False,
        appearance_interpolation='bilinear_image_nearest_support_in_reference_grid',
        appearance_warp_used_as_pose_measurement=False, pixels_restored_before_physical_verification=True)


def appearance_consensus(old, new, pa, pb, seed_pose2):
    """Image-coordinate pruning only; real range/plane solve is still required."""
    import cv2
    if len(pa) < 24: return np.zeros(len(pa), bool)
    f, d = appearance_seed_transform(old, new, seed_pose2)
    predicted = pb@f.T+d
    _, keep = cv2.estimateAffinePartial2D(predicted.astype(np.float32), pa.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=8., maxIters=2048, confidence=.995, refineIters=10)
    return np.zeros(len(pa), bool) if keep is None else keep.ravel().astype(bool)
