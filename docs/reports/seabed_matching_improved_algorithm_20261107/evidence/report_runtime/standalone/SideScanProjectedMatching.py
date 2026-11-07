"""Native-resolution tiled appearance matching, verified by real scan planes.

Projection arranges image pixels only. A query's projected XY is never used
as a horizontal observation; map-feature XY and received source planes are.
"""
from __future__ import annotations

import numpy as np

from PureSideScanFeatures import normalize_feature_image
from SideScanFeatureMap import pose_matrix
from SideScanPlaneRegistration import fit_observed_scan_planes, fit_to_world_pose
from SideScanProjectedSubmap import projected_feature_observations


def tile_bounds(shape, tile_shape=(256, 384), overlap=32):
    h, w = shape; th, tw = tile_shape
    if min(h, w) < 1 or not 0 <= overlap < min(th, tw): raise ValueError('Positive native tiles required.')
    for r in range(0, h, th-overlap):
        for c in range(0, w, tw-overlap):
            yield r, min(r+th, h), c, min(c+tw, w)


def tiled_feature_matches(query, prior_image, prior_mask, matcher, pad_rows, pad_columns,
                           *, tile_shape=(256, 384), overlap=32):
    image = query['image']; support = query['support'].astype(np.uint8)*255
    points, map_points, confidence = [], [], []
    tiles, attempted, covered = 0, 0, np.zeros(query['support'].shape, bool)
    for r0, r1, c0, c1 in tile_bounds(image.shape, tile_shape, overlap):
        tiles += 1
        covered[r0:r1, c0:c1] = True
        qm = support[r0:r1, c0:c1]
        mm = prior_mask[r0:r1+2*pad_rows, c0:c1+2*pad_columns]
        if np.count_nonzero(qm) < 24 or np.count_nonzero(mm) < 24: continue
        q, m, score = matcher(image[r0:r1, c0:c1], prior_image[r0:r1+2*pad_rows, c0:c1+2*pad_columns], qm, mm)
        attempted += 1
        points.append(q+np.array([c0, r0])); map_points.append(m+np.array([c0, r0])); confidence.append(score)
    if points:
        q, m, score = np.concatenate(points), np.concatenate(map_points), np.concatenate(confidence)
        if len(q):
            # Overlapping tiles must not duplicate identical correspondences.
            order = np.argsort(-score, kind='stable')
            rounded = np.rint(np.column_stack([q[order], m[order]])).astype(np.int64)
            _, unique = np.unique(rounded, axis=0, return_index=True)
            keep = order[np.sort(unique)]
            q, m, score = q[keep], m[keep], score[keep]
    else:
        q, m, score = np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    return q, m, score, dict(tiles=tiles, attempted_tiles=attempted, native_resolution_preserved=True,
        all_query_pixels_covered=bool(covered.all()), model_input_rescaled=False)


def match_projected_prior(feature_map, query, seed_pose2, matcher, *, margin_m=30.):
    if query['scale'] != 1: raise ValueError('This native-grid prototype requires scale=1, no silent resizing.')
    step, step_y = query['step'], query['grid_step_y_m']
    pad_rows, pad_columns = int(np.ceil(margin_m/step)), int(np.ceil(margin_m/step_y))
    h, w = query['support'].shape
    xs = query['x0']-pad_rows*step+np.arange(h+2*pad_rows)*step
    ys = query['grid_y_max_m']+pad_columns*step_y-np.arange(w+2*pad_columns)*step_y
    x, y = np.meshgrid(xs, ys, indexing='ij'); seed = pose_matrix(seed_pose2)
    amplitude = feature_map.sample(seed[0, 0]*x+seed[0, 1]*y+seed[0, 2],
                                   seed[1, 0]*x+seed[1, 1]*y+seed[1, 2])
    valid = np.isfinite(amplitude)&(amplitude > 0)
    prior_image, prior_mask = normalize_feature_image(np.where(valid, amplitude, 0.), valid, 1)
    qp, mp, confidence, tiling = tiled_feature_matches(query, prior_image, prior_mask, matcher, pad_rows, pad_columns)
    result = dict(matches=len(qp), query_feature_count=len(qp), map_feature_count=len(mp),
                  query_pixels=qp, map_pixels=mp, query_image=query['image'],
                  map_image=prior_image, tiling=tiling, projected_XY_used_as_measurement=False)
    if len(qp) < 24: return dict(result, accepted=False, reason='insufficient_projected_feature_matches')
    observed = projected_feature_observations(query, qp)
    map_xy = np.column_stack([xs[0]+mp[:, 1]*step, ys[0]-mp[:, 0]*step_y])
    normals = observed['normals']; offsets = np.einsum('ij,ij->i', normals, observed['origins'][:, :2])
    fit = fit_observed_scan_planes(map_xy, normals, offsets, seed=query['patch_end_id'])
    result.update({k: v for k, v in fit.items() if k not in ('information', 'parameters', 'inlier_mask')})
    if fit['accepted']:
        beta = fit['parameters']
        if np.linalg.norm(beta[:2]) > margin_m or abs(beta[2]) > np.deg2rad(8):
            return dict(result, accepted=False, reason='projected_fit_outside_existing_search_support')
        pose, info = fit_to_world_pose(seed_pose2, beta, fit['information'])
        result.update(matched_pose2=pose, information=info, parameters=beta,
            proposal_covariance=np.linalg.inv(info+np.diag([1/20.**2, 1/20.**2, 1/np.deg2rad(5.)**2])),
            landmark_xy=map_xy[fit['inlier_mask']]@seed[:2, :2].T+seed[:2, 2],
            scan_normals=normals[fit['inlier_mask']], scan_offsets=offsets[fit['inlier_mask']],
            score=fit['inliers']/max(len(qp), 1), query_feature_count=len(qp), map_feature_count=len(mp),
            measured_height_used=False, raw_intensity_correlation_used=False, optical_flow_used=False,
            all_query_pixels=qp, all_map_pixels=mp,
            query_pixels=qp[fit['inlier_mask']], map_pixels=mp[fit['inlier_mask']],
            inlier_query_pixels=qp[fit['inlier_mask']], inlier_map_pixels=mp[fit['inlier_mask']],
            scan_plane_normals=normals[fit['inlier_mask']], scan_plane_offsets=offsets[fit['inlier_mask']])
    return result
