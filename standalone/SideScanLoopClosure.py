"""Historical SSS submaps, view-normalized local features and physical loops.

No map/DEM/GPS/true pose interface. Appearance transforms are undone before
range/plane estimation: a 2-D homography is NEVER an SE(2) pose measurement.
"""
from __future__ import annotations

import numpy as np

from SideScanFeatureMap import pose_matrix, matrix_pose
from SideScanLoopGeometry import fit_range_plane_loop, lift_loop_to_graph_nodes, wrap


def feature_observations(query, pixels):
    row = (pixels[:, 1]+.5)/query['scale']-.5
    normal = np.column_stack([np.interp(row, np.arange(len(query['row_normals'])), query['row_normals'][:, k]) for k in range(2)])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    origin = np.column_stack([np.interp(row, np.arange(len(query['row_scan_origins'])), query['row_scan_origins'][:, k]) for k in range(3)])
    bins = query['bins']; side = np.where(pixels[:, 0] < bins, 1., -1.)
    slant = np.where(side > 0, bins-.5-pixels[:, 0], pixels[:, 0]-bins+.5)*query['max_range_m']/bins
    return dict(origins=origin, normals=normal, ranges=slant, sides=side)


def overlap_supported(a, ta, b, tb, margin=30.):
    boxes = []
    for q, t in [(a, ta), (b, tb)]:
        extra = q['max_range_m']*float(np.max(abs(q['row_normals'][:, 1])))
        x0, x1 = q['x0']-margin-extra, q['x1']+margin+extra
        local = np.array([[x, y] for x in [x0, x1] for y in [-q['max_range_m'], q['max_range_m']]])
        boxes.append(local@t[:2, :2].T+t[:2, 2])
    for axis in [ta[:2, 0], ta[:2, 1], tb[:2, 0], tb[:2, 1]]:
        aa, bb = boxes[0]@axis, boxes[1]@axis
        if aa.max() < bb.min() or bb.max() < aa.min(): return False
    return True


def row_compatible_pairs(a, b, seed_yaw):
    # Same physical row compatibility as sonar-specific SIFT SLAM. Slant
    # shear absorbs image deformation only; it is not measured heading.
    sign = 1. if np.cos(seed_yaw) >= 0 else -1.
    delta = a['origins'][:, 0]-sign*b['origins'][:, 0]
    design = np.column_stack([np.ones(len(delta)), a['ranges']*a['sides']/100.])
    rng = np.random.default_rng(31); best = np.zeros(len(delta), bool)
    if len(delta) < 24: return best
    for _ in range(128):
        ii = rng.choice(len(delta), 2, replace=False)
        if abs(np.linalg.det(design[ii])) < 1e-7: continue
        beta = np.linalg.solve(design[ii], delta[ii])
        keep = abs(delta-design@beta) <= 1.
        if keep.sum() > best.sum(): best = keep
    return best


class SideScanHistoricalLoops:
    def __init__(self, image_matcher, *, minimum_separation_s=180.):
        self.matcher = image_matcher; self.minimum_separation_s = minimum_separation_s
        self.frames = []; self.pending = None; self.used_pairs = set()
        self.attempts = self.geometry_accepts = self.confirmed = self.added_edges = 0
        self.last = dict(status='no_historical_submaps')
        self.examples = []
        self.trace = []

    @staticmethod
    def descriptor(query):
        d = query['descriptors']
        v = d.mean(axis=0) if len(d) else np.zeros(128)
        return v/max(float(np.linalg.norm(v)), 1e-12)

    @staticmethod
    def frame_pose(frame, graph):
        return pose_matrix(graph.poses[frame['node']])@frame['offset']

    def candidates(self, query, current_pose, graph):
        descriptor = self.descriptor(query); available = []
        for frame in self.frames:
            old = frame['query']
            if (query['patch_start_s']-old['timestamp'] < self.minimum_separation_s
                    or query['patch_start_id'] <= old['patch_end_id']): continue
            pose = self.frame_pose(frame, graph)
            if not overlap_supported(old, pose, query, current_pose): continue
            old_centre = pose@np.array([(old['x0']+old['x1'])/2, 0., 1.])
            new_centre = current_pose@np.array([(query['x0']+query['x1'])/2, 0., 1.])
            distance = float(np.linalg.norm(old_centre[:2]-new_centre[:2]))
            score = float(frame['descriptor']@descriptor)-distance/500.
            available.append((score, frame))
        return [frame for _, frame in sorted(available, key=lambda x: x[0], reverse=True)[:3]]

    def match_pair(self, old, query, seed):
        a = old['query']; reverse = np.cos(seed[2]) < 0
        im = np.flip(query['image'], axis=(0, 1)).copy() if reverse else query['image']
        mask = np.repeat(query['support'], query['scale'], axis=0).astype(np.uint8)*255
        if reverse: mask = np.flip(mask, axis=(0, 1)).copy()
        ma = np.repeat(a['support'], a['scale'], axis=0).astype(np.uint8)*255
        pa, pb, confidence = self.matcher(a['image'], im, ma, mask)
        if reverse:
            pb = np.array([im.shape[1]-1, im.shape[0]-1])-pb
        if len(pa) < 24: return dict(accepted=False, reason='too_few_historical_feature_matches', matches=len(pa))
        oa, ob = feature_observations(a, pa), feature_observations(query, pb)
        keep = row_compatible_pairs(oa, ob, seed[2])
        if keep.sum() < 24:
            return dict(accepted=False, reason='no_historical_row_consensus', matches=len(pa), inliers=int(keep.sum()))
        fit = fit_range_plane_loop({k: v[keep] for k, v in oa.items()}, {k: v[keep] for k, v in ob.items()}, seed,
            float(query['anchor_vertical_z_m']-a['anchor_vertical_z_m']), minimum_features=24)
        selected = fit.get('selected_feature_indices', np.arange(int(keep.sum())))
        fit.update(matches=len(pa), row_consensus_matches=int(keep.sum()),
                   inliers=fit.get('features', int(keep.sum())), appearance_reversed_for_matching=reverse,
                   old_pixels=pa[keep][selected], new_pixels=pb[keep][selected])
        return fit

    def observe(self, query, raw_pose2, graph, raw_keyframe_poses, timestamp):
        node = len(graph.poses)-1
        offset = np.linalg.inv(raw_keyframe_poses[node])@pose_matrix(raw_pose2)
        current = dict(query=query, node=node, offset=offset, raw_pose=np.asarray(raw_pose2).copy(),
                       descriptor=self.descriptor(query), timestamp=float(timestamp))
        world_pose = self.frame_pose(current, graph); selected = None
        accepted = []; attempts = []
        for old in self.candidates(query, world_pose, graph):
            self.attempts += 1
            seed = matrix_pose(np.linalg.inv(self.frame_pose(old, graph))@world_pose)
            fit = self.match_pair(old, query, seed)
            attempts.append(dict(old_timestamp_s=old['timestamp'], **{k: v for k, v in fit.items()
                if not isinstance(v, np.ndarray)}))
            if not fit['accepted']:
                self.last = {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}
                continue
            self.geometry_accepts += 1
            corrected = self.frame_pose(old, graph)@pose_matrix(fit['relative_pose2'])
            correction = corrected@np.linalg.inv(pose_matrix(raw_pose2))
            accepted.append((fit['standardized_rmse'], -fit['inliers'], old, fit, correction))
        # Update temporal confirmation ONCE per current window. Multiple
        # candidate matches from one window are not independent confirmations.
        # Choose by measurement support only, never true pose/error labels.
        if accepted:
            _, _, old, fit, correction = min(accepted, key=lambda x: x[:2])
            # Two separate current AND historical windows must support a
            # compatible map-to-odom correction. Never confirm one pair twice.
            independent = (self.pending is not None
                and query['patch_start_id'] > self.pending['new_end']
                and (old['query']['patch_end_id'] < self.pending['old_start']
                     or old['query']['patch_start_id'] > self.pending['old_end']))
            delta = matrix_pose(np.linalg.inv(self.pending['correction'])@correction) if independent else None
            compatible = independent and float(np.sum((delta/np.array([10., 10., np.deg2rad(5.)]))**2)) < 11.345
            self.last = dict(status='confirmed' if compatible else 'pending_independent_historical_evidence',
                old_timestamp_s=old['timestamp'], new_timestamp_s=float(timestamp), matches=fit['matches'],
                inliers=fit['inliers'], observable_rank=fit['observable_rank'],
                standardized_rmse=fit['standardized_rmse'], landmark_depth_prior_used=False,
                motion_seed_information_added_to_measurement=False)
            self.pending = dict(correction=correction, new_end=query['patch_end_id'],
                old_start=old['query']['patch_start_id'], old_end=old['query']['patch_end_id'])
            if compatible:
                dt = abs(query['timestamp']-graph._timestamps[node])+abs(old['query']['timestamp']-graph._timestamps[old['node']])
                cov = np.diag([1., 1., np.deg2rad(.8)**2])*dt/30.
                measurement, info = lift_loop_to_graph_nodes(fit['relative_pose2'], fit['information'],
                    old['offset'], offset, offset_covariance=cov)
                selected = dict(source=old['node'], target=node, measurement=measurement, information=info,
                    score=float(fit['inliers']/fit['matches']))
                if (old['node'], node) not in self.used_pairs and old['node'] != node:
                    graph.add_edge(old['node'], node, measurement, info, kind='loop', score=selected['score'])
                    self.used_pairs.add((old['node'], node)); self.confirmed += 1; self.added_edges += 1
                    if len(self.examples) < 16:
                        self.examples.append(dict(old_image=old['query']['image'], new_image=query['image'],
                            old_pixels=fit['old_pixels'], new_pixels=fit['new_pixels'],
                            old_timestamp_s=old['timestamp'], new_timestamp_s=float(timestamp)))
                else:
                    selected = None
        elif not attempts:
            self.last = dict(status='no_time_separated_overlapping_historical_submap')
        self.trace.append(dict(timestamp_s=float(timestamp), candidates=attempts, decision=self.last.copy(),
            actual_loop_edge_added=selected is not None,
            factor=None if selected is None else dict(source=selected['source'], target=selected['target'],
                measurement=selected['measurement'].tolist(), information=selected['information'].tolist())))
        # Store compact historical observation geometry, not the full world
        # raster or every descriptor/float image. No age-based deletion.
        keys = ['image', 'support', 'row_normals', 'row_scan_origins', 'anchor_vertical_z_m',
                'bins', 'scale', 'step', 'x0', 'x1', 'max_range_m', 'timestamp',
                'patch_start_s', 'patch_start_id', 'patch_end_id']
        current['query'] = {k: query[k] for k in keys}
        self.frames.append(current)
        return selected

    def diagnostics(self):
        return dict(historical_submaps=len(self.frames), candidate_attempts=self.attempts,
            geometric_candidates=self.geometry_accepts, independent_confirmations=self.confirmed,
            actual_loop_edges=self.added_edges, last=self.last,
            measured_heights_used=False, image_homography_used_as_pose=False,
            confirmations_require_disjoint_ping_windows=True,
            temporal_consistency_is_an_association_heuristic_not_a_truth_error_gate=True,
            caveat='Conservative loop/map correlation handling; not a fully joint landmark graph.')
