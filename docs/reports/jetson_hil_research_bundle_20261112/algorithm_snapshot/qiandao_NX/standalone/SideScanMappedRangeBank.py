"""Causal, one-use received-range associations through confirmed map features.

Only confirmed feature correspondences enter the bank. Reference XY is used
for association and the factor, never map elevations. Legacy image assembly
must supply a real dominant-ping provenance label; averaged row geometry is
not silently reinterpreted as a raw range. No query is a reference-map crop.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from SideScanMappedRangeChange import MappedRangeChange, conservative_range_change_sigma
from SideScanFeatureMap import pose_matrix


def received_feature_ranges(query, pixels):
    if 'source_phase_centre' in query:
        from SideScanProjectedSubmap import projected_feature_observations, projected_pixel_indices
        result = projected_feature_observations(query, pixels)
        yy, xx, _ = projected_pixel_indices(query, pixels)
        result['range_columns'] = query['source_range_column'][yy, xx]
        return result
    if 'received_phase_centre' not in query:
        raise ValueError('Actual received phase/range provenance required, not averaged scan origins')
    p = np.asarray(pixels, float).reshape(-1, 2)
    if not np.isfinite(p).all(): raise ValueError('Finite received feature pixels required')
    x = np.rint(p[:, 0]).astype(int)
    y = np.rint((p[:, 1]+.5)/query['scale']-.5).astype(int)
    h, w = query['received_phase_centre'].shape
    if np.any((x < 0)|(x >= w)|(y < 0)|(y >= h)):
        raise ValueError('Features outside received support')
    phase = query['received_phase_centre'][y, x]
    if np.any(phase < 0): raise ValueError('Features in holes cannot invent received ranges')
    col = query['received_range_column'][y, x].astype(int)
    bins = query['bins']
    ranges = np.where(col < bins, bins-.5-col, col-bins+.5)*query['max_range_m']/bins
    return dict(origins=query['received_phase_origins'][phase], ranges=ranges,
        packet_ids=query['received_packet_ids'][phase], beam_rows=query['received_beam_rows'][phase],
        timestamps_s=query['received_timestamps_s'][phase], range_columns=col)


class MappedRangeBank:
    def __init__(self, *, association_radius_m=1., minimum_separation_s=180., minimum_features=6):
        if association_radius_m <= 0 or minimum_separation_s < 0 or minimum_features < 3:
            raise ValueError('Positive spatial support and independent feature/window requirements')
        self.radius = float(association_radius_m)
        self.minimum_separation = float(minimum_separation_s)
        self.minimum_features = int(minimum_features)
        self.buckets = defaultdict(list)
        self.used_tokens = set()
        self.observed_tokens = set()
        self.windows = self.factors = self.pairs = self.ambiguous = 0
        self.last_timestamp = -np.inf
        self.trace = []

    def cell(self, xy):
        return tuple(np.floor(np.asarray(xy)/self.radius).astype(int))

    def nearby(self, xy):
        x, y = self.cell(xy)
        return [record for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for record in self.buckets.get((x+dx, y+dy), [])]

    def observe(self, query, match, raw_pose2, graph, raw_keyframe_poses):
        if not match.get('accepted'):
            raise ValueError('Only confirmed accepted map associations may populate this bank')
        if query['timestamp'] <= self.last_timestamp:
            raise ValueError('Strictly causal nonduplicate map windows required')
        self.last_timestamp = float(query['timestamp']); self.windows += 1
        observations = received_feature_ranges(query, match['query_pixels'])
        xy = np.asarray(match['landmark_xy'], float)
        if xy.shape != (len(observations['ranges']), 2) or not np.isfinite(xy).all():
            raise ValueError('Each received feature needs its associated map XY')
        node = len(graph.poses)-1
        offset = np.linalg.inv(raw_keyframe_poses[node])@pose_matrix(raw_pose2)
        origins = observations['origins'][:, :2]@offset[:2, :2].T+offset[:2, 2]
        z = observations['origins'][:, 2]+query['anchor_vertical_z_m']
        current = []
        for i, (packet, beam, column) in enumerate(zip(observations['packet_ids'],
                observations['beam_rows'], observations['range_columns'])):
            token = (int(packet), int(beam), int(column))
            if token in self.used_tokens or token in self.observed_tokens: continue
            self.observed_tokens.add(token)
            current.append(dict(xy=xy[i].copy(), origin=origins[i].copy(), z=float(z[i]),
                range=float(observations['ranges'][i]), token=token, node=node,
                timestamp=float(observations['timestamps_s'][i]), window_start=query['patch_start_id'],
                window_end=query['patch_end_id']))
        reserved = set(); groups = defaultdict(list)
        for record in current:
            candidates = []
            for old in self.nearby(record['xy']):
                if (old['node'] >= node or old['token'] in self.used_tokens or old['token'] in reserved
                        or old['window_end'] >= record['window_start']
                        or record['timestamp']-old['timestamp'] < self.minimum_separation): continue
                distance = float(np.linalg.norm(record['xy']-old['xy']))
                if distance <= self.radius: candidates.append((distance, old))
            if not candidates: continue
            candidates.sort(key=lambda x: x[0])
            # Several historical records can be views of the SAME landmark.
            # Do not call them ambiguous solely because it was revisited.
            # Within one old window, a near-tie means distinct associations.
            if (len(candidates) > 1 and candidates[0][1]['window_end'] == candidates[1][1]['window_end']
                    and candidates[1][0] < 1.25*max(candidates[0][0], .05)):
                self.ambiguous += 1; continue
            old = candidates[0][1]
            reserved.add(old['token'])
            groups[old['node']].append((old, record))
        added = 0
        for old_node, pairs in groups.items():
            if len(pairs) < self.minimum_features: continue
            # Spread range support before capping compute, never select by
            # hidden position error, seabed label, or a true-pose residual.
            order = np.argsort([a['range'] for a, _ in pairs])
            if len(order) > 48:
                order = order[np.linspace(0, len(order)-1, 48).round().astype(int)]
            pairs = [pairs[i] for i in order]
            old, new = [a for a, _ in pairs], [b for _, b in pairs]
            ra = np.array([a['range'] for a in old]); rb = np.array([a['range'] for a in new])
            baseline = float(np.linalg.norm(graph.poses[node, :2]-graph.poses[old_node, :2]))
            data = MappedRangeChange(
                np.array([(a['xy']+b['xy'])/2 for a, b in pairs]),
                np.array([a['origin'] for a in old]), np.array([a['origin'] for a in new]),
                np.array([a['z'] for a in old]), np.array([a['z'] for a in new]), ra, rb,
                conservative_range_change_sigma(ra, rb, baseline), maximum_range_m=query['max_range_m'])
            graph.add_mapped_range_change(old_node, node, data, score=float(match.get('score', 0.)))
            tokens = [list(a['token']) for pair in pairs for a in pair]
            self.used_tokens.update(tuple(x) for x in tokens)
            self.trace.append(dict(source=old_node, target=node, features=len(pairs),
                timestamp_s=float(query['timestamp']), received_tokens=tokens,
                known_landmark_height_used=False))
            self.pairs += len(pairs); self.factors += 1; added += 1
        # A received range is used ONCE, not once per future revisit. Used
        # current receptions are not reinserted as new independent history.
        for record in current:
            if record['token'] in self.used_tokens: continue
            bucket = self.buckets[self.cell(record['xy'])]
            bucket[:] = [a for a in bucket if a['token'] not in self.used_tokens]
            bucket.append(record)
            if len(bucket) > 4: bucket[1:-2] = []  # first and two recent views
        return added

    def diagnostics(self):
        return dict(enabled=True, confirmed_windows=self.windows, factors=self.factors,
            unique_feature_pairs=self.pairs, used_received_range_tokens=len(self.used_tokens),
            ambiguous_associations_skipped=self.ambiguous,
            original_range_sample_reused_in_multiple_new_factors=False,
            known_landmark_height_used=False, DEM_or_altitude_used=False,
            geometry='mapped_XY_two_received_ranges_profiled_unknown_height_interval',
            correlation_policy='new range precision budget .5; legacy policy reported separately; correlations remain approximate')
