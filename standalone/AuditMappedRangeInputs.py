"""Independent EVALUATION of mapped-range factor provenance, not estimation.

Reconstruct receiver offsets from archived Local states, pressure depths and
physical channel calibration. Truth positions are never needed by this
check. Every two-view range is checked against the actual input bin and a
global one-use token list, rather than trusting an algorithm's counter.
"""
from __future__ import annotations

import numpy as np


def audit_mapped_range_inputs(graph, trace, source_states, calibration, amplitudes=None):
    factors = graph.get('mapped_range_changes', [])
    actual = [(i, e) for i, e in enumerate(graph['edges']) if e['kind'] == 'mapped_range_change']
    if not (len(factors) == len(actual) == len(trace)):
        raise ValueError('Actual nonlinear factor, exported observations and input trace counts differ')
    times = np.asarray(source_states['timestamps'])
    poses = np.asarray(source_states['positions'])[:, 2]
    yaw = np.asarray(source_states['yaw_rad'])[:, 2]
    node_time = np.asarray(graph['pose_timestamps_s'])
    node_indices = np.searchsorted(times, node_time)
    if np.any(node_indices >= len(times)) or not np.array_equal(times[node_indices], node_time):
        raise ValueError('Every graph node must be bound to an actually executed Local timestamp')
    bins = calibration['bins']; rmax = calibration['max_range_m']
    centres = np.asarray(calibration['row_centres_m'])
    used = set(); points = 0
    for record, (index, edge), evidence in zip(factors, actual, trace):
        if (record['edge_index'] != index or record['source'] != edge['source']
                or record['target'] != edge['target'] or evidence['source'] != edge['source']
                or evidence['target'] != edge['target']):
            raise ValueError('Range-factor endpoints differ from their actual graph edge')
        data = record['observations']; n = len(data['range_a'])
        if n != evidence['features'] or len(evidence['received_tokens']) != 2*n:
            raise ValueError('Missing paired received sample identities')
        tokens = np.asarray(evidence['received_tokens'], int)
        if tokens.shape != (2*n, 3) or any(tuple(t) in used for t in tokens):
            raise ValueError('A received sample was reused in multiple new constraints')
        if len({tuple(t) for t in tokens}) != len(tokens):
            raise ValueError('Duplicate sample in the same factor')
        used.update(tuple(t) for t in tokens)
        previous_times = None
        for label, target_node, selected in [('a', edge['source'], tokens[::2]),
                                               ('b', edge['target'], tokens[1::2])]:
            ping, beam, column = selected.T; pi = ping-1
            if np.any((pi < 0)|(pi >= len(times))|(beam < 0)|(beam >= len(centres))
                      |(column < 0)|(column >= 2*bins)):
                raise ValueError('Invalid actual input sample address')
            if np.any(times[pi] > node_time[target_node]+1e-10):
                raise ValueError('A future received sample entered an earlier graph node')
            if previous_times is not None and np.any(times[pi]-previous_times < 180.-1e-8):
                raise ValueError('Range revisit has no independent temporal baseline')
            previous_times = times[pi]
            r = np.where(column < bins, bins-.5-column, column-bins+.5)*rmax/bins
            np.testing.assert_allclose(data['range_'+label], r, rtol=0, atol=1e-12)
            node_i = node_indices[target_node]
            c, s = np.cos(yaw[node_i]), np.sin(yaw[node_i])
            relative = poses[pi, :2]-poses[node_i, :2]
            relative += centres[beam, None]*np.column_stack([np.cos(yaw[pi]), np.sin(yaw[pi])])
            # Row-vector world to Local graph-node coordinates.
            expected = relative@np.array([[c, -s], [s, c]])
            np.testing.assert_allclose(data['origin_'+label], expected, rtol=0, atol=1e-8)
            np.testing.assert_allclose(data['sensor_z_'+label], poses[pi, 2], rtol=0, atol=1e-9)
            if amplitudes is not None:
                side = (column >= bins).astype(int)
                sample = np.where(side == 0, bins-1-column, column-bins)
                if np.any(amplitudes[pi, side, beam, sample] <= 0):
                    raise ValueError('A missing echo was used as an observed range')
        if (data.get('DEM_or_altitude_used') is not False
                or data.get('motion_seed_as_measurement') is not False
                or data.get('landmark_height_profiled_over_physical_interval') is not True):
            raise ValueError('New constraints exceeded the declared pure-side-scan sensor scope')
        points += n
    return dict(status='verified_actual_received_ranges_offsets_pressure_and_one_use_tokens',
        factors=len(factors), feature_pairs=points, unique_received_samples=len(used),
        Local_not_truth_used_to_reconstruct_offsets=True, source_bins_and_unsupported_echoes_checked=amplitudes is not None,
        raw_data_uploaded=False)
