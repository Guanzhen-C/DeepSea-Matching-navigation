from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AuditMappedRangeInputs import audit_mapped_range_inputs


def example():
    n = 6; times = np.arange(n)*100.
    positions = np.zeros((n, 4, 3)); positions[:, 2, 0] = np.arange(n)*4.
    positions[:, 2, 1] = np.arange(n)*6.; positions[:, 2, 2] = -8.
    source = dict(timestamps=times, positions=positions, yaw_rad=np.zeros((n, 4)))
    centres = (np.arange(8)-3.5)*.15
    calibration = dict(bins=512, max_range_m=250., row_centres_m=centres.tolist())
    tokens = [[2, 0, 10], [6, 0, 12], [2, 1, 30], [6, 1, 32]]
    ra = (511.5-np.array([10, 30]))*250/512
    rb = (511.5-np.array([12, 32]))*250/512
    data = dict(range_a=ra.tolist(), range_b=rb.tolist(),
        origin_a=np.column_stack([centres[:2], np.zeros(2)]).tolist(),
        origin_b=np.column_stack([centres[:2], np.zeros(2)]).tolist(),
        sensor_z_a=[-8., -8.], sensor_z_b=[-8., -8.], DEM_or_altitude_used=False,
        motion_seed_as_measurement=False, landmark_height_profiled_over_physical_interval=True)
    graph = dict(pose_timestamps_s=[100., 500.], edges=[dict(source=0, target=1, kind='mapped_range_change')],
        mapped_range_changes=[dict(source=0, target=1, edge_index=0, observations=data)])
    trace = [dict(source=0, target=1, features=2, received_tokens=tokens)]
    return graph, trace, source, calibration


def test_received_offsets_and_ranges_are_reconstructed_without_truth_positions():
    graph, trace, source, calibration = example()
    source['positions'][:, 0] = np.nan  # irrelevant evaluation truth, never used
    result = audit_mapped_range_inputs(graph, trace, source, calibration, np.ones((6, 2, 8, 512)))
    assert result['feature_pairs'] == 2 and result['unique_received_samples'] == 4


@pytest.mark.parametrize('field', ['range_a', 'origin_b', 'sensor_z_a'])
def test_fabricated_or_misaligned_measurement_is_rejected(field):
    graph, trace, source, calibration = example()
    data = graph['mapped_range_changes'][0]['observations']
    data[field] = (np.asarray(data[field])+1.).tolist()
    with pytest.raises(AssertionError): audit_mapped_range_inputs(graph, trace, source, calibration)


def test_duplicate_received_range_is_not_counted_as_new_independent_information():
    graph, trace, source, calibration = example()
    trace[0]['received_tokens'][2] = trace[0]['received_tokens'][0]
    with pytest.raises(ValueError, match='Duplicate'):
        audit_mapped_range_inputs(graph, trace, source, calibration)


def test_missing_echo_cannot_be_a_received_range():
    graph, trace, source, calibration = example()
    with pytest.raises(ValueError, match='missing echo'):
        audit_mapped_range_inputs(graph, trace, source, calibration, np.zeros((6, 2, 8, 512)))


def test_receiver_offsets_are_checked_in_rotated_node_frames_not_world_axes():
    graph, trace, source, calibration = example()
    source['yaw_rad'][:, 2] = np.linspace(-.7, 2.9, 6)
    # Move the old actual reception earlier than its node to exercise both
    # translation and receiver lever arm, not just a rotated colocated beam.
    trace[0]['received_tokens'][0][0] = 1
    trace[0]['received_tokens'][2][0] = 1
    data = graph['mapped_range_changes'][0]['observations']
    for label, node_index, ping_index in [('a', 1, 0), ('b', 5, 5)]:
        delta = source['positions'][ping_index, 2, :2]-source['positions'][node_index, 2, :2]
        offsets = []
        node_yaw = source['yaw_rad'][node_index, 2]
        direction = np.array([np.cos(source['yaw_rad'][ping_index, 2]),
                              np.sin(source['yaw_rad'][ping_index, 2])])
        c, s = np.cos(node_yaw), np.sin(node_yaw)
        for centre in calibration['row_centres_m'][:2]:
            world = delta+centre*direction
            offsets.append([c*world[0]+s*world[1], -s*world[0]+c*world[1]])
        data['origin_'+label] = offsets
    result = audit_mapped_range_inputs(graph, trace, source, calibration)
    assert result['feature_pairs'] == 2
