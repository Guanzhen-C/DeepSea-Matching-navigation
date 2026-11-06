from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanHistoricalMatching import (appearance_grid, appearance_seed_transform, match_typed_historical,
                                      appearance_consensus)
from SideScanLoopClosure import SideScanHistoricalLoops, feature_observations, overlap_supported
from SideScanProjectedSubmap import projected_feature_observations


def projected(h=64, w=64):
    return dict(image=np.full((h, w), 20, np.uint8), support=np.ones((h, w), bool),
        scale=1, step=1., x0=0., x1=float(h-1), grid_step_y_m=1., grid_y_min_m=float(1-w), grid_y_max_m=0.,
        bins=32, max_range_m=32., source_phase_centre=np.zeros((h, w), np.int32),
        source_range_column=np.full((h, w), 3, np.int32), phase_origins=np.array([[1., 2., 0.]]),
        phase_normals=np.array([[.8, .6]]), phase_side_sign=np.array([1.]), phase_packet_ids=np.array([1]),
        phase_beam_rows=np.array([4]), phase_timestamps_s=np.array([0.]), anchor_vertical_z_m=-8.,
        timestamp=49., patch_start_s=0., patch_start_id=1, patch_end_id=150,
        descriptors=np.ones((30, 128), np.float32))


def legacy():
    return dict(image=np.full((80, 128), 20, np.uint8), support=np.ones((80, 128), bool),
        row_normals=np.tile([1., 0.], (80, 1)), row_scan_origins=np.column_stack([np.arange(80)*.5-40, np.zeros((80, 2))]),
        scale=1, step=.5, bins=64, max_range_m=32., x0=-40., x1=-.5,
        anchor_vertical_z_m=-8., timestamp=49., patch_start_s=0., patch_start_id=1, patch_end_id=150,
        descriptors=np.ones((30, 128), np.float32))


def test_image_warp_maps_both_native_coordinate_systems_without_becoming_a_measurement():
    a, b = legacy(), projected(); seed = np.array([2., -3., .7])
    f, d = appearance_seed_transform(a, b, seed); aa, oa = appearance_grid(a); ab, ob = appearance_grid(b)
    p = np.array([[12., 14.], [27., 22.]])
    c, s = np.cos(seed[2]), np.sin(seed[2]); r = np.array([[c, -s], [s, c]])
    np.testing.assert_allclose((p@f.T+d)@aa.T+oa, (p@ab.T+ob)@r.T+seed[:2], atol=1e-12)


def test_rotated_tile_features_are_restored_before_source_geometry_lookup():
    a, b = projected(128, 128), projected(64, 64)
    b['image'][36, 24] = 255; a['image'][34, 64] = 255
    def matcher(query_image, old_image, qm, om):
        qy, qx = np.unravel_index(query_image.argmax(), query_image.shape)
        py, px = np.unravel_index(old_image.argmax(), old_image.shape)
        return np.array([[qx, qy]], float), np.array([[px, py]], float), np.ones(1)
    pa, pb, _, info = match_typed_historical(a, b, [10., -100., np.pi/2], matcher,
        margin_m=4., tile_shape=(64, 64), overlap=0)
    np.testing.assert_allclose(pa, [[64., 34.]], atol=1e-10)
    np.testing.assert_allclose(pb, [[24., 36.]], atol=1e-10)
    assert not info['appearance_warp_used_as_pose_measurement']
    assert not info['global_image_downsampling_used']
    assert info['appearance_interpolation'] == 'bilinear_image_nearest_support_in_reference_grid'
    assert info['pixels_restored_before_physical_verification']


def test_projected_loop_observations_use_source_labels_not_strip_pixel_rows():
    q = projected(); p = np.array([[20., 30.], [25., 17.]])
    actual = feature_observations(q, p); wanted = projected_feature_observations(q, p)
    assert set(actual) == {'origins', 'normals', 'ranges', 'sides'}
    for key in actual: np.testing.assert_array_equal(actual[key], wanted[key])
    np.testing.assert_array_equal(actual['origins'], [[1., 2., 0.], [1., 2., 0.]])


def test_mixed_history_support_boxes_and_compact_storage_keep_real_geometry():
    q = projected(); old = legacy()
    assert overlap_supported(old, np.eye(3), q, np.eye(3))
    tracker = SideScanHistoricalLoops(None)
    graph = SimpleNamespace(poses=np.zeros((1, 3)))
    assert tracker.observe(q, np.zeros(3), graph, [np.eye(3)], 49.) is None
    compact = tracker.frames[0]['query']
    for k in ['source_phase_centre', 'source_range_column', 'phase_origins', 'phase_normals', 'phase_packet_ids']:
        np.testing.assert_array_equal(compact[k], q[k])
    assert 'raw_amplitude' not in compact and 'elevation' not in compact
    assert tracker.diagnostics()['historical_projected_submaps'] == 1


def test_affine_consensus_prunes_matches_but_is_not_used_to_form_loop_pose():
    rng = np.random.default_rng(31); a = projected(); b = projected()
    pb = rng.uniform(5, 50, (40, 2)); pa = pb+np.array([3., -2.])
    pa[30:] = rng.uniform(200, 600, (10, 2))
    keep = appearance_consensus(a, b, pa, pb, np.zeros(3))
    assert keep[:30].all() and not keep[30:].any()


def test_typed_match_dispatch_never_uses_legacy_row_consensus_as_a_pose_model(monkeypatch):
    import SideScanHistoricalMatching as matching
    import SideScanLoopClosure as loops
    old, q = legacy(), projected()
    pa = np.column_stack([np.linspace(5, 100, 30), np.linspace(5, 70, 30)])
    pb = np.column_stack([np.linspace(5, 50, 30), np.linspace(5, 50, 30)])
    monkeypatch.setattr(matching, 'match_typed_historical', lambda *args: (pa, pb, np.ones(30), {}))
    monkeypatch.setattr(matching, 'appearance_consensus', lambda *args: np.ones(30, bool))
    monkeypatch.setattr(loops, 'row_compatible_pairs', lambda *args: (_ for _ in ()).throw(AssertionError('Legacy row model forbidden.')))
    captured = {}
    def physical_fit(a, b, seed, dz, **kw):
        captured['b'] = b
        return dict(accepted=True, features=30, selected_feature_indices=np.arange(30),
            relative_pose2=np.array([2., 3., .4]), information=np.diag([1., 0., 10.]))
    monkeypatch.setattr(loops, 'fit_range_plane_loop', physical_fit)
    result = SideScanHistoricalLoops(None).match_pair(dict(query=old), q, np.zeros(3))
    np.testing.assert_array_equal(captured['b']['origins'], np.tile([1., 2., 0.], (30, 1)))
    np.testing.assert_array_equal(result['relative_pose2'], [2., 3., .4])
    assert result['typed_source_geometry'] and result['appearance_warp_is_not_a_loop_pose_measurement']
