from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanFeatureMap import IntensityFeatureMap, pose_matrix, match_feature_prior
from PureSideScanFeatureNavigation import FeatureScanPlaneLikelihood


def test_intensity_map_does_not_load_measured_elevation(tmp_path, monkeypatch):
    image = np.arange(81, dtype=np.float32).reshape(9, 9)/100
    mask = np.ones((9, 9), bool)
    np.save(tmp_path/'backscatter.npy', image)
    np.save(tmp_path/'backscatter_valid_mask.npy', mask)
    (tmp_path/'metadata.json').write_text(json.dumps(dict(resolution_m=1.)))
    load = np.load
    def check(path, *args, **kwargs):
        assert Path(path).name in ('backscatter.npy', 'backscatter_valid_mask.npy')
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', check)
    prior = IntensityFeatureMap.from_path(tmp_path)
    np.testing.assert_allclose(prior.sample(np.array([0., .5]), np.array([0., .5])), [.4, .45], atol=1e-7)
    assert np.isnan(prior.sample(np.array([20.]), np.array([0.]))).all()
    with pytest.raises(AssertionError): prior.seabed_height(0, 0)
    with pytest.raises(AssertionError): prior.normal(0, 0)


def test_feature_particle_likelihood_is_scan_plane_only_with_a_true_null_direction():
    landmarks = np.column_stack([np.linspace(-10, 10, 50), np.linspace(-50, 50, 50)])
    matched = dict(landmark_xy=landmarks, scan_normals=np.tile([1., 0.], (50, 1)),
                   scan_offsets=landmarks[:, 0])
    likelihood = FeatureScanPlaneLikelihood(matched)
    values = likelihood(np.array([[0, 0, 0], [5, 0, 0], [0, 100, 0]]))
    assert values[0] > values[1]
    # No fake lateral/height measurement is manufactured from slant pixels.
    assert values[0] == values[2]
    assert not hasattr(likelihood, 'measured_height')


def test_feature_scan_plane_fit_and_returned_pose_use_same_translation_frame(monkeypatch):
    """Synthetic known correspondences test geometry, not descriptor success."""
    import SideScanFeatureMap as module
    rng = np.random.default_rng(57)
    points = np.column_stack([rng.uniform(-34, -4, 80), rng.uniform(-80, 80, 80)])
    dx, dyaw = 3., np.deg2rad(2.)
    c, s = np.cos(dyaw), np.sin(dyaw)
    rotation = np.array([[c, -s], [s, c]])
    transformed = points@rotation
    transformed[:, 0] -= dx
    pixels = np.column_stack([512-.5-points[:, 1]*512/250,
                              (points[:, 0]+70)/.15*4+1.5])
    query = dict(step=.15, scale=4, bins=512, max_range_m=250., x0=-40., x1=0., patch_end_id=150,
        normals=np.tile([1., 0.], (80, 1)), plane_offset_m=transformed[:, 0],
        pixels=pixels.copy(), descriptors=np.ones((80, 128), np.float32),
        image=np.ones((32, 1024), np.uint8))
    class DummyMap:
        def sample(self, x, y):
            return np.ones_like(x)
    monkeypatch.setattr(module, 'normalize_feature_image', lambda *a: (query['image'], query['image']))
    monkeypatch.setattr(module, 'extract_features', lambda *a, **k: (pixels, query['descriptors']))
    monkeypatch.setattr(module, 'descriptor_pairs', lambda *a: np.column_stack([np.arange(80)]*2))
    seed = np.array([7., -5., np.deg2rad(17.)])
    matched = match_feature_prior(DummyMap(), query, seed)
    assert matched['accepted']
    expected = seed.copy(); expected[2] += dyaw
    expected[:2] += pose_matrix(expected)[:2, 0]*dx
    np.testing.assert_allclose(matched['matched_pose2'], expected, atol=1e-7)
    assert np.linalg.matrix_rank(matched['information']) == 2
    cross_axis = np.r_[pose_matrix(expected)[:2, 1], 0.]
    np.testing.assert_allclose(matched['information']@cross_axis, 0., atol=1e-10)
    likelihood = FeatureScanPlaneLikelihood(matched)
    assert likelihood(expected[None])[0] == pytest.approx(50.)
    shifted = expected.copy(); shifted[:2] += pose_matrix(expected)[:2, 0]*8.
    assert likelihood(shifted[None])[0] < likelihood(expected[None])[0]


def test_learned_local_features_share_physical_scan_plane_verification_without_depth():
    rng = np.random.default_rng(72)
    points = np.column_stack([rng.uniform(-35, -5, 80), rng.uniform(-80, 80, 80)])
    dx, angle = 2., np.deg2rad(1.)
    c, s = np.cos(angle), np.sin(angle)
    offsets = (points@np.array([[c, -s], [s, c]]))[:, 0]-dx
    bins, step = 512, .5
    query = dict(step=step, scale=1, bins=bins, max_range_m=250., x0=-40., x1=0., patch_end_id=150,
        support=np.ones((81, 1024), bool), image=np.ones((81, 1024), np.uint8),
        row_normals=np.tile([1., 0.], (81, 1)))
    map_pixels = np.column_stack([bins-.5-points[:, 1]*bins/250, (points[:, 0]+70)/step])
    query_pixels = np.column_stack([map_pixels[:, 0], (offsets+40)/step])
    class TexturedMap:
        def sample(self, x, y):
            return 1+.2*np.sin(x*2)+.1*np.cos(y)
    def matches(a, b, qm, mm):
        assert qm.shape == query['image'].shape
        return query_pixels.copy(), map_pixels.copy(), np.ones(80)
    seed = np.array([10., 20., .3])
    result = match_feature_prior(TexturedMap(), query, seed, image_matcher=matches)
    assert result['accepted'] and result['observable_rank'] == 2
    expected = seed.copy(); expected[2] += angle
    expected[:2] += pose_matrix(expected)[:2, 0]*dx
    np.testing.assert_allclose(result['matched_pose2'], expected, atol=1e-7)


def test_user_cancelled_secondary_denoising_window_uses_only_overlap_average(monkeypatch):
    import SideScanFeatureMap as module
    from PureSideScanFeatures import SideScanFeaturePacket
    from SideScanSubmapDenoising import fuse_aligned_rows
    window = module.SideScanFeatureWindow()
    for stamp, value in enumerate([1., 1., 100.]):
        packet = SideScanFeaturePacket(timestamp=float(stamp), ranges_m=np.array([2., 4.]),
            amplitudes=np.full(2, value), rows=np.zeros(2, int), sides=np.ones(2, int),
            row_centres_m=(np.arange(8)-3.5)*.15, sample_sequence=np.arange(2))
        window.add(packet, np.zeros(3), stamp+1)
    monkeypatch.setattr(module, 'resample_supported_echoes',
        lambda packet, bins: np.full((8, 2*bins), packet.amplitudes[0]))
    monkeypatch.setattr(module, 'normalize_feature_image',
        lambda image, support, scale: (np.zeros(image.shape, np.uint8), support.astype(np.uint8)))
    monkeypatch.setattr(module, 'extract_features',
        lambda *a: (np.empty((0, 2)), np.empty((0, 128), np.float32)))
    q = window.build(np.zeros(3), bins=32)
    np.testing.assert_allclose(q['assembled_amplitude'][q['support']], 34.)
    assert not json.loads(q['assembly_json'])['secondary_denoising_used']
