from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from SideScanLearnedFeatures import coarse_context_mask
from SideScanProjectedMatching import tile_bounds, tiled_feature_matches


def test_coarse_context_mask_does_not_lose_sparse_native_support_or_mark_empty_tiles():
    mask = np.zeros((17, 19), np.uint8); mask[7, 7] = 255; mask[16, 18] = 255
    context = coarse_context_mask(mask, (24, 24))
    assert context[:8, :8].all() and context[16:24, 16:24].all()
    assert not context[8:16].any()
    assert not context[:8, 8:].any()


def test_native_tiling_covers_every_pixel_including_irregular_last_tiles():
    covered = np.zeros((73, 89), np.int32)
    for r0, r1, c0, c1 in tile_bounds(covered.shape, (32, 40), 8): covered[r0:r1, c0:c1] += 1
    assert covered.min() > 0


def test_tiled_match_coordinates_are_restored_to_global_native_grids():
    query = dict(image=np.ones((36, 48), np.uint8), support=np.ones((36, 48), bool))
    prior = np.ones((44, 60), np.uint8); mask = prior*255
    def matcher(q, p, qm, pm):
        assert p.shape == (q.shape[0]+8, q.shape[1]+12)
        return np.array([[2., 2.]]), np.array([[8., 6.]]), np.array([.9])
    q, m, _, report = tiled_feature_matches(query, prior, mask, matcher, 4, 6, tile_shape=(24, 32), overlap=8)
    np.testing.assert_allclose(m-q, np.tile([6., 4.], (len(q), 1)))
    assert report['all_query_pixels_covered'] and report['native_resolution_preserved']
    assert len(q) == report['attempted_tiles']


def test_matcher_context_masks_are_resize_compatible_but_do_not_invent_fine_support():
    import torch
    from kornia.geometry.transform import resize
    from SideScanLearnedFeatures import LoFTRSideScanMatcher
    matcher = object.__new__(LoFTRSideScanMatcher)
    matcher.device = torch.device('cpu'); matcher.context_masks = True
    def model(data):
        assert data['mask0'].dtype == torch.float32
        coarse = resize(data['mask0'], (2, 2), interpolation='nearest')
        assert coarse.shape == (1, 2, 2)
        assert data['mask0'][0, 7, 7] == 1  # same valid receptive tile
        return dict(keypoints0=torch.tensor([[3., 3.], [7., 7.], [15., 15.]]),
            keypoints1=torch.tensor([[3., 3.], [7., 7.], [15., 15.]]), confidence=torch.ones(3))
    matcher.model = model
    image = np.zeros((13, 13), np.uint8); image[3, 3] = 200
    native = np.zeros((13, 13), np.uint8); native[3, 3] = 255
    q, m, confidence = matcher(image, image, native, native)
    np.testing.assert_array_equal(q, [[3., 3.]])
    np.testing.assert_array_equal(m, [[3., 3.]])
    assert len(confidence) == 1


def test_projected_match_pose_comes_from_map_XY_and_source_planes_not_query_projection_or_ranges(monkeypatch):
    import SideScanProjectedMatching as module
    from SideScanPlaneRegistration import plane_residual_jacobian, fit_to_world_pose
    rng = np.random.default_rng(19); number = 80
    points = np.column_stack([rng.uniform(-35, -5, number), rng.uniform(-80, 80, number)])
    angle = np.linspace(-.5, .5, number); normals = np.column_stack([np.cos(angle), np.sin(angle)])
    beta = np.array([2., -3., .03])
    offsets = plane_residual_jacobian(beta, points, normals, np.zeros(number))[0]
    origins = np.column_stack([normals*offsets[:, None], np.zeros(number)])
    # Deliberately unrelated image coordinates: they select actual stored
    # observations, not observed ground XY. Only reference-map pixels locate P.
    qp = np.column_stack([20+np.arange(number)*10, 5+np.arange(number)%50]).astype(float)
    labels = np.full((96, 1024), -1, np.int32)
    labels[qp[:, 1].astype(int), qp[:, 0].astype(int)] = np.arange(number)
    query = dict(image=np.zeros_like(labels, np.uint8), support=labels >= 0, source_phase_centre=labels,
        source_range_column=np.full(labels.shape, 100, np.int32), phase_origins=origins, phase_normals=normals,
        phase_side_sign=np.ones(number), phase_packet_ids=np.arange(number), phase_beam_rows=np.zeros(number, int),
        phase_timestamps_s=np.arange(number), bins=512, max_range_m=250., scale=1, step=.5,
        grid_step_y_m=250/512, grid_y_max_m=249.755859375, x0=-40., patch_end_id=150)
    mp = np.column_stack([(249.755859375+62*250/512-points[:, 1])/(250/512), (points[:, 0]+70)/.5])
    monkeypatch.setattr(module, 'tiled_feature_matches', lambda *args: (qp, mp, np.ones(number), {}))
    monkeypatch.setattr(module, 'normalize_feature_image', lambda a, m, s: (np.zeros(a.shape, np.uint8), m))
    class MapOnly:
        def sample(self, x, y): return np.ones_like(x)
        def seabed_height(self, *args): raise AssertionError('No measured height allowed.')
    seed = np.array([7., -5., .3]); r = module.match_projected_prior(MapOnly(), query, seed, None)
    assert r['accepted'] and r['observable_rank'] == 3
    expected, _ = fit_to_world_pose(seed, beta, np.eye(3))
    np.testing.assert_allclose(r['matched_pose2'], expected, atol=1e-8)
    query['source_range_column'][:] = 200
    different_ranges = module.match_projected_prior(MapOnly(), query, seed, None)
    np.testing.assert_array_equal(different_ranges['matched_pose2'], r['matched_pose2'])
    assert not r['projected_XY_used_as_measurement'] and not r['measured_height_used']
