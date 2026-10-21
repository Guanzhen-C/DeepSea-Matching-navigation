from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry
from standalone.hil_stable_correspondences import build_canonical_klt, canonical_pairs


def test_correspondence_set_is_unchanged_and_independent_of_confidence_permutation():
    previous = SimpleNamespace(keypoint_pixels=np.array([[2., 3.], [5., 2.], [7., 1.]]))
    current = SimpleNamespace(keypoint_pixels=np.array([[8., 2.], [1., 1.], [4., 1.]]))
    pairs = [(0, 2), (1, 0), (2, 1)]
    a = canonical_pairs(pairs, previous, current)
    b = canonical_pairs(pairs[::-1], previous, current)
    assert a == b == [(1, 0), (2, 1), (0, 2)]
    assert set(a) == set(pairs)


def test_physical_order_does_not_depend_on_detector_array_indices():
    previous = SimpleNamespace(keypoint_pixels=np.array([[2., 3.], [5., 2.], [7., 1.]]))
    current = SimpleNamespace(keypoint_pixels=np.array([[8., 2.], [1., 1.], [4., 1.]]))
    pairs = [(0, 2), (1, 0), (2, 1)]
    order = [2, 0, 1]
    reindexed = SimpleNamespace(keypoint_pixels=current.keypoint_pixels[order])
    changed = [(order.index(i), j) for i, j in pairs]
    a = canonical_pairs(pairs, previous, current)
    b = canonical_pairs(changed, previous, reindexed)
    assert [tuple(current.keypoint_pixels[i]) for i, _ in a] == [tuple(reindexed.keypoint_pixels[i]) for i, _ in b]
    assert [j for _, j in a] == [j for _, j in b]


def test_recognized_builder_is_not_automatically_installed():
    original = FeatureSonarOdometry._klt_pairs
    method = build_canonical_klt(original)
    assert FeatureSonarOdometry._klt_pairs is original
    assert method.frozen_method_sha256


def test_unknown_association_source_fails_closed():
    def different(self, previous, current, prediction):
        return []
    with pytest.raises(ValueError, match="unrecognized"):
        build_canonical_klt(different)
