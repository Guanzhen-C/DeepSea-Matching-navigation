from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometry, FeatureSonarOdometryConfig
from standalone.hil_feature_batching import build_batched_canonical


def synthetic_ping(x_offset):
    ranges = np.linspace(1., 245., 256)
    y = np.tile(np.r_[ranges, -ranges], 8)
    x = np.repeat((np.arange(8)-3.5)*.15, 512)
    points = np.c_[x, y, np.full_like(x, -40.)]
    directions = np.c_[np.zeros_like(x), y, np.full_like(x, -40.)]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    wx = x + x_offset
    intensity = np.clip(.42 + .15*np.sin(.7*wx+.035*y) + .13*np.cos(.31*wx-.11*y)
                        + .12*np.sin(.018*y*y+.9*wx) + .08*np.cos(.23*y), .01, 1.)
    return SimpleNamespace(timestamp=0., imaging=SimpleNamespace(
        along_track_row_indices=np.repeat(np.arange(8), 512), points_sensor=points,
        intensities=intensity, beam_directions=directions, hit_mask=np.ones(len(x), dtype=bool)),
        side_scan=SimpleNamespace(range_bin_centres=(np.arange(12500)+.5)*.02))


@pytest.mark.parametrize("offset", [0., .2, 100.])
@pytest.mark.parametrize("duplicate_corners", [False, True])
def test_batched_builder_preserves_all_frame_fields_exactly(offset, duplicate_corners, monkeypatch):
    import cv2
    if duplicate_corners:
        original_corners = cv2.goodFeaturesToTrack
        def duplicates(*args, **kwargs):
            corners = original_corners(*args, **kwargs)
            return None if corners is None else np.repeat(corners, 2, axis=0)
        monkeypatch.setattr(cv2, "goodFeaturesToTrack", duplicates)
    feature = FeatureSonarOdometry(FeatureSonarOdometryConfig(compute_backend="cpu"))
    original = FeatureSonarOdometry._canonical_frame
    expected = original(feature, synthetic_ping(offset))
    actual = build_batched_canonical(original)(feature, synthetic_ping(offset))
    assert len(actual.descriptors) > 0
    for field in fields(expected):
        a, b = getattr(expected, field.name), getattr(actual, field.name)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)


def test_batched_builder_rejects_unknown_source():
    def unrelated(self, ping):
        return ping
    with pytest.raises(ValueError, match="unrecognized"):
        build_batched_canonical(unrelated)
