from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam.feature_sonar_odometry import FeatureSonarOdometryConfig, _FeatureFrame, _PairEstimate
from standalone.audit_hil_feature_pair import load_pair, save_pair


def test_feature_pair_storage_preserves_arrays_without_pickle(tmp_path):
    feature = SimpleNamespace(config=FeatureSonarOdometryConfig(compute_backend="cpu"),
                              _body_from_sensor=np.eye(4), active_compute_backend="cpu")
    frame = _FeatureFrame(timestamp=3.6, image=np.arange(64, dtype=np.uint8).reshape(8, 8),
        keypoint_pixels=np.zeros((2, 2)), keypoint_points_sensor=np.zeros((2, 3)),
        descriptors=np.ones((2, 64), dtype=np.float32), rotation_world_sensor=np.eye(3))
    result = _PairEstimate(False, "synthetic_fixture", 2, 0, 0., None, None)
    save_pair(tmp_path, feature, frame, frame, np.eye(4), result)
    metadata, frames, prediction, body = load_pair(tmp_path, _FeatureFrame)
    assert metadata["no_truth_input"]
    for actual in frames:
        for field in fields(frame):
            a, b = getattr(frame, field.name), getattr(actual, field.name)
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(prediction, np.eye(4))
    np.testing.assert_array_equal(body, np.eye(4))
    with np.load(tmp_path / "pair_inputs.npz", allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)


def test_feature_pair_audit_rejects_changed_data(tmp_path):
    feature = SimpleNamespace(config=FeatureSonarOdometryConfig(compute_backend="cpu"),
                              _body_from_sensor=np.eye(4), active_compute_backend="cpu")
    frame = _FeatureFrame(1., np.zeros((8, 8), dtype=np.uint8), np.empty((0, 2)),
                          np.empty((0, 3)), np.empty((0, 64)))
    save_pair(tmp_path, feature, frame, frame, None,
              _PairEstimate(False, "synthetic_fixture", 0, 0, 0., None, None))
    with (tmp_path / "pair_inputs.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="changed pair"):
        load_pair(tmp_path, _FeatureFrame)
