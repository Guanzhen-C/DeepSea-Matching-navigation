from pathlib import Path
import os

import numpy as np
import pytest

from oceansim.slam import (
    BBS3DCandidateVerifier,
    BBS3DVerifierConfig,
    InertialState,
)
from oceansim.slam.math3d import rotation_from_rpy
from oceansim.slam.real_bathymetry import MeasuredBathymetryGrid


def _terrain(size: int = 121, resolution_m: float = 5.0):
    axis = (np.arange(size) - 0.5 * (size - 1)) * resolution_m
    x, y = np.meshgrid(axis, axis)
    elevation = (
        -100.0
        + 0.0008 * x**2
        + 0.0011 * y**2
        + 8.0 * np.sin(x / 47.0) * np.cos(y / 61.0)
        + 3.0 * np.sin((x + y) / 23.0)
    )
    return MeasuredBathymetryGrid(elevation, resolution_m=resolution_m)


class _FakeNative:
    def __init__(self, correction=(0.0, 0.0, 0.0), score_fraction=0.8):
        self.correction = correction
        self.score_fraction = score_fraction
        self.calls = []

    def localize(self, target, source, **kwargs):
        self.calls.append((target.copy(), source.copy(), dict(kwargs)))
        yaw = float(self.correction[2])
        transform = np.eye(4, dtype=np.float64)
        transform[:2, :2] = [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
        transform[:2, 3] = self.correction[:2]
        return {
            "transform": transform,
            "score": int(round(self.score_fraction * source.shape[0])),
            "score_fraction": self.score_fraction,
            "elapsed_milliseconds": 12.5,
            "localized": True,
            "timed_out": False,
        }


def _body_points(count=256):
    along = np.linspace(-80.0, 80.0, count)
    across = 65.0 * np.sin(np.linspace(-2.0, 2.0, count))
    down = -45.0 + 3.0 * np.cos(np.linspace(-3.0, 3.0, count))
    return np.column_stack((along, across, down))


def test_bbs3d_adapter_holds_depth_roll_and_pitch_out_of_native_search():
    backend = _FakeNative(correction=(-10.0, 5.0, np.deg2rad(-3.0)))
    verifier = BBS3DCandidateVerifier(
        _terrain(),
        BBS3DVerifierConfig(
            maximum_translation_m=25.0,
            maximum_yaw_deg=8.0,
            minimum_source_points=64,
        ),
        native_backend=backend,
    )
    roll, pitch = np.deg2rad([2.0, -3.0])
    candidate = np.array([55.0, -40.0, np.deg2rad(31.0)])
    state = InertialState(
        rotation=rotation_from_rpy(roll, pitch, np.deg2rad(28.0)),
        position=np.array([999.0, -777.0, -50.0]),
    )
    points = _body_points()

    result = verifier.verify(points, state, candidate)

    assert result.accepted
    np.testing.assert_allclose(result.matched_pose2[:2], [45.0, -35.0])
    assert np.rad2deg(result.matched_pose2[2]) == pytest.approx(28.0)
    target, source, arguments = backend.calls[0]
    assert target.shape[1] == source.shape[1] == 3
    np.testing.assert_allclose(
        arguments["minimum_translation_xyz"], [-25.0, -25.0, 0.0]
    )
    np.testing.assert_allclose(
        arguments["maximum_translation_xyz"], [25.0, 25.0, 0.0]
    )
    np.testing.assert_allclose(arguments["minimum_rpy"][:2], [0.0, 0.0])
    np.testing.assert_allclose(arguments["maximum_rpy"][:2], [0.0, 0.0])
    expected_rotation = rotation_from_rpy(roll, pitch, candidate[2])
    expected_source = points @ expected_rotation.T
    expected_source[:, 2] += state.position[2]
    np.testing.assert_allclose(source, expected_source, atol=1.0e-5)


def test_bbs3d_adapter_rejects_native_solution_on_search_boundary():
    backend = _FakeNative(correction=(-40.0, 0.0, 0.0), score_fraction=0.9)
    verifier = BBS3DCandidateVerifier(
        _terrain(),
        BBS3DVerifierConfig(
            maximum_translation_m=25.0,
            minimum_source_points=64,
        ),
        native_backend=backend,
    )
    result = verifier.verify(
        _body_points(), InertialState(position=np.array([0.0, 0.0, -50.0])), [0, 0, 0]
    )
    assert not result.accepted
    assert result.rejection_reason == "search_boundary_reached"


def test_bbs3d_adapter_rejects_weak_geometry_score():
    backend = _FakeNative(correction=(0.0, 0.0, 0.0), score_fraction=0.2)
    verifier = BBS3DCandidateVerifier(
        _terrain(),
        BBS3DVerifierConfig(
            minimum_score_fraction=0.3,
            minimum_source_points=64,
        ),
        native_backend=backend,
    )
    result = verifier.verify(
        _body_points(), InertialState(position=np.array([0.0, 0.0, -50.0])), [0, 0, 0]
    )
    assert not result.accepted
    assert result.rejection_reason == "geometric_score_below_threshold"
    assert result.score_fraction == pytest.approx(0.2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"maximum_level": -1},
        {"minimum_score_fraction": 0.0},
        {"maximum_yaw_deg": 45.0},
        {"minimum_source_points": 100, "maximum_source_points": 99},
    ],
)
def test_bbs3d_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        BBS3DVerifierConfig(**overrides)


def test_pinned_native_bbs3d_recovers_a_synthetic_planar_candidate_when_built():
    library = (
        Path(__file__).resolve().parents[1]
        / "native"
        / "bbs3d_bridge"
        / "build"
        / "liboceansim_bbs3d.so"
    )
    if os.environ.get("OCEANSIM_BBS3D_LIBRARY"):
        library = Path(os.environ["OCEANSIM_BBS3D_LIBRARY"])
        assert library.is_file(), "explicit native 3D-BBS library is missing"
    if not library.is_file():
        pytest.skip("optional 3D-BBS bridge has not been built")

    terrain = _terrain(size=201)
    grid = terrain.navigation_elevation_m
    axis_x = (
        np.arange(grid.shape[1]) - 0.5 * (grid.shape[1] - 1)
    ) * terrain.resolution_m
    axis_y = (
        np.arange(grid.shape[0]) - 0.5 * (grid.shape[0] - 1)
    ) * terrain.resolution_m
    x, y = np.meshgrid(axis_x, axis_y)
    truth = np.array([45.0, -35.0, np.deg2rad(28.0)])
    rotation = rotation_from_rpy(np.deg2rad(2.0), np.deg2rad(-3.0), truth[2])
    vehicle_position = np.array([truth[0], truth[1], -50.0])
    world = np.column_stack((x.ravel(), y.ravel(), grid.ravel()))
    body = (world - vehicle_position) @ rotation
    selected = (
        (np.linalg.norm(body[:, :2], axis=1) < 145.0)
        & (body[:, 0] > -120.0)
        & (body[:, 0] < 120.0)
    )
    body = body[selected][::2]
    verifier = BBS3DCandidateVerifier(
        terrain,
        BBS3DVerifierConfig(
            maximum_translation_m=25.0,
            maximum_yaw_deg=8.0,
            minimum_score_fraction=0.2,
            minimum_source_points=128,
            timeout_milliseconds=10_000,
        ),
        library_path=library,
        require_native=True,
    )
    candidate = truth + np.array([10.0, -5.0, np.deg2rad(3.0)])

    result = verifier.verify(
        body,
        InertialState(rotation=rotation, position=vehicle_position),
        candidate,
    )

    assert result.accepted
    np.testing.assert_allclose(result.matched_pose2[:2], truth[:2], atol=5.1)
    assert abs(np.rad2deg(result.matched_pose2[2] - truth[2])) <= 2.1
    assert result.score_fraction > 0.9
