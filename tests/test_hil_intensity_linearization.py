from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam.esikf import AcousticESIKF
from oceansim.slam.math3d import rotation_from_rpy
from standalone.hil_intensity_linearization import build_batched_linearization


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("anchor", [None, 3])
@pytest.mark.parametrize("count", [0, 1, 200])
def test_batched_measurements_preserve_every_output_bit(planar, anchor, count):
    rng = np.random.default_rng(31)
    estimator = SimpleNamespace(sensor_rotation_body=rotation_from_rpy(.02, -.01, .1),
        sensor_translation_body=np.array([.3, -.2, .1]), config=SimpleNamespace(
            intensity_planar_only=planar, intensity_noise_std=.04, intensity_surface_consistency_std=.2))
    state = SimpleNamespace(rotation=rotation_from_rpy(.06, -.03, .7), position=np.array([10., -20., -40.]))
    points = rng.normal(0., 30., (count, 3)); measured = rng.uniform(0., 1., count)
    samples = [None if i % 7 == 0 else SimpleNamespace(gradient_xy=rng.normal(0., .02, 2),
        intensity=float(rng.uniform(0., 1.)), variance=float(rng.uniform(.01, .1)),
        surface_z=np.nan if i % 5 == 0 else float(rng.normal(-40., 1.)),
        surface_variance=float(rng.uniform(-.01, .1)), reference_observation_id=i//3, anchor_id=i//9)
        for i in range(count)]
    model = SimpleNamespace(query_batch=lambda *a, **k: list(samples),
                            query_fixed_anchor_batch=lambda *a, **k: list(samples))
    original = AcousticESIKF._intensity_linearization
    kwargs = dict(anchor_id=anchor, maximum_observation_id=100, affine_group_ids=np.arange(count)%3)
    expected = original(estimator, state, points, measured, model, **kwargs)
    actual = build_batched_linearization(original)(estimator, state, points, measured, model, **kwargs)
    for a, b in zip(expected, actual):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def test_unknown_linearization_source_fails_closed():
    def different(self, state):
        return state
    with pytest.raises(ValueError, match="unrecognized"):
        build_batched_linearization(different)
