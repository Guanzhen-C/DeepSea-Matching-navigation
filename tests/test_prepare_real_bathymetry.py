from __future__ import annotations

import numpy as np
import pytest

from standalone.prepare_real_bathymetry import (
    _nearest_registered_resample,
    _robust_unit_normalize,
    _valid_mask,
)


def test_valid_mask_respects_each_products_nodata_not_intensity_sign() -> None:
    float_nodata = np.float32(-3.402823e38)
    em300 = np.array([float_nodata, -60.8, -35.2, 0.0], dtype=np.float32)
    assert _valid_mask(em300, float(float_nodata)).tolist() == [
        False,
        True,
        True,
        True,
    ]

    reson_7125 = np.array([0, 1, 254, 255], dtype=np.uint8)
    assert _valid_mask(reson_7125, 255.0).tolist() == [True, True, True, False]

    csumb_swath = np.array([0, 2, 128, 254], dtype=np.uint8)
    assert _valid_mask(csumb_swath, 0.0).tolist() == [False, True, True, True]


def test_exact_grid_registration_preserves_negative_em300_db() -> None:
    values = np.array([[-42.0, -35.0], [-30.0, -20.0]], dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    world = (5.0, 0.0, 0.0, -5.0, 583532.5, 4078337.5)

    registered, registered_valid = _nearest_registered_resample(
        values, valid, world, values.shape, world
    )

    np.testing.assert_array_equal(registered, values)
    np.testing.assert_array_equal(registered_valid, valid)


def test_robust_normalization_is_monotonic_and_keeps_invalid_as_nan() -> None:
    values = np.array([[np.nan, -60.0, -40.0, -20.0, 0.0]], dtype=np.float32)
    valid = np.isfinite(values)

    normalized, low, high = _robust_unit_normalize(
        values, valid, lower_percentile=0.0, upper_percentile=100.0
    )

    assert low == pytest.approx(-60.0)
    assert high == pytest.approx(0.0)
    assert np.isnan(normalized[0, 0])
    np.testing.assert_allclose(normalized[0, 1:], [0.0, 1 / 3, 2 / 3, 1.0])

