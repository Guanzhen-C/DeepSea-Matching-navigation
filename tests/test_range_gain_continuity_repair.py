from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from RangeGainContinuityRepair import repair, smooth_parts


def inputs(n=240, zero_tail=13):
    a = np.full((n, 1024), .4, np.float64)
    old = np.full(a.shape, -.7, np.float32)
    if zero_tail:
        a[:, -zero_tail:] = 0
        old[:, -zero_tail:] = 0
    return old, a, [n//2, n-n//2], 5000-zero_tail


def test_constant_matching_gain_is_identity_without_brightness_reset():
    old, a, parts, end = inputs()
    values, meta = repair(old, a, parts, full_width=5000, valid_end=end)
    for value in values.values():
        np.testing.assert_array_equal(value, old[:, -512:])
    assert meta['statistical_radiometric_correction']


def test_removes_known_boundary_gain_ramp_and_preserves_zeros_and_body():
    old, a, parts, end = inputs()
    before = old.copy()
    old[:, -141:-13] = 0
    values, _ = repair(old, a, parts, full_width=5000, valid_end=end)
    for value in values.values():
        np.testing.assert_array_equal(value[:, -141:-13], before[:, -141:-13])
        np.testing.assert_array_equal(value[:, -13:], 0)
        np.testing.assert_array_equal(value[:, :115], before[:, -512:-397])
    np.testing.assert_array_equal(a[:, -13:], 0)


def test_known_commonmode_stripes_removed_but_local_range_contrast_retained():
    old, a, parts, end = inputs(n=640)
    t = np.arange(len(a))
    perturb = .4*np.sin(2*np.pi*.16*t)
    a[:, -141:-13] *= np.exp(perturb[:, None])
    # Localised target influences far fewer than half the reference cells.
    a[100:105, -45:-40] *= 3
    values, _ = repair(old, a, parts, full_width=5000, valid_end=end)
    clean_log = np.log(.4)-.7
    out = np.log(a[:, -141:-13])+values['gain_and_commonmode'][:, -141:-13]
    interior = slice(40, 280)
    assert np.std(np.median(out, axis=1)[interior]-clean_log) < .005
    # Same-row local contrast is exactly retained; absolute target amplitude
    # is not claimed universally correct without actual clean observations.
    np.testing.assert_allclose(out[100:105, -32:-27]-out[100:105, -42:-37], np.log(3), atol=2e-7)


def test_parts_are_not_mixed_and_inputs_not_mutated():
    x = np.r_[np.ones(60), np.ones(60)*20]
    np.testing.assert_allclose(smooth_parts(x, [60, 60]), x, atol=1e-14)
    old, a, parts, end = inputs()
    old_before, a_before = old.copy(), a.copy()
    repair(old, a, parts, full_width=5000, valid_end=end)
    np.testing.assert_array_equal(old, old_before)
    np.testing.assert_array_equal(a, a_before)


@pytest.mark.parametrize('fault', ['dtype', 'missing', 'end', 'parts', 'nan'])
def test_invalid_or_incomplete_input_rejected(fault):
    old, a, parts, end = inputs()
    if fault == 'dtype':
        old = old.astype(np.float64)
    elif fault == 'missing':
        a[20, -100] = 0
    elif fault == 'end':
        end -= 300
    elif fault == 'parts':
        parts = [120, 119]
    else:
        old[1, 1] = np.nan
    with pytest.raises(ValueError):
        repair(old, a, parts, full_width=5000, valid_end=end)
