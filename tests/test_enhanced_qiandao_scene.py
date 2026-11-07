from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from PrepareQiandaoFullFrameJointScene import appearance_values
from NativeSonarPaletteStream import amplitude_indices


def test_appearance_is_exact_original_native_display_quantizer_then_fixed_lut():
    lookup = np.linspace(0, 1, 256)**.7
    values = np.array([0, 1e-9, 1/256-1e-12, 1/256, .5, 1., 5.])
    np.testing.assert_array_equal(appearance_values(values, lookup), lookup[amplitude_indices(values)])
    assert appearance_values(np.zeros(12), lookup).sum() == 0
    assert np.all(appearance_values(values, lookup) <= 1)


@pytest.mark.parametrize('bad', [np.array([-1.]), np.array([np.nan]), np.array([np.inf])])
def test_nonphysical_native_values_are_not_filled_or_silently_clipped(bad):
    with pytest.raises(ValueError):
        appearance_values(bad, np.linspace(0, 1, 256))


def test_appearance_mapping_cannot_create_intensity_out_of_missing_black():
    lookup = np.linspace(.01, 1, 256)
    with pytest.raises(ValueError):
        appearance_values(np.zeros(12), lookup)


def test_nonmonotonic_or_misshaped_appearance_is_rejected():
    lookup = np.linspace(0, 1, 256)
    lookup[100] = 0
    for bad in (lookup, np.linspace(0, 1, 255)):
        with pytest.raises(ValueError):
            appearance_values(np.array([.1]), bad)
