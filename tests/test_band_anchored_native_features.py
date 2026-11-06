from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from LocalRelativeSonarFeatures import local_features
from BandAnchoredNativeSonarFeatures import band_features,validate_reference


def field():
    rng=np.random.default_rng(423)
    return (rng.normal(size=(37,237))+1j*rng.normal(size=(37,237))).astype(np.complex64)


def test_amplitude_channel_labels_and_support_are_bitwise_unchanged():
    z=field();z[:7,:13]=0
    a,q,m=local_features(z);b,r,n=band_features(z)
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(q,r);np.testing.assert_array_equal(m,n)
    assert np.max(abs(a[1:]-b[1:]))>.5
    assert np.all(b[:,~m]==0)


def test_amplitude_units_and_constant_phase_remain_invariant_but_frequency_shift_does_not():
    z=field().astype(np.complex128);a,_,m=band_features(z)
    b,_,_=band_features(z*17*np.exp(.48j))
    np.testing.assert_allclose(a,b,atol=3e-7,rtol=3e-6)
    c,_,_=band_features(z*np.exp(.6j*np.arange(z.shape[1])))
    np.testing.assert_allclose(a[0],c[0],atol=3e-7,rtol=3e-6)
    assert np.sqrt(np.mean((a[1:]-c[1:])**2))>.3


def test_real_predecessor_matches_whole_field_and_fixed_reference_rejects_invalid_band():
    z=field();full,_,_=band_features(z)
    context,_,_=band_features(z[5:29,48:198],first_column=49)
    np.testing.assert_array_equal(context[:,2:-2,32:-32],full[:,7:27,81:166])
    for band in ([1,2,3],[15000,-3000],[-20000,-3000],[np.nan,0]):
        with pytest.raises(ValueError):validate_reference(37650,band)
