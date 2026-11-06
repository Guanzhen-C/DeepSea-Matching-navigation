from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from LocalRelativeSonarFeatures import local_features, local_mean


def source(shape=(33,237)):
    rng=np.random.default_rng(928)
    return rng.normal(size=shape)+1j*rng.normal(size=shape)


def test_actual_predecessor_and_finite_context_match_whole_source_interior():
    z=source();full,q,mask=local_features(z)
    cropped,q1,m1=local_features(z[5:28,48:198],first_column=49)
    np.testing.assert_array_equal(cropped[:,2:-2,32:-32],full[:,7:26,81:166])
    np.testing.assert_array_equal(q1,q[5:28,49:198])
    np.testing.assert_array_equal(m1,mask[5:28,49:198])
    assert np.count_nonzero(cropped[1:,:,0])>0


def test_amplitude_units_and_linear_carrier_reference_do_not_change_features():
    z=source();z[:3,:8]=0
    phase=np.exp(1.7j+.31j*np.arange(z.shape[1]))
    before,q,mask=local_features(z);after,q2,mask2=local_features(z*17*phase)
    np.testing.assert_allclose(after,before,atol=3e-7,rtol=3e-6)
    np.testing.assert_allclose((q2-q)[mask],np.log(17),atol=7e-7)
    np.testing.assert_array_equal(mask,mask2)
    assert np.all(before[:,~mask]==0)


def test_mask_normalized_mean_does_not_treat_missing_support_as_zero_observation():
    mask=np.zeros((9,91),bool);mask[2:7,20:70]=True
    a=np.where(mask,7.,999999.)
    mean=local_mean(a,mask)
    np.testing.assert_allclose(mean[mask],7.,rtol=1e-13,atol=1e-13)
    features,q,m=local_features(np.zeros((9,91),np.complex128))
    assert not np.any(features) and not np.any(q) and not np.any(m)


def test_never_mutates_original_and_rejects_invalid_context():
    z=source();before=z.copy();local_features(z)
    np.testing.assert_array_equal(before,z)
    with pytest.raises(ValueError):local_features(z.real)
    with pytest.raises(ValueError):local_features(z,first_column=-1)
