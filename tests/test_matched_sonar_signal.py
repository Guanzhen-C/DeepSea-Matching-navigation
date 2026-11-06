from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from MatchedSonarSignal import noise_variants,negative_frontend,source_logs
from LocalRelativeSonarFeatures import local_features


def test_acquisition_filter_modifies_noise_only_and_realized_RMS_is_matched():
    rng=np.random.default_rng(81)
    reference=(rng.normal(size=(16,4096))+1j*rng.normal(size=(16,4096))).astype(np.complex64)
    delta=rng.normal(size=reference.shape)+1j*rng.normal(size=reference.shape)
    before=reference.copy();fields,meta=noise_variants(reference,delta,.08)
    np.testing.assert_array_equal(reference,before)
    for name in fields:assert abs(meta[name]['actual_added_complex_noise_RMS_over_source_RMS']-.08)<1e-7
    f=np.fft.fftfreq(reference.shape[1]);window=np.hanning(reference.shape[1])
    powers={name:abs(np.fft.fft((z-reference)*window,axis=-1))**2 for name,z in fields.items()}
    assert powers['wide'][:,abs(f)>.45].sum()/powers['wide'].sum()>.08
    assert powers['limited'][:,abs(f)>.45].sum()/powers['limited'].sum()<1e-6
    zero,_=noise_variants(reference,delta,0)
    for a in zero.values():np.testing.assert_array_equal(a,reference)


def test_true_global_FIR_edges_not_artificial_context_edges_and_source_never_overwritten():
    rng=np.random.default_rng(11);z=(rng.normal(size=(8,2048))+1j*rng.normal(size=(8,2048))).astype(np.complex64)
    z[:2,:200]=0;before=z.copy();full=negative_frontend(z)
    inner=negative_frontend(z[:,399:1700],first_range_bin=399,full_range_bins=2048)
    np.testing.assert_allclose(inner[:,128:-128],full[:,527:1572],rtol=3e-6,atol=2e-7)
    np.testing.assert_array_equal(full[:,:128],z[:,:128]);np.testing.assert_array_equal(full[:,-128:],z[:,-128:])
    np.testing.assert_array_equal(z,before);np.testing.assert_array_equal(full[abs(z)==0],z[abs(z)==0])
    q,mask=source_logs(z);_,expected,m=local_features(z)
    np.testing.assert_array_equal(q,expected);np.testing.assert_array_equal(mask,m)
