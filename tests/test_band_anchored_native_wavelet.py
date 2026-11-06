from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from BandAnchoredNativeSonarFeatures import band_features
from BandAnchoredNativeSonarWavelet import BandAnchoredNativeSonarWavelet,infer_band_native
from LocalRelativeSonarFeatures import local_features
from DiagnoseNativeCarrierInformation import ideal_fields


def test_opposite_band_full_ideal_controls_are_now_distinguishable():
    a,b,_,_=ideal_fields();old,_,_=local_features(a);old2,_,_=local_features(b)
    assert np.max(abs(old-old2))<1e-6
    new,_,_=band_features(a);new2,_,_=band_features(b)
    assert np.sqrt(np.mean((new[1:]-new2[1:])**2))>.9


def test_amp_ablation_ignores_changed_phase_channels_and_native_tiling_is_consistent():
    torch.set_num_threads(2);torch.manual_seed(9182)
    model=BandAnchoredNativeSonarWavelet(4)
    rng=np.random.default_rng(91);z=(rng.normal(size=(197,633))+1j*rng.normal(size=(197,633))).astype(np.complex64)
    new,_,mask=band_features(z);old,_,_=local_features(z)
    with torch.no_grad():
        model.network.heads[1].weight.normal_(0,.02)
        model.network.input.weight[:,1:].normal_(0,.03)
        a=model.feature_residual(torch.from_numpy(new[None]),torch.from_numpy(mask[None,None]),use_phase=False)
        b=model.feature_residual(torch.from_numpy(old[None]),torch.from_numpy(mask[None,None]),use_phase=False)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a=infer_band_native(model,z,'cpu',core_shape=(64,256))
    b=infer_band_native(model,z,'cpu',core_shape=(128,512))
    np.testing.assert_allclose(a,b,atol=8e-7,rtol=3e-5)
