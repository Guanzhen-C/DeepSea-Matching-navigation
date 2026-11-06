from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from MatchedSonarArtifacts import check_full_cases
from MatchedSonarWavelet import MatchedSonarWavelet,infer_group


def test_full_native_case_gate_requires_both_laws_all_beams_and_all_range_samples():
    sources=[dict(native_records=744,native_range_bins=12189)]*2
    manifest=dict(sources=sources);previous=dict(reserved_source_indices=[0,1])
    cases=[dict(source_index=i,beam_0based=b,test_noise_law=law,native_values=744*12189,
        source_observation_is_not_clean_truth=True,missing_support_delta_exact_zero=True)
        for i in range(2) for b in range(8) for law in ('wide','limited')]
    check_full_cases(cases,previous,manifest)
    with pytest.raises(ValueError):check_full_cases(cases[:-1],previous,manifest)
    bad=deepcopy(cases);bad[0]['native_values']-=1
    with pytest.raises(ValueError):check_full_cases(bad,previous,manifest)


def test_selected_subset_replay_does_not_change_any_model_prediction_or_neural_digest():
    torch.set_num_threads(2);torch.manual_seed(71)
    rng=np.random.default_rng(23);z=(rng.normal(size=(65,1001))+1j*rng.normal(size=(65,1001))).astype(np.complex64)
    models={}
    for name,front in (('a',False),('b',False),('c',True)):
        model=MatchedSonarWavelet(4)
        with torch.no_grad():model.network.heads[1].weight.normal_(0,.01)
        models[name]=(model,front)
    full,info=infer_group(models,z,'cpu',core_shape=(64,512))
    selected,subset=infer_group({n:models[n] for n in ('a','c')},z,'cpu',core_shape=(64,512))
    for name in selected:
        np.testing.assert_array_equal(selected[name],full[name])
        assert subset['neural'][name]==info['neural'][name]


def test_learned_local_selector_receives_gradients_without_gain_clipping():
    torch.set_num_threads(2);torch.manual_seed(93)
    model=MatchedSonarWavelet(4);f=torch.randn(1,3,32,128);mask=torch.ones(1,1,32,128,dtype=torch.bool)
    with torch.no_grad():model.network.heads[1].weight.normal_(0,.02)
    model.feature_residual(f,mask,use_phase=True).square().mean().backward()
    assert model.local_gate[-1].weight.grad.abs().sum()>0
