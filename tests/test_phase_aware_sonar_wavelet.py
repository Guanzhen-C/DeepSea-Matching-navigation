from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from PhaseAwareSonarWavelet import PhaseAwareSonarWavelet
from PublicEnvelopeCoherentModel import PublicEnvelopeCoherentModel
from PublicSonarComplexCorruption import complex_phase_features
from PublicComplexPhaseExperiment import (ComplexObservation,CompleteComplexBank,corrupt_complete,
    restore_complete,evaluate_pair,file_sha)
from test_acoustic_stripe_augmentation import PROFILE


def pair():
    torch.set_num_threads(2);torch.manual_seed(81)
    previous=PublicEnvelopeCoherentModel(width=2)
    torch.nn.init.normal_(previous.network.heads[1].weight,std=.02)
    model=PhaseAwareSonarWavelet.from_envelope(dict(model_family=previous.family,width=2,model=previous.state_dict()))
    return previous,model


def observation(tmp_path):
    rng=np.random.default_rng(82)
    values=(rng.normal(size=(49,229))+1j*rng.normal(size=(49,229))).astype(np.complex64)
    values[:,:3]=0;path=tmp_path/'native_complex.npy';np.save(path,values)
    row=dict(sample_id='public_fixture',domain='complex_field',split='train_aux',is_clean_truth=False,cache_sha256=file_sha(path))
    return ComplexObservation.read(row,path,torch.device('cpu'))


def test_warm_start_reparameterization_preserves_previous_amplitude_function():
    previous,model=pair();torch.manual_seed(83)
    z=torch.complex(torch.randn(1,1,27,225),torch.randn(1,1,27,225))
    f,q,mask,origin=complex_phase_features(z)
    with torch.no_grad():
        a=previous.residual(q,mask,origin=origin)
        b=model.feature_residual(f,mask,use_phase=False)
        c=model.feature_residual(f,mask,use_phase=True)
    torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(b,c,rtol=0,atol=0)
    assert torch.count_nonzero(model.network.input.weight[:,1:])==0
    with pytest.raises(ValueError,match='feature_residual'):model.residual(q,mask)


def test_phase_feature_weights_learn_and_control_cannot_use_phase():
    _,model=pair();torch.manual_seed(84)
    f=torch.randn(1,3,25,225);mask=torch.ones(1,1,25,225,dtype=torch.bool)
    model.feature_residual(f,mask,use_phase=True).square().mean().backward()
    assert model.network.input.weight.grad[:,1:].abs().sum()>0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():model.network.input.weight[:,1:].normal_(std=.01)
    a=model.feature_residual(f,mask,use_phase=False)
    changed=f.clone();changed[:,1:]*=3
    b=model.feature_residual(changed,mask,use_phase=False)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    c=model.feature_residual(changed,mask,use_phase=True)
    assert (c-a).abs().max()>1e-6


def test_full_source_feature_crops_keep_observed_left_phase_neighbour(tmp_path):
    ref=observation(tmp_path);before=file_sha(ref.path)
    bank=CompleteComplexBank([ref],PROFILE,'cpu');record=bank.refresh(0)
    f,q,target,mask,meta=bank.sample(random.Random(85),height=16,width=32,halo=4)
    top,left,bottom,right=meta['native_context']
    source=ref.features if meta['identity'] else bank.fields[0].features
    np.testing.assert_array_equal(f[0].numpy(),source[...,top:bottom,left:right])
    assert left>3 and np.count_nonzero(f[0,1:,0,0])>0
    assert record['fields'][0]['native_shape']==[49,229]
    assert file_sha(ref.path)==before
    assert np.array_equal(q[~mask].numpy(),target[~mask].numpy())


def test_native_phase_feature_tiling_matches_complete_field_inference(tmp_path):
    ref=observation(tmp_path);_,model=pair()
    with torch.no_grad():model.network.input.weight[:,1:].normal_(std=.01)
    f=torch.from_numpy(ref.features.copy())[None];mask=torch.from_numpy(ref.support.copy())[None]
    with torch.inference_mode():full=model.feature_residual(f,mask,use_phase=True)[0].numpy()
    tiled=restore_complete(model,ref.features,ref.support,'cpu',use_phase=True,core=32)
    np.testing.assert_allclose(full,tiled,rtol=3e-4,atol=3e-5)
    assert tiled.shape==(1,49,229) and np.count_nonzero(tiled[~ref.support])==0


def test_complete_pair_validation_reports_all_pixels_not_a_fake_scene_holdout(tmp_path):
    ref=observation(tmp_path);_,model=pair()
    field=corrupt_complete(ref,PROFILE,'cpu',86,.75)
    reports=evaluate_pair({'control':(model,'cpu',False),'phase':(model,'cpu',True)},[(ref,field)])
    assert reports['control']==reports['phase']
    assert reports['control']['complete_native_values_processed']==49*229
    assert reports['control']['all_validation_textures_also_in_training']
    assert not reports['control']['independent_scene_generalization']
