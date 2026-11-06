from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from MatchedSonarWavelet import MatchedSonarWavelet,infer_matched,infer_group
from MatchedSonarSignal import negative_frontend,source_logs
from PublicEnvelopeCoherentModel import DirectionalNoiseConsensus


def test_mandatory_log_gain_averaging_distorts_even_known_correct_bright_target_gain():
    torch.set_num_threads(2)
    reference=torch.ones(1,1,21,257);reference[:,:,:,128]=10
    observed=reference+1
    oracle=(reference/observed).log();mask=torch.ones_like(oracle,dtype=torch.bool)
    changed=DirectionalNoiseConsensus()(oracle,mask)
    ideal=observed*oracle.exp();averaged=observed*changed.exp()
    torch.testing.assert_close(ideal,reference)
    ratio=float((averaged[0,0,10,128]/reference[0,0,10,128]).detach())
    assert .59<ratio<.61
    # A local path can retain that oracle result. This is a synthetic operator
    # counterexample, not a claim that a trained model is an oracle or cannot
    # precompensate the old aggregation.
    near_local=.999999*oracle+.000001*changed
    assert abs(float((observed[0,0,10,128]*near_local[0,0,10,128].exp()/10).detach())-1)<1e-6


def test_composed_frontend_and_nonzero_network_have_no_native_partition_seams():
    torch.set_num_threads(2);torch.manual_seed(42)
    rng=np.random.default_rng(93);z=(rng.normal(size=(173,1001))+1j*rng.normal(size=(173,1001))).astype(np.complex64)
    model=MatchedSonarWavelet(4)
    with torch.no_grad():model.network.heads[1].weight.normal_(0,.01)
    for front in (False,True):
        a,_=infer_matched(model,z,'cpu',frontend=front,core_shape=(64,256))
        b,_=infer_matched(model,z,'cpu',frontend=front,core_shape=(128,512))
        np.testing.assert_allclose(a,b,atol=2e-6,rtol=1e-5)


def test_zero_neural_contribution_is_not_called_learned_frontend_success():
    rng=np.random.default_rng(45);z=(rng.normal(size=(32,1409))+1j*rng.normal(size=(32,1409))).astype(np.complex64)
    total,neural=infer_matched(None,z,'cpu',frontend=True,zero_neural=True)
    before,_=source_logs(z);prepared,_=source_logs(negative_frontend(z))
    np.testing.assert_allclose(total,prepared-before,atol=1e-6,rtol=1e-5)
    assert not np.any(neural) and np.max(abs(total))>.1


def test_full_source_preparation_matches_composed_context_inference_for_each_operator():
    torch.set_num_threads(2);torch.manual_seed(123)
    rng=np.random.default_rng(42);z=(rng.normal(size=(137,1001))+1j*rng.normal(size=(137,1001))).astype(np.complex64)
    model=MatchedSonarWavelet(4)
    with torch.no_grad():model.network.heads[1].weight.normal_(0,.01)
    grouped,meta=infer_group({'raw':(model,False),'front':(model,True)},z,'cpu',core_shape=(64,512))
    for name,front in (('raw',False),('front',True)):
        actual,_=infer_matched(model,z,'cpu',frontend=front,core_shape=(64,256))
        np.testing.assert_allclose(actual,grouped[name],rtol=1e-5,atol=2e-6)
    assert set(meta['prepared_fields'])=={'raw','front'} and meta['neural']['raw']['RMS']>0
