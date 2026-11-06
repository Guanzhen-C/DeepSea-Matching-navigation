from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from FullMeasuredSonarProtocol import supported_total_delta,overlap_block,overlap_tail
from RecordedSubbandNativeStream import overlap_block as previous_block,overlap_tail as previous_tail
from InferMeasuredPublicSonarWavelet import supported_delta as previous_support
from MatchedSonarWavelet import MatchedSonarWavelet,infer_group,context_prediction
from NativeLocalSonarWavelet import context_box


def test_water_support_is_exact_previous_operator_not_gain_clipping():
    rng=np.random.default_rng(7);a=rng.uniform(0,1,(16,1409));a[:,:3]=0
    delta=rng.normal(size=a.shape).astype(np.float32);bounds=np.tile([5.,10.],(16,1))
    expected=previous_support(a,delta,bounds,np.arange(1409),1480.,37650.)
    actual=supported_total_delta(a,delta,bounds,np.arange(1409),1480.,37650.)
    np.testing.assert_array_equal(actual,expected)
    assert not actual[:,:3].any()


def test_part_boundaries_and_final_four_beams_are_exact_and_partition_invariant():
    rng=np.random.default_rng(17);a=rng.random((2,27,8,53));all_rows,tail=previous_block(a)
    expected=np.concatenate((all_rows,previous_tail(tail)))
    output=[];tail=None
    for start,stop in ((0,7),(7,16),(16,27)):
        rows,tail=overlap_block(a[:,start:stop],tail);output.append(rows)
    output.append(overlap_tail(tail))
    np.testing.assert_array_equal(np.concatenate(output),expected)


@pytest.mark.parametrize('front',[False,True])
def test_predeclared_neural_context_audit_replays_full_source_field_with_real_neighbors(front):
    torch.set_num_threads(2);torch.manual_seed(11)
    rng=np.random.default_rng(37);z=(rng.normal(size=(137,1409))+1j*rng.normal(size=(137,1409))).astype(np.complex64)
    model=MatchedSonarWavelet(4)
    with torch.no_grad():model.network.heads[1].weight.normal_(0,.01)
    prediction,_=infer_group({'a':(model,front)},z,'cpu',core_shape=(64,512))
    core=(40,768,32,128);box,inner=context_box(z.shape,core,halo=(80,416));top,left,bottom,right=box
    with torch.inference_mode():
        independently,_=context_prediction(model,z[top:bottom,max(0,left-1):right],'cpu',
            first_model_column=left,full_range_bins=z.shape[-1],frontend=front,core=inner)
    y,x,h,w=core
    np.testing.assert_allclose(prediction['a'][y:y+h,x:x+w],independently,atol=5e-6,rtol=2e-5)
