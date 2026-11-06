from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from TrainBandAnchoredNativeSonar import compare_example,record_epoch
from TrainNativeLocalSonarWavelet import array_sha


def example(identity=False):
    target=np.arange(24,dtype=np.float32).reshape(4,6)
    noisy=target+.7;mask=np.ones(target.shape,bool)
    item=dict(old_feature_SHA256=['a'*64,'b'*64],target=target,noisy=noisy,mask=mask,
        context=[0,8,4,14],scale=.5,source_complex128_context_SHA256='c'*64,parameters_SHA256='d'*64)
    saved=dict(identity=identity,features_SHA256=item['old_feature_SHA256'][0 if identity else 1],
        target_log_SHA256=array_sha(target),noisy_log_SHA256=array_sha(target if identity else noisy),
        support_SHA256=array_sha(mask),parameters_SHA256='d'*64,
        source_complex128_context_SHA256='c'*64,context=[0,8,4,14],scale=.5)
    return saved,item


@pytest.mark.parametrize('identity',[False,True])
def test_source_noise_features_and_labels_are_all_checked_before_new_training(identity):
    saved,item=example(identity);assert compare_example(saved,item)==(0 if identity else 1)
    for key in ('target_log_SHA256','noisy_log_SHA256','features_SHA256','support_SHA256',
                'parameters_SHA256','source_complex128_context_SHA256'):
        bad=deepcopy(saved);bad[key]='changed'
        with pytest.raises(ValueError):compare_example(bad,item)
    bad=deepcopy(saved);bad['context'][1]+=1
    with pytest.raises(ValueError):compare_example(bad,item)


def test_exact_frozen_generation_schedule_is_not_inferred_from_time_or_status():
    for epoch in (0,1,4):
        row=dict(seed=9163000+epoch*100+4*8+3,source_index=4,beam=3)
        assert record_epoch(row)==epoch
        row['seed']+=1
        with pytest.raises(ValueError):record_epoch(row)
