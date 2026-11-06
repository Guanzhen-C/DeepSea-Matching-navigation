from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"standalone"))
from PublicSonarWavelet import PublicSonarWavelet, haar, inverse_haar


def test_haar_roundtrip_and_energy():
    torch.set_num_threads(2)
    x=torch.randn(2,3,18,26,dtype=torch.float64)
    y=haar(x)
    torch.testing.assert_close(inverse_haar(y),x,atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(y.square().sum(),x.square().sum(),atol=1e-10,rtol=1e-14)


@pytest.mark.parametrize("domain",range(3))
def test_exact_identity_initialization_with_native_odd_sizes(domain):
    torch.set_num_threads(2)
    model=PublicSonarWavelet(width=4)
    x=torch.randn(1,3,17,29)
    base=torch.randn_like(x) if domain==2 else x
    torch.testing.assert_close(model(x,domain,base=base),base,rtol=0,atol=0)


def test_complex_phase_features_are_not_mistaken_for_log_amplitude_base():
    model=PublicSonarWavelet(width=4)
    with pytest.raises(ValueError,match="explicit"):
        model(torch.randn(1,3,8,12),2)


def test_training_gradients_reach_the_correct_domain_head():
    torch.set_num_threads(2)
    model=PublicSonarWavelet(width=4)
    x=torch.randn(1,3,16,24)
    output=model(x,1)
    (output-(x+.1)).square().mean().backward()
    assert model.heads[1].weight.grad.abs().sum()>0
    assert model.heads[0].weight.grad is None
    assert model.heads[2].weight.grad is None
