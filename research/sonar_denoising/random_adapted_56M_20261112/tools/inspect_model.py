"""CPU/meta model inspection only: no weights, measurements, CUDA or training."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
import torch
from NoFIRSonarWavelet import NoFIRSonarMWCNN


def main():
    torch.set_num_threads(2)
    with torch.device('meta'):
        formal=NoFIRSonarMWCNN(96)
    total=sum(p.numel() for p in formal.parameters())
    assert total==56083513
    small=NoFIRSonarMWCNN(2).cpu().eval()
    x=torch.randn(1,3,17,29);mask=torch.ones_like(x[:,:1],dtype=torch.bool)
    with torch.inference_mode():y=small(x,mask,3)
    assert y.shape==(1,1,17,29) and not torch.count_nonzero(y)
    print(json.dumps(dict(model='NoFIRSonarMWCNN',width=96,parameters=total,
        domain_embedding_parameters=formal.domain_affine.weight.numel(),
        formal_parameter_storage_device='meta',small_CPU_shape_test_passed=True,
        initialized_zero_residual_verified=True,actual_pretrained_weights_loaded=False,
        training_started=False,GPU_used=False),indent=2))


if __name__=='__main__':main()
