"""A bounded 300-update execution check, not a denoising quality claim."""
from pathlib import Path
import argparse
import subprocess
import sys

from NoFIRSonarData import read, save
from RunRandomRippleSonarTrial import reuse_admission

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--admission-root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--world-size', type=int, default=4)
    a = p.parse_args()
    a.root.mkdir(exist_ok=False)
    reuse_admission(a)
    save(a.root / 'random_ripple_profile.json', read(a.admission_root / 'random_ripple_profile.json'))
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node', str(a.world_size), str(Path(__file__).with_name('TrainNoFIRSonarDistributed.py')),
        '--root', str(a.root), '--phase', 'pretrain', '--steps', str(a.steps), '--width', '96',
        '--random-ripple', '--smoke'], check=True)
