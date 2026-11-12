"""The same whole-coordinate random ripple recipe for optimization and tests."""
from pathlib import Path
import hashlib
import random

import numpy as np
import torch

from NoFIRSonarData import Sources, plan_source
from NoFIRSonarWavelet import raw_features
from RandomTriggeredSonarNoise import apply_scenario, draw_scenario


def example(sources, item, index, phase, device, profile, *, validation=False, force_identity=False):
    seed = 17618000 + index + (0 if phase == 'pretrain' else 2000000)
    z, left, inner, core_global = sources.context(item, seed, center=validation)
    cy, cx, h, w = inner
    y, x, _, _ = core_global
    origin = (y - cy, x - cx - int(left > 0))
    support = None
    if item.get('noise_support_file'):
        path = Path(item['noise_support_file'])
        stamp = path.stat()
        if [stamp.st_size, stamp.st_mtime_ns] != item['noise_support_file_guard']:
            raise ValueError('Reviewed augmentation support changed during training.')
        full_support = np.load(path, mmap_mode='r', allow_pickle=False)
        support = np.array(full_support[origin[0]:origin[0] + z.shape[0], origin[1]:origin[1] + z.shape[1]], copy=True)
    scenario = draw_scenario(profile, item['shape'], item['domain'], seed, side=item.get('native_side'))
    # Do not alias index%5 with the 4-native/1-public source schedule: that
    # would leave every fourth native beam with identity samples ONLY.
    identity = force_identity or (not validation and random.Random(seed + 713).random() < .2)
    observed, noise = apply_scenario(torch.from_numpy(z).to(device), profile, scenario,
        origin=origin, support=support, reference_RMS=item['global_reference_RMS'],
        beam=item.get('beam'), identity=identity)
    observed = observed.cpu().numpy()
    f, q, mask = raw_features(observed, item['domain'], first_column=left)
    _, target, target_mask = raw_features(z, item['domain'], first_column=left)
    if not np.array_equal(mask, target_mask):
        raise ValueError('Reference positive support changed by ripple augmentation.')
    tensors = (torch.from_numpy(f[None]).to(device), torch.from_numpy(mask[None, None]).to(device),
               torch.from_numpy(q[None, None]).to(device), torch.from_numpy(target[None, None]).to(device))
    meta = dict(id=item['id'], domain=item['domain'], seed=seed, identity=identity,
        native_core=list(core_global), context_shape=list(q.shape), noise=noise,
        observed_SHA256=hashlib.sha256(observed.tobytes()).hexdigest())
    return tensors, (slice(None), slice(None), slice(cy, cy + h), slice(cx, cx + w)), meta
