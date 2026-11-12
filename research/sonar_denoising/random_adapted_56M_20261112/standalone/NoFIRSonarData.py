"""Verified multitype data and new no-FIR augmentation, NOT clean truth.

Only synthetic fields are spatially interpolated. Original measured pixels,
reference amplitudes, acquisition splits and actual available phase stay put.
This is an appearance/noise coverage hypothesis, not identified device noise.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from NoFIRSonarWavelet import raw_features, native_context


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_text(value, **kwargs):
    """Strict finite JSON; NumPy metadata never terminates actual training."""
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f'unsupported experiment metadata: {type(item).__name__}')
    return json.dumps(value, default=convert, allow_nan=False, ensure_ascii=False, **kwargs)


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json_text(value, indent=2)+'\n')
    tmp.replace(path)


def plan_source(records, phase, index):
    """Global deterministic draws shared across ranks; no held-out admission."""
    if phase not in ('pretrain', 'adapt') or index < 0:
        raise ValueError('explicit nonnegative training draw required')
    if phase == 'adapt' and index % 5 != 4:
        pool = [r for r in records if r['role'] == 'native_train']
        return pool[(index-index//5) % len(pool)]
    public_index = index if phase == 'pretrain' else index//5
    domain = (0, 0, 2, 1)[public_index % 4]
    pool = [r for r in records if r['role'] == 'public_train' and r['domain'] == domain]
    if not pool:
        raise ValueError('all three public training domains required')
    ordinal = public_index//4
    if domain == 0:
        ordinal = 2*ordinal+public_index % 2
    return pool[ordinal % len(pool)]


class Sources:
    def __init__(self):
        self.cache = OrderedDict()

    def array(self, item):
        key = item['id']
        path = Path(item['path'])
        current = path.stat()
        if [current.st_size, current.st_mtime_ns] != item['file_guard']:
            raise ValueError('admitted source changed during training')
        if key not in self.cache:
            if item['domain'] == 0:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'), np.float64)/255.
                value = np.einsum('hwc,c->hw', rgb, np.array([.2126, .7152, .0722]))
            else:
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                if 'beam' in item:
                    value = value[item['beam']]
            if list(value.shape) != item['shape']:
                raise ValueError('actual native shape differs from admission manifest')
            self.cache[key] = value
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def context(self, item, seed, *, center=False):
        values = self.array(item)
        rng = random.Random(seed)
        h, w = min(96, values.shape[0]), min(384, values.shape[1])
        y = (values.shape[0]-h)//2 if center else rng.randrange(values.shape[0]-h+1)
        x = (values.shape[1]-w)//2 if center else rng.randrange(values.shape[1]-w+1)
        # No-data is not a clean training label. Sampling depends only on
        # observed positive support, never model error or a true terrain mask.
        for _ in range(8):
            if np.any(np.abs(values[y:y+h, x:x+w]) > 0):
                break
            y, x = rng.randrange(values.shape[0]-h+1), rng.randrange(values.shape[1]-w+1)
        if not np.any(np.abs(values[y:y+h, x:x+w]) > 0):
            positive = np.abs(values) > 0
            if not positive.any():
                raise ValueError('admitted source has no observed support')
            py, px = np.unravel_index(int(positive.argmax()), values.shape)
            y, x = int(min(py, values.shape[0]-h)), int(min(px, values.shape[1]-w))
        box, inner = native_context(values.shape, (y, x, h, w))
        top, left, bottom, right = box
        z = np.array(values[top:bottom, max(0, left-1):right], copy=True)
        if item['domain'] >= 2:
            z = z.astype(np.complex64)
        else:
            z = z.astype(np.float32)
        return z, left, inner, (y, x, h, w)


@torch.no_grad()
def augment(values, domain, generator, profile, *, identity=False):
    """No frequency/bandpass FIR even on synthetic noise in this recipe."""
    if identity:
        return values.clone(), dict(identity=True, frequency_FIR=False)
    if values.ndim != 2 or (domain >= 2) != values.is_complex():
        raise ValueError('actual typed native observations required')
    amplitude = values.abs()
    support = amplitude > 0
    dtype, device = amplitude.dtype, amplitude.device
    def rand(lo, hi):
        return lo+(hi-lo)*float(torch.rand((), device=device, generator=generator))
    h, w = values.shape
    x = torch.arange(w, device=device, dtype=dtype)[None, :]-(w-1)/2
    y = torch.arange(h, device=device, dtype=dtype)[:, None]
    periods = profile['reference_period_mosaic_rows' if domain == 0 else 'reference_period_records']
    k = min(len(periods)-1, int(rand(0, len(periods))))
    period = periods[k]*rand(.6, 1.6)
    slope = profile['reference_abs_native_slopes'][k]*rand(.6, 1.6)
    if domain != 0:
        slope /= 4
    if rand(0, 1) < .5:
        slope = -slope
    margin = abs(slope)*w+5*period
    n = int((h+2*margin)/(.78*period))+2
    centers = torch.cumsum(.78+.44*torch.rand(n, device=device, generator=generator), 0)*period-margin
    widths = period*(.35+.30*torch.rand(n, device=device, generator=generator))/2.354820045
    strengths = .35+1.45*torch.rand(n, device=device, generator=generator)
    field = torch.zeros_like(amplitude)
    bend = rand(-.15, .15)*x.square()/max(w/2, 1)
    wobble = period*rand(.01, .12)*torch.sin(2*torch.pi*x/max(w, 1)*rand(.5, 2.5)+rand(-3.14, 3.14))
    coordinate = y-slope*(x+bend)-wobble
    for start in range(0, n, 12):
        c, s, a = (v[start:start+12, None, None] for v in (centers, widths, strengths))
        field += (a*torch.exp(-.5*((coordinate[None]-c)/s).square())).sum(0)
    knots = .25+.75*torch.rand((1, 1, 4, 8), device=device, generator=generator)
    field *= F.interpolate(knots, size=(h, w), mode='bilinear', align_corners=True)[0, 0]
    levels = (.03, .08, .2, .5, 1., 1.25)
    scale = levels[min(len(levels)-1, int(rand(0, len(levels))))]
    maximum = amplitude[support].max()
    rms = maximum*torch.sqrt((amplitude[support]/maximum).square().mean())
    if domain >= 2:
        re = torch.randn((h, w), device=device, generator=generator)
        im = torch.randn((h, w), device=device, generator=generator)
        phase = 2*torch.pi*rand(-.49, .49)*x+rand(-3.14, 3.14)
        carrier = torch.complex(phase.cos(), phase.sin())
        carrier_fraction = rand(0., 1.)
        field_complex = field*(carrier_fraction*carrier+(1-carrier_fraction)*torch.complex(re, im)/2**.5)
        field_complex = torch.where(support, field_complex, 0)
        norm = torch.sqrt(field_complex.abs()[support].square().mean())
        observed = values+field_complex*(scale*rms/norm)
    else:
        grain = torch.sqrt(-torch.rand((h, w), device=device, generator=generator).clamp_min(1e-20).log())
        delta = field*grain
        norm = torch.sqrt(delta[support].square().mean())
        observed = values+delta*(scale*rms/norm)
        if domain == 0:
            observed = observed.clamp(0, 1)  # display saturation, not physical amplitude clipping
    observed = torch.where(support, observed, values)
    if not torch.isfinite(observed).all():
        raise ValueError('nonfinite synthetic native observation')
    return observed, dict(identity=False, scale=scale, period=period, slope=slope,
        frequency_FIR=False, source_noise_calibrated=False,
        patch_augmentation_NOT_identical_whole_field_noise=True)


def example(sources, item, index, phase, device, profile, *, validation=False, force_identity=False):
    seed = 9518000+index+(0 if phase == 'pretrain' else 2000000)
    z, left, inner, core_global = sources.context(item, seed, center=validation)
    reference = z
    g = torch.Generator(device=device).manual_seed(seed)
    identity = force_identity or (not validation and index % 5 == 0)
    observed, noise = augment(torch.from_numpy(z).to(device), item['domain'], g, profile, identity=identity)
    observed = observed.cpu().numpy()
    f, q, mask = raw_features(observed, item['domain'], first_column=left)
    _, target, target_mask = raw_features(reference, item['domain'], first_column=left)
    if not np.array_equal(mask, target_mask):
        raise ValueError('source support changed by augmentation')
    cy, cx, h, w = inner
    tensors = (torch.from_numpy(f[None]).to(device), torch.from_numpy(mask[None, None]).to(device),
               torch.from_numpy(q[None, None]).to(device), torch.from_numpy(target[None, None]).to(device))
    meta = dict(id=item['id'], domain=item['domain'], seed=seed, identity=identity,
                native_core=list(core_global), context_shape=list(q.shape), noise=noise,
                observed_SHA256=hashlib.sha256(observed.tobytes()).hexdigest())
    return tensors, (slice(None), slice(None), slice(cy, cy+h), slice(cx, cx+w)), meta
