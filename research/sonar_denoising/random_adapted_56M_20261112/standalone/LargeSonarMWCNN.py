"""Independent MWCNN-style sonar adaptation, never an official pretrained model.

Architecture reference: Liu et al., Multi-Level Wavelet Convolutional Neural
Networks, https://github.com/lpj-github-io/MWCNNv2 (model/mwcnn.py).
Three Haar scales and the encoder/decoder's (2,1)/(2,3,3,2) dilation schedule
are retained. Differences are explicit: three observed sonar feature channels,
one zero-initialized log-gain head, the existing sonar residual consensus/gate,
device/dtype-safe Haar operators, and native boundary padding. No upstream
weights or source files are copied. This family cannot load the frozen model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from BandAnchoredNativeSonarFeatures import PROTOCOL
from NativeLocalSonarWavelet import SparseDirectionalNoiseConsensus
from PublicSonarWavelet import haar, inverse_haar


def conv(in_channels, out_channels, dilation=1):
    return nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                     dilation=dilation)


def stage(in_channels, out_channels, dilations):
    layers = []
    for dilation in dilations:
        layers.extend((conv(in_channels, out_channels, dilation), nn.ReLU()))
        in_channels = out_channels
    return nn.Sequential(*layers)


class SonarMWCNNBackbone(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        if not isinstance(width, int) or width < 1:
            raise ValueError('positive integral channel width required')
        self.width = width
        self.head = stage(3, width, (1,))
        self.encode0 = stage(width, width, (2, 1))
        self.encode1 = stage(width*4, width*2, (1, 2, 1))
        self.encode2 = stage(width*8, width*4, (1, 2, 1))
        self.middle = stage(width*16, width*8, (1, 2, 3, 3, 2))
        self.middle_out = stage(width*8, width*16, (1,))
        self.decode2 = stage(width*4, width*4, (2, 1))
        self.up2 = stage(width*4, width*8, (1,))
        self.decode1 = stage(width*2, width*2, (2, 1))
        self.up1 = stage(width*2, width*4, (1,))
        self.decode0 = stage(width, width, (2, 1))
        self.tail = conv(width, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, features):
        height, width = features.shape[-2:]
        x = F.pad(features, (0, (-width) % 8, 0, (-height) % 8),
                  mode='replicate')
        e0 = self.encode0(self.head(x))
        e1 = self.encode1(haar(e0))
        e2 = self.encode2(haar(e1))
        deep = inverse_haar(self.middle_out(self.middle(haar(e2)))) + e2
        d1 = inverse_haar(self.up2(self.decode2(deep))) + e1
        d0 = inverse_haar(self.up1(self.decode1(d1))) + e0
        return self.tail(self.decode0(d0))[..., :height, :width]


class LargeSonarMWCNN(nn.Module):
    family = 'experimental_three_level_mwcnn_native_sonar_log_gain_v1'
    feature_protocol = PROTOCOL
    # Conservative graph support: backbone radius < 160; consensus adds at
    # most 4 records / 96 bins. Features/FIR are prepared on COMPLETE fields.
    native_halo = (208, 288)
    lattice = 8

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.network = SonarMWCNNBackbone(width)
        self.consensus = SparseDirectionalNoiseConsensus()
        self.local_gate = nn.Sequential(nn.Conv2d(5, 8, 1), nn.GELU(),
                                        nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

    def feature_residual(self, features, support, *, use_phase=True):
        if (features.ndim != 4 or features.shape[1] != 3
                or support.shape != features[:, :1].shape
                or not isinstance(use_phase, bool)):
            raise ValueError('aligned native three-channel features/support required')
        if not use_phase:
            features = torch.cat((features[:, :1], torch.zeros_like(features[:, 1:])), 1)
        local = self.network(features) * support
        coherent = self.consensus(local, support)
        weight = torch.sigmoid(self.local_gate(torch.cat((features, local, coherent), 1)))
        return (weight*local + (1-weight)*coherent) * support


def native_window(shape, core, halo=LargeSonarMWCNN.native_halo):
    """Origins on the GLOBAL eight-pixel Haar lattice, not each core's origin."""
    height, width = shape
    y, x, h, w = core
    hy, hx = halo
    if min(y, x, hy, hx) < 0 or min(h, w) < 1 or y+h > height or x+w > width:
        raise ValueError('bounded positive native core required')
    top, left = max(0, (y-hy)//8*8), max(0, (x-hx)//8*8)
    bottom, right = min(height, (y+h+hy+7)//8*8), min(width, (x+w+hx+7)//8*8)
    return (top, left, bottom, right), (y-top, x-left, h, w)


@torch.inference_mode()
def infer_features(model, features, support, device, *, core_shape=(128, 1024)):
    """No image resize, output averaging, gain clipping, or invented phase."""
    f, mask = np.asarray(features), np.asarray(support)
    if f.shape != (3, *mask.shape) or mask.ndim != 2 or not np.isfinite(f).all():
        raise ValueError('complete finite observed feature field required')
    ch, cw = core_shape
    if min(ch, cw) < 1:
        raise ValueError('positive native inference tile required')
    result = np.empty(mask.shape, np.float32)
    model.eval()
    for y in range(0, mask.shape[0], ch):
        for x in range(0, mask.shape[1], cw):
            h, w = min(ch, mask.shape[0]-y), min(cw, mask.shape[1]-x)
            box, core = native_window(mask.shape, (y, x, h, w))
            top, left, bottom, right = box
            cy, cx, _, _ = core
            ft = torch.from_numpy(np.array(f[:, top:bottom, left:right], copy=True)[None]).to(device)
            mt = torch.from_numpy(np.array(mask[top:bottom, left:right], copy=True)[None, None]).to(device)
            value = model.feature_residual(ft, mt, use_phase=True)
            result[y:y+h, x:x+w] = value[0, 0, cy:cy+h, cx:cx+w].cpu().numpy()
    if not np.isfinite(result).all() or np.any(result[~mask.astype(bool)]):
        raise ValueError('nonfinite output or changed missing-observation support')
    return result
