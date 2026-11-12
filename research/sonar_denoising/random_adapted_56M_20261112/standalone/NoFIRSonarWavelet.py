"""Raw-input multi-domain MWCNN experiment; NO frequency-filter front end.

Display luminance, measured envelope, public complex SAS and private complex
sonar are different explicit domains. Missing phase is marked by the domain
and zero feature planes, never fabricated as a measured complex waveform.
Masked local centering constructs features only; it never replaces the output
base. The ONLY output correction is the learned scalar log-amplitude residual.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
import torch
from torch import nn

from LargeSonarMWCNN import SonarMWCNNBackbone


PROTOCOL = 'raw_multidomain_logcontrast_actual_phase_NO_FIR_v1'
DOMAINS = ('display_luminance', 'linear_envelope', 'public_complex_SAS', 'native_complex')
HALO = (168, 200)  # backbone radius <160, feature radii2/32, predecessor1


def raw_features(values, domain, *, first_column=0):
    if domain not in range(4):
        raise ValueError('explicit reviewed measurement domain required')
    z = np.asarray(values)
    if (z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all()
            or first_column < 0 or (domain >= 2) != np.iscomplexobj(z)):
        raise ValueError('actual complex phase or explicit noncomplex amplitude required')
    if domain < 2 and (z < 0).any():
        raise ValueError('nonnegative observed brightness/envelope required')
    amplitude = np.abs(z).astype(np.float64)
    all_mask = amplitude > 0
    q = np.zeros(amplitude.shape, np.float64)
    np.log(amplitude, out=q, where=all_mask)
    if first_column:
        q, mask = q[:, 1:], all_mask[:, 1:]
    else:
        mask = all_mask
    weights = uniform_filter(mask.astype(np.float64), (5, 65), mode='constant')
    total = uniform_filter(q*mask, (5, 65), mode='constant')
    mean = np.divide(total, weights, out=np.zeros_like(total), where=weights > 0)
    features = np.zeros((3, *q.shape), np.float32)
    features[0] = (q-mean)*mask
    if domain >= 2:
        unit = np.divide(z, amplitude, out=np.zeros(z.shape, np.complex128), where=all_mask)
        gradient = unit[:, 1:]*unit[:, :-1].conj()
        valid = all_mask[:, 1:] & all_mask[:, :-1]
        if not first_column:
            gradient = np.pad(gradient, ((0, 0), (1, 0)))
            valid = np.pad(valid, ((0, 0), (1, 0)))
        features[1] = np.where(valid, gradient.real, 0)
        features[2] = np.where(valid, gradient.imag, 0)
    return features, q.astype(np.float32), mask


def native_context(shape, core):
    height, width = shape
    y, x, h, w = core
    if min(y, x) < 0 or min(h, w) < 1 or y+h > height or x+w > width:
        raise ValueError('actual bounded native core required')
    top, left = max(0, (y-HALO[0])//8*8), max(0, (x-HALO[1])//8*8)
    bottom = min(height, (y+h+HALO[0]+7)//8*8)
    right = min(width, (x+w+HALO[1]+7)//8*8)
    return (top, left, bottom, right), (y-top, x-left, h, w)


class NoFIRSonarMWCNN(nn.Module):
    family = 'multisource_three_level_mwcnn_raw_NO_FIR_v1'
    feature_protocol = PROTOCOL

    def __init__(self, width=96):
        super().__init__()
        self.width = width
        self.network = SonarMWCNNBackbone(width)
        self.domain_affine = nn.Embedding(4, 6)
        nn.init.zeros_(self.domain_affine.weight)

    def forward(self, features, support, domain=3):
        if (domain not in range(4) or features.ndim != 4 or features.shape[1] != 3
                or features[:, :1].shape != support.shape):
            raise ValueError('explicit aligned domain/features/support required')
        gain, bias = self.domain_affine.weight[domain].chunk(2)
        adjusted = features*(1+gain[None, :, None, None])+bias[None, :, None, None]
        return self.network(adjusted)*support

    def feature_residual(self, features, support, *, domain=3, use_phase=True):
        if domain < 2 and use_phase:
            raise ValueError('unmeasured public phase must not be called observed')
        if not use_phase:
            features = torch.cat((features[:, :1], torch.zeros_like(features[:, 1:])), 1)
        return self(features, support, domain)


@torch.inference_mode()
def infer_raw(model, values, device, *, domain=3, core_shape=(96, 768)):
    """Full-field observed features, global8 Haar origins, native sample output."""
    features, _, mask = raw_features(values, domain)
    height, width = mask.shape
    result = np.empty(mask.shape, np.float32)
    model.eval()
    for y in range(0, height, core_shape[0]):
        for x in range(0, width, core_shape[1]):
            h, w = min(core_shape[0], height-y), min(core_shape[1], width-x)
            box, inner = native_context(mask.shape, (y, x, h, w))
            top, left, bottom, right = box
            cy, cx, _, _ = inner
            f = torch.from_numpy(features[:, top:bottom, left:right].copy()[None]).to(device)
            m = torch.from_numpy(mask[top:bottom, left:right].copy()[None, None]).to(device)
            prediction = model(f, m, domain)
            result[y:y+h, x:x+w] = prediction[0, 0, cy:cy+h, cx:cx+w].cpu().numpy()
    if not np.isfinite(result).all() or np.any(result[~mask]):
        raise ValueError('invalid learned raw-input restoration')
    return result
