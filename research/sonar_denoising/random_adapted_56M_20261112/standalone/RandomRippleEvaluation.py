"""Whole-field tests of exactly the augmentation used during optimization."""
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_propagation

from NoFIRSonarData import read, sha
from RandomTriggeredSonarNoise import draw_scenario, apply_scenario, scenario_sha
from MeasuredStripeNoiseModel import noise_support_from_bounds


def prepare_fields(original, profile, device, item, source_index, beam):
    stamp = Path(item['path']).stat()
    if [stamp.st_size, stamp.st_mtime_ns] != item['file_guard']:
        raise ValueError('Held-out original source changed since admission.')
    gate = np.load(item['noise_support_file'], mmap_mode='r')
    if sha(item['noise_support_file']) != item['noise_support_SHA256']:
        raise ValueError('Held-out augmentation support changed.')
    fields = []
    for variant in range(2):
        seed = 68127000 + source_index * 100 + variant
        scenario = draw_scenario(profile, original.shape, 3, seed, side=item['native_side'])
        noisy = np.empty(original.shape, np.complex64)
        for start in range(0, original.shape[0], 64):
            stop = min(start + 64, original.shape[0])
            result, _ = apply_scenario(torch.from_numpy(original[start:stop].copy()).to(device),
                profile, scenario, origin=(start, 0), support=np.array(gate[start:stop], copy=True),
                reference_RMS=item['global_reference_RMS'], beam=beam)
            noisy[start:stop] = result.cpu().numpy()
        fields.append(dict(law=f'random_ripple_{variant}', noisy=noisy, scenario=scenario,
                           scenario_SHA256=scenario_sha(scenario)))
    return fields


def prepare_real_support(root, voyage):
    values = np.load(voyage / 'native_source/measured_complex128.npy', mmap_mode='r')
    bounds = np.load(voyage / 'metadata/bottom_first_ranges.npy')
    meta = read(voyage / 'metadata/envelope_cache_summary.json')
    c, fs = (meta['parts'][0]['parameters'][k] for k in ('c', 'BeamSigFs'))
    output = np.lib.format.open_memmap(root / 'evaluation/real_observed_noise_support.npy',
        mode='w+', dtype=np.float32, shape=(2, *values.shape[2:]))
    for side in range(2):
        gate = noise_support_from_bounds(bounds[side], c, fs, values.shape[-1])
        protected = gate == 0
        seeds = protected.copy()
        for beam in range(8):
            dark = np.abs(values[side, beam]) <= 8 / 255
            protected |= binary_propagation(seeds & dark, mask=dark)
        gate[protected] = 0
        output[side] = gate
        output.flush()


def load_previous_model(sonar_root, device):
    from NoFIRSonarWavelet import NoFIRSonarMWCNN, PROTOCOL
    root = sonar_root / 'nofir_multisource_20261108_v2/formal/adapt'
    selected = read(root / 'run_summary.json')['selected']
    path = root / selected['file']
    if sha(path) != selected['SHA256']:
        raise ValueError('Prior 56M model changed.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['model_family'] != NoFIRSonarMWCNN.family or state['feature_protocol'] != PROTOCOL:
        raise ValueError('Actual compatible previous model required.')
    model = NoFIRSonarMWCNN(state['width']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval(), dict(**selected, source_root=str(root))
