"""Original v7.3 MAT -> frozen fusion_wide_front -> edge repair -> Shotwell.

Portable entry point for the reviewed measured-sonar processing chain. Reuses
the original frozen scientific operators and checkpoint without new training.
No historical server, remote receipt, GPS, reference PNG or envelope cache is
required. Stages may use separate Python environments, all on the local host.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from BathymetrySideScanDrawing import PingFile
from FreezeSonarDenoisingRelease import sha, verify
from FullMeasuredSonarProtocol import overlap_block, overlap_tail, supported_total_delta
from NativeSonarPaletteStream import NativePaletteWriter, amplitude_indices
from RangeGainContinuityRepair import repair
from caris_orangecopper import caris_orangecopper


ROOT = Path(__file__).resolve().parents[1]
NAME = 'fusion_wide_front'
WEIGHT = ROOT / 'releases/sonar_denoise_20261106/weights/fusion_wide_front.pt'
WEIGHT_SHA = 'faece15dfd9c9a4b028a409b6d8ee583c93287e1b6ec81afd0c274c4686073e0'
EDGE_INDEX_SHA = '46872ac0f29166315bd0cee8055a87848667ec9e3ad307dedace503d9a7e6721'
SIDES = ('Port', 'Stbd')
REQUIRED = ('SysPara.mat', 'PortNormalModeBeamformingTotal.mat',
            'StbdNormalModeBeamformingTotal.mat', 'PortNormalModeDepthTotal.mat',
            'StbdNormalModeDepthTotal.mat')


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.incomplete')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def guard(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def inspect_source(source):
    import h5py
    source = Path(source).resolve()
    if all((source / f).is_file() for f in REQUIRED):
        parts = [source]
    else:
        parts = sorted(p for p in source.glob('Data[0-9]*') if p.is_dir())
    if not parts:
        raise ValueError('Source must contain Data01/Data02 or the five original MAT files')
    infos, guards, bins, reference = [], {}, None, None
    for part in parts:
        if not all((part / f).is_file() for f in REQUIRED):
            raise ValueError(f'Missing required MAT files in {part.name}')
        for name in REQUIRED:
            guards[str(part / name)] = guard(part / name)
        with h5py.File(part / 'SysPara.mat', 'r') as handle:
            pars = {k: float(handle['SysPara'][k][0, 0]) for k in ('c', 'BeamSigFs')}
        if pars['BeamSigFs'] != 37650. or not np.isfinite(pars['c']) or pars['c'] <= 0:
            raise ValueError('Frozen recorded-band model requires fs=37650 Hz and positive sound speed')
        if reference is not None and pars != reference:
            raise ValueError('Acquisition parameters differ across parts')
        reference = pars
        with ExitStack() as stack:
            readers = [stack.enter_context(PingFile(part / f'{s}NormalMode{k}Total.mat'))
                       for k in ('Beamforming', 'Depth') for s in SIDES]
            count = len(readers[0])
            if count < 2 or any(len(r) != count for r in readers):
                raise ValueError('Four side/echo/depth counts must agree and each part needs >=2 records')
            for i in range(count):
                identity = readers[0].identity(i)
                if any(r.identity(i) != identity for r in readers[1:]):
                    raise ValueError(f'Record identities differ in {part.name}, cell {i+1}')
            shape = readers[0].beam_sum(0, 'Port').shape
            if shape[0] != 8 or shape[1] < 2048 or (bins is not None and bins != shape[1]):
                raise ValueError('Require eight beams, constant native range width >=2048')
            bins = shape[1]
            infos.append(dict(name=part.name, source=str(part), count=count,
                              offset=sum(x['count'] for x in infos),
                              first_identity=readers[0].identity(0),
                              last_identity=readers[0].identity(count-1)))
    return dict(parts=infos, parameters=reference, source_guards=guards,
                native_shape=[2, 8, sum(x['count'] for x in infos), bins],
                selection='every original record, no silent skip', source_MAT_read_only=True)


def extract(source, out):
    meta = inspect_source(source)
    directory = Path(out) / 'raw'
    directory.mkdir(parents=True, exist_ok=False)
    shape = tuple(meta['native_shape'])
    if shutil.disk_usage(directory).free < np.prod(shape) * 20 + 2 * 1024**3:
        raise OSError('Need space for native complex128, float32 gains and display outputs')
    signal = np.lib.format.open_memmap(directory / 'native_complex128.npy', mode='w+',
                                      dtype=np.complex128, shape=shape)
    bounds = np.empty((2, shape[2], 2), np.float64)
    identities = np.empty((shape[2], 3), np.int64)
    for part in meta['parts']:
        with ExitStack() as stack:
            readers = [stack.enter_context(PingFile(Path(part['source']) / f'{s}NormalMode{k}Total.mat'))
                       for k in ('Beamforming', 'Depth') for s in SIDES]
            for index in range(part['count']):
                position = part['offset'] + index
                identities[position] = readers[0].identity(index)
                for side, name in enumerate(SIDES):
                    z = readers[side].beam_sum(index, name)
                    if z.shape != (8, shape[-1]) or not np.isfinite(z).all():
                        raise ValueError('Nonfinite or variable-width original echo; do not skip silently')
                    signal[side, :, position] = z
                    depth = readers[side+2].record(index)
                    bottom = float(depth['BottomTrackingInfo'][0, 0])
                    first = float(np.median(np.asarray(depth['TargetDelayInfo'])[0]) *
                                  meta['parameters']['c'] / 2)
                    if not np.isfinite([bottom, first]).all() or not 0 < bottom < first:
                        raise ValueError('Invalid supplied bottom/first-return protection bounds')
                    bounds[side, position] = bottom, first
                if (index+1) % 128 == 0 or index+1 == part['count']:
                    signal.flush()
                    print(f"[MAT] {part['name']} {index+1}/{part['count']} records, all16 beams", flush=True)
    signal.flush()
    np.save(directory / 'bottom_first_ranges.npy', bounds)
    np.save(directory / 'PingNum_PingS_PingUs.npy', identities)
    if any(guard(p) != stamp for p, stamp in meta['source_guards'].items()):
        raise ValueError('Original MAT changed during extraction')
    meta['status'] = 'completed_read_only_all_native_complex128_records'
    dump(directory / 'manifest.json', meta)
    return meta


def load_model(device):
    import torch
    from BandAnchoredNativeSonarFeatures import PROTOCOL
    from MatchedSonarWavelet import MatchedSonarWavelet
    if sha(WEIGHT) != WEIGHT_SHA:
        raise ValueError('Fixed selected checkpoint hash changed')
    state = torch.load(WEIGHT, map_location=device, weights_only=True)
    expected = dict(noise='wide', frontend=True, aggregation='learned_fusion')
    if (state['condition_name'] != NAME or state['condition'] != expected or state['step'] != 400
            or state['feature_protocol'] != PROTOCOL or state['model_family'] != MatchedSonarWavelet.family):
        raise ValueError('Checkpoint interpretation differs from reviewed fusion_wide_front')
    model = MatchedSonarWavelet(state['width'], aggregation='learned_fusion').to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval()


def infer(out, device_name='auto'):
    import torch
    from MatchedSonarWavelet import infer_group
    out = Path(out)
    meta = read(out / 'raw/manifest.json')
    if meta['status'] != 'completed_read_only_all_native_complex128_records':
        raise ValueError('Complete original MAT extraction is required')
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu')
                          if device_name == 'auto' else device_name)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(device)
    directory = out / 'inference'
    directory.mkdir(exist_ok=False)
    z = np.load(out / 'raw/native_complex128.npy', mmap_mode='r')
    limits = np.load(out / 'raw/bottom_first_ranges.npy')
    parts = [p['count'] for p in meta['parts']]
    if list(z.shape) != meta['native_shape'] or z.dtype != np.complex128:
        raise ValueError('Native complex precision/shape differs')
    gain = np.lib.format.open_memmap(directory / 'baseline_log_gain.npy', mode='w+',
                                    dtype=np.float32, shape=z.shape)
    patch = np.lib.format.open_memmap(directory / 'gain_and_commonmode_last512.npy', mode='w+',
                                     dtype=np.float32, shape=z.shape[:-1] + (512,))
    report = dict(status='running_frozen_inference', checkpoint_SHA256=WEIGHT_SHA,
                  device=str(device), torch=str(torch.__version__), numpy=np.__version__,
                  core_shape=[256, 2048], source_shape=list(z.shape), beams=[])
    for side in range(2):
        for beam in range(8):
            original = np.asarray(z[side, beam])
            amplitude = np.abs(original)
            model_input = np.asarray(original, np.complex64)
            input_sha = hashlib.sha256(memoryview(np.ascontiguousarray(model_input)).cast('B')).hexdigest()
            values, neural = infer_group({NAME: (model, True)}, model_input, device)
            delta = supported_total_delta(amplitude, values[NAME], limits[side],
                                          np.arange(z.shape[-1]),
                                          meta['parameters']['c'], meta['parameters']['BeamSigFs'])
            gain[side, beam] = delta
            context = amplitude[:, -1024:]
            present = context > 1e-12
            ends = z.shape[-1] - np.argmax(present[:, ::-1], axis=1)
            if not present.any(axis=1).all() or not np.all(ends == ends[0]):
                raise ValueError('Edge protocol needs a common observed endpoint, not inferred black image pixels')
            repaired, adapter = repair(delta[:, -1024:], context, parts,
                                       full_width=z.shape[-1], valid_end=int(ends[0]))
            patch[side, beam] = repaired['gain_and_commonmode']
            gain.flush()
            patch.flush()
            report['beams'].append(dict(side=side, beam=beam, edge=adapter,
                                        original_complex64_SHA256=input_sha,
                                        network=neural))
            dump(directory / 'manifest.json', report)
            print(f'[frozen model + edge] beam {len(report["beams"])}/16', flush=True)
    report.update(status='completed_all16_frozen_model_and_gain_and_commonmode',
                  gain_SHA256=sha(directory / 'baseline_log_gain.npy'),
                  edge_SHA256=sha(directory / 'gain_and_commonmode_last512.npy'))
    dump(directory / 'manifest.json', report)
    return report


def render_native_fields(signal, gain, patch, target, *, signal_is_complex=True):
    """Shared native renderer; real amplitude cache may independently audit it."""
    z = signal
    if (z.ndim != 4 or z.shape[:2] != (2, 8) or gain.shape != z.shape
            or patch.shape != z.shape[:-1] + (512,)
            or gain.dtype != np.float32 or patch.dtype != np.float32
            or (signal_is_complex and z.dtype != np.complex128)
            or (not signal_is_complex and z.dtype != np.float64)):
        raise ValueError('Native signal, full gain and edge patch are not aligned')
    palette = (caris_orangecopper(256)*255).astype(np.uint8)
    writer = NativePaletteWriter(target, 2*z.shape[-1], 4*z.shape[2]+4, palette)
    previous = None
    try:
        for start in range(0, z.shape[2], 16):
            end = min(start+16, z.shape[2])
            values = z[:, :, start:end]
            amplitude = (np.abs(values) if signal_is_complex else values).transpose(0, 2, 1, 3)
            d = np.array(gain[:, :, start:end], dtype=np.float64)
            d[..., -512:] = patch[:, :, start:end]
            restored = amplitude * np.exp(d.transpose(0, 2, 1, 3))
            rows, previous = overlap_block(restored, previous)
            writer.append(amplitude_indices(rows))
        writer.append(amplitude_indices(overlap_tail(previous)))
        encoding = writer.finish()
    finally:
        writer.close()
    return encoding


def render(out, reference_batch=False):
    out = Path(out)
    meta = read(out / 'raw/manifest.json')
    status = read(out / 'inference/manifest.json')
    if status['status'] != 'completed_all16_frozen_model_and_gain_and_commonmode':
        raise ValueError('Complete full inference required')
    z = np.load(out / 'raw/native_complex128.npy', mmap_mode='r')
    gain = np.load(out / 'inference/baseline_log_gain.npy', mmap_mode='r')
    patch = np.load(out / 'inference/gain_and_commonmode_last512.npy', mmap_mode='r')
    if list(z.shape) != meta['native_shape']:
        raise ValueError('Native signal metadata differs')
    directory = out / 'render'
    directory.mkdir(exist_ok=False)
    target = directory / 'native_gain_and_commonmode.png'
    encoding = render_native_fields(z, gain, patch, target)
    matches = encoding['index_SHA256'] == EDGE_INDEX_SHA
    report = dict(status='completed_native_edge_candidate', candidate=str(target),
                  SHA256=sha(target), encoding=encoding,
                  reference_batch_index_SHA256_match=matches)
    dump(directory / 'manifest.json', report)
    if reference_batch and not matches:
        raise ValueError('Reference batch pixels differ; output preserved, do not claim exact reproduction')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, help='Original Data01/Data02 parent directory')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=('all', 'extract', 'infer', 'render', 'enhance'), default='all')
    p.add_argument('--device', default='auto')
    p.add_argument('--mat-python', default=sys.executable)
    p.add_argument('--ml-python', default=sys.executable)
    p.add_argument('--display-python', default='/usr/bin/python3')
    p.add_argument('--reference-batch', action='store_true', help='Require bit-exact reviewed edge image indices')
    a = p.parse_args()
    out = a.out.resolve()
    if a.stage in ('all', 'extract'):
        if a.source is None:
            p.error('--source is required to read the original MAT files')
        source = a.source.resolve()
        if out == source or source in out.parents:
            raise ValueError('Output must be separate from original data')
    verify(ROOT, ROOT / 'releases/sonar_denoise_20261106')
    if a.stage == 'extract':
        return extract(a.source, out)
    if a.stage == 'infer':
        return infer(out, a.device)
    if a.stage == 'render':
        return render(out, a.reference_batch)
    if a.stage == 'enhance':
        from EnhanceEdgeSonarShotwell import enhance
        candidate = out / 'render/native_gain_and_commonmode.png'
        report = read(out / 'render/manifest.json')
        return enhance(candidate, out / 'enhanced', expected_sha=report['SHA256'])
    if out.exists():
        raise FileExistsError('Choose a fresh output directory; do not overwrite completed or failed runs')
    out.mkdir(parents=True)
    report = dict(status='running_raw_MAT_to_enhanced_native', stages=[],
                  checkpoint_SHA256=WEIGHT_SHA, source=str(source),
                  source_MAT_modified=False, training=False)
    started = time.monotonic()
    try:
        for stage, python in [('extract', a.mat_python), ('infer', a.ml_python),
                              ('render', a.mat_python), ('enhance', a.display_python)]:
            command = [python, '-u', str(Path(__file__).resolve()), '--stage', stage,
                       '--source', str(source), '--out', str(out), '--device', a.device]
            if a.reference_batch:
                command.append('--reference-batch')
            subprocess.run(command, check=True)
            report['stages'].append(stage)
            dump(out / 'pipeline_manifest.json', report)
        report['status'] = 'completed_raw_MAT_to_edge_repaired_Shotwell_enhanced_native'
    except BaseException as exc:
        report.update(status='failed_preserved_no_automatic_restart', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        dump(out / 'pipeline_manifest.json', report)


if __name__ == '__main__':
    main()
