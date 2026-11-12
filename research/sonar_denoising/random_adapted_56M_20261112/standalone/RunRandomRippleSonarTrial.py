"""Admit existing data, randomize lateral sources, train, and evaluate full fields.

No original data are uploaded or modified. Every optimizer input is recorded.
The public corpus and native recordings are observations, NOT clean truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import binary_propagation

from NoFIRSonarData import Sources, read, save, sha
from RunNoFIRSonarTrial import admit
from ObservedBottomSupportArtifacts import verified_metadata
from MeasuredStripeNoiseModel import noise_support_from_bounds
from RandomTriggeredSonarNoise import PROTOCOL


def reference_rms(values, gate=None):
    energy, count = 0., 0
    for start in range(0, values.shape[0], 64):
        a = np.abs(np.asarray(values[start:start + 64])).astype(np.float64)
        mask = a > 0
        if gate is not None:
            mask &= gate[start:start + 64] > 0
        energy += float(np.square(a[mask]).sum())
        count += int(mask.sum())
    if count == 0:
        raise ValueError('No observed source support for augmentation.')
    return float(np.sqrt(energy / count))


def enrich_manifest(args, manifest):
    teacher = args.sonar_root / 'front_target_alignment_20261105_v1/teacher_bundle'
    arrays, metadata, evidence = verified_metadata(args.sonar_root / 'observed_bottom_support_20261106_v1',
                                                  teacher / 'reports/native_processed_manifest.json')
    support_root = args.root / 'observed_augmentation_support'
    support_root.mkdir()
    loader = Sources()
    gate_paths = {}
    for index in range(8):
        items = [r for r in manifest['records'] if r['id'].startswith(f'qiandao_s{index}_')]
        source = metadata['sources'][index]
        if not arrays[index]['valid_detection_bool'].all():
            raise ValueError('Invalid native range boundaries require explicit review.')
        gate = noise_support_from_bounds(arrays[index]['bottom_first_float64'],
            source['sound_speed_m_s'], source['sampling_frequency_Hz'], source['native_range_bins'])
        # Protect near-black water connected to OBSERVED water seeds, not all
        # isolated dark seabed pixels. This input support never paints output.
        protected = gate == 0
        values = np.load(items[0]['path'], mmap_mode='r', allow_pickle=False)
        for beam in range(8):
            dark = np.abs(values[beam]) <= 8 / 255
            protected |= binary_propagation((gate == 0) & dark, mask=dark)
        gate[protected] = 0
        path = support_root / f'source_{index:02d}_support_float32.npy'
        np.save(path, gate, allow_pickle=False)
        stamp = path.stat()
        gate_paths[index] = dict(path=str(path), SHA256=sha(path),
            file_guard=[stamp.st_size, stamp.st_mtime_ns], protected_cells=int((gate == 0).sum()))
        for item in items:
            item.update(noise_support_file=str(path), noise_support_SHA256=gate_paths[index]['SHA256'],
                noise_support_file_guard=gate_paths[index]['file_guard'], native_side=index % 2,
                global_reference_RMS=reference_rms(loader.array(item), gate))
        print(dict(stage='native_ripple_support_and_global_RMS', source=index), flush=True)
    for item in manifest['records']:
        if item['domain'] != 3:
            item['global_reference_RMS'] = reference_rms(loader.array(item))
    manifest.update(random_ripple_protocol=PROTOCOL, global_reference_RMS_computed_on_whole_sources=True,
        original_source_pixels_modified=False, augmentation_support_evidence=evidence,
        native_augmentation_support=gate_paths)
    save(args.root / 'admission_manifest.json', manifest)
    return manifest


def reuse_admission(args):
    source = args.admission_root
    parent = read(source / 'admission_manifest.json')
    if parent.get('random_ripple_protocol') != PROTOCOL:
        raise ValueError('Completed random-ripple source admission required.')
    for item in parent['records']:
        path = Path(item['path'])
        stamp = path.stat()
        if [stamp.st_size, stamp.st_mtime_ns] != item['file_guard']:
            raise ValueError('Previously payload-verified source changed; do not silently re-admit.')
    for info in parent['native_augmentation_support'].values():
        path = Path(info['path'])
        if sha(path) != info['SHA256']:
            raise ValueError('Native noise support changed since admission.')
    parent['reused_verified_admission_SHA256'] = sha(source / 'admission_manifest.json')
    parent['source_guards_rechecked_for_this_run'] = True
    save(args.root / 'admission_manifest.json', parent)
    shutil.copy2(source / 'aggregate_appearance_profile.json', args.root / 'aggregate_appearance_profile.json')
    return parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'public-root', 'sonar-root', 'noise-profile'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--admission-root', type=Path)
    p.add_argument('--world-size', type=int, default=4)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--pretrain-steps', type=int, default=24000)
    p.add_argument('--adapt-steps', type=int, default=8000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.world_size, a.width, a.pretrain_steps, a.adapt_steps) <= 0:
        raise ValueError('Positive actual training schedule required.')
    a.root = a.root.absolute()
    a.root.mkdir(exist_ok=False)
    start = time.monotonic()
    report = dict(status='admitting_random_ripple_training_data', PID=os.getpid(), smoke=a.smoke,
        protocol=PROTOCOL, noise_profile_parent_SHA256=sha(a.noise_profile), width=a.width, world_size=a.world_size,
        actual_training_completed=False, quality_accepted=False, originals_modified=False,
        input_FIR=False, output_FIR=False, training_noise_is_NOT_identified_device_truth=True,
        previous_models_and_experiments_preserved=True,
        architecture_decision='Retrain existing 56M three-level wavelet CNN first; isolate noise-model effect',
        alternatives_considered=[dict(name='MWCNN', source='https://arxiv.org/abs/1805.07071'),
            dict(name='MambaIRv2', source='https://github.com/csguoh/MambaIR',
                 reason_not_selected='Published generic restoration results do not establish advantage on our curved sonar stripes')],
        source_code_SHA256={f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
        pretrain_steps_requested=2 if a.smoke else a.pretrain_steps,
        adaptation_steps_requested=2 if a.smoke else a.adapt_steps)
    def update():
        report['elapsed_s'] = time.monotonic() - start
        save(a.root / 'run_summary.json', report)
    update()
    try:
        manifest = reuse_admission(a) if a.admission_root else enrich_manifest(a, admit(a))
        report['admitted_counts'] = manifest['counts']
        report['admission_manifest_SHA256'] = sha(a.root / 'admission_manifest.json')
        profile = read(a.noise_profile)
        profile['random_ripple_training_protocol'] = PROTOCOL
        profile['user_authorized_randomized_source_augmentation'] = True
        profile['noise_parameter_distribution_is_augmentation_hypothesis_NOT_physical_truth'] = True
        save(a.root / 'random_ripple_profile.json', profile)
        update()
        for phase, steps in (('pretrain', report['pretrain_steps_requested']), ('adapt', report['adaptation_steps_requested'])):
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc-per-node', str(a.world_size), str(Path(__file__).with_name('TrainNoFIRSonarDistributed.py')),
                '--root', str(a.root), '--phase', phase, '--steps', str(steps), '--width', str(a.width), '--random-ripple']
            if a.smoke:
                command.append('--smoke')
            report.update(status='training_random_ripple_' + phase, current_command=command)
            update()
            subprocess.run(command, check=True)
            stage = read(a.root / phase / 'run_summary.json')
            if stage['steps_completed'] != steps:
                raise ValueError('Requested real optimizer updates were not completed.')
            report[phase + '_selected'] = stage['selected']
            report[phase + '_all_training_sources_augmented'] = stage['all_declared_training_sources_actually_augmented']
            update()
        report['actual_training_completed'] = True
        if a.smoke:
            report['status'] = 'completed_random_ripple_two_phase_distributed_smoke_NOT_quality'
            return
        report['status'] = 'evaluating_random_ripple_full_fields_and_real_voyage'
        update()
        subprocess.run([sys.executable, str(Path(__file__).with_name('EvaluateRandomRippleSonar.py')),
            '--root', str(a.root), '--sonar-root', str(a.sonar_root), '--workers', str(a.world_size)], check=True)
        report['status'] = 'completed_random_ripple_training_full_evaluation_and_render_NOT_quality_acceptance'
    except BaseException as exc:
        report.update(status='failed_preserved_random_ripple_run_NO_automatic_restart', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
