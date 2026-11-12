"""Admit existing authorized sources, run immutable staged training/evaluation.

Public data are already on172; no new private recording transfer is needed.
All caches are read-only. Only this NEW experiment directory is writable.
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
from PIL import Image

from NoFIRSonarData import read, save, sha


def admit(args):
    records, seen = [], {}
    public = args.public_root
    manifests = (
        public/'output/real_sonar/public_corpus_20261104/learning_manifest_v1/manifest.json',
        public/'output/real_sonar/public_long_envelopes_20261105/run_summary.json',
        public/'output/real_sonar/public_sassed_native_20261105/run_summary.json')
    corpus = read(manifests[0])['records']
    gavia_val = sorted((r for r in corpus if r['dataset'] == 'Gavia' and r['split'] == 'validation'
                         and r['include_in_learning']), key=lambda r: r['sample_id'])
    selected_val = {r['sample_id'] for r in gavia_val[::max(1, len(gavia_val)//32)][:32]}
    original = []
    original.extend((r, 0) for r in corpus if r['dataset'] == 'Gavia' and r['include_in_learning']
                    and (r['split'] == 'train' or r['sample_id'] in selected_val))
    original.extend((r, 1) for r in read(manifests[1])['records'] if r['split'] in ('train', 'validation'))
    original.extend((r, 2) for r in read(manifests[2])['records'])
    for row, domain in original:
        path = public/Path(row['path']).relative_to('[local-path]/OceanSim')
        if not path.resolve().is_relative_to(public.resolve()):
            raise ValueError('public path outside authorized corpus')
        digest = sha(path)
        expected = row.get('cache_sha256', row.get('source_sha256'))
        if digest != expected:
            raise ValueError('original qualified public bytes changed')
        if domain == 0:
            with Image.open(path) as image:
                a = np.asarray(image.convert('RGB'))
            shape = list(a.shape[:2])
            positive = bool(np.any(a))
        else:
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            shape = list(a.shape)
            positive = bool(np.any(abs(a) > 0))
        if not positive:
            raise ValueError('blank source needs an explicit admission exclusion, not silent replacement')
        role = 'public_train' if row['split'] in ('train', 'train_aux') else 'public_validation'
        if digest in seen:
            if seen[digest] != role:
                raise ValueError('exact source leakage across training/validation')
            raise ValueError('duplicate public source needs explicit deduplication')
        seen[digest] = role
        stamp = path.stat()
        records.append(dict(id=row['sample_id'], domain=domain, role=role, path=str(path),
            shape=shape, file_SHA256=digest, file_guard=[stamp.st_size, stamp.st_mtime_ns],
            group=row.get('acquisition_group_id', row['group_id']), original_split=row['split'], source=row['source'],
            license=row['license'], unknown_original_noise=True,
            public_unknown_SAS_acquisitions_are_training_only=domain == 2))
    teacher = args.sonar_root/'front_target_alignment_20261105_v1/teacher_bundle'
    recovery = args.sonar_root/'matched_processing_recovery_20261105_v1'
    native_manifest = read(teacher/'reports/native_processed_manifest.json')
    if native_manifest['split_coverage']['same_physical_part_crosses_split']:
        raise ValueError('private physical split leakage')
    for i, source in enumerate(native_manifest['sources']):
        folder = teacher if i < 6 else recovery
        path = folder/'native_arrays'/f'source_{i:02d}_complex128.npy'
        receipt = read(folder/'native_arrays'/f'source_{i:02d}_receipt.json')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        import hashlib
        payload_sha = hashlib.sha256(memoryview(np.asarray(array)).cast('B')).hexdigest()
        if payload_sha != receipt['source_payload_SHA256']:
            raise ValueError('original authorized native payload changed')
        stamp = path.stat()
        for beam in range(8):
            records.append(dict(id=f'qiandao_s{i}_b{beam}', domain=3,
                role='native_train' if i < 6 else 'native_validation',
                path=str(path), shape=list(array.shape[1:]), beam=beam,
                source_payload_SHA256=payload_sha, file_guard=[stamp.st_size, stamp.st_mtime_ns],
                group=source['physical_part'], original_split=source['split'],
                source='private acoustic institute processed observation; not unfiltered ADC',
                license='private research data, NEVER publish', unknown_original_noise=True))
    groups = {}
    for item in records:
        role = 'train' if item['role'].endswith('train') else 'validation'
        key = item['domain'], item['group']
        groups.setdefault(key, set()).add(role)
    if any(len(roles) != 1 for roles in groups.values()):
        raise ValueError('acquisition group leakage')
    manifest = dict(status='admitted_existing_multisource_typed_NO_FIR_training_corpus', records=records,
        counts=dict(Counter((r['role']+':'+str(r['domain'])) for r in records)),
        source_manifest_SHA256={str(p): sha(p) for p in manifests},
        all_admitted_source_payloads_verified=True, no_private_data_published=True,
        data_are_original_observations_NOT_clean_targets=True,
        no_input_output_FIR_in_candidate=True, upstream_acquisition_processing_NOT_reversible=True,
        new_private_data_transferred=False, training_public_SAS_group_not_used_as_independent_test=True,
        public_val_full_count=len(gavia_val), public_val_selected_windows=len(selected_val),
        scene_geographic_independence_NOT_proven=True)
    save(args.root/'admission_manifest.json', manifest)
    shutil.copy2(recovery/'aggregate_appearance_profile.json', args.root/'aggregate_appearance_profile.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'public-root', 'sonar-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pretrain-steps', type=int, default=20000)
    parser.add_argument('--adapt-steps', type=int, default=6000)
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.root = args.root.absolute()
    args.root.mkdir(exist_ok=False)
    start = time.monotonic()
    report = dict(status='admitting_readonly_public_and_private_data', PID=os.getpid(), smoke=args.smoke,
        width=args.width, world_size=args.world_size, candidate_input_FIR=False,
        frozen_release_modified=False, prior_experiments_modified=False,
        pretrain_steps_requested=2 if args.smoke else args.pretrain_steps,
        adaptation_steps_requested=2 if args.smoke else args.adapt_steps,
        actual_training_completed=False, real_quality_accepted=False,
        source_code_SHA256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    def update():
        report['elapsed_s'] = time.monotonic()-start
        save(args.root/'run_summary.json', report)
    update()
    try:
        manifest = admit(args)
        report['admitted_counts'] = manifest['counts']
        report['admission_manifest_SHA256'] = sha(args.root/'admission_manifest.json')
        update()
        for phase, steps in (('pretrain', report['pretrain_steps_requested']),
                             ('adapt', report['adaptation_steps_requested'])):
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                '--nproc-per-node', str(args.world_size), str(Path(__file__).with_name('TrainNoFIRSonarDistributed.py')),
                '--root', str(args.root), '--phase', phase, '--steps', str(steps), '--width', str(args.width)]
            if args.smoke:
                command.append('--smoke')
            report['status'] = 'running_'+phase+'_NO_FIR_distributed_training'
            report['phase_command'] = command
            update()
            subprocess.run(command, check=True)
            phase_report = read(args.root/phase/'run_summary.json')
            if phase_report['steps_completed'] != steps:
                raise ValueError('declared phase did not finish all actual updates')
            report[phase+'_selected'] = phase_report['selected']
            update()
        report['actual_training_completed'] = True
        if args.smoke:
            report['status'] = 'completed_two_phase_real_data_distributed_smoke_NOT_quality'
            return
        report['status'] = 'evaluating_complete_known_inputs_and_real_native_voyage_NO_FIR'
        update()
        command = [sys.executable, str(Path(__file__).with_name('EvaluateNoFIRSonarTrial.py')),
            '--root', str(args.root), '--sonar-root', str(args.sonar_root), '--workers', str(args.world_size)]
        subprocess.run(command, check=True)
        report['status'] = 'completed_multisource_NO_FIR_training_and_full_native_outputs_NOT_quality_acceptance'
    except BaseException as exc:
        report.update(status='failed_preserved_multisource_NO_FIR_experiment_NO_automatic_restart',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
