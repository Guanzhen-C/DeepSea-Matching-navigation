"""Verify actual execution evidence and the untouched previous code snapshot."""
from pathlib import Path
import argparse
import hashlib
import tarfile

from NoFIRSonarData import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous-experiment', type=Path, required=True)
    p.add_argument('--regression', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    a = p.parse_args()
    target = a.root / 'launch_preflight.json'
    if target.exists():
        raise FileExistsError('Preserve prior preflight evidence.')
    checked = 0
    with tarfile.open(a.previous_experiment / 'code_snapshot.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe previous snapshot member.')
            expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if sha(a.previous_experiment / relative) != expected:
                raise ValueError('Previous experiment code changed.')
            checked += 1
    smoke = read(a.root / 'smoke/run_summary.json')
    if smoke['status'] != 'completed_random_ripple_two_phase_distributed_smoke_NOT_quality':
        raise ValueError('Actual complete two-phase smoke required.')
    used = ('RandomTriggeredSonarNoise.py', 'RandomRippleSonarData.py',
            'NoFIRSonarWavelet.py', 'TrainNoFIRSonarDistributed.py')
    for name in used:
        if sha(a.root / 'standalone' / name) != smoke['source_code_SHA256'][name]:
            raise ValueError('Actually smoke-tested training numerics changed.')
    for phase in ('pretrain', 'adapt'):
        stage = read(a.root / 'smoke' / phase / 'run_summary.json')
        if stage['steps_completed'] != 2 or len(stage['rank_ledgers']) != 4:
            raise ValueError('Every rank must complete actual updates.')
        if stage['initial_update_gradient_checks'][1]['backbone_gradient_absolute_sum'] <= 0:
            raise ValueError('Backbone did not actually receive gradients.')
        for rank in range(4):
            receipt = read(a.root / 'smoke' / phase / f'rank_{rank}_receipt.json')
            if receipt['steps'] != 2 or receipt['examples'] != 2:
                raise ValueError('Incomplete rank receipt.')
    regression = read(a.regression / 'pretrain/run_summary.json')
    if regression['steps_completed'] != 300 or not regression['all_declared_training_sources_seen']:
        raise ValueError('300 actual consecutive updates and public-source coverage required.')
    qualification = read(a.qualification / 'run_summary.json')
    if qualification['status'] != 'passed_actual_source_whole_crop_noise_consistency' or len(qualification['cases']) != 12:
        raise ValueError('All twelve predeclared full-source augmentation checks required.')
    report = dict(status='passed_preflight_before_formal_training_NOT_quality',
        old_snapshot_files_checked=checked, old_snapshot_files_changed=0,
        old_snapshot_archive_SHA256=sha(a.previous_experiment / 'code_snapshot.tar.gz'),
        actually_smoke_tested_training_modules_unchanged=list(used),
        smoke_report_SHA256=sha(a.root / 'smoke/run_summary.json'),
        regression_report_SHA256=sha(a.regression / 'pretrain/run_summary.json'),
        regression_actual_updates=300, regression_actual_unique_sources=regression['actual_unique_sources'],
        qualification_report_SHA256=sha(a.qualification / 'run_summary.json'),
        full_native_whole_crop_consistency_cases=12, old_models_overwritten=False,
        final_experiment_code_SHA256={f.name: sha(f) for f in (a.root / 'standalone').glob('*.py')},
        real_clean_truth=False, real_quality_accepted=False)
    save(target, report)
    print(report['status'], 'old_code_unchanged', checked, flush=True)


if __name__ == '__main__':
    main()
