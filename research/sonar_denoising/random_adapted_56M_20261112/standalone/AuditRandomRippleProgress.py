"""Read-only audit of committed optimizer-input ledgers, not model quality.

This utility is deployed outside the frozen training source directory. It
checks executed input records, not merely configuration or intended sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess

from NoFIRSonarData import plan_source, read, save, sha
from RandomTriggeredSonarNoise import PROTOCOL, draw_scenario, scenario_sha


def complete_records(path):
    # Capture a finite prefix: a concurrently growing ledger is not a snapshot.
    with path.open('rb') as stream:
        payload = stream.read(path.stat().st_size)
    lines = payload.splitlines(keepends=True)
    rows = []
    for number, line in enumerate(lines):
        if not line.endswith(b'\n'):
            if number != len(lines) - 1:
                raise ValueError('Interior ledger record is incomplete.')
            break
        rows.append(json.loads(line))
    return rows


def phase_admission_path(root, phase):
    """Preserve the unchanged live public phase's original data admission."""
    root = Path(root)
    current = root / 'admission_manifest.json'
    snapshot = root / 'pretrain_admission_snapshot.json'
    if phase == 'pretrain' and snapshot.exists():
        old, new = read(snapshot), read(current)
        public_old = [r for r in old['records'] if r['domain'] != 3]
        public_new = [r for r in new['records'] if r['domain'] != 3]
        if public_old != public_new:
            raise ValueError('Public inputs changed during the frozen public training phase.')
        return snapshot
    return current


def audit_prefix(root, phase):
    root = Path(root)
    stage = read(root / phase / 'run_summary.json')
    profile = read(root / 'random_ripple_profile.json')
    admission_path = phase_admission_path(root, phase)
    admission = read(admission_path)
    if stage.get('random_ripple_augmentation') is not True:
        raise ValueError('Actual random-ripple stage required.')
    if (stage['source_manifest_SHA256'] != sha(admission_path)
            or stage['noise_profile_SHA256'] != sha(root / 'random_ripple_profile.json')):
        raise ValueError('Training profile or source admission changed.')
    world = stage['world_size']
    ledgers = [complete_records(root / phase / f'actual_rank_{rank}_inputs.jsonl') for rank in range(world)]
    prefix = min(map(len, ledgers))
    reported = stage['steps_completed']
    if prefix < reported:
        raise ValueError('Reported updates exceed the completely flushed rank ledgers.')
    # Entries beyond the current all-rank report can be uncommitted. Only
    # verify the publicly committed prefix and record the larger flushed one.
    exposures, augmented, identity, domains, levels, counts = (Counter() for _ in range(6))
    phases, positions, spacings = [], [], []
    changed_values = 0
    for rank, rows in enumerate(ledgers):
        for step, entry in enumerate(rows[:reported], 1):
            if entry['rank'] != rank or entry['step'] != step:
                raise ValueError('Rank/optimizer-step sequence differs.')
            index = (step - 1) * world + rank
            expected = plan_source(admission['records'], phase, index)
            if entry['id'] != expected['id'] or entry['domain'] != expected['domain']:
                raise ValueError('Executed input does not match the admitted source schedule.')
            if not expected['role'].endswith('train'):
                raise ValueError('Held-out input entered optimization.')
            seed = 17618000 + index + (0 if phase == 'pretrain' else 2000000)
            expected_identity = random.Random(seed + 713).random() < .2
            if entry['seed'] != seed or entry['identity'] != expected_identity:
                raise ValueError('Actual draw/identity sampling differs from its seed.')
            noise = entry['noise']
            scenario = draw_scenario(profile, expected['shape'], expected['domain'], seed,
                                     side=expected.get('native_side'))
            if noise['protocol'] != PROTOCOL or noise['scenario_SHA256'] != scenario_sha(scenario):
                raise ValueError('Actual noise scenario differs from reproducible source geometry.')
            if noise['frequency_FIR'] is not False:
                raise ValueError('Undeclared reference FIR preprocessing.')
            changed = noise['actual_changed_values']
            if not isinstance(changed, int) or changed < 0 or (expected_identity and changed != 0):
                raise ValueError('Invalid actual-corruption exposure count.')
            if len(entry['observed_SHA256']) != 64 or any(c not in '0123456789abcdef' for c in entry['observed_SHA256']):
                raise ValueError('Missing actual optimizer-input digest.')
            if not math.isfinite(entry['loss']) or not math.isfinite(entry['gradient_norm']):
                raise ValueError('Nonfinite actual loss/gradient norm.')
            y, x, h, w = entry['native_core']
            if min(y, x) < 0 or min(h, w) <= 0 or y+h > expected['shape'][0] or x+w > expected['shape'][1]:
                raise ValueError('Optimizer core outside original native observation.')
            exposures[entry['id']] += 1
            domains[entry['domain']] += 1
            changed_values += changed
            if expected_identity:
                identity[entry['id']] += 1
            elif changed:
                augmented[entry['id']] += 1
            levels[scenario['relative_noise_RMS']] += 1
            counts[len(scenario['events'])] += 1
            for event in scenario['events']:
                phases.append(event['phase'])
                positions.append((event['centre_record'], event['centre_range']))
                spacings.append(event['spacing'])
    expected_role = 'public_train' if phase == 'pretrain' else 'native_train'
    expected_sources = {item['id'] for item in admission['records'] if item['role'] == expected_role}
    return dict(status='passed_actual_committed_input_prefix_NOT_training_completion_or_quality',
        phase=phase, training_stage_status=stage['status'], actual_updates_verified=reported,
        all_rank_flushed_updates_available=prefix, actual_global_examples_verified=reported * world,
        world_size=world, planned_phase_updates=stage['steps_requested'],
        source_manifest_SHA256=stage['source_manifest_SHA256'], noise_profile_SHA256=stage['noise_profile_SHA256'],
        actual_phase_admission_snapshot=str(admission_path),
        actual_unique_sources=len(exposures), actual_unique_augmented_sources=len(augmented),
        all_declared_phase_sources_seen=expected_sources <= set(exposures),
        all_declared_phase_sources_actually_augmented=expected_sources <= set(augmented),
        missing_augmented_phase_sources=sorted(expected_sources - set(augmented)),
        identity_examples=sum(identity.values()), actually_corrupted_examples=sum(augmented.values()),
        actual_changed_context_values=changed_values, actual_domain_exposures=dict(domains),
        actual_source_exposures=dict(exposures), actual_augmented_source_exposures=dict(augmented),
        randomized_source_count_histogram=dict(counts), randomized_relative_strength_histogram=dict(levels),
        source_phase_range=[min(phases), max(phases)] if phases else [],
        source_spacing_range=[min(spacings), max(spacings)] if spacings else [],
        distinct_source_positions=len(set(positions)),
        heldout_optimizer_inputs=0, seed_and_scenario_matches_verified=reported * world,
        finite_actual_losses_and_gradients=True, original_source_pixels_read_or_modified_by_audit=False,
        actual_input_SHA_recorded_NOT_full_signal_regenerated_by_audit=True,
        training_completed=False, real_quality_accepted=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=('pretrain', 'adapt'), required=True)
    p.add_argument('--unit', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous observations.')
    result = audit_prefix(a.root, a.phase)
    state = subprocess.check_output(['systemctl', '--user', 'show', a.unit,
        '-p', 'MainPID', '-p', 'ExecMainStatus', '-p', 'SubState'], text=True)
    result['authoritative_unit_observation'] = dict(line.split('=', 1) for line in state.splitlines())
    result['unit'] = a.unit
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save(a.output, result)
    print(json.dumps({key: result[key] for key in ('status', 'actual_updates_verified',
        'actual_global_examples_verified', 'actual_unique_sources', 'actual_unique_augmented_sources',
        'all_declared_phase_sources_actually_augmented', 'authoritative_unit_observation')}), flush=True)


if __name__ == '__main__':
    main()
