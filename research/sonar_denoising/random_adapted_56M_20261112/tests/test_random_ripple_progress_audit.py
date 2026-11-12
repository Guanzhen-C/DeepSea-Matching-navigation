from pathlib import Path
import json
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from AuditRandomRippleProgress import audit_prefix, complete_records
from NoFIRSonarData import plan_source, save, sha
from RandomTriggeredSonarNoise import draw_scenario, scenario_sha, PROTOCOL


def fixtures(root):
    (root / 'pretrain').mkdir()
    records = [dict(id=f'public_{domain}', role='public_train', domain=domain, shape=[100, 300])
               for domain in (0, 1, 2)]
    profile = dict(triggered_source_model=dict(sides={str(s): dict(events=[dict(wave_spacing_scaled_samples=4.5)]) for s in (0, 1)}))
    save(root / 'admission_manifest.json', dict(records=records))
    save(root / 'random_ripple_profile.json', profile)
    save(root / 'pretrain/run_summary.json', dict(random_ripple_augmentation=True,
        source_manifest_SHA256=sha(root / 'admission_manifest.json'),
        noise_profile_SHA256=sha(root / 'random_ripple_profile.json'), world_size=4,
        steps_completed=1, steps_requested=3, status='training_actual_updates'))
    rows = []
    for rank in range(4):
        item = plan_source(records, 'pretrain', rank)
        seed = 17618000 + rank
        identity = random.Random(seed + 713).random() < .2
        scenario = draw_scenario(profile, item['shape'], item['domain'], seed)
        entry = dict(rank=rank, step=1, id=item['id'], domain=item['domain'], seed=seed,
            identity=identity, observed_SHA256='0' * 64, loss=.3, gradient_norm=.1,
            native_core=[0, 0, 50, 100], noise=dict(protocol=PROTOCOL,
                scenario_SHA256=scenario_sha(scenario), frequency_FIR=False,
                actual_changed_values=0 if identity else 100))
        (root / f'pretrain/actual_rank_{rank}_inputs.jsonl').write_text(json.dumps(entry) + '\n')
        rows.append(entry)
    return rows


def test_committed_prefix_checks_every_rank_seed_source_and_draw(tmp_path):
    fixtures(tmp_path)
    report = audit_prefix(tmp_path, 'pretrain')
    assert report['actual_updates_verified'] == 1
    assert report['actual_global_examples_verified'] == 4
    assert report['actual_unique_sources'] == 3
    assert report['heldout_optimizer_inputs'] == 0
    assert not report['training_completed']
    assert report['actual_input_SHA_recorded_NOT_full_signal_regenerated_by_audit']


def test_partial_last_write_is_not_mistaken_for_a_committed_optimizer_record(tmp_path):
    fixtures(tmp_path)
    path = tmp_path / 'pretrain/actual_rank_0_inputs.jsonl'
    with path.open('a') as stream:
        stream.write('{"step": 2')
    assert len(complete_records(path)) == 1
    assert audit_prefix(tmp_path, 'pretrain')['actual_global_examples_verified'] == 4


@pytest.mark.parametrize('kind', ['bad_seed', 'bad_scenario', 'nonfinite_loss', 'incorrect_source', 'wrong_rank', 'phantom_corruption'])
def test_invalid_input_evidence_is_rejected(tmp_path, kind):
    rows = fixtures(tmp_path)
    entry = rows[0]
    if kind == 'bad_seed':
        entry['seed'] += 1
    elif kind == 'bad_scenario':
        entry['noise']['scenario_SHA256'] = '0' * 64
    elif kind == 'nonfinite_loss':
        entry['loss'] = float('nan')
    elif kind == 'incorrect_source':
        entry['id'] = 'heldout_masquerading_as_train'
    elif kind == 'wrong_rank':
        entry['rank'] = 7
    else:
        # Seed 17618000 might select either identity or noisy. Force a
        # negative changed count, never valid for either case.
        entry['noise']['actual_changed_values'] = -10
    (tmp_path / 'pretrain/actual_rank_0_inputs.jsonl').write_text(json.dumps(entry) + '\n')
    with pytest.raises(ValueError):
        audit_prefix(tmp_path, 'pretrain')


def test_configuration_changes_after_training_are_not_accepted_as_same_run(tmp_path):
    fixtures(tmp_path)
    save(tmp_path / 'random_ripple_profile.json', dict(changed=True))
    with pytest.raises(ValueError, match='changed'):
        audit_prefix(tmp_path, 'pretrain')


def test_summary_cannot_claim_more_updates_than_all_rank_ledgers(tmp_path):
    fixtures(tmp_path)
    path = tmp_path / 'pretrain/run_summary.json'
    report = json.loads(path.read_text())
    report['steps_completed'] = 2
    save(path, report)
    with pytest.raises(ValueError, match='exceed'):
        audit_prefix(tmp_path, 'pretrain')


def test_future_native_metadata_does_not_rewrite_completed_public_inputs(tmp_path):
    import shutil
    fixtures(tmp_path)
    current = tmp_path / 'admission_manifest.json'
    snapshot = tmp_path / 'pretrain_admission_snapshot.json'
    shutil.copy2(current, snapshot)
    candidate = json.loads(current.read_text())
    candidate['future_native_support_version'] = 1  # no public records changed
    save(current, candidate)
    report = audit_prefix(tmp_path, 'pretrain')
    assert report['actual_phase_admission_snapshot'] == str(snapshot)
    assert report['actual_global_examples_verified'] == 4


def test_public_input_mutation_cannot_be_hidden_by_a_phase_snapshot(tmp_path):
    import shutil
    fixtures(tmp_path)
    current = tmp_path / 'admission_manifest.json'
    shutil.copy2(current, tmp_path / 'pretrain_admission_snapshot.json')
    candidate = json.loads(current.read_text())
    candidate['records'][0]['shape'][0] += 1
    save(current, candidate)
    with pytest.raises(ValueError, match='Public inputs changed'):
        audit_prefix(tmp_path, 'pretrain')
