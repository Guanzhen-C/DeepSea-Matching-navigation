import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from AnalyzeMappedRangeNavigation import verify_frozen_dependencies, verify_preserve_prefix


def dependency_fixture(tmp_path):
    core = {'esikf.py', 'global_tracking.py', 'pose_graph.py', 'map_odom.py'}
    prior = {'backscatter.npy', 'backscatter_valid_mask.npy', 'metadata.json'}
    old = dict(original_core_source_SHA256={}, prior_intensity_inputs_SHA256={})
    replay = dict(frozen_core_and_prior_dependency_SHA256={})
    for name in sorted(core | prior):
        path = tmp_path/('estimator_bundle/isaacsim/oceansim/slam' if name in core else 'scene/reference_baseline')/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        old['original_core_source_SHA256' if name in core else 'prior_intensity_inputs_SHA256'][name] = digest
        replay['frozen_core_and_prior_dependency_SHA256'][str(path)] = digest
    return old, replay


def test_dependencies_are_all_rehashed_and_not_just_accepted_from_summary(tmp_path):
    old, replay = dependency_fixture(tmp_path)
    assert verify_frozen_dependencies(old, replay)['core_files'] == 4
    Path(next(iter(replay['frozen_core_and_prior_dependency_SHA256']))).write_bytes(b'changed')
    with pytest.raises(ValueError, match='dependency changed'):
        verify_frozen_dependencies(old, replay)


@pytest.mark.parametrize('mutation', ['missing_original', 'missing_executed', 'duplicate_name', 'wrong_role', 'wrong_hash'])
def test_incomplete_duplicate_or_wrong_role_dependencies_are_rejected(tmp_path, mutation):
    old, replay = dependency_fixture(tmp_path)
    deps = replay['frozen_core_and_prior_dependency_SHA256']
    first = next(iter(deps))
    if mutation == 'missing_original': old['original_core_source_SHA256'].pop('esikf.py')
    elif mutation == 'missing_executed': deps.pop(first)
    elif mutation == 'duplicate_name': deps[str(tmp_path/'duplicate'/Path(first).name)] = deps[first]
    elif mutation == 'wrong_role': deps[str(tmp_path/'wrong_role'/Path(first).name)] = deps.pop(first)
    else: deps[first] = '0'*64
    with pytest.raises(ValueError): verify_frozen_dependencies(old, replay)


def prefix_fixture():
    old = dict(requested_feature_matcher='LoFTR', historical_side_scan_loops_requested=True,
               submap_representation='adaptive_projected')
    replay = dict(mapped_range_policy='preserve_legacy', loop_closure=True,
        submap_representation='adaptive_projected', maximum_Global_difference_before_first_range_m=0.)
    source = dict(timestamps=np.arange(5.), positions=np.zeros((5, 4, 3)))
    # Truth is deliberately nowhere near the estimates. The audit must not
    # use it or reject improvement/regression by a ground-truth error gate.
    source['positions'][:, 0] = 1.e6
    state = dict(positions=np.zeros((5, 3, 3)))
    state['positions'][3:, 2] = 9.
    trace = [dict(timestamp_s=3.)]
    return old, replay, source, state, trace


def test_pre_range_prefix_is_recomputed_from_estimates_and_ignores_truth():
    args = prefix_fixture()
    result = verify_preserve_prefix(*args)
    assert result['samples'] == 3
    assert result['maximum_coordinate_difference_m'] == 0.
    args[2]['positions'][:, 0] *= -10
    assert verify_preserve_prefix(*args) == result


@pytest.mark.parametrize('mutation', ['changed_prefix', 'wrong_claim', 'missing_claim', 'protocol_change', 'bad_time'])
def test_false_prefix_or_different_protocol_cannot_pass_preservation_claim(mutation):
    old, replay, source, state, trace = prefix_fixture()
    if mutation == 'changed_prefix': state['positions'][1, 2, 0] = .001
    elif mutation == 'wrong_claim': replay['maximum_Global_difference_before_first_range_m'] = .001
    elif mutation == 'missing_claim': replay.pop('maximum_Global_difference_before_first_range_m')
    elif mutation == 'protocol_change': replay['loop_closure'] = False
    else: trace[0]['timestamp_s'] = np.nan
    with pytest.raises(ValueError): verify_preserve_prefix(old, replay, source, state, trace)


def test_no_new_factor_requires_the_entire_original_Global_trajectory():
    old, replay, source, state, _ = prefix_fixture()
    state['positions'][:] = 0.
    assert verify_preserve_prefix(old, replay, source, state, [])['samples'] == 5
    state['positions'][-1, 2, 0] = .01
    with pytest.raises(ValueError): verify_preserve_prefix(old, replay, source, state, [])


def test_tempered_profile_is_not_mislabeled_as_preserved():
    old, replay, source, state, trace = prefix_fixture()
    replay['mapped_range_policy'] = 'tempered_legacy'
    assert verify_preserve_prefix(old, replay, source, state, trace)['status'].startswith('not_applicable')
