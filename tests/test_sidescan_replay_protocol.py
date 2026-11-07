from pathlib import Path
import itertools
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from ReplayPureSideScanNavigation import same_executed_configuration


def test_reproduction_requires_the_actual_source_protocol_for_all_matrix_cells():
    for source_rep, source_loop in itertools.product(('legacy_strip', 'adaptive_projected'), (False, True)):
        original = dict(requested_feature_matcher='LoFTR', historical_side_scan_loops_requested=source_loop,
                        submap_representation=source_rep)
        for rep, loop in itertools.product(('legacy_strip', 'adaptive_projected'), (False, True)):
            assert same_executed_configuration(original, loop_closure=loop, submap_representation=rep) == (
                source_rep == rep and source_loop == loop)


def test_loop_off_is_an_ablation_not_a_failed_reproduction_of_a_loop_on_source():
    original = dict(requested_feature_matcher='LoFTR', historical_side_scan_loops_requested=True)
    assert not same_executed_configuration(original)
    assert same_executed_configuration(original, loop_closure=True)


def test_changed_canonical_geometry_or_matcher_is_not_claimed_as_original_protocol():
    original = dict(requested_feature_matcher='LoFTR')
    assert same_executed_configuration(original)
    assert not same_executed_configuration(original, canonical_submaps=True)
    assert not same_executed_configuration(original, scan_plane_registration='observable_subspace')
    assert not same_executed_configuration(dict(original, requested_feature_matcher='RootSIFT_baseline'))


def test_new_range_constraints_are_an_explicit_ablation_not_original_reproduction():
    original = dict(requested_feature_matcher='LoFTR')
    assert not same_executed_configuration(original, mapped_range_changes=True)
    upgraded = dict(original, mapped_range_changes=True)
    assert same_executed_configuration(upgraded, mapped_range_changes=True)
    assert not same_executed_configuration(upgraded)
    assert not same_executed_configuration(upgraded, mapped_range_changes=True,
                                           mapped_range_policy='preserve_legacy')
