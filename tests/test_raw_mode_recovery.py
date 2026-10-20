from types import SimpleNamespace

import numpy as np
import pytest

from oceansim.slam import GlobalCohortTracker, GlobalPoseEvidence, ParticleLikelihoodObservation
from oceansim.slam.raw_mode_recovery import RawModeRecovery


def decision(*, confirmed=False, independent=False, ready=False):
    return SimpleNamespace(confirmed_with_new_independent_evidence=independent,
        confirmed_cohort=SimpleNamespace(last_measurement_is_graph_ready=ready) if confirmed else None)


def observation():
    return ParticleLikelihoodObservation(evaluator=lambda x: np.zeros(len(x)),
        patch_start_timestamp_s=60.2, patch_end_timestamp_s=90.,
        patch_start_observation_id=301, patch_end_observation_id=450)


class FakeRaw:
    config = SimpleNamespace(bathymetry_reference_normalized_loss=1., bathymetry_fit_temperature=1.)

    def __init__(self):
        self.seeds = []

    def refine_pose(self, pose):
        self.seeds.append(np.array(pose))
        return np.array(pose) + [12., 0., 0.]

    def approximate_local_measurement(self, pose):
        return SimpleNamespace(pose2=pose, covariance=np.diag([2., 3., .01]),
            information=np.diag([.5, 1/3, 0.]), diagnostics=SimpleNamespace(
                best_bathymetry_normalized_loss=.2, best_backscatter_correlation=.8,
                bathymetry_observability=1., backscatter_observability=1.))


def test_default_off_never_evaluates_or_consumes_rng():
    policy = RawModeRecovery()
    for _ in range(100):
        policy.observe(decision())
    raw = FakeRaw()
    assert policy.generate(raw, [(7, np.zeros(3))], observation(), source_index=1) == ()
    assert raw.seeds == []


def test_trigger_uses_only_past_independently_usable_pf_fixes():
    policy = RawModeRecovery(enabled=True)
    raw = FakeRaw()
    for _ in range(2):
        policy.observe(decision())
    assert not policy.generate(raw, [(7, np.zeros(3))], observation(), source_index=1)
    # Confirmation of old/reused pings is not a fresh independent fix.
    policy.observe(decision(confirmed=True, ready=True))
    assert policy.generate(raw, [(7, np.zeros(3))], observation(), source_index=1)
    policy.observe(decision(confirmed=True, independent=True, ready=True))
    assert policy.failed_windows == 0
    assert not policy.generate(raw, [(7, np.zeros(3))], observation(), source_index=1)


def test_new_modes_are_unassigned_current_window_measurements_not_old_cohorts():
    policy = RawModeRecovery(enabled=True, after_failed_windows=1)
    policy.observe(decision())
    seed = np.zeros(3)
    evidence, = policy.generate(FakeRaw(), [(7, seed)], observation(), source_index=10)
    assert evidence.target_cohort_id is None
    assert evidence.source_index == 10
    assert evidence.patch_start_observation_id == 301
    assert evidence.patch_end_observation_id == 450
    np.testing.assert_array_equal(seed, np.zeros(3))
    np.testing.assert_array_equal(evidence.map_pose2, [12., 0., 0.])
    np.testing.assert_array_equal(evidence.information, np.diag([.5, 1/3, 0.]))
    # A newborn still cannot become a confirmed fix after one acquisition.
    result = GlobalCohortTracker().update(timestamp_s=90., raw_pose2=np.zeros(3),
        evidences=[evidence], particle_observation=observation())
    assert not result.confirmed
    assert result.cohorts[0].independent_evidence == 1


def test_recovery_does_not_override_provenance_nms_or_relax_alias_separation():
    tracker = GlobalCohortTracker()
    original = GlobalPoseEvidence(map_pose2=np.zeros(3), bathymetry_score=.1,
        backscatter_score=.1, target_cohort_id=7, source_index=0)
    near = GlobalPoseEvidence(map_pose2=np.array([8., 0., 0.]), bathymetry_score=.99,
        backscatter_score=.99, source_index=1)
    distant = GlobalPoseEvidence(map_pose2=np.array([25., 0., 0.]), bathymetry_score=.99,
        backscatter_score=.99, source_index=2)
    retained = tracker._deduplicate_same_observation((original, near, distant))
    assert [e.source_index for e in retained] == [0, 2]


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_interval(value):
    with pytest.raises(ValueError):
        RawModeRecovery(after_failed_windows=value)


def test_invalid_enable_flag():
    with pytest.raises(TypeError):
        RawModeRecovery(enabled=1)
