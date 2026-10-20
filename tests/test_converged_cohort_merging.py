"""Conservative persistent-basin consolidation; no threshold relaxation."""
import math

import numpy as np
import pytest

from oceansim.slam import GlobalCohortTracker, GlobalCohortTrackerConfig, GlobalPoseEvidence


def populated_tracker(enabled=True, *, yaw=0.0):
    tracker = GlobalCohortTracker(GlobalCohortTrackerConfig(
        merge_converged_cohorts=enabled, particles_per_cohort=2048,
    ))
    # First create genuinely separated hypotheses through the public API.
    tracker.update(timestamp_s=1, raw_pose2=np.zeros(3), evidences=[
        GlobalPoseEvidence(
            map_pose2=np.array([i * 40., 0., yaw]),
            bathymetry_score=.9, backscatter_score=.9,
            position_std_m=1., yaw_std_deg=.2,
            source_index=i, patch_start_timestamp_s=0,
            patch_end_timestamp_s=1, patch_start_observation_id=0,
            patch_end_observation_id=5,
        ) for i in range(3)
    ])
    # Controlled post-update posterior: all three tracks have converged.
    # Existing one-window NMS explicitly leaves persistent IDs separate.
    for i, cohort in enumerate(tracker._cohorts):
        cohort.particles[:, 0] -= cohort.map_pose2[0] - (i - 1) * 2.
        tracker._refresh_particle_statistics(cohort)
        cohort.log_probability = math.log(.99 / 3)
        cohort.last_measurement_pose2 = np.array([(i - 1) * 2., 0., yaw])
        cohort.last_measurement_covariance = np.diag([1., 1., .001])
        cohort.last_measurement_information = np.linalg.inv(cohort.last_measurement_covariance)
        cohort.total_evidence = cohort.independent_evidence = 5
        cohort.last_measurement_is_graph_ready = True
        cohort.last_update_was_independent = True
    tracker._null_log_probability = math.log(.01)
    return tracker


def test_converged_mass_is_added_without_multiplying_correlated_information():
    tracker = populated_tracker()
    assert not tracker._decision().confirmed
    covariance_before = tracker._cohorts[0].covariance.copy()
    measurement_info_before = tracker._cohorts[0].last_measurement_information.copy()
    tracker._merge_converged_cohorts()
    tracker._normalize_and_prune()
    decision = tracker._decision({tracker._cohorts[0].cohort_id})
    assert decision.confirmed
    assert decision.confirmed_with_new_independent_evidence
    assert len(decision.cohorts) == 1
    assert decision.best_probability == pytest.approx(.99)
    assert decision.null_probability == pytest.approx(.01)
    cohort = decision.confirmed_cohort
    assert cohort.independent_evidence == 5  # not 15
    assert cohort.total_evidence == 5
    assert cohort.particle_count == 2048
    assert cohort.covariance[0, 0] > covariance_before[0, 0] * 2
    assert cohort.last_measurement_information[0, 0] < measurement_info_before[0, 0]
    assert not np.allclose(cohort.covariance, cohort.last_measurement_covariance)
    assert tracker.converged_cohorts_merged == 2


def test_disabled_merging_is_exact_no_op_including_rng():
    tracker = populated_tracker(False)
    before = [(c.particles.copy(), c.log_weights.copy(), c.log_probability)
              for c in tracker._cohorts]
    rng_before = str(tracker._rng.bit_generator.state)
    tracker._merge_converged_cohorts()
    assert len(tracker._cohorts) == len(before)
    for cohort, (particles, weights, probability) in zip(tracker._cohorts, before):
        np.testing.assert_array_equal(cohort.particles, particles)
        np.testing.assert_array_equal(cohort.log_weights, weights)
        assert cohort.log_probability == probability
    assert str(tracker._rng.bit_generator.state) == rng_before


@pytest.mark.parametrize('difference', ['position', 'yaw', 'measurement', 'stale', 'window'])
def test_distinct_or_unverified_hypotheses_are_not_erased(difference):
    tracker = populated_tracker()
    last = tracker._cohorts[-1]
    if difference == 'position':
        last.map_pose2[0] += 100
    elif difference == 'yaw':
        last.map_pose2[2] += .3
    elif difference == 'measurement':
        last.last_measurement_pose2[0] += 100
    elif difference == 'stale':
        last.last_measurement_is_graph_ready = False
    else:
        last.last_processed_patch_end_observation_id += 1
    tracker._merge_converged_cohorts()
    assert len(tracker._cohorts) == 2
    tracker._normalize_and_prune()
    assert not tracker._decision().confirmed


def test_newborn_cannot_inherit_five_independent_acquisitions():
    tracker = populated_tracker()
    tracker._cohorts[-1].total_evidence = 1
    tracker._cohorts[-1].independent_evidence = 1
    tracker._merge_converged_cohorts()
    tracker._normalize_and_prune()
    assert tracker._cohorts[0].independent_evidence == 1
    assert not tracker._decision().confirmed


def test_complete_link_does_not_chain_distinct_peaks():
    tracker = populated_tracker()
    for i, cohort in enumerate(tracker._cohorts):
        cohort.map_pose2[0] = i * 8.
        cohort.last_measurement_pose2[0] = i * 8.
    tracker._merge_converged_cohorts()
    assert len(tracker._cohorts) == 2  # 0--8--16 must not all merge


def test_yaw_wrap_and_null_mass_are_preserved():
    tracker = populated_tracker(yaw=np.pi - .005)
    tracker._cohorts[-1].map_pose2[2] = -np.pi + .005
    tracker._cohorts[-1].last_measurement_pose2[2] = -np.pi + .005
    tracker._merge_converged_cohorts()
    tracker._normalize_and_prune()
    assert len(tracker._cohorts) == 1
    assert tracker.null_probability == pytest.approx(.01)
    assert abs(tracker._cohorts[0].map_pose2[2]) > 3.


def test_merge_flag_is_boolean_and_off_by_default():
    assert not GlobalCohortTrackerConfig().merge_converged_cohorts
    with pytest.raises(TypeError, match='boolean'):
        GlobalCohortTrackerConfig(merge_converged_cohorts=1)


def test_null_information_direction_survives_consolidation():
    tracker = populated_tracker()
    for cohort in tracker._cohorts:
        cohort.last_measurement_information = np.diag([1., 1., 0.])
        cohort.last_measurement_observable_rank = 2
    tracker._merge_converged_cohorts()
    assert len(tracker._cohorts) == 1
    assert tracker._cohorts[0].last_measurement_information[2, 2] == 0.
    assert tracker._cohorts[0].last_measurement_observable_rank == 2


def test_separated_repetitive_terrain_peaks_still_cannot_confirm():
    tracker = GlobalCohortTracker(GlobalCohortTrackerConfig(merge_converged_cohorts=True))
    for epoch in range(1, 8):
        decision = tracker.update(timestamp_s=epoch * 30., raw_pose2=np.zeros(3),
            evidences=[GlobalPoseEvidence(
                map_pose2=np.array([x, 0., 0.]), bathymetry_score=.9, backscatter_score=.9,
                position_std_m=2., yaw_std_deg=1., source_index=i,
                patch_start_timestamp_s=epoch * 30. - 29.,
                patch_end_timestamp_s=epoch * 30.,
                patch_start_observation_id=epoch * 150 - 149,
                patch_end_observation_id=epoch * 150,
            ) for i, x in enumerate((0., 200.))])
    assert not decision.confirmed
    assert len(decision.cohorts) == 2
    assert tracker.converged_cohorts_merged == 0
