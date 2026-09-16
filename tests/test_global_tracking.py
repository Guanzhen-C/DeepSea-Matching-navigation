from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oceansim.slam import (
    GlobalCohortTracker,
    GlobalCohortTrackerConfig,
    GlobalPoseEvidence,
    ParticleLikelihoodObservation,
    reporting_covariance_from_information,
    select_terrain_navigation_output_source,
)


def _evidence(
    pose: tuple[float, float, float],
    *,
    source_index: int,
    start_id: int,
    end_id: int,
    bathymetry_score: float = 0.85,
    backscatter_score: float = 0.85,
    bathymetry_observability: float = 1.0,
    backscatter_observability: float = 1.0,
    geometry_score: float | None = 0.9,
    cross_modal_consistent: bool = True,
    position_std_m: float = 8.0,
    yaw_std_deg: float = 2.0,
) -> GlobalPoseEvidence:
    return GlobalPoseEvidence(
        map_pose2=np.asarray(pose, dtype=np.float64),
        bathymetry_score=bathymetry_score,
        backscatter_score=backscatter_score,
        bathymetry_observability=bathymetry_observability,
        backscatter_observability=backscatter_observability,
        geometry_score=geometry_score,
        cross_modal_consistent=cross_modal_consistent,
        position_std_m=position_std_m,
        yaw_std_deg=yaw_std_deg,
        patch_start_timestamp_s=float(start_id),
        patch_end_timestamp_s=float(end_id),
        patch_start_observation_id=start_id,
        patch_end_observation_id=end_id,
        source_index=source_index,
    )


def _permissive_config(**overrides) -> GlobalCohortTrackerConfig:
    values = dict(
        minimum_total_evidence=3,
        minimum_independent_evidence=3,
        confirmation_probability=0.90,
        confirmation_odds_over_second=8.0,
        confirmation_max_position_std_m=20.0,
        confirmation_max_yaw_std_deg=5.0,
    )
    values.update(overrides)
    return GlobalCohortTrackerConfig(**values)


def _particle_observation(
    centre: tuple[float, float, float],
    *,
    start_id: int,
    end_id: int,
    strength: float = 8.0,
) -> ParticleLikelihoodObservation:
    target = np.asarray(centre, dtype=np.float64)

    def evaluate(particles: np.ndarray) -> np.ndarray:
        residual = particles - target
        residual[:, 2] = (
            residual[:, 2] + np.pi
        ) % (2.0 * np.pi) - np.pi
        return strength - 0.5 * (
            np.sum(residual[:, :2] ** 2, axis=1) / 4.0**2
            + residual[:, 2] ** 2 / np.deg2rad(3.0) ** 2
        )

    return ParticleLikelihoodObservation(
        evaluator=evaluate,
        patch_start_timestamp_s=float(start_id),
        patch_end_timestamp_s=float(end_id),
        patch_start_observation_id=start_id,
        patch_end_observation_id=end_id,
    )


@pytest.mark.parametrize(
    ("global_count", "ordinary_count", "expected"),
    [
        (2, 4, "global_terrain_particle_map"),
        (0, 2, "ordinary_terrain_particle_map"),
        (0, 1, "pose_graph"),
        (0, 0, "pose_graph"),
    ],
)
def test_online_output_source_policy(
    global_count, ordinary_count, expected
) -> None:
    assert (
        select_terrain_navigation_output_source(
            global_active_cohorts=global_count,
            ordinary_active_cohorts=ordinary_count,
        )
        == expected
    )


def test_formal_feature_launcher_uses_gtsam_multimodal_backend() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "standalone"
        / "run_v2_full_map_slam_isaac.sh"
    ).read_text(encoding="utf-8")

    assert "--feature-compute-backend cuda" in launcher
    assert "--pose-graph-fixed-lag-s 180" in launcher
    assert "--terrain-global-particles-per-cohort 256" in launcher
    assert "--local-frontend" not in launcher
    assert "--pose-graph-backend" not in launcher
    assert "--terrain-particle-online-output" not in launcher


def test_active_global_recovery_cohort_drives_provisional_output() -> None:
    """An active whole-map recovery mode has priority over ordinary output."""

    tracker = GlobalCohortTracker(_permissive_config(random_seed=41))
    decision = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (45.0, -20.0, 0.2),
                source_index=0,
                start_id=0,
                end_id=20,
            )
        ],
    )

    assert not decision.confirmed
    assert tracker.maximum_a_posteriori_pose(np.zeros(3)) is not None
    assert (
        select_terrain_navigation_output_source(
            global_active_cohorts=tracker.active_cohort_count,
            ordinary_active_cohorts=0,
        )
        == "global_terrain_particle_map"
    )


def test_repeated_independent_evidence_confirms_one_persistent_cohort() -> None:
    tracker = GlobalCohortTracker(_permissive_config())
    raw_poses = (
        np.array([0.0, 0.0, 0.0]),
        np.array([80.0, 0.0, 0.0]),
        np.array([160.0, 0.0, 0.0]),
    )
    map_poses = (
        (1000.0, -300.0, 0.10),
        (1079.6, -292.0, 0.10),
        (1159.2, -284.0, 0.10),
    )

    decisions = []
    for index, (raw, mapped) in enumerate(zip(raw_poses, map_poses)):
        start = index * 100
        decisions.append(
            tracker.update(
                timestamp_s=10.0 + index * 100.0,
                raw_pose2=raw,
                evidences=[
                    _evidence(
                        mapped,
                        source_index=index,
                        start_id=start,
                        end_id=start + 20,
                    )
                ],
            )
        )

    assert not decisions[0].confirmed
    assert not decisions[1].confirmed
    assert decisions[2].confirmed
    cohort = decisions[2].confirmed_cohort
    assert cohort is not None
    assert cohort.total_evidence == 3
    assert cohort.independent_evidence == 3
    assert cohort.probability > 0.90
    assert np.linalg.norm(cohort.map_pose2[:2] - map_poses[-1][:2]) < 1.0
    assert np.isclose(
        sum(item.probability for item in decisions[2].cohorts)
        + decisions[2].null_probability,
        1.0,
    )


def test_two_repeatable_aliases_remain_ambiguous_instead_of_selecting_top_one() -> None:
    tracker = GlobalCohortTracker(_permissive_config())
    final = None
    for index in range(4):
        displacement = 60.0 * index
        start = 100 * index
        final = tracker.update(
            timestamp_s=20.0 + 100.0 * index,
            raw_pose2=np.array([displacement, 0.0, 0.0]),
            evidences=[
                _evidence(
                    (500.0 + displacement, 100.0, 0.0),
                    source_index=0,
                    start_id=start,
                    end_id=start + 10,
                ),
                _evidence(
                    (-700.0 + displacement, 400.0, 0.0),
                    source_index=1,
                    start_id=start,
                    end_id=start + 10,
                ),
            ],
        )

    assert final is not None
    assert not final.confirmed
    assert final.status == "tracking"
    assert len(final.cohorts) == 2
    assert final.cohorts[0].independent_evidence == 4
    assert final.cohorts[1].independent_evidence == 4
    assert final.odds_over_second < 2.0
    assert tracker.ambiguous_updates >= 1


def test_same_window_refined_duplicates_cannot_split_posterior_or_evidence() -> None:
    tracker = GlobalCohortTracker(_permissive_config())
    decisions = []
    for index in range(3):
        displacement = 80.0 * index
        start = 100 * index
        decisions.append(
            tracker.update(
                timestamp_s=10.0 + 100.0 * index,
                raw_pose2=np.array([displacement, 0.0, 0.0]),
                evidences=[
                    _evidence(
                        (1000.0 + displacement, -200.0, 0.10),
                        source_index=0,
                        start_id=start,
                        end_id=start + 20,
                        bathymetry_score=0.80,
                        backscatter_score=0.80,
                    ),
                    # A second coarse yaw seed converged to the same metric
                    # basin in this very same acquisition.
                    _evidence(
                        (1000.4 + displacement, -200.3, 0.105),
                        source_index=1,
                        start_id=start,
                        end_id=start + 20,
                        bathymetry_score=0.92,
                        backscatter_score=0.93,
                    ),
                ],
            )
        )

    assert all(len(decision.cohorts) == 1 for decision in decisions)
    assert not decisions[0].confirmed
    assert not decisions[1].confirmed
    assert decisions[2].confirmed
    cohort = decisions[2].confirmed_cohort
    assert cohort is not None
    assert cohort.total_evidence == 3
    assert cohort.independent_evidence == 3
    assert cohort.last_source_index == 1
    assert tracker.duplicate_evidences_suppressed == 3


def test_overlapping_rolling_windows_cannot_manufacture_confirmation() -> None:
    tracker = GlobalCohortTracker(_permissive_config())
    decision = None
    for index in range(8):
        decision = tracker.update(
            timestamp_s=10.0 + index,
            raw_pose2=np.array([float(index), 0.0, 0.0]),
            evidences=[
                _evidence(
                    (101.0 + index, 20.0, 0.0),
                    source_index=0,
                    start_id=index,
                    end_id=100 + index,
                )
            ],
        )

    assert decision is not None
    assert not decision.confirmed
    assert decision.cohorts[0].total_evidence == 8
    assert decision.cohorts[0].independent_evidence == 1


def test_rolling_windows_eventually_add_independent_evidence() -> None:
    """Overlaps must not move the last independent acquisition boundary."""

    tracker = GlobalCohortTracker(_permissive_config())
    decisions = []
    for start in range(0, 241, 30):
        decisions.append(
            tracker.update(
                timestamp_s=1000.0 + start,
                raw_pose2=np.array([float(start), 0.0, 0.0]),
                evidences=[
                    _evidence(
                        (1000.0 + start, -200.0, 0.1),
                        source_index=0,
                        start_id=start,
                        end_id=start + 100,
                    )
                ],
            )
        )

    assert decisions[3].cohorts[0].independent_evidence == 1  # start 90
    assert decisions[4].cohorts[0].independent_evidence == 2  # start 120
    assert decisions[7].cohorts[0].independent_evidence == 2  # start 210
    assert decisions[8].cohorts[0].independent_evidence == 3  # start 240
    assert decisions[8].confirmed


def test_one_shot_false_peak_decays_back_to_null_hypothesis() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            maximum_missed_observations=3,
            minimum_retained_probability=1.0e-4,
        )
    )
    first = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (800.0, -600.0, 0.2),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
    )
    assert first.cohorts

    decision = first
    for index in range(1, 6):
        decision = tracker.update(
            timestamp_s=1.0 + index,
            raw_pose2=np.array([float(index), 0.0, 0.0]),
            evidences=[],
        )

    assert not decision.confirmed
    assert decision.status == "no_fix"
    assert decision.null_probability > 0.99
    assert not decision.cohorts


def test_modality_observability_adapts_likelihood_instead_of_fixed_weighting() -> None:
    config = _permissive_config(
        minimum_total_evidence=1,
        minimum_independent_evidence=1,
        confirmation_probability=0.50,
        confirmation_odds_over_second=1.1,
    )
    good_sss = GlobalCohortTracker(config)
    decision = good_sss.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (20.0, 30.0, 0.0),
                source_index=0,
                start_id=0,
                end_id=10,
                bathymetry_score=-0.8,
                backscatter_score=0.9,
                bathymetry_observability=0.0,
                backscatter_observability=1.0,
                geometry_score=None,
            )
        ],
    )
    assert decision.confirmed

    inconsistent = GlobalCohortTracker(config)
    rejected = inconsistent.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (20.0, 30.0, 0.0),
                source_index=0,
                start_id=0,
                end_id=10,
                cross_modal_consistent=False,
            )
        ],
    )
    assert not rejected.confirmed
    assert rejected.null_probability > rejected.best_probability


def test_yaw_wrap_and_local_odometry_propagation_preserve_cohort_identity() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            minimum_total_evidence=2,
            minimum_independent_evidence=2,
            confirmation_probability=0.80,
            confirmation_odds_over_second=3.0,
        )
    )
    first = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.array([0.0, 0.0, np.deg2rad(179.0)]),
        evidences=[
            _evidence(
                (100.0, 50.0, np.deg2rad(179.5)),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
    )
    cohort_id = first.cohorts[0].cohort_id
    second = tracker.update(
        timestamp_s=101.0,
        raw_pose2=np.array([-20.0, 0.0, np.deg2rad(-179.0)]),
        evidences=[
            _evidence(
                (80.0, 49.3, np.deg2rad(-178.5)),
                source_index=0,
                start_id=100,
                end_id=110,
            )
        ],
    )

    assert len(second.cohorts) == 1
    assert second.cohorts[0].cohort_id == cohort_id
    assert second.confirmed


def test_each_cohort_contains_a_weighted_particle_set() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(particles_per_cohort=128, random_seed=7)
    )
    decision = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (120.0, -80.0, 0.2),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
    )

    cohort = decision.cohorts[0]
    assert cohort.particle_count == 128
    assert cohort.effective_sample_size == pytest.approx(128.0)
    assert np.all(np.linalg.eigvalsh(cohort.covariance) > 0.0)
    diagnostics = tracker.diagnostics()
    assert diagnostics["active_particles"] == 128
    assert diagnostics["particles_per_cohort"] == 128


def test_nonmutating_prediction_seed_propagates_local_se2_motion() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=256,
            random_seed=13,
            process_translation_std_per_m=0.0,
            process_translation_std_per_s=0.0,
            process_yaw_std_deg_per_m=0.0,
            process_yaw_std_deg_per_s=0.0,
        )
    )
    first = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.array([4.0, -2.0, 0.2]),
        evidences=[
            _evidence(
                (100.0, 50.0, 0.3),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
    )
    before = first.cohorts[0].map_pose2.copy()
    current_raw = np.array([14.0, 3.0, 0.25])

    seeds = tracker.prediction_seeds(current_raw)

    assert len(seeds) == 1
    cohort_id, predicted = seeds[0]
    assert cohort_id == first.cohorts[0].cohort_id
    raw_delta = current_raw.copy()
    raw_delta[:2] -= np.array([4.0, -2.0])
    # Compare to an explicit transform composition through public geometry.
    map_to_odom_yaw = before[2] - 0.2
    c, s = np.cos(map_to_odom_yaw), np.sin(map_to_odom_yaw)
    expected_xy = before[:2] + np.array(
        [c * raw_delta[0] - s * raw_delta[1],
         s * raw_delta[0] + c * raw_delta[1]]
    )
    # Propagating the full yaw-distributed particle cloud is not exactly the
    # same as propagating only its circular mean (Jensen's inequality), but it
    # must remain close to that intuitive rigid-transform check.
    np.testing.assert_allclose(predicted[:2], expected_xy, atol=0.02)
    assert predicted[2] == pytest.approx(before[2] + 0.05)
    propagated = tracker.update(
        timestamp_s=2.0,
        raw_pose2=current_raw,
        evidences=[],
    )
    np.testing.assert_allclose(
        propagated.cohorts[0].map_pose2, predicted, atol=1.0e-10
    )


def test_map_output_uses_spatial_map_and_rejects_null_dominant_state() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=128,
            random_seed=19,
            process_translation_std_per_m=0.0,
            process_translation_std_per_s=0.0,
            process_yaw_std_deg_per_m=0.0,
            process_yaw_std_deg_per_s=0.0,
        )
    )
    raw = np.array([4.0, -2.0, 0.2])
    assert tracker.active_cohort_count == 0
    assert tracker.maximum_a_posteriori_pose(raw) is None

    decision = tracker.update(
        timestamp_s=1.0,
        raw_pose2=raw,
        evidences=[
            _evidence(
                (100.0, 50.0, 0.3),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
    )
    current_raw = np.array([14.0, 3.0, 0.25])
    output = tracker.maximum_a_posteriori_pose(current_raw)
    assert output is not None
    assert tracker.active_cohort_count == 1
    np.testing.assert_allclose(
        output, tracker.prediction_seeds(current_raw)[0][1]
    )
    assert not output.flags.writeable
    assert decision.best_probability > decision.null_probability

    null_tracker = GlobalCohortTracker(
        _permissive_config(particles_per_cohort=128, random_seed=23)
    )
    rejected = null_tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (100.0, 50.0, 0.3),
                source_index=0,
                start_id=0,
                end_id=10,
                bathymetry_score=-1.0,
                backscatter_score=-1.0,
                geometry_score=0.0,
                cross_modal_consistent=False,
            )
        ],
    )
    assert rejected.null_probability > rejected.best_probability
    assert null_tracker.maximum_a_posteriori_pose(np.zeros(3)) is None


def test_sharp_measurement_triggers_regularized_systematic_resampling() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=256,
            random_seed=11,
            resample_effective_sample_fraction=0.90,
        )
    )
    tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (100.0, 50.0, 0.1),
                source_index=0,
                start_id=0,
                end_id=10,
                position_std_m=20.0,
                yaw_std_deg=8.0,
            )
        ],
    )
    decision = tracker.update(
        timestamp_s=101.0,
        raw_pose2=np.array([20.0, 0.0, 0.0]),
        evidences=[
            _evidence(
                (119.9, 52.0, 0.1),
                source_index=0,
                start_id=100,
                end_id=110,
                position_std_m=5.0,
                yaw_std_deg=2.0,
            )
        ],
    )

    cohort = decision.cohorts[0]
    assert cohort.resampling_events == 1
    assert cohort.effective_sample_size < 0.90 * cohort.particle_count
    assert tracker.diagnostics()["particle_resampling_events"] == 1
    assert np.all(np.linalg.eigvalsh(cohort.covariance) > 0.0)


def test_particle_filter_is_reproducible_for_a_fixed_seed() -> None:
    config = _permissive_config(particles_per_cohort=64, random_seed=19)
    trackers = (GlobalCohortTracker(config), GlobalCohortTracker(config))
    decisions = []
    for tracker in trackers:
        tracker.update(
            timestamp_s=1.0,
            raw_pose2=np.zeros(3),
            evidences=[
                _evidence(
                    (200.0, 300.0, -0.2),
                    source_index=0,
                    start_id=0,
                    end_id=10,
                )
            ],
        )
        decisions.append(
            tracker.update(
                timestamp_s=101.0,
                raw_pose2=np.array([40.0, 3.0, 0.01]),
                evidences=[
                    _evidence(
                        (240.0, 299.0, -0.19),
                        source_index=0,
                        start_id=100,
                        end_id=110,
                    )
                ],
            )
        )

    assert np.array_equal(
        decisions[0].cohorts[0].map_pose2,
        decisions[1].cohorts[0].map_pose2,
    )
    assert np.array_equal(
        decisions[0].cohorts[0].covariance,
        decisions[1].cohorts[0].covariance,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"maximum_cohorts": 0}, "positive integer"),
        ({"birth_probability": 1.0}, "lie in"),
        ({"score_temperature": 0.0}, "positive"),
        ({"same_observation_nms_position_m": 0.0}, "positive"),
        ({"particles_per_cohort": 0}, "positive integer"),
        ({"random_seed": -1}, "nonnegative integer"),
        ({"resample_effective_sample_fraction": 1.0}, "lie in"),
        ({"resample_kernel_bandwidth": 0.0}, "lie in"),
        (
            {"minimum_total_evidence": 2, "minimum_independent_evidence": 3},
            "must not exceed",
        ),
    ],
)
def test_tracker_config_rejects_invalid_policies(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GlobalCohortTrackerConfig(**kwargs)


def test_update_rejects_duplicate_candidate_indices_and_nonmonotonic_time() -> None:
    tracker = GlobalCohortTracker()
    evidence = _evidence(
        (0.0, 0.0, 0.0), source_index=0, start_id=0, end_id=1
    )
    with pytest.raises(ValueError, match="unique"):
        tracker.update(
            timestamp_s=1.0,
            raw_pose2=np.zeros(3),
            evidences=[evidence, evidence],
        )
    tracker.update(timestamp_s=1.0, raw_pose2=np.zeros(3), evidences=[evidence])
    with pytest.raises(ValueError, match="increase"):
        tracker.update(timestamp_s=1.0, raw_pose2=np.zeros(3), evidences=[])


def test_direct_particle_likelihood_retains_the_nonlinear_map_measurement() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=1024,
            random_seed=23,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.50,
            confirmation_odds_over_second=1.1,
            resample_effective_sample_fraction=0.01,
        )
    )
    decision = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (100.0, 50.0, 0.1),
                source_index=0,
                start_id=0,
                end_id=10,
                position_std_m=15.0,
                yaw_std_deg=8.0,
            )
        ],
        particle_observation=_particle_observation(
            (104.0, 47.0, 0.13), start_id=0, end_id=10
        ),
    )

    assert decision.confirmed
    assert decision.confirmed_with_new_independent_evidence
    cohort = decision.confirmed_cohort
    assert cohort is not None
    assert np.linalg.norm(
        cohort.map_pose2[:2] - np.array([104.0, 47.0])
    ) < 1.0
    assert cohort.independent_evidence == 1
    assert tracker.diagnostics()["direct_map_likelihood_updates"] == 1


def test_current_measurement_is_separate_from_recursive_particle_posterior() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=4096,
            random_seed=29,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.05,
            confirmation_odds_over_second=0.01,
            confirmation_max_position_std_m=100.0,
            confirmation_max_yaw_std_deg=30.0,
            resample_effective_sample_fraction=0.01,
        )
    )
    candidate = _evidence(
        (0.0, 0.0, 0.0),
        source_index=0,
        start_id=0,
        end_id=10,
        position_std_m=12.0,
        yaw_std_deg=10.0,
    )
    tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[candidate],
        particle_observation=_particle_observation(
            (-4.0, 0.0, 0.0), start_id=0, end_id=10, strength=10.0
        ),
    )
    decision = tracker.update(
        timestamp_s=101.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (4.0, 0.0, 0.0),
                source_index=0,
                start_id=100,
                end_id=110,
                position_std_m=12.0,
                yaw_std_deg=10.0,
            )
        ],
        particle_observation=_particle_observation(
            (4.0, 0.0, 0.0), start_id=100, end_id=110, strength=10.0
        ),
    )

    cohort = decision.cohorts[0]
    # The recursive posterior compromises between both epochs.  The retained
    # graph measurement deliberately represents only the second acquisition.
    assert cohort.last_measurement_pose2[0] > cohort.map_pose2[0] + 1.0
    assert cohort.last_measurement_pose2[0] == pytest.approx(4.0)
    assert cohort.last_measurement_is_graph_ready
    assert cohort.last_update_was_independent


def test_persistent_proposal_provenance_prevents_compact_measurement_theft() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=512,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.05,
            confirmation_odds_over_second=0.01,
            confirmation_max_position_std_m=100.0,
            confirmation_max_yaw_std_deg=30.0,
        )
    )
    first = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (0.0, 0.0, 0.0),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
        particle_observation=_particle_observation(
            (0.0, 0.0, 0.0), start_id=0, end_id=10
        ),
    )
    cohort_id = first.cohorts[0].cohort_id
    persistent = GlobalPoseEvidence(
        map_pose2=np.array([1.0, 0.0, 0.0]),
        bathymetry_score=0.85,
        backscatter_score=0.85,
        geometry_score=0.9,
        position_std_m=100.0,
        yaw_std_deg=20.0,
        patch_start_timestamp_s=100.0,
        patch_end_timestamp_s=110.0,
        patch_start_observation_id=100,
        patch_end_observation_id=110,
        source_index=0,
        target_cohort_id=cohort_id,
    )
    compact_alias = GlobalPoseEvidence(
        map_pose2=np.array([0.0, 0.0, 0.0]),
        bathymetry_score=0.85,
        backscatter_score=0.85,
        geometry_score=0.9,
        position_std_m=1.0,
        yaw_std_deg=1.0,
        patch_start_timestamp_s=100.0,
        patch_end_timestamp_s=110.0,
        patch_start_observation_id=100,
        patch_end_observation_id=110,
        source_index=1,
    )

    decision = tracker.update(
        timestamp_s=101.0,
        raw_pose2=np.zeros(3),
        evidences=[persistent, compact_alias],
        particle_observation=_particle_observation(
            (1.0, 0.0, 0.0), start_id=100, end_id=110
        ),
    )

    original = next(
        item for item in decision.cohorts if item.cohort_id == cohort_id
    )
    np.testing.assert_allclose(
        original.last_measurement_pose2, persistent.map_pose2
    )
    assert original.last_source_index == persistent.source_index
    assert original.last_measurement_is_graph_ready
    # This compact proposal is in the same metric basin, so same-observation
    # NMS removes it only after the provenance-labelled proposal has been
    # reserved.  It cannot overwrite the persistent measurement.
    assert len(decision.cohorts) == 1
    assert tracker.duplicate_evidences_suppressed == 1


def test_neutral_direct_observations_restore_probability_to_null_state() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=256,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.50,
            confirmation_odds_over_second=1.1,
            cohort_survival_probability=0.70,
            null_survival_probability=0.95,
        )
    )
    decision = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (30.0, 20.0, 0.0),
                source_index=0,
                start_id=0,
                end_id=10,
            )
        ],
        particle_observation=_particle_observation(
            (30.0, 20.0, 0.0), start_id=0, end_id=10, strength=12.0
        ),
    )
    assert decision.confirmed

    for index in range(1, 13):
        start_id = 100 * index
        neutral = ParticleLikelihoodObservation(
            evaluator=lambda particles: np.zeros(particles.shape[0]),
            patch_start_timestamp_s=float(start_id),
            patch_end_timestamp_s=float(start_id + 10),
            patch_start_observation_id=start_id,
            patch_end_observation_id=start_id + 10,
        )
        decision = tracker.update(
            timestamp_s=1.0 + index,
            raw_pose2=np.zeros(3),
            evidences=[],
            particle_observation=neutral,
        )

    assert not decision.confirmed
    assert decision.status == "no_fix"
    assert decision.null_probability > decision.best_probability


def test_overlapping_direct_window_cannot_authorize_a_fresh_graph_factor() -> None:
    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=512,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.50,
            confirmation_odds_over_second=1.1,
        )
    )
    candidate = _evidence(
        (20.0, 10.0, 0.0), source_index=0, start_id=0, end_id=100
    )
    first = tracker.update(
        timestamp_s=1.0,
        raw_pose2=np.zeros(3),
        evidences=[candidate],
        particle_observation=_particle_observation(
            (20.0, 10.0, 0.0), start_id=0, end_id=100, strength=12.0
        ),
    )
    assert first.confirmed_with_new_independent_evidence

    second = tracker.update(
        timestamp_s=2.0,
        raw_pose2=np.array([1.0, 0.0, 0.0]),
        evidences=[],
        particle_observation=_particle_observation(
            (21.0, 10.0, 0.0), start_id=1, end_id=101, strength=12.0
        ),
    )
    assert second.confirmed
    assert not second.confirmed_with_new_independent_evidence
    assert not second.cohorts[0].last_update_was_independent
    assert not second.cohorts[0].last_measurement_is_graph_ready
    assert second.cohorts[0].last_observation_novel_fraction == pytest.approx(
        1.0 / 101.0
    )


def test_full_measurement_covariance_is_preserved_for_graph_handoff() -> None:
    covariance = np.array(
        [
            [9.0, 1.5, 0.02],
            [1.5, 16.0, -0.01],
            [0.02, -0.01, np.deg2rad(4.0) ** 2],
        ]
    )
    evidence = GlobalPoseEvidence(
        map_pose2=np.array([10.0, 20.0, 0.1]),
        bathymetry_score=0.8,
        backscatter_score=0.8,
        covariance_matrix=covariance,
    )

    np.testing.assert_allclose(evidence.covariance, covariance)
    assert not np.shares_memory(evidence.covariance, covariance)


def test_rank_deficient_measurement_information_is_preserved_for_graph_handoff() -> None:
    covariance = np.diag([9.0, 16.0, 1.0e6])
    information = np.diag([1.0 / 9.0, 1.0 / 16.0, 0.0])
    evidence = GlobalPoseEvidence(
        map_pose2=np.array([10.0, 20.0, 0.1]),
        bathymetry_score=0.8,
        backscatter_score=0.8,
        covariance_matrix=covariance,
        information_matrix=information,
    )

    np.testing.assert_allclose(evidence.information, information)
    assert np.linalg.matrix_rank(evidence.information) == 2


def test_rank_one_measurement_is_safe_but_cannot_authorize_pose_factor() -> None:
    information = np.diag([0.25, 0.0, 0.0])
    covariance = reporting_covariance_from_information(information)
    evidence = GlobalPoseEvidence(
        map_pose2=np.array([10.0, 20.0, 0.1]),
        bathymetry_score=0.8,
        backscatter_score=0.0,
        bathymetry_observability=0.5,
        backscatter_observability=0.0,
        covariance_matrix=covariance,
        information_matrix=information,
        patch_start_timestamp_s=0.0,
        patch_end_timestamp_s=10.0,
        patch_start_observation_id=0,
        patch_end_observation_id=10,
        source_index=0,
    )
    tracker = GlobalCohortTracker(
        _permissive_config(
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
        )
    )

    decision = tracker.update(
        timestamp_s=10.0,
        raw_pose2=np.zeros(3),
        evidences=[evidence],
    )

    assert decision.cohorts
    assert decision.cohorts[0].last_measurement_observable_rank == 1
    assert not decision.cohorts[0].last_measurement_is_graph_ready
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_joint_raw_likelihood_can_reacquire_after_a_confident_wrong_mode() -> None:
    """A current-map candidate must compete with the incumbent in one PMF."""

    tracker = GlobalCohortTracker(
        _permissive_config(
            particles_per_cohort=512,
            random_seed=5,
            minimum_total_evidence=1,
            minimum_independent_evidence=1,
            confirmation_probability=0.50,
            confirmation_odds_over_second=1.1,
            confirmation_max_position_std_m=100.0,
            confirmation_max_yaw_std_deg=30.0,
            resample_effective_sample_fraction=0.01,
        )
    )

    def normalized_observation(
        centre_x: float, start_id: int, call_counter: list[int] | None = None
    ) -> ParticleLikelihoodObservation:
        def evaluate(particles: np.ndarray) -> np.ndarray:
            if call_counter is not None:
                call_counter[0] += 1
            loss = (
                (particles[:, 0] - centre_x) ** 2 / 4.0**2
                + particles[:, 1] ** 2 / 4.0**2
                + particles[:, 2] ** 2 / np.deg2rad(3.0) ** 2
            )
            # This mirrors a raw map likelihood whose relative term is
            # normalized by the best particle in the supplied population.
            return 4.0 + float(np.min(loss)) - loss

        return ParticleLikelihoodObservation(
            evaluator=evaluate,
            patch_start_timestamp_s=float(start_id),
            patch_end_timestamp_s=float(start_id + 10),
            patch_start_observation_id=start_id,
            patch_end_observation_id=start_id + 10,
        )

    # Build deliberately overwhelming history around the wrong terrain alias.
    for index in range(6):
        start_id = 100 * index
        decision = tracker.update(
            timestamp_s=float(index + 1),
            raw_pose2=np.zeros(3),
            evidences=[
                _evidence(
                    (0.0, 0.0, 0.0),
                    source_index=0,
                    start_id=start_id,
                    end_id=start_id + 10,
                    position_std_m=4.0,
                    yaw_std_deg=2.0,
                )
            ],
            particle_observation=normalized_observation(0.0, start_id),
        )
    assert decision.best_probability > 0.99
    wrong_cohort_id = decision.cohorts[0].cohort_id

    evaluator_calls = [0]
    start_id = 700
    decision = tracker.update(
        timestamp_s=7.0,
        raw_pose2=np.zeros(3),
        evidences=[
            _evidence(
                (80.0, 0.0, 0.0),
                source_index=0,
                start_id=start_id,
                end_id=start_id + 10,
                position_std_m=4.0,
                yaw_std_deg=2.0,
            )
        ],
        particle_observation=normalized_observation(
            80.0, start_id, evaluator_calls
        ),
    )

    assert evaluator_calls == [1]
    assert decision.confirmed_cohort is not None
    assert decision.confirmed_cohort.cohort_id != wrong_cohort_id
    assert decision.confirmed_cohort.map_pose2[0] == pytest.approx(
        80.0, abs=1.0
    )
    assert decision.confirmed_cohort.probability > 0.90
    diagnostics = tracker.diagnostics()
    assert diagnostics["joint_particle_likelihood_updates"] == 7
    assert diagnostics["candidate_birth_mass_allocations"] == 2
    assert diagnostics["posterior_mode_switches"] == 1
    assert diagnostics["last_joint_log_likelihood_span"] > 10.0
