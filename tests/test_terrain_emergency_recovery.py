from __future__ import annotations

import numpy as np
import pytest

from oceansim.slam import TerrainEmergencyRecoveryController


def _drive_to_emergency(
    controller: TerrainEmergencyRecoveryController,
    *,
    final_reason: str = "yaw_unobservable",
):
    request = None
    for index in range(7):
        request = controller.observe_preceding_recovery_outcome(
            timestamp=70.0 * index,
            raw_pose2=np.array([100.0 * index, 0.0, 0.0]),
            prior_added=False,
            preceding_recovery_invoked=index == 6,
            preceding_recovery_rejection_reason=(
                final_reason if index == 6 else "yaw_unobservable"
            ),
        )
        if index < 6:
            assert not request.should_attempt
    assert request is not None
    return request


def _next_observation(
    controller: TerrainEmergencyRecoveryController,
    *,
    x_m: float,
    reason: str = "search_boundary_reached",
):
    return controller.observe_preceding_recovery_outcome(
        timestamp=70.0 * controller.observation_index,
        raw_pose2=np.array([x_m, 0.0, 0.0]),
        prior_added=False,
        preceding_recovery_invoked=True,
        preceding_recovery_rejection_reason=reason,
    )


def _consider(
    controller: TerrainEmergencyRecoveryController,
    **kwargs,
):
    """Supply a unique synthetic 30-ping acquisition interval by default."""

    end_timestamp = float(controller._current_timestamp)
    end_observation_id = 30 * controller.observation_index
    kwargs.setdefault("patch_start_timestamp", end_timestamp - 20.0)
    kwargs.setdefault("patch_end_timestamp", end_timestamp)
    kwargs.setdefault(
        "patch_start_observation_id", end_observation_id - 29
    )
    kwargs.setdefault("patch_end_observation_id", end_observation_id)
    return controller.consider_recovery_match(**kwargs)


def test_emergency_requires_all_three_outage_thresholds() -> None:
    controller = TerrainEmergencyRecoveryController()
    request = _drive_to_emergency(controller)

    assert request.should_attempt
    assert request.reason == "ordinary_allowed"
    assert request.translation_window_m == 150.0
    assert request.strict_rank_three
    assert controller.consecutive_without_prior == 7
    assert controller.raw_arc_since_prior_m == pytest.approx(600.0)
    assert controller.time_since_prior_s == pytest.approx(420.0)

    arc_limited = TerrainEmergencyRecoveryController()
    for index in range(7):
        request = arc_limited.observe_preceding_recovery_outcome(
            timestamp=70.0 * index,
            raw_pose2=np.array([99.0 * index, 0.0, 0.0]),
            prior_added=False,
            preceding_recovery_invoked=index == 6,
            preceding_recovery_rejection_reason="yaw_unobservable",
        )
    assert not request.should_attempt
    assert request.reason == "raw_arc_threshold"

    time_limited = TerrainEmergencyRecoveryController()
    for index in range(7):
        request = time_limited.observe_preceding_recovery_outcome(
            timestamp=(399.0 / 6.0) * index,
            raw_pose2=np.array([100.0 * index, 0.0, 0.0]),
            prior_added=False,
            preceding_recovery_invoked=index == 6,
            preceding_recovery_rejection_reason="yaw_unobservable",
        )
    assert not request.should_attempt
    assert request.reason == "time_threshold"

    with pytest.raises(ValueError, match="must not exceed 150"):
        TerrainEmergencyRecoveryController(
            recovery_translation_window_m=150.001
        )


def test_emergency_never_bypasses_an_uninvoked_preceding_recovery() -> None:
    controller = TerrainEmergencyRecoveryController()
    for index in range(12):
        request = controller.observe_preceding_recovery_outcome(
            timestamp=70.0 * index,
            raw_pose2=np.array([100.0 * index, 0.0, 0.0]),
            prior_added=False,
            preceding_recovery_invoked=False,
            # Even a boundary reason is not proof that +/-100 m ran.
            preceding_recovery_rejection_reason="search_boundary_reached",
        )
        assert not request.should_attempt
        assert request.reason == "preceding_recovery_not_invoked"
    assert controller.consecutive_without_prior == 12
    assert controller.raw_arc_since_prior_m == pytest.approx(1100.0)

    invoked = controller.observe_preceding_recovery_outcome(
        timestamp=840.0,
        raw_pose2=np.array([1200.0, 0.0, 0.0]),
        prior_added=False,
        preceding_recovery_invoked=True,
        preceding_recovery_rejection_reason="search_boundary_reached",
    )
    assert invoked.should_attempt
    assert invoked.reason == "search_boundary_immediate"


@pytest.mark.parametrize(
    "reason",
    ["yaw_unobservable", "unobservable_terrain", "insufficient_inliers"],
)
def test_ordinary_allowlist_reasons_can_trigger(reason: str) -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(
        controller, final_reason=reason
    ).should_attempt


@pytest.mark.parametrize(
    "reason",
    [
        "ambiguous_match",
        "insufficient_initial_map_overlap",
        "insufficient_final_map_overlap",
        "coarse_search_failed",
        None,
    ],
)
def test_ambiguity_overlap_and_unlisted_reasons_fail_closed(reason) -> None:
    controller = TerrainEmergencyRecoveryController()
    request = _drive_to_emergency(controller, final_reason=reason)
    assert not request.should_attempt
    assert request.reason == "reason_not_allowed"
    assert controller.recovery_attempts == 0


def test_ordinary_throttle_and_boundary_bypass() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt

    throttled = _next_observation(
        controller, x_m=610.0, reason="insufficient_inliers"
    )
    assert not throttled.should_attempt
    assert throttled.reason == "ordinary_throttle"

    immediate = _next_observation(controller, x_m=620.0)
    assert immediate.should_attempt
    assert immediate.reason == "search_boundary_immediate"
    assert immediate.bypassed_throttle

    throttled_again = _next_observation(
        controller, x_m=630.0, reason="unobservable_terrain"
    )
    assert not throttled_again.should_attempt
    assert throttled_again.reason == "ordinary_throttle"
    allowed_again = _next_observation(
        controller, x_m=640.0, reason="unobservable_terrain"
    )
    assert not allowed_again.should_attempt
    assert allowed_again.reason == "ordinary_throttle"
    allowed_again = _next_observation(
        controller, x_m=650.0, reason="unobservable_terrain"
    )
    assert allowed_again.should_attempt


@pytest.mark.parametrize(
    ("observable_rank", "translation_rank", "yaw_observable"),
    [(2, 2, True), (3, 1, True), (3, 2, False)],
)
def test_only_complete_strict_rank_three_can_create_pending_hypothesis(
    observable_rank: int,
    translation_rank: int,
    yaw_observable: bool,
) -> None:
    controller = TerrainEmergencyRecoveryController()
    request = _drive_to_emergency(controller)
    assert request.should_attempt

    outcome = _consider(controller,
        accepted=True,
        observable_rank=observable_rank,
        translation_observable_rank=translation_rank,
        yaw_observable=yaw_observable,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.1]),
    )
    assert not outcome.fuse_current_match
    assert outcome.status == "strict_rank_three_required"
    assert not outcome.pending
    assert not controller.has_pending_hypothesis



def test_rank_two_cannot_confirm_an_existing_pending_hypothesis() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.1]),
    )
    assert _next_observation(controller, x_m=660.0).should_attempt
    rank_two_second = _consider(controller,
        accepted=True,
        observable_rank=2,
        translation_observable_rank=2,
        yaw_observable=False,
        seed_pose2=np.array([660.0, 0.0, 0.0]),
        matched_pose2=np.array([70.0, 20.0, 0.1]),
    )
    assert not rank_two_second.fuse_current_match
    assert rank_two_second.status == "strict_rank_three_required"
    assert rank_two_second.pending
    assert controller.has_pending_hypothesis


def test_two_strict_matches_confirm_only_current_then_insertion_resets() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    first = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert not first.fuse_current_match
    assert first.status == "pending_created"
    assert first.pending

    request = _next_observation(
        controller, x_m=620.0, reason="yaw_unobservable"
    )
    assert request.should_attempt
    assert request.reason == "pending_confirmation"
    assert request.bypassed_throttle
    too_near = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([620.0, 0.0, 0.0]),
        matched_pose2=np.array([30.0, 20.0, 0.0]),
    )
    assert not too_near.fuse_current_match
    assert too_near.status == "pending_wait_distance"
    assert too_near.pending

    request = _next_observation(
        controller, x_m=660.0, reason="insufficient_inliers"
    )
    assert request.should_attempt
    assert request.reason == "pending_confirmation"
    assert request.bypassed_throttle
    second = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([660.0, 0.0, 0.0]),
        # Propagated prediction is [70, 20, 0].  Both gates pass.
        matched_pose2=np.array([85.0, 32.0, np.deg2rad(4.0)]),
    )
    assert second.fuse_current_match
    assert second.status == "confirmed_current_match"
    assert second.position_disagreement_m == pytest.approx(
        np.hypot(15.0, 12.0)
    )
    assert second.yaw_disagreement_deg == pytest.approx(4.0)
    assert second.reference_observation_index == 7
    assert second.acquisition_windows_disjoint
    assert not second.pending
    assert controller.consecutive_without_prior == 9
    assert controller.raw_arc_since_prior_m == pytest.approx(660.0)

    # Confirmation alone is not an insertion and therefore did not reset.
    controller.record_prior_insertion()
    assert controller.consecutive_without_prior == 0
    assert controller.raw_arc_since_prior_m == 0.0
    assert controller.last_prior_timestamp == pytest.approx(560.0)
    assert not controller.has_pending_hypothesis

    after_reset = _next_observation(
        controller, x_m=760.0, reason="search_boundary_reached"
    )
    assert not after_reset.should_attempt
    assert after_reset.reason == "failure_threshold"
    assert controller.consecutive_without_prior == 1


def test_confirmation_requires_a_nonoverlapping_time_window_despite_pose_jump() -> None:
    controller = TerrainEmergencyRecoveryController(
        confirmation_minimum_time_separation_s=20.0
    )
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )

    # A local-filter jump can exceed the 60 m arc gate without providing a
    # new, independent rolling sonar strip.  Ten seconds is still inside the
    # configured 30-ping/1.5 Hz acquisition window and cannot self-confirm.
    request = controller.observe_preceding_recovery_outcome(
        timestamp=430.0,
        raw_pose2=np.array([800.0, 0.0, 0.0]),
        prior_added=False,
        preceding_recovery_rejection_reason="search_boundary_reached",
        preceding_recovery_invoked=True,
    )
    assert request.should_attempt
    outcome = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([800.0, 0.0, 0.0]),
        matched_pose2=np.array([210.0, 20.0, 0.0]),
    )
    assert not outcome.fuse_current_match
    assert outcome.status == "pending_wait_independent_strip"
    assert controller.has_pending_hypothesis


@pytest.mark.parametrize(
    ("patch_start_timestamp", "patch_start_observation_id"),
    [
        # New IDs but a retained acquisition time from the first patch.
        (410.0, 211),
        # New acquisition times but retained ping IDs from the first patch.
        (470.0, 200),
    ],
)
def test_confirmation_requires_actual_patch_time_and_id_nonoverlap(
    patch_start_timestamp: float,
    patch_start_observation_id: int,
) -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(
        controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert _next_observation(controller, x_m=660.0).should_attempt

    outcome = _consider(
        controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([660.0, 0.0, 0.0]),
        matched_pose2=np.array([70.0, 20.0, 0.0]),
        patch_start_timestamp=patch_start_timestamp,
        patch_end_timestamp=490.0,
        patch_start_observation_id=patch_start_observation_id,
        patch_end_observation_id=240,
    )

    assert not outcome.fuse_current_match
    assert outcome.status == "pending_wait_nonoverlapping_patch"
    assert not outcome.acquisition_windows_disjoint
    assert controller.has_pending_hypothesis


def test_confirmation_uses_seed_se2_increment_not_world_translation() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, np.deg2rad(90.0)]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert _next_observation(controller, x_m=660.0).should_attempt

    # Seed moved +60 world X while facing +90 degrees, hence its local
    # increment is -60 Y.  Applied at matched yaw zero it predicts [10,-40].
    outcome = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([660.0, 0.0, np.deg2rad(90.0)]),
        matched_pose2=np.array([10.0, -40.0, 0.0]),
    )
    assert outcome.fuse_current_match
    assert outcome.position_disagreement_m == pytest.approx(0.0)
    assert outcome.yaw_disagreement_deg == pytest.approx(0.0)


def test_inconsistent_confirmation_clears_pending_and_cools_three_observations() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert _next_observation(controller, x_m=660.0).should_attempt
    inconsistent = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([660.0, 0.0, 0.0]),
        matched_pose2=np.array([100.0, 100.0, np.deg2rad(8.0)]),
    )
    assert not inconsistent.fuse_current_match
    assert inconsistent.status == "confirmation_inconsistent"
    assert not controller.has_pending_hypothesis
    assert controller.inconsistent_hypotheses == 1

    for x_m in (670.0, 680.0, 690.0):
        request = _next_observation(controller, x_m=x_m)
        assert not request.should_attempt
        assert request.reason == "cooldown"
    assert _next_observation(controller, x_m=700.0).should_attempt


def test_pending_expires_after_five_future_observations_then_cools() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )

    for offset in range(1, 6):
        _next_observation(controller, x_m=600.0 + 10.0 * offset)
        assert controller.has_pending_hypothesis
    expired = _next_observation(controller, x_m=660.0)
    assert not expired.should_attempt
    assert expired.reason == "pending_expired"
    assert not controller.has_pending_hypothesis
    assert controller.expired_hypotheses == 1

    for x_m in (670.0, 680.0, 690.0):
        request = _next_observation(controller, x_m=x_m)
        assert not request.should_attempt
        assert request.reason == "cooldown"
    assert _next_observation(controller, x_m=700.0).should_attempt


def test_confirmed_factor_rejection_starts_cooldown_without_reset() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert _next_observation(controller, x_m=660.0).should_attempt
    confirmed = _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([660.0, 0.0, 0.0]),
        matched_pose2=np.array([70.0, 20.0, 0.0]),
    )
    assert confirmed.fuse_current_match
    failures = controller.consecutive_without_prior
    arc = controller.raw_arc_since_prior_m

    with pytest.raises(RuntimeError, match="insertion or rejection"):
        _next_observation(controller, x_m=670.0)
    controller.record_factor_rejection()
    assert controller.consecutive_without_prior == failures
    assert controller.raw_arc_since_prior_m == arc
    assert controller.cooldown_remaining_observations == 3
    with pytest.raises(RuntimeError, match="no confirmed"):
        controller.record_factor_rejection()

    for x_m in (670.0, 680.0, 690.0):
        request = _next_observation(controller, x_m=x_m)
        assert not request.should_attempt
        assert request.reason == "cooldown"
    assert _next_observation(controller, x_m=700.0).should_attempt


def test_preceding_factor_rejection_clears_pending_cools_and_does_not_reset() -> None:
    controller = TerrainEmergencyRecoveryController()
    assert _drive_to_emergency(controller).should_attempt
    _consider(controller,
        accepted=True,
        observable_rank=3,
        translation_observable_rank=2,
        yaw_observable=True,
        seed_pose2=np.array([600.0, 0.0, 0.0]),
        matched_pose2=np.array([10.0, 20.0, 0.0]),
    )
    assert controller.has_pending_hypothesis
    failures = controller.consecutive_without_prior
    arc = controller.raw_arc_since_prior_m

    not_inserted = controller.observe_preceding_recovery_outcome(
        timestamp=70.0 * controller.observation_index,
        raw_pose2=np.array([610.0, 0.0, 0.0]),
        prior_added=False,
        preceding_recovery_invoked=False,
        preceding_recovery_rejection_reason=None,
        preceding_factor_rejected=True,
    )
    assert not not_inserted.should_attempt
    assert not_inserted.reason == "preceding_factor_rejected"
    assert not controller.has_pending_hypothesis
    assert controller.cooldown_remaining_observations == 3
    assert controller.preceding_factor_rejections == 1
    assert controller.consecutive_without_prior == failures + 1
    assert controller.raw_arc_since_prior_m > arc

    with pytest.raises(ValueError, match="cannot both be true"):
        controller.observe_preceding_recovery_outcome(
            timestamp=70.0 * controller.observation_index,
            raw_pose2=np.array([620.0, 0.0, 0.0]),
            prior_added=True,
            preceding_recovery_invoked=False,
            preceding_recovery_rejection_reason=None,
            preceding_factor_rejected=True,
        )

    inserted = controller.observe_preceding_recovery_outcome(
        timestamp=70.0 * controller.observation_index,
        raw_pose2=np.array([620.0, 0.0, 0.0]),
        prior_added=True,
        preceding_recovery_invoked=False,
        preceding_recovery_rejection_reason=None,
    )
    assert not inserted.should_attempt
    assert inserted.reason == "prior_inserted"
    assert controller.consecutive_without_prior == 0
    assert controller.raw_arc_since_prior_m == 0.0
