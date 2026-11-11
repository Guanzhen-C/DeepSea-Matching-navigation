"""Bounded raw-sonar proposals after loss of independently usable fixes.

This is proposal generation, not a second measurement update. A broader search
may discover another mode, but never moves an existing cohort or inherits its
confirmation history. The existing NMS, joint raw-likelihood update, birth mass,
null hypothesis and multi-window confirmation remain authoritative. In
particular a nearby proposal may intentionally be suppressed by existing NMS.
No truth, graph posterior covariance, or expected position error is an input.
"""
from __future__ import annotations

import numpy as np

from .global_tracking import GlobalPoseEvidence


class RawModeRecovery:
    def __init__(self, *, enabled=False, after_failed_windows=3):
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError('enabled must be boolean')
        if (isinstance(after_failed_windows, bool)
                or int(after_failed_windows) != after_failed_windows
                or after_failed_windows < 1):
            raise ValueError('after_failed_windows must be a positive integer')
        self.enabled = bool(enabled)
        self.after_failed_windows = int(after_failed_windows)
        self.failed_windows = 0
        self.searches = 0
        self.proposals = 0

    def observe(self, decision):
        cohort = decision.confirmed_cohort
        usable = bool(decision.confirmed_with_new_independent_evidence
                      and cohort is not None
                      and cohort.last_measurement_is_graph_ready)
        self.failed_windows = 0 if usable else self.failed_windows + 1

    def generate(self, raw, prediction_seeds, observation, *, source_index):
        if not self.enabled or self.failed_windows < self.after_failed_windows:
            return ()
        result = []
        config = raw.config
        for _cohort_id, predicted in prediction_seeds:
            # Refit curvature using THIS window at the broader-search mode;
            # never reuse a compact or recursive Gaussian at a shifted mean.
            approximation = raw.approximate_local_measurement(raw.refine_pose(predicted))
            diagnostics = approximation.diagnostics
            covariance = approximation.covariance
            result.append(GlobalPoseEvidence(
                map_pose2=approximation.pose2,
                bathymetry_score=float(np.tanh((config.bathymetry_reference_normalized_loss
                    - diagnostics.best_bathymetry_normalized_loss) / config.bathymetry_fit_temperature))
                    if np.isfinite(diagnostics.best_bathymetry_normalized_loss) else 0.,
                backscatter_score=float(np.clip(diagnostics.best_backscatter_correlation, -1., 1.))
                    if np.isfinite(diagnostics.best_backscatter_correlation) else 0.,
                bathymetry_observability=diagnostics.bathymetry_observability,
                backscatter_observability=diagnostics.backscatter_observability,
                position_std_m=float(np.sqrt(np.linalg.eigvalsh(covariance[:2, :2])[-1])),
                yaw_std_deg=float(np.rad2deg(np.sqrt(covariance[2, 2]))),
                covariance_matrix=covariance, information_matrix=approximation.information,
                target_cohort_id=None, source_index=source_index + len(result),
                **{key: getattr(observation, key) for key in (
                    'patch_start_timestamp_s', 'patch_end_timestamp_s',
                    'patch_start_observation_id', 'patch_end_observation_id')},
            ))
            self.searches += 1
        self.proposals += len(result)
        return tuple(result)

    def diagnostics(self):
        return dict(enabled=self.enabled, after_failed_windows=self.after_failed_windows,
                    failed_windows=self.failed_windows, searches=self.searches,
                    proposals=self.proposals, existing_nms_and_confirmation_unchanged=True)
