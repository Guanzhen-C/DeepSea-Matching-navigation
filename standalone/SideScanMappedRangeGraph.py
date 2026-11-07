"""Opt-in SSS range-change factors on the UNMODIFIED frozen GTSAM graph.

Legacy map and loop precisions can be preserved or use the explicitly named
historical half-budget experiment. New ranges retain their own conservative
precision. Neither policy proves all map/loop correlations are modelled.
Odometry is unchanged. All new
factors join TWO variable poses: a historical estimate is never frozen and
reused as a supposedly independent absolute measurement.
"""
from __future__ import annotations

import numpy as np

from oceansim.slam.pose_graph import GTSAMPlanarPoseGraph
from SideScanMappedRangeChange import MappedRangeChange, profiled_range_residual_jacobians


class MappedRangePoseGraph(GTSAMPlanarPoseGraph):
    def __init__(self, config=None, *, legacy_precision_budget=.5):
        if not np.isfinite(legacy_precision_budget) or not 0 < legacy_precision_budget <= 1:
            raise ValueError('Legacy precision budget must be positive and cannot strengthen the baseline')
        self.legacy_precision_budget = float(legacy_precision_budget)
        self.mapped_range_changes = {}
        super().__init__(config)

    def add_edge(self, source, target, measurement, information, *, kind, score=0.):
        # Do not strengthen any original factor. This branch is opt-in only.
        budget = self.legacy_precision_budget if kind == 'loop' else 1.
        super().add_edge(source, target, measurement, np.asarray(information)*budget,
                         kind=kind, score=score)

    def add_prior(self, target, measurement, information, **kwargs):
        super().add_prior(target, measurement, np.asarray(information)*self.legacy_precision_budget, **kwargs)

    def add_mapped_range_change(self, source, target, observations, *, score=0.):
        if not isinstance(observations, MappedRangeChange):
            raise TypeError('Only validated immutable received range observations allowed')
        # Zero legacy SE2 precision is intentional: this is NOT a relative
        # pose pseudo-measurement. The actual nonlinear CustomFactor below
        # uses the raw ranges/XY and both variable endpoint poses.
        super().add_edge(source, target, np.zeros(3), np.zeros((3, 3)),
                         kind='mapped_range_change', score=score)
        self.mapped_range_changes[id(self.edges[-1])] = observations

    def _edge_factor(self, edge):
        if edge.kind != 'mapped_range_change':
            return super()._edge_factor(edge)
        data = self.mapped_range_changes[id(edge)]
        source_key, target_key = self._key(edge.source), self._key(edge.target)

        def error_function(_factor, values, jacobians):
            a = self._pose_array(values.atPose2(source_key))
            b = self._pose_array(values.atPose2(target_key))
            error, ja, jb = profiled_range_residual_jacobians(a, b, data)
            if jacobians is not None:
                jacobians[0] = np.asfortranarray(ja@self._local_delta_to_world(a))
                jacobians[1] = np.asfortranarray(jb@self._local_delta_to_world(b))
            return error

        return self._gtsam.CustomFactor(self._gtsam.noiseModel.Isotropic.Sigma(data.features, 1.),
                                        [source_key, target_key], error_function)

    def _cost(self, poses):
        cost = super()._cost(poses)
        for edge in self.edges:
            if edge.kind == 'mapped_range_change':
                error, _, _ = profiled_range_residual_jacobians(
                    poses[edge.source], poses[edge.target], self.mapped_range_changes[id(edge)])
                cost += float(error@error)
        return cost

    def factor_stream_payload(self):
        payload = super().factor_stream_payload()
        payload['mapped_range_changes'] = [dict(edge_index=i, source=e.source, target=e.target,
            observations=self.mapped_range_changes[id(e)].payload())
            for i, e in enumerate(self.edges) if e.kind == 'mapped_range_change']
        payload['mapped_range_change_policy'] = dict(
            schema='oceansim.sss.mapped_range_change_graph.v1',
            nonlinear_two_variable_pose_factors=True,
            legacy_zero_SE2_information_is_only_a_store_placeholder=True,
            original_map_and_loop_precision_budget=self.legacy_precision_budget,
            unknown_height_profiled_not_measured=True,
            frozen_core_modified=False)
        return payload
