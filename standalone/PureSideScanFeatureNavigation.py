"""Side-scan FEATURE observation wiring for the original filter/PF/GTSAM.

This is a new sensor-ablation adapter, not the unchanged full joint engine.
There are no MBES/DOA/height inputs, no optical flow, and no DVL-altitude input.
Known source terrain remains private to the simulator. The prior supplied
here is XY-indexed intensity only; its geocoding assumptions remain relevant.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import json
import numpy as np

from PureSideScanFeatures import image_from_packet, robust_feature_row_motion
from SideScanFeatureMap import pose_matrix, matrix_pose, SideScanFeatureWindow, match_feature_prior


def state_pose2(state):
    return np.r_[state.position[:2], np.arctan2(state.rotation[1, 0], state.rotation[0, 0])]


def add_side_scan_feature_prior(graph, target, pose2, information, *, score):
    """Reuse the original validated unary-Pose2 math with truthful labeling.

    The frozen graph's public add_prior accepts only the historic name
    terrain_prior, although its math is a generic planar absolute constraint.
    Validate through that API, then relabel the NEW, not-yet-synchronized
    immutable factor. No frozen source or existing factor is changed.
    """
    graph.add_prior(target, pose2, information, kind='terrain_prior', score=score, robust=True)
    graph.priors[-1] = replace(graph.priors[-1], kind='side_scan_feature_prior')


class FeatureScanPlaneLikelihood:
    def __init__(self, match):
        self.landmarks = match['landmark_xy']
        self.normals = match['scan_normals']
        self.offsets = match['scan_offsets']

    def __call__(self, particles):
        poses = np.asarray(particles).reshape(-1, 3)
        result = np.empty(len(poses))
        for start in range(0, len(poses), 128):
            p = poses[start:start+128]
            delta = self.landmarks[None]-p[:, None, :2]
            c, s = np.cos(p[:, 2, None]), np.sin(p[:, 2, None])
            bx = c*delta[:, :, 0]+s*delta[:, :, 1]
            by = -s*delta[:, :, 0]+c*delta[:, :, 1]
            residual = bx*self.normals[None, :, 0]+by*self.normals[None, :, 1]-self.offsets
            # Student-t scan-plane residual vs a broad uniform null. At most
            #24 independent-equivalent features, not every descriptor pixel.
            ratio = np.log(40/.7)-2.5*np.log1p((residual/.7)**2/4)
            result[start:start+len(p)] = np.clip(min(24, len(self.landmarks))*ratio.mean(axis=1), -50, 50)
        return result


class PureSideScanFeatureNavigator:
    def __init__(self, intensity_map, initial, startup_specific_force, image_matcher=None, *, loop_closure=False,
                 scan_plane_registration='rank2_baseline', canonical_submaps=False,
                 submap_representation='legacy_strip'):
        from oceansim.slam import AcousticESIKF, ESIKFConfig, gauge_anchored_initial_covariance
        from oceansim.slam.pose_graph import GTSAMPlanarPoseGraph, PoseGraphConfig
        from oceansim.slam.global_tracking import GlobalCohortTracker, GlobalCohortTrackerConfig
        from oceansim.slam.map_odom import MapOdomSoftCorrector
        covariance = gauge_anchored_initial_covariance(gyro_bias_std_rad_s=np.deg2rad(1)/3600/2)
        self.local = AcousticESIKF(ESIKFConfig(gyro_bias_max_norm=np.deg2rad(1)/3600))
        self.dr = AcousticESIKF(ESIKFConfig(gyro_bias_max_norm=np.deg2rad(1)/3600))
        for f in (self.local, self.dr):
            f.initialize(0., rotation=initial.rotation, position=initial.position,
                         velocity=initial.velocity, covariance=covariance)
        self.graph = GTSAMPlanarPoseGraph(PoseGraphConfig(fixed_lag_seconds=180.,
            terrain_prior_enabled=False, intensity_factor_enabled=False,
            odometry_reference_interval_s=30., odometry_xy_std_m=1.))
        self.graph.add_pose(state_pose2(self.local.state), timestamp=0.)
        self.raw_keyframe_poses = [pose_matrix(state_pose2(self.local.state))]
        self.last_keyframe_s = 0.
        self.tracker = GlobalCohortTracker(GlobalCohortTrackerConfig())
        self.soft = MapOdomSoftCorrector()
        self.map = intensity_map
        self.image_matcher = image_matcher
        if scan_plane_registration not in ('rank2_baseline', 'observable_subspace'):
            raise ValueError('Explicit scan-plane registration mode required.')
        self.scan_plane_registration = scan_plane_registration
        if submap_representation not in ('legacy_strip', 'adaptive_projected'):
            raise ValueError('Explicit received-submap representation required.')
        if submap_representation != 'legacy_strip' and (loop_closure or image_matcher is None):
            raise ValueError('First isolate projected-map evidence with LoFTR and loops OFF; historical-image compatibility is not yet validated.')
        self.submap_representation = submap_representation
        self.projected_map_attempts = self.projected_map_accepts = 0
        if loop_closure:
            if image_matcher is None:
                raise ValueError('Historical side-scan loops require a local-feature matcher.')
            from SideScanLoopClosure import SideScanHistoricalLoops
            self.historical_loops = SideScanHistoricalLoops(image_matcher)
        else:
            self.historical_loops = None
        self.map_priors_replaced_by_loop = 0
        self.shared_feature_windows = 0
        self.latest_gyro = np.zeros(3)
        self.latest_acceleration = np.mean(startup_specific_force, axis=0)
        self.window = SideScanFeatureWindow(150, canonical_orientation=canonical_submaps)
        self.previous_frame = None
        self.previous_local_pose = None
        self.anchor_pose = None
        self.anchor_variance = 0.
        self.chain = np.eye(3); self.chain_count = 0; self.chain_variance = 0.
        self.observation_id = 0
        self.front_accepted = self.front_rejected = self.front_updates = 0
        self.map_attempts = self.map_accepts = self.graph_feature_priors = 0
        self.last_backend = dict(status='not_enough_independent_side_scan_pings')
        self.last_feature = None
        self.correspondence_examples = []
        self.map_correspondence_examples = []
        self.map_failure_examples = []
        self.backend_query_examples = []

    def imu(self, stamp, gyro, acceleration):
        self.latest_gyro, self.latest_acceleration = np.asarray(gyro).copy(), np.asarray(acceleration).copy()
        for f in (self.local, self.dr): f.propagate(stamp, gyro, acceleration)

    def dvl(self, velocity):
        for f in (self.local, self.dr): f.update_dvl(velocity)

    def depth(self, depth_m):
        for f in (self.local, self.dr): f.update_depth(depth_m)

    def raw_graph_pose(self, raw_pose):
        transform = pose_matrix(self.graph.poses[-1])@np.linalg.inv(self.raw_keyframe_poses[-1])
        return matrix_pose(transform@pose_matrix(raw_pose))

    def ensure_keyframe(self, timestamp, raw_pose, force=False):
        dt = timestamp-self.last_keyframe_s
        if dt < 1e-8 or (not force and dt < 30.):
            return
        t = pose_matrix(raw_pose)
        index = self.graph.add_pose(self.raw_graph_pose(raw_pose), timestamp=timestamp)
        relative = matrix_pose(np.linalg.inv(self.raw_keyframe_poses[-1])@t)
        time_scale = max(dt/30., 1e-6)
        info = np.diag([1/time_scale, 1/time_scale, 1/(np.deg2rad(.8)**2*time_scale)])
        self.graph.add_edge(index-1, index, relative, info, kind='odometry')
        self.raw_keyframe_poses.append(t); self.last_keyframe_s = timestamp
        self.graph.optimize()

    def frontend(self, packet):
        predicted = pose_matrix(state_pose2(self.local.state))
        try:
            frame = image_from_packet(packet)
        except ValueError as exc:
            self.previous_frame = None; self.chain_count = 0
            self.front_rejected += 1
            self.last_feature = dict(accepted=False, reason=str(exc))
            return
        if self.previous_frame is None:
            self.anchor_pose = predicted.copy(); self.chain = np.eye(3); self.chain_count = 1
            self.anchor_variance = float(np.trace(self.local.state.covariance[3:5, 3:5]))
            self.previous_frame = frame; self.previous_local_pose = predicted.copy()
            return
        relative = np.linalg.inv(self.previous_local_pose)@predicted
        result = robust_feature_row_motion(frame, self.previous_frame, relative[0, 2])
        ready = False
        if result is None:
            self.front_rejected += 1
            self.last_feature = dict(accepted=False, reason='insufficient_mutual_descriptor_scanline_consensus')
            ready = True
        else:
            self.front_accepted += 1
            relative[0, 2] += result['correction_m']
            self.chain = self.chain@relative; self.chain_count += 1
            self.chain_variance += result['variance_m2']
            self.last_feature = {k:v for k,v in result.items() if k not in ('pairs', 'inlier_mask')}
            self.last_feature['accepted'] = True
            if len(self.correspondence_examples) < 3 and self.front_accepted in (1, 40, 100):
                self.correspondence_examples.append(dict(previous_image=self.previous_frame['image'].copy(),
                    current_image=frame['image'].copy(), previous_pixels=self.previous_frame['pixels'].copy(),
                    current_pixels=frame['pixels'].copy(), pairs=result['pairs'].copy(), inliers=result['inlier_mask'].copy()))
            if self.chain_count >= 4:
                measured = self.anchor_pose@self.chain
                axis = self.anchor_pose[:2, 0]
                variance = self.chain_variance+1.5*self.anchor_variance
                self.local.update_horizontal_position_along_axis(float(axis@measured[:2, 2]), axis, variance)
                self.front_updates += 1; ready = True
        finalized = pose_matrix(state_pose2(self.local.state))
        if ready:
            self.anchor_pose = finalized.copy(); self.chain = np.eye(3); self.chain_count = 1; self.chain_variance = 0.
            self.anchor_variance = float(np.trace(self.local.state.covariance[3:5, 3:5]))
        self.previous_frame, self.previous_local_pose = frame, finalized

    def side_scan(self, packet):
        from oceansim.slam.global_tracking import GlobalPoseEvidence, ParticleLikelihoodObservation
        # Original frozen engine propagates held IMU to every sonar stamp,
        # including stamps falling between the 50 Hz IMU events.
        if self.local.state.timestamp < packet.timestamp-1e-12:
            for f in (self.local, self.dr):
                f.propagate(packet.timestamp, self.latest_gyro, self.latest_acceleration)
        self.frontend(packet)
        raw = state_pose2(self.local.state)
        self.observation_id += 1
        self.window.add(packet, raw, self.observation_id, vertical_position_m=self.local.state.position[2])
        self.ensure_keyframe(packet.timestamp, raw)
        target = self.raw_graph_pose(raw)
        if self.observation_id % 150 == 0:
            self.map_attempts += 1
            # Observation-only diagnostics: retain periodic/turn examples
            # using ESTIMATED history, never programmed route labels/truth.
            historical_yaw = np.unwrap([entry[1][2] for entry in self.window.scans])
            turning_window = float(np.ptp(historical_yaw)) > np.deg2rad(15.)
            sample_debug = self.observation_id % 1500 == 0 or turning_window
            try:
                # Use projection only when the legacy strip would discard or
                # reverse acquired scan directions. This is representation
                # support, never a true-pose/error-based acceptance threshold.
                needs_projection = (self.submap_representation == 'adaptive_projected'
                    and np.any(np.cos(historical_yaw-historical_yaw[-1]) < .5))
                if needs_projection:
                    from SideScanProjectedSubmap import build_projected_submap
                    from SideScanProjectedMatching import match_projected_prior
                    self.projected_map_attempts += 1
                    query = build_projected_submap(self.window, raw)
                    matched = match_projected_prior(self.map, query, target, self.image_matcher)
                    self.projected_map_accepts += int(matched['accepted'])
                else:
                    query = self.window.build(raw)
                    matched = match_feature_prior(self.map, query, target, image_matcher=self.image_matcher,
                                                   registration_mode=self.scan_plane_registration)
                if len(self.backend_query_examples) < 3 or (sample_debug and len(self.backend_query_examples) < 32):
                    self.backend_query_examples.append(dict(query, seed_pose2=target.copy()))
            except ValueError as exc:
                matched = dict(accepted=False, reason=str(exc), matches=0)
                query = None
            evidence, observation = [], None
            if not matched['accepted'] and 'map_image' in matched and (len(self.map_failure_examples) < 3
                    or (sample_debug and len(self.map_failure_examples) < 32)):
                example = {k: matched[k] for k in
                    ('query_image', 'map_image', 'query_pixels', 'map_pixels',
                     'query_feature_count', 'map_feature_count', 'reason')}
                example.update(timestamp_s=packet.timestamp, observation_id=self.observation_id)
                self.map_failure_examples.append(example)
            if matched['accepted']:
                self.map_accepts += 1
                if len(self.map_correspondence_examples) < 3 or (sample_debug and len(self.map_correspondence_examples) < 64):
                    example = {k: matched[k].copy() for k in ('query_image', 'map_image', 'query_pixels', 'map_pixels')}
                    example.update(timestamp_s=packet.timestamp, observation_id=self.observation_id)
                    self.map_correspondence_examples.append(example)
                evidence = [GlobalPoseEvidence(map_pose2=matched['matched_pose2'], bathymetry_score=0.,
                    bathymetry_observability=0., backscatter_score=float(matched['score']),
                    backscatter_observability=1., covariance_matrix=matched['proposal_covariance'],
                    information_matrix=matched['information'], position_std_m=20., yaw_std_deg=5.,
                    patch_start_timestamp_s=query['patch_start_s'], patch_end_timestamp_s=packet.timestamp,
                    patch_start_observation_id=query['patch_start_id'], patch_end_observation_id=query['patch_end_id'])]
                observation = ParticleLikelihoodObservation(FeatureScanPlaneLikelihood(matched),
                    query['patch_start_s'], packet.timestamp, query['patch_start_id'], query['patch_end_id'])
            decision = self.tracker.update(timestamp_s=packet.timestamp, raw_pose2=raw,
                evidences=evidence, particle_observation=observation)
            self.last_backend = dict(accepted=matched['accepted'], reason=matched.get('reason'),
                matches=matched['matches'], inliers=matched.get('inliers', 0),
                query_features=matched.get('query_feature_count'), map_features=matched.get('map_feature_count'),
                observable_rank=matched.get('observable_rank'),
                submap_assembly=None if query is None else json.loads(query['assembly_json']),
                status=decision.status, null_probability=decision.null_probability,
                cohorts=len(decision.cohorts), feature_likelihood_only=True,
                measured_depth_likelihood_terms=0, measured_depth_graph_factors=0,
                DVL_altitude_used=False, optical_flow_used=False)
            cohort = decision.confirmed_cohort if decision.confirmed_with_new_independent_evidence else None
            graph_ready = cohort is not None and cohort.last_measurement_is_graph_ready
            if graph_ready:
                self.ensure_keyframe(packet.timestamp, raw, force=True)
            loop = None
            if query is not None and self.historical_loops is not None:
                loop = self.historical_loops.observe(query, raw, self.graph, self.raw_keyframe_poses, packet.timestamp,
                    shared_current_window=bool(graph_ready))
            if loop is not None:
                # Preserve new absolute anchoring instead of replacing it by
                # a relative revisit to a potentially drifted historical pose.
                # Shared current pixels get HALF information in each factor.
                if graph_ready:
                    add_side_scan_feature_prior(self.graph, len(self.graph.poses)-1, cohort.last_measurement_pose2,
                        cohort.last_measurement_information*loop['current_window_information_weight'],
                        score=matched.get('score', 0))
                    self.graph_feature_priors += 1
                    self.shared_feature_windows += 1
                self.graph.optimize()
            elif graph_ready:
                add_side_scan_feature_prior(self.graph, len(self.graph.poses)-1, cohort.last_measurement_pose2,
                    cohort.last_measurement_information, score=matched.get('score', 0))
                self.graph.optimize(); self.graph_feature_priors += 1
            target = self.raw_graph_pose(raw)
        published = self.soft.update(packet.timestamp, raw, target)
        position = self.local.state.position.copy(); position[:2] = published[:2]
        return dict(DR=self.dr.state.position.copy(), Local=self.local.state.position.copy(),
                    Global=position, raw_pose2=raw, global_pose2=published)

    def diagnostics(self):
        return dict(mode='pure_side_scan_descriptor_features', feature_detector='GFTT+subpixel',
            descriptor='SIFT', matching='mutual_ratio+scanline_RANSAC', optical_flow_calls=0,
            DVL_altitude_observations=0, MBES_points_consumed=0, measured_height_likelihood_terms=0,
            measured_height_graph_factors=0, raw_intensity_correlation_calls=0,
            frontend=dict(accepted=self.front_accepted, rejected=self.front_rejected, updates=self.front_updates),
            feature_map=dict(attempts=self.map_attempts, accepts=self.map_accepts,
                             graph_feature_priors=self.graph_feature_priors, last=self.last_backend),
            optimizer_backend=self.graph.backend_name,
            backend_feature_matcher=None if self.image_matcher is None else self.image_matcher.provenance,
            secondary_submap_denoising_used=False,
            scan_plane_registration=self.scan_plane_registration,
            canonical_submap_orientation=self.window.canonical_orientation,
            submap_representation=self.submap_representation,
            projected_map=dict(attempts=self.projected_map_attempts, accepts=self.projected_map_accepts,
                projected_XY_used_as_measurement=False),
            loop_closure=dict(enabled=False, actual_loop_edges=0) if self.historical_loops is None else
                dict(enabled=True, **self.historical_loops.diagnostics(),
                     map_priors_replaced_by_loop=self.map_priors_replaced_by_loop,
                     shared_current_windows=self.shared_feature_windows,
                     shared_observation_policy='half_information_each_for_loop_and_map_not_independent_full_factors'),
            global_cohort_tracker_updates=self.tracker.updates,
            map_to_odom=self.soft.diagnostics(),
            observation_limit='Scan-plane geometry only; actual null directions retained. Slant pixels are not horizontal observations.',
            short_test_is_not_validated_whole_map_unknown_pose_relocalization=True)
