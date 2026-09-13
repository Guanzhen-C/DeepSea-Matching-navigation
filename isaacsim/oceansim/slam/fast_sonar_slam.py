"""Runnable FAST-LIVO2-style underwater acoustic SLAM front end."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .esikf import (
    AcousticESIKF,
    ESIKFConfig,
    InertialState,
    IntensityUpdateResult,
    SonarUpdateResult,
)
from .intensity_map import AcousticIntensityMap
from .map_odom import MapOdomSoftCorrectionConfig, MapOdomSoftCorrector
from .math3d import transform_points
from .pose_graph import (
    AcousticPoseGraph,
    PoseGraphConfig,
    PoseGraphDiagnostics,
    sidescan_descriptor,
)
from .surfel_map import SurfelVoxelMap

if TYPE_CHECKING:
    from ..utils.acoustic_sonar import BathymetricSideScanPing


@dataclass
class FastSonarSLAMConfig:
    filter: ESIKFConfig = field(default_factory=ESIKFConfig)
    pose_graph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    voxel_size: float = 1.8
    surfel_min_points: int = 5
    surfel_max_planarity: float = 0.14
    surfel_max_plane_distance: float = 1.25
    bootstrap_pings: int = 14
    mbes_group_pings: int = 5
    map_insert_stride: int = 1
    # Keep the full incremental map, but pack only the current sensor-reach
    # neighbourhood for accelerated association. This is the non-destructive
    # counterpart of FAST-LIVO2's sliding local map.
    association_map_half_extent_m: float = 120.0
    # FAST-LIVO2 updates its incremental geometry map from the converged
    # LiDAR state.  A rejected acoustic registration has no such converged
    # state, so admitting it to the reference map creates a self-consistent
    # but drifting surface.  Keep those points in a separate quarantine map
    # instead of using them for subsequent measurement association.
    # Strict quarantine is useful for diagnosing corrupt registrations, but
    # it cannot be the default on a forward-only strip survey: FAST-LIVO2
    # must keep extending its incremental map into newly observed space.
    # The production path therefore admits the propagated scan and lets the
    # probabilistic map/update weighting represent its uncertainty.
    quality_gate_map_updates: bool = False
    map_min_range: float = 0.5
    map_max_range: float = 80.0
    sidescan_max_age_s: float = 2.0
    sidescan_min_range: float = 1.0
    sidescan_max_range: float = 80.0
    sidescan_descriptor_history_lines: int = 8
    intensity_voxel_size: float = 3.0
    intensity_bootstrap_pings: int = 18
    intensity_min_separation_pings: int = 75
    # A legacy side-scan ping is a one-dimensional range line and may need
    # temporal recombination.  The integrated V2 product already supplies eight
    # simultaneous along-track rows, so its launcher uses one acquisition per
    # update.  Values greater than one remain available for single-row sensors.
    intensity_group_pings: int = 8
    intensity_min_group_span_m: float = 1.0
    # Local direct alignment should not fragment its intensity submap at every
    # global pose-graph keyframe.  A positive interval keeps one existing
    # keyframe as the map owner for at least this long; zero preserves the
    # legacy one-anchor-per-keyframe behaviour.
    intensity_submap_anchor_interval_s: float = 0.0
    intensity_updates_enabled: bool = True
    integrated_timestamp_tolerance_s: float = 0.002
    # Retaining every IMU propagation as a separate NumPy object is useful
    # for short interactive traces but unnecessarily costly on multi-hour
    # missions.  A positive interval decimates only the stored visualization
    # trajectory; estimator propagation and sensor updates remain unchanged.
    trajectory_sample_interval_s: float = 0.0
    # Suppress the single exactly-nadir beam in the image update.  It remains
    # available to the bathymetric point-to-plane update, but its port/
    # starboard image coordinate is poorly conditioned.
    coregistered_min_cross_track_m: float = 0.25
    # Pose-graph corrections belong in map->odom, never in the local ESIKF.
    # Enabling this output layer prevents a newly confirmed global factor from
    # teleporting the published vehicle pose while preserving the raw graph
    # target for matching and final optimization.
    map_to_odom_soft_correction_enabled: bool = False
    map_to_odom_translation_time_constant_s: float = 5.0
    map_to_odom_yaw_time_constant_s: float = 5.0
    map_to_odom_maximum_translation_rate_mps: float = 5.0
    map_to_odom_maximum_yaw_rate_deg_s: float = 5.0


@dataclass(frozen=True)
class SLAMDiagnostics:
    ping_count: int
    accepted_sonar_updates: int
    rejected_sonar_updates: int
    map_points: int
    total_geometry_points_committed: int
    map_voxels: int
    association_map_voxels: int
    mature_surfels: int
    sidescan_lines: int
    integrated_sonar_pings: int
    coregistered_intensity_samples: int
    legacy_range_attached_intensity_samples: int
    intensity_map_points: int
    intensity_map_cells: int
    accepted_intensity_updates: int
    accepted_local_intensity_updates: int
    accepted_local_intensity_updates_with_graph_factor: int
    accepted_local_intensity_updates_without_graph_factor: int
    accepted_graph_seeded_intensity_updates: int
    rejected_intensity_updates: int
    committed_geometry_groups: int
    quarantined_geometry_groups: int
    committed_intensity_batches: int
    quarantined_intensity_batches: int
    sonar_rejection_reasons: dict[str, int]
    sonar_observable_rank_counts: dict[str, int]
    intensity_rejection_reasons: dict[str, int]
    intensity_planar_rank_counts: dict[str, int]
    mean_accepted_intensity_correlation: float
    minimum_accepted_intensity_correlation: float
    mean_accepted_intensity_overlap: float
    cumulative_intensity_translation_correction_m: float
    cumulative_intensity_rotation_correction_deg: float
    pending_intensity_batches: int
    pending_intensity_points: int
    formed_intensity_strips: int
    skipped_intensity_strips: int
    last_intensity_strip_span_m: float
    last_sonar_update: SonarUpdateResult | None
    last_intensity_update: IntensityUpdateResult | None
    pose_graph: PoseGraphDiagnostics


class FastSonarSLAM:
    """Fuse navigation sensors with integrated bathymetry/backscatter.

    This class intentionally does not depend on Isaac Sim or ROS.  Sensor
    adapters feed timestamped arrays into it, which makes the exact estimator
    usable in live Kit UI, bag replay, and deterministic large-scale tests.
    """

    def __init__(
        self,
        config: FastSonarSLAMConfig | None = None,
        *,
        sensor_rotation_body: np.ndarray | None = None,
        sensor_translation_body: np.ndarray | None = None,
        surfel_associator=None,
    ) -> None:
        self.config = config or FastSonarSLAMConfig()
        if int(self.config.intensity_min_separation_pings) < 1:
            raise ValueError(
                "intensity_min_separation_pings must be at least one"
            )
        if int(self.config.mbes_group_pings) < 1:
            raise ValueError("mbes_group_pings must be at least one")
        if int(self.config.intensity_group_pings) < 1:
            raise ValueError("intensity_group_pings must be at least one")
        if (
            not np.isfinite(self.config.trajectory_sample_interval_s)
            or self.config.trajectory_sample_interval_s < 0.0
        ):
            raise ValueError(
                "trajectory_sample_interval_s must be finite and non-negative"
            )
        self.filter = AcousticESIKF(
            self.config.filter,
            sensor_rotation_body=sensor_rotation_body,
            sensor_translation_body=sensor_translation_body,
            surfel_associator=surfel_associator,
        )
        self.map = SurfelVoxelMap(
            voxel_size=self.config.voxel_size,
            min_points=self.config.surfel_min_points,
            max_planarity=self.config.surfel_max_planarity,
            max_plane_distance=self.config.surfel_max_plane_distance,
        )
        self.intensity_map = AcousticIntensityMap(
            voxel_size=self.config.intensity_voxel_size
        )
        self.quarantine_map = SurfelVoxelMap(
            voxel_size=self.config.voxel_size,
            min_points=self.config.surfel_min_points,
            max_planarity=self.config.surfel_max_planarity,
            max_plane_distance=self.config.surfel_max_plane_distance,
        )
        self.quarantine_intensity_map = AcousticIntensityMap(
            voxel_size=self.config.intensity_voxel_size
        )
        self.pose_graph = AcousticPoseGraph(self.config.pose_graph)
        self.map_to_odom_corrector = MapOdomSoftCorrector(
            MapOdomSoftCorrectionConfig(
                translation_time_constant_s=(
                    self.config.map_to_odom_translation_time_constant_s
                ),
                yaw_time_constant_s=(
                    self.config.map_to_odom_yaw_time_constant_s
                ),
                maximum_translation_rate_mps=(
                    self.config.map_to_odom_maximum_translation_rate_mps
                ),
                maximum_yaw_rate_deg_s=(
                    self.config.map_to_odom_maximum_yaw_rate_deg_s
                ),
            )
        )
        self._ping_count = 0
        self._accepted_updates = 0
        self._rejected_updates = 0
        self._last_sonar_update: SonarUpdateResult | None = None
        self._accepted_intensity_updates = 0
        self._accepted_local_intensity_updates = 0
        self._accepted_local_intensity_updates_with_graph_factor = 0
        self._accepted_local_intensity_updates_without_graph_factor = 0
        self._accepted_graph_seeded_intensity_updates = 0
        self._rejected_intensity_updates = 0
        self._last_intensity_update: IntensityUpdateResult | None = None
        self._latest_sidescan_descriptor: np.ndarray | None = None
        self._sidescan_descriptor_history: deque[np.ndarray] = deque(
            maxlen=max(
                1, int(self.config.sidescan_descriptor_history_lines)
            )
        )
        self._latest_sidescan: tuple[np.ndarray, np.ndarray] | None = None
        self._latest_sidescan_timestamp = -np.inf
        self._sidescan_lines = 0
        self._integrated_sonar_pings = 0
        self._coregistered_intensity_samples = 0
        self._legacy_range_attached_intensity_samples = 0
        self._committed_geometry_groups = 0
        self._total_geometry_points_committed = 0
        self._quarantined_geometry_groups = 0
        self._committed_intensity_batches = 0
        self._quarantined_intensity_batches = 0
        self._geometry_pose_trusted = True
        self._sonar_rejection_reasons: Counter[str] = Counter()
        self._sonar_observable_rank_counts: Counter[str] = Counter()
        self._intensity_rejection_reasons: Counter[str] = Counter()
        self._intensity_planar_rank_counts: Counter[str] = Counter()
        self._accepted_intensity_correlation_sum = 0.0
        self._accepted_intensity_correlation_min = np.inf
        self._accepted_intensity_overlap_sum = 0.0
        self._intensity_translation_correction_sum = 0.0
        self._intensity_rotation_correction_sum = 0.0
        self._pending_intensity: deque[
            tuple[int, np.ndarray, np.ndarray, float, int]
        ] = deque()
        self._pending_intensity_points = 0
        self._intensity_group: list[
            tuple[int, InertialState, np.ndarray, np.ndarray, float]
        ] = []
        self._formed_intensity_strips = 0
        self._skipped_intensity_strips = 0
        self._last_intensity_strip_span_m = 0.0
        self._intensity_submap_anchor_id: int | None = None
        self._intensity_submap_anchor_timestamp = -np.inf
        self._trajectory: list[np.ndarray] = []
        self._trajectory_timestamps: list[float] = []
        self._mbes_group: list[
            tuple[InertialState, np.ndarray, np.ndarray | None]
        ] = []

    @property
    def state(self) -> InertialState:
        return self.filter.state

    @property
    def initialized(self) -> bool:
        return self.filter.initialized

    def initialize(self, timestamp: float, **state_kwargs) -> None:
        self.filter.initialize(timestamp, **state_kwargs)
        self.map.clear()
        self.intensity_map.clear()
        self.quarantine_map.clear()
        self.quarantine_intensity_map.clear()
        self.pose_graph.reset()
        self.map_to_odom_corrector.reset(timestamp=float(timestamp))
        self._trajectory.clear()
        self._trajectory_timestamps.clear()
        self._mbes_group.clear()
        self._ping_count = 0
        self._accepted_updates = 0
        self._rejected_updates = 0
        self._last_sonar_update = None
        self._accepted_intensity_updates = 0
        self._accepted_local_intensity_updates = 0
        self._accepted_local_intensity_updates_with_graph_factor = 0
        self._accepted_local_intensity_updates_without_graph_factor = 0
        self._accepted_graph_seeded_intensity_updates = 0
        self._rejected_intensity_updates = 0
        self._last_intensity_update = None
        self._latest_sidescan_descriptor = None
        self._sidescan_descriptor_history.clear()
        self._latest_sidescan = None
        self._latest_sidescan_timestamp = -np.inf
        self._sidescan_lines = 0
        self._integrated_sonar_pings = 0
        self._coregistered_intensity_samples = 0
        self._legacy_range_attached_intensity_samples = 0
        self._committed_geometry_groups = 0
        self._total_geometry_points_committed = 0
        self._quarantined_geometry_groups = 0
        self._committed_intensity_batches = 0
        self._quarantined_intensity_batches = 0
        self._geometry_pose_trusted = True
        self._sonar_rejection_reasons.clear()
        self._sonar_observable_rank_counts.clear()
        self._intensity_rejection_reasons.clear()
        self._intensity_planar_rank_counts.clear()
        self._accepted_intensity_correlation_sum = 0.0
        self._accepted_intensity_correlation_min = np.inf
        self._accepted_intensity_overlap_sum = 0.0
        self._intensity_translation_correction_sum = 0.0
        self._intensity_rotation_correction_sum = 0.0
        self._pending_intensity.clear()
        self._pending_intensity_points = 0
        self._intensity_group.clear()
        self._formed_intensity_strips = 0
        self._skipped_intensity_strips = 0
        self._last_intensity_strip_span_m = 0.0
        self._intensity_submap_anchor_id = None
        self._intensity_submap_anchor_timestamp = -np.inf
        self._append_trajectory()

    def process_imu(
        self,
        timestamp: float,
        gyro_rad_s: np.ndarray,
        specific_force_mps2: np.ndarray,
    ) -> bool:
        propagated = self.filter.propagate(
            timestamp, gyro_rad_s, specific_force_mps2
        )
        if propagated:
            self._append_trajectory()
        return propagated

    def process_dvl(
        self,
        velocity_body_mps: np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> bool:
        return self.filter.update_dvl(velocity_body_mps, covariance)

    def process_depth(
        self, depth_m: float, variance: float | None = None
    ) -> bool:
        return self.filter.update_depth(depth_m, variance)

    def process_sidescan(
        self,
        port: np.ndarray,
        starboard: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Cache a side-scan line for direct intensity and descriptor updates."""

        port = np.nan_to_num(
            np.asarray(port, dtype=np.float64).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        starboard = np.nan_to_num(
            np.asarray(starboard, dtype=np.float64).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._latest_sidescan = (port.copy(), starboard.copy())
        line_descriptor = sidescan_descriptor(
            port, starboard, bins=self.config.pose_graph.sidescan_descriptor_bins
        )
        self._sidescan_descriptor_history.append(line_descriptor)
        # A robust temporal aggregate describes an actual waterfall patch,
        # making place recognition less sensitive to ping-level speckle while
        # retaining FAST-LIVO2's compact per-keyframe visual descriptor.
        self._latest_sidescan_descriptor = np.median(
            np.stack(tuple(self._sidescan_descriptor_history), axis=0), axis=0
        )
        self._latest_sidescan_timestamp = float(
            self.state.timestamp if timestamp is None else timestamp
        )
        self._sidescan_lines += 1

    def process_integrated_ping(
        self,
        ping: "BathymetricSideScanPing",
        *,
        sample_stride: int = 1,
    ) -> SonarUpdateResult | None:
        """Consume one physically shared bathymetry/backscatter ping.

        This is the preferred path for the V2 instrument.  It obtains geometry
        and radiometry from the same point-wise raw-return array and applies one
        deterministic decimation to both, preventing callers from accidentally
        fusing an unrelated bathymetry cloud and side-scan image.
        """

        if not bool(ping.products_share_raw_returns):
            raise ValueError(
                "integrated ping must derive bathymetry and side-scan from "
                "one raw-return grid"
            )
        stride = int(sample_stride)
        if stride < 1:
            raise ValueError("sample_stride must be at least one")
        points, values, _ = ping.valid_acoustic_samples(frame="sensor")
        points = points[::stride]
        values = values[::stride]
        return self.process_bathymetric_sidescan(
            points,
            values,
            ping.side_scan.port,
            ping.side_scan.starboard,
            timestamp=ping.timestamp,
            along_track_coverage_m=ping.along_track_coverage_m,
        )

    def process_bathymetric_sidescan(
        self,
        points_sensor: np.ndarray,
        backscatter: np.ndarray,
        port: np.ndarray,
        starboard: np.ndarray,
        *,
        timestamp: float | None = None,
        imaging_points_sensor: np.ndarray | None = None,
        imaging_backscatter: np.ndarray | None = None,
        along_track_coverage_m: float = 0.0,
    ) -> SonarUpdateResult | None:
        """Process one integrated geometry/backscatter acoustic acquisition.

        ``backscatter[i]`` must be measured by the same sparse bottom detection
        that produced ``points_sensor[i]``. A shared ping may additionally
        provide a denser, independently sampled ``imaging_points_sensor`` /
        ``imaging_backscatter`` pair. The port/starboard arrays are retained
        for the rolling waterfall descriptor; direct intensity fusion always
        uses explicit XYZ-to-radiometry registration and never guesses a
        side-scan pixel's elevation angle.
        """

        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(backscatter, dtype=np.float64).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError(
                "integrated backscatter must match bathymetric points"
            )
        if (imaging_points_sensor is None) != (imaging_backscatter is None):
            raise ValueError(
                "dense imaging points and backscatter must be supplied together"
            )
        if imaging_points_sensor is None:
            direct_points = points
            direct_values = values
        else:
            direct_points = np.asarray(
                imaging_points_sensor, dtype=np.float64
            ).reshape(-1, 3)
            direct_values = np.asarray(
                imaging_backscatter, dtype=np.float64
            ).reshape(-1)
            if direct_values.size != direct_points.shape[0]:
                raise ValueError(
                    "dense imaging backscatter must match imaging points"
                )
        observation_time = float(
            self.state.timestamp if timestamp is None else timestamp
        )
        if (
            abs(observation_time - self.state.timestamp)
            > max(0.0, self.config.integrated_timestamp_tolerance_s)
        ):
            raise ValueError(
                "integrated sonar timestamp must match the propagated state"
            )
        self.process_sidescan(
            port,
            starboard,
            timestamp=observation_time,
        )
        self._integrated_sonar_pings += 1
        return self.process_mbes(
            points,
            values,
            coregistered_backscatter=values,
            direct_intensity_points=direct_points,
            direct_intensities=direct_values,
            direct_intensity_coverage_m=along_track_coverage_m,
        )

    def process_mbes(
        self,
        points_sensor: np.ndarray,
        intensities: np.ndarray | None = None,
        *,
        coregistered_backscatter: np.ndarray | None = None,
        direct_intensity_points: np.ndarray | None = None,
        direct_intensities: np.ndarray | None = None,
        direct_intensity_coverage_m: float = 0.0,
    ) -> SonarUpdateResult | None:
        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        ranges = np.linalg.norm(np.where(finite[:, None], points, 0.0), axis=1)
        valid = (
            finite
            & (ranges >= self.config.map_min_range)
            & (ranges <= self.config.map_max_range)
        )
        points = points[valid]
        if intensities is not None:
            values = np.asarray(intensities, dtype=np.float64).reshape(-1)
            if values.size != valid.size:
                raise ValueError("intensities must match MBES points")
            values = values[valid]
        else:
            values = None
        if coregistered_backscatter is not None:
            registered_values = np.asarray(
                coregistered_backscatter, dtype=np.float64
            ).reshape(-1)
            if registered_values.size != valid.size:
                raise ValueError(
                    "coregistered_backscatter must match MBES points"
                )
            registered_values = registered_values[valid]
        else:
            registered_values = None
        if (direct_intensity_points is None) != (direct_intensities is None):
            raise ValueError(
                "direct intensity points and values must be supplied together"
            )
        if direct_intensity_points is None:
            direct_points = points
            direct_values = registered_values
        else:
            direct_points = np.asarray(
                direct_intensity_points, dtype=np.float64
            ).reshape(-1, 3)
            direct_values = np.asarray(
                direct_intensities, dtype=np.float64
            ).reshape(-1)
            if direct_values.size != direct_points.shape[0]:
                raise ValueError(
                    "direct intensity values must match direct points"
                )
        if points.size == 0:
            self._rejected_updates += 1
            return None

        current_points = points.copy()
        self._ping_count += 1
        self._mbes_group.append((self.state.copy(), points, values))
        group_size = max(1, int(self.config.mbes_group_pings))
        update: SonarUpdateResult | None = None
        geometry_to_commit: tuple[
            np.ndarray, np.ndarray | None, int, bool
        ] | None = None
        if len(self._mbes_group) >= group_size:
            grouped_points, grouped_values = self._motion_compensated_group()
            self._mbes_group.clear()
            update_attempted = False
            if (
                self._ping_count > self.config.bootstrap_pings
                and self.map.mature_voxel_count > 0
            ):
                update_attempted = True
                self.map.set_association_box(
                    self.state.position,
                    self.config.association_map_half_extent_m,
                )
                update = self.filter.update_sonar(grouped_points, self.map)
                rank_key = (
                    f"r{update.rotation_observable_rank}_"
                    f"t{update.translation_observable_rank}"
                )
                self._sonar_observable_rank_counts[rank_key] += 1
                if update.accepted:
                    self._accepted_updates += 1
                else:
                    self._rejected_updates += 1
                    self._sonar_rejection_reasons[
                        update.rejection_reason or "unspecified"
                    ] += 1

            stride = max(1, int(self.config.map_insert_stride))
            self._geometry_pose_trusted = (
                not update_attempted or bool(update and update.accepted)
            )
            commit_geometry = (
                not self.config.quality_gate_map_updates
                or self._geometry_pose_trusted
            )
            loop_points_body = transform_points(
                self.filter.sensor_rotation_body,
                self.filter.sensor_translation_body,
                grouped_points,
            )
            appearance_descriptor = None
            if (
                self._latest_sidescan_descriptor is not None
                and abs(self.state.timestamp - self._latest_sidescan_timestamp)
                <= self.config.sidescan_max_age_s
            ):
                appearance_descriptor = self._latest_sidescan_descriptor
            if commit_geometry:
                self.pose_graph.add_if_needed(
                    self.state.timestamp,
                    self.state.rotation,
                    self.state.position,
                    loop_points_body,
                    grouped_values,
                    appearance_descriptor,
                    observation_id=self._ping_count,
                )
            geometry_to_commit = (
                grouped_points,
                grouped_values,
                stride,
                commit_geometry,
            )
            self._last_sonar_update = update

        self._process_sidescan_intensity(
            direct_points if direct_values is not None else current_points,
            direct_values,
            along_track_coverage_m=direct_intensity_coverage_m,
        )
        if geometry_to_commit is not None:
            (
                grouped_points,
                grouped_values,
                stride,
                commit_geometry,
            ) = geometry_to_commit
            # Sequential geometry and intensity observations at this ping
            # share one final ESIKF state. Insert the grouped MBES patch only
            # after both have converged so it is not frozen at the pre-image
            # pose and then used to pull the next update backwards.
            points_world = self.filter.points_sensor_to_world(grouped_points)
            map_intensities = (
                None
                if grouped_values is None
                else grouped_values[::stride]
            )
            if commit_geometry:
                inserted = self.map.add_points(
                    points_world[::stride], map_intensities
                )
                self._total_geometry_points_committed += inserted
                self._committed_geometry_groups += 1
            else:
                self.quarantine_map.add_points(
                    points_world[::stride], map_intensities
                )
                self._quarantined_geometry_groups += 1
        self._replace_latest_trajectory()
        return update

    def _sidescan_support(
        self, points_sensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy adapter for genuinely separate MBES and side-scan inputs."""

        if (
            self._latest_sidescan is None
            or abs(self.state.timestamp - self._latest_sidescan_timestamp)
            > self.config.sidescan_max_age_s
        ):
            return np.empty((0, 3)), np.empty(0)
        minimum = float(self.config.sidescan_min_range)
        maximum = float(self.config.sidescan_max_range)
        if maximum <= minimum:
            raise ValueError("side-scan range limits are invalid")
        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        ranges = np.linalg.norm(points, axis=1)
        sampled = np.zeros(points.shape[0], dtype=np.float64)
        supported = np.zeros(points.shape[0], dtype=bool)
        port, starboard = self._latest_sidescan
        for side_mask, line in (
            (points[:, 1] > 0.25, port),
            (points[:, 1] < -0.25, starboard),
        ):
            if line.size < 2:
                continue
            coordinate = (
                (ranges - minimum) / (maximum - minimum) * line.size - 0.5
            )
            valid = side_mask & (coordinate >= 0.0) & (
                coordinate <= line.size - 1.0
            )
            sampled[valid] = np.interp(
                coordinate[valid], np.arange(line.size), line
            )
            supported |= valid
        # Log compression is the acoustic counterpart of exposure-normalized
        # image patches in FAST-LIVO2.
        sampled = np.log1p(8.0 * np.maximum(sampled, 0.0)) / np.log(9.0)
        supported &= sampled > 0.015
        self._legacy_range_attached_intensity_samples += int(
            np.count_nonzero(supported)
        )
        return points[supported], sampled[supported]

    def _coregistered_sidescan_support(
        self,
        points_sensor: np.ndarray,
        intensities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select radiometry already registered to bathymetric XYZ samples."""

        points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError(
                "co-registered intensity count must match bathymetric points"
            )
        ranges = np.linalg.norm(
            np.where(np.isfinite(points), points, 0.0), axis=1
        )
        supported = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(values)
            & (values > 0.0)
            & (ranges >= self.config.sidescan_min_range)
            & (ranges <= self.config.sidescan_max_range)
            & (
                np.abs(points[:, 1])
                >= max(0.0, self.config.coregistered_min_cross_track_m)
            )
        )
        # FAST-LIVO2 normalizes image exposure before its direct residual.
        # Log compression plays the same role for acoustic dynamic range while
        # preserving the exact XYZ-to-backscatter correspondence.
        values = np.log1p(8.0 * np.maximum(values, 0.0)) / np.log(9.0)
        supported &= values > 0.015
        self._coregistered_intensity_samples += int(
            np.count_nonzero(supported)
        )
        return points[supported], values[supported]

    def _process_sidescan_intensity(
        self,
        points_sensor: np.ndarray,
        coregistered_backscatter: np.ndarray | None = None,
        *,
        along_track_coverage_m: float = 0.0,
    ) -> None:
        along_track_coverage_m = float(along_track_coverage_m)
        if (
            not np.isfinite(along_track_coverage_m)
            or along_track_coverage_m < 0.0
        ):
            raise ValueError(
                "along-track coverage must be finite and non-negative"
            )
        self._promote_intensity_references()
        if coregistered_backscatter is None:
            points, values = self._sidescan_support(points_sensor)
        else:
            points, values = self._coregistered_sidescan_support(
                points_sensor, coregistered_backscatter
            )
        if points.shape[0] < self.config.filter.intensity_min_correspondences:
            return
        if (
            self._intensity_group
            and self._ping_count != self._intensity_group[-1][0] + 1
        ):
            # Never combine lines across a sensor-data gap: the resulting
            # strip would contain an unmodelled temporal discontinuity.
            self._intensity_group.clear()
        self._intensity_group.append(
            (
                self._ping_count,
                self.state.copy(),
                points,
                values,
                along_track_coverage_m,
            )
        )
        group_size = max(1, int(self.config.intensity_group_pings))
        if len(self._intensity_group) < group_size:
            return
        oldest_group_observation_id = int(self._intensity_group[0][0])
        grouped_points, grouped_values, grouped_line_ids, strip_span = (
            self._motion_compensated_intensity_group()
        )
        self._intensity_group.clear()
        self._last_intensity_strip_span_m = strip_span
        if strip_span < max(0.0, self.config.intensity_min_group_span_m):
            self._skipped_intensity_strips += 1
            self._intensity_rejection_reasons["insufficient_strip_span"] += 1
            return
        self._formed_intensity_strips += 1
        result: IntensityUpdateResult | None = None
        intensity_prior_state: InertialState | None = None
        if (
            self.config.intensity_updates_enabled
            and self._ping_count
            > max(
                self.config.intensity_bootstrap_pings,
                self.config.intensity_min_separation_pings,
            )
            and len(self.intensity_map) > 0
        ):
            intensity_prior_state = self.state.copy()
            maximum_reference_id = (
                oldest_group_observation_id
                - self.config.intensity_min_separation_pings
            )
            result = self.filter.update_intensity(
                grouped_points,
                grouped_values,
                self.intensity_map,
                maximum_observation_id=maximum_reference_id,
                affine_group_ids=grouped_line_ids,
            )
            graph_seeded = False
            graph_factor_added = False
            prior_yaw = float(
                np.arctan2(
                    intensity_prior_state.rotation[1, 0],
                    intensity_prior_state.rotation[0, 0],
                )
            )
            prior_pose2 = np.array(
                [
                    intensity_prior_state.position[0],
                    intensity_prior_state.position[1],
                    prior_yaw,
                ],
                dtype=np.float64,
            )
            points_body = transform_points(
                self.filter.sensor_rotation_body,
                self.filter.sensor_translation_body,
                grouped_points,
            )

            def add_graph_factor(candidate: IntensityUpdateResult) -> bool:
                if candidate.anchor_keyframe_index is None:
                    return False
                return self.pose_graph.add_intensity_observation(
                    timestamp=self.state.timestamp,
                    observation_id=self._ping_count,
                    anchor_keyframe_index=candidate.anchor_keyframe_index,
                    prior_pose2=prior_pose2,
                    posterior_rotation_world_body=self.state.rotation,
                    posterior_position_world=self.state.position,
                    photometric_pose2=candidate.photometric_pose2,
                    information_world=candidate.information_pose2,
                    points_body=points_body,
                    intensities=grouped_values,
                    appearance_descriptor=self._latest_sidescan_descriptor,
                    score=candidate.normalized_correlation,
                )

            if not result.accepted and self.pose_graph.loop_edges:
                # A local intensity interpolator cannot recover after metres
                # of drift because there are no initial correspondences. Once
                # a geometrically verified loop has placed the current graph
                # branch near an old anchored submap, try that global estimate
                # as a *non-committing* image alignment seed. A successful
                # match refines the graph but never jumps the live ESIKF.
                seed_candidates = self.pose_graph.intensity_seed_candidates(
                    rotation_world_body=self.state.rotation,
                    position_world=self.state.position,
                    anchor_ids=self.intensity_map.anchor_ids,
                    maximum_observation_id=maximum_reference_id,
                )
                graph_inconsistent_result: IntensityUpdateResult | None = None
                for candidate_anchor, candidate_pose in seed_candidates:
                    seeded_state = self.state.copy()
                    current_yaw = float(
                        np.arctan2(
                            seeded_state.rotation[1, 0],
                            seeded_state.rotation[0, 0],
                        )
                    )
                    yaw_delta = float(
                        np.arctan2(
                            np.sin(candidate_pose[2] - current_yaw),
                            np.cos(candidate_pose[2] - current_yaw),
                        )
                    )
                    c, s = np.cos(yaw_delta), np.sin(yaw_delta)
                    yaw_rotation = np.array(
                        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    seeded_state.rotation = (
                        yaw_rotation @ seeded_state.rotation
                    )
                    seeded_state.position[:2] = candidate_pose[:2]
                    candidate_result = self.filter.update_intensity(
                        grouped_points,
                        grouped_values,
                        self.intensity_map,
                        maximum_observation_id=maximum_reference_id,
                        initial_state=seeded_state,
                        fixed_anchor_id=candidate_anchor,
                        commit_state=False,
                        affine_group_ids=grouped_line_ids,
                    )
                    if (
                        candidate_result.accepted
                        and candidate_result.anchor_keyframe_index is not None
                    ):
                        # A non-committing graph seed has no effect unless its
                        # independently verified image factor is actually
                        # accepted by the graph.  Rejected factors are
                        # transactional, so another anchor can safely be tried.
                        if add_graph_factor(candidate_result):
                            result = candidate_result
                            graph_seeded = True
                            graph_factor_added = True
                            break
                        graph_inconsistent_result = candidate_result
                if not graph_seeded and graph_inconsistent_result is not None:
                    result = replace(
                        graph_inconsistent_result,
                        accepted=False,
                        rejection_reason="graph_factor_rejected",
                    )
            elif result.accepted:
                graph_factor_added = add_graph_factor(result)

            self._intensity_planar_rank_counts[
                f"r{result.planar_observable_rank}"
            ] += 1
            if result.accepted:
                self._accepted_intensity_updates += 1
                if graph_seeded:
                    self._accepted_graph_seeded_intensity_updates += 1
                else:
                    self._accepted_local_intensity_updates += 1
                    if graph_factor_added:
                        self._accepted_local_intensity_updates_with_graph_factor += 1
                    else:
                        self._accepted_local_intensity_updates_without_graph_factor += 1
                self._accepted_intensity_correlation_sum += (
                    result.normalized_correlation
                )
                self._accepted_intensity_correlation_min = min(
                    self._accepted_intensity_correlation_min,
                    result.normalized_correlation,
                )
                self._accepted_intensity_overlap_sum += result.overlap_ratio
                self._intensity_translation_correction_sum += (
                    result.translation_correction
                )
                self._intensity_rotation_correction_sum += (
                    result.rotation_correction_deg
                )
                if (
                    intensity_prior_state is not None
                    and not graph_seeded
                    and not graph_factor_added
                ):
                    # The local image update is still valid even when its
                    # measurement-only graph Hessian is too weak/unanchored.
                    # If this ping already made a geometry keyframe, absorb
                    # the posterior into that terminal odometry edge so the
                    # same correction is neither lost nor deferred to the
                    # following keyframe.
                    self.pose_graph.absorb_latest_local_posterior(
                        timestamp=self.state.timestamp,
                        observation_id=self._ping_count,
                        posterior_rotation_world_body=self.state.rotation,
                        posterior_position_world=self.state.position,
                    )
            else:
                self._rejected_intensity_updates += 1
                self._intensity_rejection_reasons[
                    result.rejection_reason or "unspecified"
                ] += 1
        points_world = self.filter.points_sensor_to_world(grouped_points)
        heading = float(
            np.arctan2(self.state.rotation[1, 0], self.state.rotation[0, 0])
        )
        latest_anchor_id = self.pose_graph.latest_keyframe_index
        if latest_anchor_id is None:
            anchor_id = -1
        else:
            latest_anchor_timestamp = float(
                self.pose_graph.keyframes[latest_anchor_id].timestamp
            )
            minimum_interval = max(
                0.0, float(self.config.intensity_submap_anchor_interval_s)
            )
            if (
                self._intensity_submap_anchor_id is None
                or minimum_interval <= 0.0
                or latest_anchor_timestamp
                - self._intensity_submap_anchor_timestamp
                >= minimum_interval - 1.0e-12
            ):
                self._intensity_submap_anchor_id = int(latest_anchor_id)
                self._intensity_submap_anchor_timestamp = (
                    latest_anchor_timestamp
                )
            anchor_id = int(self._intensity_submap_anchor_id)
        intensity_pose_trusted = self._geometry_pose_trusted or bool(
            result and result.accepted
        )
        commit_intensity = (
            not self.config.quality_gate_map_updates
            or intensity_pose_trusted
        )
        if commit_intensity:
            self._pending_intensity.append(
                (
                    self._ping_count,
                    points_world,
                    grouped_values,
                    heading,
                    int(anchor_id),
                )
            )
            self._pending_intensity_points += int(points_world.shape[0])
        elif not commit_intensity:
            self.quarantine_intensity_map.add_points(
                points_world,
                grouped_values,
                heading_rad=heading,
                observation_id=self._ping_count,
                anchor_id=int(anchor_id),
            )
            self._quarantined_intensity_batches += 1
        self._last_intensity_update = result

    def _promote_intensity_references(self) -> None:
        """Promote only sufficiently old acoustic patches to the map.

        Previously every ping immediately changed a cell whose
        ``first_observation_id`` happened to be old.  A later query could
        therefore compare against a mean already containing recent samples,
        despite the requested separation.  FAST-LIVO2 keeps explicit
        reference patches; this delayed queue provides the same causal
        separation for side-scan samples.
        """

        maximum_id = self._ping_count - max(
            1, int(self.config.intensity_min_separation_pings)
        )
        while (
            self._pending_intensity
            and self._pending_intensity[0][0] <= maximum_id
        ):
            observation_id, points_world, values, heading, anchor_id = (
                self._pending_intensity.popleft()
            )
            self._pending_intensity_points -= int(points_world.shape[0])
            self.intensity_map.add_points(
                points_world,
                values,
                heading_rad=heading,
                observation_id=observation_id,
                anchor_id=anchor_id,
            )
            self._committed_intensity_batches += 1

    def diagnostics(self) -> SLAMDiagnostics:
        return SLAMDiagnostics(
            ping_count=self._ping_count,
            accepted_sonar_updates=self._accepted_updates,
            rejected_sonar_updates=self._rejected_updates,
            map_points=self.map.point_count,
            total_geometry_points_committed=(
                self._total_geometry_points_committed
            ),
            map_voxels=len(self.map),
            association_map_voxels=self.map.association_voxel_count,
            mature_surfels=self.map.mature_voxel_count,
            sidescan_lines=self._sidescan_lines,
            integrated_sonar_pings=self._integrated_sonar_pings,
            coregistered_intensity_samples=(
                self._coregistered_intensity_samples
            ),
            legacy_range_attached_intensity_samples=(
                self._legacy_range_attached_intensity_samples
            ),
            intensity_map_points=self.intensity_map.point_count,
            intensity_map_cells=len(self.intensity_map),
            accepted_intensity_updates=self._accepted_intensity_updates,
            accepted_local_intensity_updates=(
                self._accepted_local_intensity_updates
            ),
            accepted_local_intensity_updates_with_graph_factor=(
                self._accepted_local_intensity_updates_with_graph_factor
            ),
            accepted_local_intensity_updates_without_graph_factor=(
                self._accepted_local_intensity_updates_without_graph_factor
            ),
            accepted_graph_seeded_intensity_updates=(
                self._accepted_graph_seeded_intensity_updates
            ),
            rejected_intensity_updates=self._rejected_intensity_updates,
            committed_geometry_groups=self._committed_geometry_groups,
            quarantined_geometry_groups=self._quarantined_geometry_groups,
            committed_intensity_batches=self._committed_intensity_batches,
            quarantined_intensity_batches=self._quarantined_intensity_batches,
            sonar_rejection_reasons=dict(self._sonar_rejection_reasons),
            sonar_observable_rank_counts=dict(
                self._sonar_observable_rank_counts
            ),
            intensity_rejection_reasons=dict(
                self._intensity_rejection_reasons
            ),
            intensity_planar_rank_counts=dict(
                self._intensity_planar_rank_counts
            ),
            mean_accepted_intensity_correlation=(
                self._accepted_intensity_correlation_sum
                / max(1, self._accepted_intensity_updates)
            ),
            minimum_accepted_intensity_correlation=(
                self._accepted_intensity_correlation_min
                if self._accepted_intensity_updates > 0
                else 0.0
            ),
            mean_accepted_intensity_overlap=(
                self._accepted_intensity_overlap_sum
                / max(1, self._accepted_intensity_updates)
            ),
            cumulative_intensity_translation_correction_m=(
                self._intensity_translation_correction_sum
            ),
            cumulative_intensity_rotation_correction_deg=(
                self._intensity_rotation_correction_sum
            ),
            pending_intensity_batches=len(self._pending_intensity),
            pending_intensity_points=self._pending_intensity_points,
            formed_intensity_strips=self._formed_intensity_strips,
            skipped_intensity_strips=self._skipped_intensity_strips,
            last_intensity_strip_span_m=self._last_intensity_strip_span_m,
            last_sonar_update=self._last_sonar_update,
            last_intensity_update=self._last_intensity_update,
            pose_graph=self.pose_graph.diagnostics(),
        )

    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._trajectory:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )
        timestamps = np.asarray(
            self._trajectory_timestamps, dtype=np.float64
        )
        positions = np.asarray(self._trajectory, dtype=np.float64)
        if abs(timestamps[-1] - self.state.timestamp) > 1.0e-9:
            timestamps = np.append(timestamps, float(self.state.timestamp))
            positions = np.concatenate(
                (positions, self.state.position.reshape(1, 3)), axis=0
            )
        return timestamps, positions

    def global_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the loop-corrected trajectory without disturbing the ESIKF."""

        timestamps, raw = self.trajectory()
        return timestamps, self.pose_graph.correct_trajectory(timestamps, raw)

    def raw_global_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest unsmoothed pose-graph target pose."""

        return self.pose_graph.corrected_pose(
            self.state.rotation, self.state.position
        )

    def global_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the online ``map`` pose without disturbing the ESIKF.

        When configured, only this published map-frame output is softened.
        :meth:`raw_global_pose` and :meth:`global_trajectory` retain the exact
        graph solution for matcher seeds and final scientific evaluation.
        """

        target_rotation, target_position = self.raw_global_pose()
        if not self.config.map_to_odom_soft_correction_enabled:
            return target_rotation, target_position
        return self.map_to_odom_corrector.correct_pose(
            self.state.timestamp,
            self.state.rotation,
            self.state.position,
            target_rotation,
            target_position,
        )

    def optimized_map_points(self) -> np.ndarray:
        """Return keyframe MBES points transformed by optimized poses."""

        return self.pose_graph.optimized_map_points()

    def reset_map(self) -> None:
        self.map.clear()
        self.intensity_map.clear()
        self.quarantine_map.clear()
        self.quarantine_intensity_map.clear()
        self._ping_count = 0
        self._accepted_updates = 0
        self._rejected_updates = 0
        self._last_sonar_update = None
        self._accepted_intensity_updates = 0
        self._accepted_local_intensity_updates = 0
        self._accepted_local_intensity_updates_with_graph_factor = 0
        self._accepted_local_intensity_updates_without_graph_factor = 0
        self._accepted_graph_seeded_intensity_updates = 0
        self._rejected_intensity_updates = 0
        self._last_intensity_update = None
        self._mbes_group.clear()
        self._latest_sidescan_descriptor = None
        self._sidescan_descriptor_history.clear()
        self._latest_sidescan = None
        self._latest_sidescan_timestamp = -np.inf
        self._sidescan_lines = 0
        self._integrated_sonar_pings = 0
        self._coregistered_intensity_samples = 0
        self._legacy_range_attached_intensity_samples = 0
        self._committed_geometry_groups = 0
        self._total_geometry_points_committed = 0
        self._quarantined_geometry_groups = 0
        self._committed_intensity_batches = 0
        self._quarantined_intensity_batches = 0
        self._geometry_pose_trusted = True
        self._sonar_rejection_reasons.clear()
        self._sonar_observable_rank_counts.clear()
        self._intensity_rejection_reasons.clear()
        self._intensity_planar_rank_counts.clear()
        self._accepted_intensity_correlation_sum = 0.0
        self._accepted_intensity_correlation_min = np.inf
        self._accepted_intensity_overlap_sum = 0.0
        self._intensity_translation_correction_sum = 0.0
        self._intensity_rotation_correction_sum = 0.0
        self._pending_intensity.clear()
        self._pending_intensity_points = 0
        self._intensity_group.clear()
        self._formed_intensity_strips = 0
        self._skipped_intensity_strips = 0
        self._last_intensity_strip_span_m = 0.0
        self._intensity_submap_anchor_id = None
        self._intensity_submap_anchor_timestamp = -np.inf
        self.pose_graph.reset()
        self.map_to_odom_corrector.reset(
            timestamp=float(self.state.timestamp)
        )

    def _append_trajectory(self) -> None:
        interval = float(self.config.trajectory_sample_interval_s)
        if (
            interval > 0.0
            and self._trajectory_timestamps
            and self.state.timestamp - self._trajectory_timestamps[-1]
            < interval - 1.0e-12
        ):
            return
        self._trajectory_timestamps.append(float(self.state.timestamp))
        self._trajectory.append(self.state.position.copy())

    def _replace_latest_trajectory(self) -> None:
        if (
            self._trajectory_timestamps
            and abs(self._trajectory_timestamps[-1] - self.state.timestamp)
            < 1.0e-9
        ):
            self._trajectory[-1] = self.state.position.copy()
        else:
            self._append_trajectory()

    def _motion_compensated_group(
        self,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Deskew a short MBES group into the current sensor frame.

        A down-looking MBES ping is only a one-dimensional fan and is
        intrinsically degenerate.  Grouping several motion-compensated pings
        creates the local surface patch required for a stable point-to-plane
        update, analogous to deskewing a spinning LiDAR scan.
        """

        current = self.state
        rotation_bs = self.filter.sensor_rotation_body
        translation_bs = self.filter.sensor_translation_body
        rotation_ws_current = current.rotation @ rotation_bs
        translation_ws_current = (
            current.position + current.rotation @ translation_bs
        )
        grouped_points: list[np.ndarray] = []
        grouped_intensities: list[np.ndarray] = []
        has_intensity = all(item[2] is not None for item in self._mbes_group)
        for historical, points_sensor, intensities in self._mbes_group:
            rotation_ws_historical = historical.rotation @ rotation_bs
            translation_ws_historical = (
                historical.position + historical.rotation @ translation_bs
            )
            points_world = (
                points_sensor @ rotation_ws_historical.T
                + translation_ws_historical
            )
            points_current_sensor = (
                points_world - translation_ws_current
            ) @ rotation_ws_current
            grouped_points.append(points_current_sensor)
            if has_intensity:
                grouped_intensities.append(intensities)
        points = np.ascontiguousarray(np.concatenate(grouped_points), dtype=np.float64)
        values = (
            np.ascontiguousarray(np.concatenate(grouped_intensities), dtype=np.float64)
            if has_intensity
            else None
        )
        return points, values

    def _motion_compensated_intensity_group(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Deskew acquisitions into one strip, including intra-ping aperture."""

        current = self.state
        rotation_bs = self.filter.sensor_rotation_body
        translation_bs = self.filter.sensor_translation_body
        rotation_ws_current = current.rotation @ rotation_bs
        translation_ws_current = (
            current.position + current.rotation @ translation_bs
        )
        grouped_points: list[np.ndarray] = []
        grouped_values: list[np.ndarray] = []
        grouped_line_ids: list[np.ndarray] = []
        sensor_origins: list[np.ndarray] = []
        simultaneous_spans: list[float] = []
        for line_id, (
            _, historical, points_sensor, values, simultaneous_span
        ) in enumerate(self._intensity_group):
            rotation_ws_historical = historical.rotation @ rotation_bs
            translation_ws_historical = (
                historical.position + historical.rotation @ translation_bs
            )
            points_world = (
                points_sensor @ rotation_ws_historical.T
                + translation_ws_historical
            )
            grouped_points.append(
                (points_world - translation_ws_current) @ rotation_ws_current
            )
            grouped_values.append(values)
            grouped_line_ids.append(
                np.full(values.size, line_id, dtype=np.int64)
            )
            sensor_origins.append(translation_ws_historical[:2])
            simultaneous_spans.append(float(simultaneous_span))

        origins = np.asarray(sensor_origins, dtype=np.float64)
        vehicle_origin_span = float(
            np.sum(np.linalg.norm(np.diff(origins, axis=0), axis=1))
        )
        # A V2 ping is already a 2-D strip: its eight synchronous row centres
        # cover 1.2 m (standard) or 0.6 m (high resolution).  Do not report a
        # zero-width strip merely because all rows share one acquisition time.
        span = vehicle_origin_span + max(simultaneous_spans, default=0.0)
        return (
            np.ascontiguousarray(
                np.concatenate(grouped_points), dtype=np.float64
            ),
            np.ascontiguousarray(
                np.concatenate(grouped_values), dtype=np.float64
            ),
            np.ascontiguousarray(
                np.concatenate(grouped_line_ids), dtype=np.int64
            ),
            span,
        )
