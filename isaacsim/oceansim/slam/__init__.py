"""FAST-LIVO2-inspired underwater acoustic-inertial SLAM components.

The package is dependency-light on purpose: the estimator and map only need
NumPy, which allows the same implementation to run in Isaac Sim, ROS 2 nodes,
and deterministic unit tests.
"""

from .esikf import (
    AcousticESIKF,
    ESIKFConfig,
    InertialState,
    IntensityUpdateResult,
    SonarUpdateResult,
)
from .evaluation import (
    map_to_odom_change_metrics,
    sanitize_for_strict_json,
    se2_relative_pose_error_by_distance,
    se3_relative_pose_error_by_distance,
)
from .fast_sonar_slam import FastSonarSLAM, FastSonarSLAMConfig
from .intensity_map import AcousticIntensityMap, IntensityMapSample
from .initialization import (
    NavigationInitialization,
    gauge_anchored_initial_covariance,
    initialize_from_imu_dvl_depth,
)
from .global_relocalization import (
    GlobalTerrainCandidate,
    GlobalTerrainRelocalizer,
    GlobalTerrainRelocalizerConfig,
    GlobalTerrainSearchResult,
)
from .pose_graph import (
    AcousticKeyframe,
    AcousticPoseGraph,
    GTSAMPlanarPoseGraph,
    PlanarPoseGraph,
    PoseGraphConfig,
    PoseGraphDiagnostics,
    PoseGraphEdge,
    RegistrationResult,
    acoustic_descriptor,
    register_keyframes,
    sidescan_descriptor,
)
from .real_bathymetry import (
    MeasuredBathymetryGrid,
    TerrainFollowingSurvey,
    WaypointTerrainFollowingSurvey,
)
from .surfel_map import (
    Surfel,
    SurfelAssociationBatch,
    SurfelMapSnapshot,
    SurfelVoxelMap,
)
from .terrain_matching import (
    BackscatterMapMatcher,
    BackscatterMatcherConfig,
    BackscatterMatchResult,
    BathymetricTerrainMatcher,
    TerrainEmergencyRecoveryConfirmation,
    TerrainEmergencyRecoveryController,
    TerrainEmergencyRecoveryRequest,
    TerrainRecoveryController,
    TerrainMatcherConfig,
    TerrainMatchResult,
    apply_terrain_match,
    fixed_lag_fused_deskew_states,
    make_terrain_recovery_config,
    match_bathymetric_terrain,
    motion_compensate_to_body,
    uniform_full_swath_scan_indices,
    voxel_average_backscatter_patch,
    voxel_average_terrain_patch,
)

__all__ = [
    "AcousticESIKF",
    "AcousticIntensityMap",
    "AcousticKeyframe",
    "AcousticPoseGraph",
    "BackscatterMapMatcher",
    "BackscatterMatcherConfig",
    "BackscatterMatchResult",
    "BathymetricTerrainMatcher",
    "ESIKFConfig",
    "FastSonarSLAM",
    "FastSonarSLAMConfig",
    "GTSAMPlanarPoseGraph",
    "GlobalTerrainCandidate",
    "GlobalTerrainRelocalizer",
    "GlobalTerrainRelocalizerConfig",
    "GlobalTerrainSearchResult",
    "InertialState",
    "IntensityMapSample",
    "IntensityUpdateResult",
    "NavigationInitialization",
    "MeasuredBathymetryGrid",
    "PlanarPoseGraph",
    "PoseGraphConfig",
    "PoseGraphDiagnostics",
    "PoseGraphEdge",
    "RegistrationResult",
    "SonarUpdateResult",
    "Surfel",
    "SurfelAssociationBatch",
    "SurfelMapSnapshot",
    "SurfelVoxelMap",
    "TerrainFollowingSurvey",
    "TerrainEmergencyRecoveryConfirmation",
    "TerrainEmergencyRecoveryController",
    "TerrainEmergencyRecoveryRequest",
    "TerrainMatcherConfig",
    "TerrainMatchResult",
    "TerrainRecoveryController",
    "WaypointTerrainFollowingSurvey",
    "acoustic_descriptor",
    "apply_terrain_match",
    "fixed_lag_fused_deskew_states",
    "gauge_anchored_initial_covariance",
    "initialize_from_imu_dvl_depth",
    "map_to_odom_change_metrics",
    "sanitize_for_strict_json",
    "se2_relative_pose_error_by_distance",
    "se3_relative_pose_error_by_distance",
    "make_terrain_recovery_config",
    "match_bathymetric_terrain",
    "motion_compensate_to_body",
    "uniform_full_swath_scan_indices",
    "voxel_average_backscatter_patch",
    "voxel_average_terrain_patch",
    "register_keyframes",
    "sidescan_descriptor",
]
