#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Experimental water-adapted geometry/appearance frontend.  The underlying
# full-map launcher still supplies the validated V2 5 Hz sensor timing,
# MBES/GTSAM/UTAN backend, and UI settings.  This wrapper adds:
#   * IMU/DVL-seeded KLT tracks on the eight simultaneous side-scan rows;
#   * exact XYZ lifting through the co-registered MBES returns;
#   * CUDA-batched geometric consensus with float64 CPU final refinement;
#   * an exactly one-dimensional side-scan state update along the row aperture;
#   * no direct point-to-deforming-local-map MBES state write (append
#     ``--feature-retain-direct-mbes-state-update`` for that hybrid ablation);
#   * GTSAM incremental fixed-lag graph optimization, cohort particle terrain
#     tracking, 3-D BBS verification, and map-to-odom soft output correction;
#     unconfirmed particle modes remain internal and only independently
#     confirmed measurements can affect the published GTSAM solution;
#   * a transparent subgrid scattering model because the public USGS mosaic is
#     5 m while the documented V2 along-track sampling is 0.15 m.
# The regional bathymetry and low-frequency backscatter envelope remain the
# measured USGS products.  The subgrid field is explicitly reported as
# synthetic unresolved sediment scattering and can be disabled by appending
# ``--subgrid-reflectivity-amplitude 0``.
# The candidate run starts at the difficult 30 km route section with exact
# simulation initialization so Local drift is measured without an initial
# gauge error; truth is not fused again after initialization.
exec "${SCRIPT_DIR}/run_v2_full_map_slam_isaac.sh" \
    --survey-start-distance-m 30000 \
    --initial-position-offset-x-m 0 \
    --initial-position-offset-y-m 0 \
    --initial-yaw-offset-deg 0 \
    --initialize-from-truth \
    --local-frontend feature \
    --feature-subframe-pings 4 \
    --feature-maximum-features 900 \
    --feature-minimum-matches 24 \
    --feature-ransac-residual-m 0.70 \
    --feature-compute-backend cuda \
    --subgrid-reflectivity-amplitude 0.18 \
    --pose-graph-backend gtsam-fixed-lag-isam2 \
    --pose-graph-fixed-lag-s 180 \
    --pose-graph-isam-relinearize-threshold 0.01 \
    --pose-graph-isam-relinearize-skip 1 \
    --enable-terrain-cohort-particle-backend \
    --enable-global-cohort-tracking \
    --enable-bbs3d-global-verification \
    --terrain-global-yaw-window-deg 180 \
    --terrain-global-maximum-cohorts 48 \
    --terrain-global-particles-per-cohort 256 \
    --terrain-global-particle-resample-ess-fraction 0.50 \
    --terrain-global-particle-kernel-bandwidth 0.15 \
    --terrain-global-minimum-independent-evidence 3 \
    --terrain-global-cohort-confirmation-probability 0.97 \
    --terrain-global-cohort-confirmation-odds 20 \
    --terrain-particle-online-output graph \
    --enable-map-to-odom-soft-correction \
    "$@"
