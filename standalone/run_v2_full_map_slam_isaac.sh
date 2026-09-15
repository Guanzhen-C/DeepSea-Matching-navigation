#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This is a closed, approximately 72.35 km 3-D traversal of the largest connected
# Monterey Canyon region having both measured bathymetry and backscatter.  It
# spans about 22.75 x 16.38 km, but is not exhaustive lawnmower sonar coverage.
# At 1.5 m/s and 5 pings/s, consecutive integrated bathymetry/side-scan
# acquisitions advance 0.30 m.  The documented 1.2 m standard-mode footprint
# therefore overlaps by 0.90 m (75%); six of its eight 0.15 m image rows
# coincide spatially with the next ping.  Two distinguishable pings in water
# are an explicit scheduling assumption required for 5 Hz at 250 m range.
#
# Defaults preserve every sensor and estimator event while accelerating only
# presentation.  The event loop remains at IMU/DVL/depth/sonar timestamps;
# operator images are capped at 2 simulated Hz and only the latest ping before
# a rendered frame is uploaded.  The raw waterfall still accumulates every
# ping, while each presented image advances by at most 0.75 m at 1.5 m/s.
# Each ping already contains eight simultaneous along-track rows per side.
# The production front end therefore registers exactly one physical ping once
# at the native 5 Hz rate.  A 30 s causal fixed-lag surfel map supplies about
# 45 m of historical along-track aperture; the current ping enters that map
# only after its update and is removed when it leaves the lag.  Adjacent
# 1.2 m footprints overlap by 75%, so their full residuals are retained while
# information is continuously weighted by the newly covered fraction.  The
# single-ping MBES correction writes position only; yaw is intentionally left
# to the inertial propagation and the multi-ping Global matcher.  The direct
# intensity residual excludes the latest four pings from its reference map:
# at 1.5 m/s this supplies a 1.2 m causal baseline, exactly one documented
# standard-mode strip, so duplicated along-track rows cannot be reused as
# independent evidence. Every current ping still updates against the fixed-lag
# causal local map and every converged ping is written back. The 512 XYZI samples are
# correlated inside the 5 m source mosaic, so their aggregate filter weight is
# capped at 32 independent-equivalent samples.  A 120 s stable intensity-map
# anchor mirrors FAST-LIVO2's persistent local map instead of fragmenting it
# at every 30 s global keyframe.  These are continuous baseline/covariance/map
# parameters; no extra acceptance threshold is introduced. One 30 s
# pose-graph interval contains 150 pings at 5 Hz. Retaining the
# complete interval supplies about 45 m of along-track terrain-match aperture;
# voxel centroids keep the numerical matcher budget fixed at 3072 points. The
# co-registered side-scan strip independently refines accepted bathymetric
# fixes. Accepted texture fixes provide the absolute planar pose; bathymetry
# remains the seed and the fallback when texture matching is rejected. This
# avoids carrying the measured low-pass bathymetry bias back into an accepted
# side-scan result through an unjustified independence assumption.
# Rare whole-map recovery searches a broad yaw interval around inertial yaw;
# ordinary local updates do not use that expensive absolute search.
# A 20x render time scale was measurably faster than both 8x and 30x on the
# validated UI workload, while retaining 50% more rendered frames than 30x.
# Append command-line arguments to override these presentation settings.
exec "${SCRIPT_DIR}/run_large_scale_slam_isaac.sh" \
    --survey-pattern full-map-route \
    --survey-route "${SCRIPT_DIR}/../data/monterey_canyon_full_5m/full_map_traversal_route.json" \
    --survey-laps 1.01 \
    --survey-speed-mps 1.5 \
    --survey-altitude-m 40 \
    --survey-maximum-pitch-deg 12 \
    --initial-position-offset-x-m 5 \
    --initial-position-offset-y-m -4 \
    --initial-yaw-offset-deg 1.5 \
    --sonar-profile v2-standard \
    --ping-hz 5 \
    --sonar-pings-in-water 2 \
    --sidescan-imaging-rays-per-side 256 \
    --sidescan-slam-stride 8 \
    --slam-voxel-size-m 5.0 \
    --intensity-voxel-size-m 3.0 \
    --map-insert-stride 2 \
    --mbes-group-pings 1 \
    --local-map-history-s 30 \
    --enable-mbes-footprint-overlap-weighting \
    --local-observability-mode continuous \
    --local-sonar-state-update-policy position \
    --local-intensity-surface-consistency-std-m 0.75 \
    --intensity-group-pings 1 \
    --intensity-min-group-span-m 1.0 \
    --intensity-reference-delay-pings 4 \
    --intensity-effective-independent-samples 32 \
    --intensity-submap-anchor-interval-s 120 \
    --intensity-factor-min-interval-s 300 \
    --intensity-factor-optimize-interval 20 \
    --disable-local-intensity-state-update \
    --trajectory-sample-interval-s 0.2 \
    --enable-terrain-map-matching \
    --enable-backscatter-map-matching \
    --backscatter-match-yaw-window-deg 2.5 \
    --backscatter-boundary-yaw-retry-window-deg 5 \
    --backscatter-ci-bathymetry-weight -1 \
    --cross-modal-large-correction-m 5 \
    --terrain-require-yaw-observable \
    --terrain-initialize-global-gauge \
    --terrain-gauge-confirmation-patches 2 \
    --terrain-confirm-every-prior \
    --terrain-gauge-minimum-baseline-m 40 \
    --terrain-gauge-maximum-position-residual-m 5 \
    --terrain-gauge-maximum-yaw-residual-deg 3 \
    --terrain-match-window-pings 150 \
    --terrain-patch-motion-source acoustic-predictive \
    --terrain-match-maximum-points 3072 \
    --terrain-prior-surface low-pass \
    --terrain-prior-yaw-std-floor-deg 3 \
    --terrain-match-translation-window-m 30 \
    --terrain-recovery-translation-window-m 100 \
    --terrain-recovery-after-rejections 3 \
    --terrain-emergency-recovery-translation-window-m 150 \
    --enable-global-terrain-relocalization \
    --terrain-global-window-pings 600 \
    --terrain-global-pyramid-stride 4 \
    --terrain-global-yaw-window-deg 90 \
    --terrain-global-yaw-step-deg 5 \
    --terrain-global-minimum-correlation 0.30 \
    --terrain-global-maximum-bathymetry-rmse-m 6 \
    --terrain-global-bathymetry-absolute-weight 0.50 \
    --terrain-global-minimum-score-gap 0.08 \
    --terrain-global-confirmation-position-tolerance-m 25 \
    --terrain-global-confirmation-yaw-tolerance-deg 5 \
    --terrain-match-yaw-window-deg 8 \
    --terrain-prior-optimize-interval 1 \
    --association-map-half-extent-m 275 \
    --sonar-terrain-stride 1 \
    --keyframe-distance-m 150 \
    --keyframe-max-interval-s 30 \
    --pose-graph-odometry-reference-interval-s 30 \
    --pose-graph-odometry-xy-std-m 1.0 \
    --time-scale 20 \
    --ui-sonar-refresh-hz 2 \
    --progress-distance-m 250 \
    --camera-mode overview \
    --metrics "${SCRIPT_DIR}/../demo/large_scale_slam/full_map_live_metrics.json" \
    "$@"
