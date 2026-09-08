#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The route lies wholly inside measured bathymetry and measured backscatter,
# with more than 250 m clearance from the joint NoData boundary.  It covers a
# 2.42 x 1.81 km envelope and follows roughly 7.94 km per lap.
exec "${SCRIPT_DIR}/run_large_scale_slam_isaac.sh" \
    --sonar-profile v2-standard \
    --ping-hz 2.5 \
    --survey-speed-mps 1.5 \
    --survey-center-x-m -3500 \
    --survey-center-y-m -1500 \
    --survey-x-amplitude-m 1200 \
    --survey-y-amplitude-m 700 \
    --survey-pattern-yaw-deg 165 \
    --survey-altitude-m 28 \
    --association-map-half-extent-m 275 \
    --keyframe-distance-m 25 \
    --keyframe-max-interval-s 20 \
    --time-scale 4 \
    --camera-mode chase \
    --metrics "${SCRIPT_DIR}/../demo/large_scale_slam/v2_large_area_live_metrics.json" \
    "$@"
