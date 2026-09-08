#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Candidate half of the controlled backend A/B experiment. All sensor,
# front-end, map-matcher, route, rendering, and random-seed settings come from
# the same V2 full-route launcher used by the NumPy baseline. Arguments below
# change only the graph solver, make the tested initialization explicit, and
# select an independent output file.
exec "${SCRIPT_DIR}/run_v2_full_map_slam_isaac.sh" \
    --seed 31 \
    --initialize-from-truth \
    --initial-position-offset-x-m 0 \
    --initial-position-offset-y-m 0 \
    --initial-yaw-offset-deg 0 \
    --pose-graph-backend gtsam-fixed-lag-isam2 \
    --pose-graph-fixed-lag-s 180 \
    --pose-graph-isam-relinearize-threshold 0.01 \
    --pose-graph-isam-relinearize-skip 1 \
    --metrics /tmp/oceansim_gtsam_fixed_lag_full_route_truth_ui.json \
    "$@"
