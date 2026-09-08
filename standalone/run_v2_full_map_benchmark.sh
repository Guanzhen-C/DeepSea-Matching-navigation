#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fast, non-visual execution of the same approximately 72.35 km 3-D traversal.
# The high time scale only reduces frame/render overhead; sensor events remain
# scheduled at their configured simulated-time rates.  Trailing arguments win.
exec "${SCRIPT_DIR}/run_v2_full_map_slam_isaac.sh" \
    --headless \
    --time-scale 3600 \
    --terrain-display-stride 32 \
    --metrics "${SCRIPT_DIR}/../demo/large_scale_slam/full_map_metrics.json" \
    "$@"
