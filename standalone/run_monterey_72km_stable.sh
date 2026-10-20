#!/usr/bin/env bash
set -euo pipefail

# Frozen historical production code, not the Chelan experimental worktree.
# Its original multimodal online-output policy is intentionally preserved.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STABLE_DIR="/home/cgz/OceanSim_worktrees/monterey-72km-stable-20260918"
EXPECTED_SHA="8b5df53cb57f9de2506d78d90b16e33bf5e96510"
if [[ ! -d "${STABLE_DIR}" ]] || [[ "$(git -C "${STABLE_DIR}" rev-parse HEAD)" != "${EXPECTED_SHA}" ]]; then
    echo "The frozen 72 km worktree is missing or has a different revision." >&2
    exit 1
fi
if [[ -n "$(git -C "${STABLE_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "The frozen 72 km tracked source was modified; refusing to run." >&2
    exit 1
fi
export OCEANSIM_BBS3D_LIBRARY="${ROOT_DIR}/native/bbs3d_bridge/build/liboceansim_bbs3d.so"
exec bash "${STABLE_DIR}/standalone/run_v2_full_map_slam_isaac.sh" \
    --bathymetry "${ROOT_DIR}/data/monterey_canyon_full_5m" \
    --survey-route "${ROOT_DIR}/data/monterey_canyon_full_5m/full_map_traversal_route.json" \
    --metrics "${ROOT_DIR}/demo/large_scale_slam/monterey_72km_frozen_metrics.json" \
    "$@"
