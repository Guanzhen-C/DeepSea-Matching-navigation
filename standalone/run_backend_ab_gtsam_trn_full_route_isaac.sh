#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preferred entry point for the independently implemented operational TRN
# backend.  It retains the Fast-LIVO2-style acoustic/inertial front end and
# fixed-lag GTSAM graph, then runs raw MBES/side-scan likelihoods through one
# jointly normalized multimodal point-mass/particle filter.  The legacy
# Global-UTAN launcher name remains available for reproducible old commands.
exec "${SCRIPT_DIR}/run_backend_ab_gtsam_global_utan_full_route_isaac.sh" \
    --metrics /tmp/oceansim_gtsam_trn_full_route_truth_ui.json \
    "$@"
