#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Revalidation run after adding fail-closed whole-map relocalization.  The
# underlying controlled launcher retains the same route, sensors, random seed,
# exact initial state, UI settings, and GTSAM fixed-lag backend as the original
# candidate.  Only the recovery implementation and output path differ.
exec "${SCRIPT_DIR}/run_backend_ab_gtsam_full_route_isaac.sh" \
    --metrics /tmp/oceansim_gtsam_fixed_lag_relocalized_full_route_truth_ui.json \
    "$@"
