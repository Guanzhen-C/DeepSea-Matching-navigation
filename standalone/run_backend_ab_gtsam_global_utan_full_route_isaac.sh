#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build_bbs3d_bridge.sh"

# Controlled full-route candidate for the Global-UTAN-inspired recovery path.
# This is an OceanSim implementation of the published architectural ideas,
# not the authors' unavailable reference implementation.  The sensor model,
# route, seed, exact initial state, local estimator and GTSAM backend remain
# identical to the fixed-lag control.  The candidate places persistent
# particle cohorts, an explicit null state, raw MBES/side-scan map
# likelihoods, and independent-window confirmation in front of both ordinary
# terrain factors and whole-map recovery.  A deterministic basin-local raw-
# likelihood MAP/Laplace approximation supplies the current acquisition's
# rank-aware information to GTSAM even when the compact matcher has no
# proposal; the recursive PF posterior is never reused as a unary factor.
# Provenance-labelled persistent proposals cannot be overwritten by an
# unlabelled whole-map alias during association.  Ordinary
# terrain priors retain the baseline Huber loss; only a cohort-confirmed,
# re-seeded global relocalization uses the separate Cauchy policy.
exec "${SCRIPT_DIR}/run_backend_ab_gtsam_full_route_isaac.sh" \
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
    --enable-map-to-odom-soft-correction \
    --metrics /tmp/oceansim_gtsam_global_utan_full_route_truth_ui.json \
    "$@"
