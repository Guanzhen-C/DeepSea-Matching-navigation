#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {chelan|jenkinson|wallops|sheboygan} [OceanSim options...]" >&2
    exit 2
fi

ALIAS="$1"
shift
case "${ALIAS}" in
    chelan)
        DATASET_ID="lake_chelan_usgs_2023_3m"
        ;;
    jenkinson)
        DATASET_ID="jenkinson_lake_usgs_2023"
        ;;
    wallops)
        DATASET_ID="wallops_assawoman_usgs_2024_1m"
        ;;
    sheboygan)
        DATASET_ID="lake_michigan_sheboygan_noaa_2018"
        ;;
    *)
        echo "Unknown dataset '${ALIAS}'. Choose chelan, jenkinson, wallops, or sheboygan." >&2
        exit 2
        ;;
esac

MAP_DIR="${ROOT_DIR}/data/underwater_terrain_datasets/${DATASET_ID}/prepared"
ROUTE="${MAP_DIR}/full_map_traversal_route.json"
if [[ ! -f "${MAP_DIR}/metadata.json" || ! -f "${ROUTE}" ]]; then
    echo "Prepared map is missing for '${ALIAS}'. Run:" >&2
    echo "  python ${SCRIPT_DIR}/prepare_public_terrain_datasets.py --dataset ${ALIAS}" >&2
    exit 1
fi

# The production launcher retains the validated CUDA feature-sonar front end,
# cohort particle terrain tracker, GTSAM fixed-lag backend, and bounded
# map-to-odom correction.  The options below only substitute the measured map,
# its verified in-coverage route, the fresh start point, and output file.
exec "${SCRIPT_DIR}/run_v2_full_map_slam_isaac.sh" \
    --bathymetry "${MAP_DIR}" \
    --survey-pattern full-map-route \
    --survey-route "${ROUTE}" \
    --survey-laps 1.0 \
    --survey-start-distance-m 0 \
    --survey-segment-distance-m 0 \
    --metrics "${ROOT_DIR}/demo/large_scale_slam/public_${ALIAS}_metrics.json" \
    "$@"
