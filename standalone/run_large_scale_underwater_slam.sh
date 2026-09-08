#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="/home/cgz/.conda/envs/isaacsim-6.0/bin/python"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Isaac Sim Python was not found at ${ISAAC_PYTHON}" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/isaacsim${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAAC_PYTHON}" "${SCRIPT_DIR}/large_scale_underwater_slam.py" "$@"
