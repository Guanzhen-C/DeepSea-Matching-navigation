#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="/home/cgz/.conda/envs/isaacsim-6.0/bin/python"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "Isaac Sim 6.0 Python was not found at ${ISAAC_PYTHON}" >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
exec "${ISAAC_PYTHON}" "${SCRIPT_DIR}/large_scale_slam_isaac_demo.py" "$@"
