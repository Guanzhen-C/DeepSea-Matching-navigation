#!/usr/bin/env bash
# Keep ROS Humble's Python 3.10 outside Isaac Sim's Python 3.12 process.
set -euo pipefail
if [[ "${1:-}" != "--side" || $# -lt 3 ]]; then
    echo 'Usage: run_ros.sh --side {pc|jetson} [ros_bridge.py options]' >&2
    exit 2
fi
HIL_SIDE="$2"
shift 2
HIL_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
export PYTHONPATH="${HIL_CODE_ROOT}:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=177 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${HIL_SIDE}" in
    pc)
        export FASTRTPS_DEFAULT_PROFILES_FILE="${HIL_CODE_ROOT}/standalone/hil/fastdds_pc.xml"
        HIL_PYTHON=/usr/bin/python3
        ;;
    jetson)
        export FASTRTPS_DEFAULT_PROFILES_FILE="${HIL_CODE_ROOT}/standalone/hil/fastdds_jetson.xml"
        HIL_DEP_ROOT=/media/cgz/data/oceansim_hil_20261021
        export OCEANSIM_BBS3D_LIBRARY="${HIL_DEP_ROOT}/build/bbs3d-bridge-arm/liboceansim_bbs3d.so"
        export LD_LIBRARY_PATH="${HIL_DEP_ROOT}/gtsam/lib:${LD_LIBRARY_PATH:-}"
        HIL_PYTHON="${HIL_DEP_ROOT}/venv/bin/python"
        ;;
    *) echo 'side must be pc or jetson' >&2; exit 2 ;;
esac
exec "${HIL_PYTHON}" -m standalone.hil.ros_bridge "$@"
