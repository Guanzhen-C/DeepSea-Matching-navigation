#!/usr/bin/env bash
set -euo pipefail

readonly ISAACSIM_PYTHON="${OCEANSIM_ISAAC_PYTHON:-/home/cgz/.conda/envs/isaacsim-6.0/bin/python}"
readonly DEMO_SCRIPT="/home/cgz/OceanSim/standalone/acoustic_slam_isaac_demo.py"

if [[ ! -x "${ISAACSIM_PYTHON}" ]]; then
    echo "Isaac Sim Python environment is missing: ${ISAACSIM_PYTHON}" >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:0}"
export PYTHONNOUSERSITE=1

# This workstation normally sources a ROS/Gazebo workspace in every shell.
# Keep its Python 3.10 packages and CUDA libraries out of Isaac's Python 3.12
# process; the live demo itself does not require the ROS bridge.
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH ROS_DISTRO ROS_VERSION

exec "${ISAACSIM_PYTHON}" "${DEMO_SCRIPT}" "$@"
