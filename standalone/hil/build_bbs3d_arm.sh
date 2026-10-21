#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/media/cgz/data/oceansim_hil_20261021"
UPSTREAM="${TASK_ROOT}/build/3d_bbs_upstream"
export PATH="/usr/local/cuda/bin:${PATH}"
export TMPDIR="${TASK_ROOT}/tmp"
test "$(uname -m)" = aarch64
test -f "${TASK_ROOT}/wheels/bbs3d-pinned-source.tar.gz"
if [[ ! -f "${UPSTREAM}/CMakeLists.txt" ]]; then
    tar -xzf "${TASK_ROOT}/wheels/bbs3d-pinned-source.tar.gz" -C "${TASK_ROOT}/build"
fi
cmake -S "${UPSTREAM}" -B "${TASK_ROOT}/build/bbs3d-arm" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    '-DCUDA_NVCC_FLAGS=-gencode=arch=compute_87,code=sm_87'
cmake --build "${TASK_ROOT}/build/bbs3d-arm" --parallel 2
cmake -S "${TASK_ROOT}/build/bbs3d_bridge" -B "${TASK_ROOT}/build/bbs3d-bridge-arm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBBS3D_SOURCE_DIR="${UPSTREAM}" -DBBS3D_BUILD_DIR="${TASK_ROOT}/build/bbs3d-arm" \
    -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build "${TASK_ROOT}/build/bbs3d-bridge-arm" --parallel 2
file "${TASK_ROOT}/build/bbs3d-bridge-arm/liboceansim_bbs3d.so"
