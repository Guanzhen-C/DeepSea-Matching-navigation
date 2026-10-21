#!/usr/bin/env bash
# Invoke in a memory-limited user unit; build/install only in the HIL NVMe tree.
set -euo pipefail
TASK_ROOT="/media/cgz/data/oceansim_hil_20261021"
SOURCE_COMMIT="835f791f72dc7710dac5a453e2d60ec60d705659"
SOURCE_ROOT="${TASK_ROOT}/build/gtsam-${SOURCE_COMMIT}"
export TMPDIR="${TASK_ROOT}/tmp"
export PIP_CACHE_DIR="${TASK_ROOT}/pip-cache"
test "$(uname -m)" = aarch64
test -x "${TASK_ROOT}/venv/bin/python"
test -f "${TASK_ROOT}/wheels/gtsam-${SOURCE_COMMIT}.tar.gz"
test "$(df --output=avail -k "${TASK_ROOT}" | tail -1)" -gt 12582912
if [[ ! -f "${SOURCE_ROOT}/CMakeLists.txt" ]]; then
    tar -xzf "${TASK_ROOT}/wheels/gtsam-${SOURCE_COMMIT}.tar.gz" -C "${TASK_ROOT}/build"
fi
cmake -S "${SOURCE_ROOT}" -B "${TASK_ROOT}/build/gtsam-arm" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${TASK_ROOT}/gtsam" \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=ON \
    -DGTSAM_WITH_TBB=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_BUILD_PYTHON=ON \
    -DPYTHON_EXECUTABLE="${TASK_ROOT}/venv/bin/python" \
    -DGTSAM_PYTHON_VERSION=3.10
# Binding translation units can each exceed 8 GiB at -O3 on ARM. Compile
# sequentially; this preserves native optimization and does not change math.
cmake --build "${TASK_ROOT}/build/gtsam-arm" --parallel 1
cmake --install "${TASK_ROOT}/build/gtsam-arm"
cd "${TASK_ROOT}/build/gtsam-arm/python"
"${TASK_ROOT}/venv/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation .
"${TASK_ROOT}/venv/bin/python" -c 'import gtsam, gtsam_unstable; assert hasattr(gtsam_unstable, "IncrementalFixedLagSmoother"); print("GTSAM fixed-lag import OK")'
