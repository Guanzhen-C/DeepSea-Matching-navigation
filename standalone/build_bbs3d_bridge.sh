#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${ROOT}/downloads/3d_bbs_upstream"
UPSTREAM_BUILD="${UPSTREAM}/build"
BRIDGE_BUILD="${ROOT}/native/bbs3d_bridge/build"
PINNED_COMMIT="41529a34a2fb9618b5ff560fb3c2363f1615666d"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  git clone https://github.com/KOKIAOKI/3d_bbs.git "${UPSTREAM}"
fi

CURRENT_COMMIT="$(git -C "${UPSTREAM}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  git -C "${UPSTREAM}" fetch origin "${PINNED_COMMIT}"
  git -C "${UPSTREAM}" checkout --detach "${PINNED_COMMIT}"
fi

/usr/bin/cmake \
  -S "${UPSTREAM}" \
  -B "${UPSTREAM_BUILD}" \
  -DBUILD_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build "${UPSTREAM_BUILD}" --parallel

/usr/bin/cmake \
  -S "${ROOT}/native/bbs3d_bridge" \
  -B "${BRIDGE_BUILD}" \
  -DBBS3D_SOURCE_DIR="${UPSTREAM}" \
  -DBBS3D_BUILD_DIR="${UPSTREAM_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build "${BRIDGE_BUILD}" --parallel

echo "Built ${BRIDGE_BUILD}/liboceansim_bbs3d.so"
