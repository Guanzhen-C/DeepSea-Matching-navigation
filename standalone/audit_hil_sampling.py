#!/usr/bin/env python3
"""Read-only numerical portability audit of the frozen Gaussian sampler."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh numerical audit output required")
    sys.path.insert(0, str(args.bundle / "isaacsim"))
    from oceansim.slam.global_tracking import TerrainNavigationFilter
    covariance = np.load(args.inputs, allow_pickle=False)["covariance"]
    draws, vectors, eigenvalues, principal_draws, normals = [], [], [], [], []
    for value in covariance:
        symmetric = .5 * (value + value.T)
        w, v = np.linalg.eigh(symmetric)
        standard = np.random.default_rng(31).standard_normal((256, 3))
        standard -= standard.mean(axis=0, keepdims=True)
        # Exercise the ORIGINAL method, without changing any algorithm object.
        raw = TerrainNavigationFilter._draw_centered_gaussian(
            SimpleNamespace(_rng=np.random.default_rng(31)), value, 256)
        principal_root = (v * np.sqrt(np.maximum(w, 0.))) @ v.T
        draws.append(raw)
        vectors.append(v)
        eigenvalues.append(w)
        normals.append(standard)
        principal_draws.append(standard @ principal_root.T)
    np.savez_compressed(args.output, original_draws=np.array(draws), eigenvectors=np.array(vectors),
                        eigenvalues=np.array(eigenvalues), standard_normals=np.array(normals),
                        diagnostic_principal_draws=np.array(principal_draws),
                        numpy_version=np.array(np.__version__), architecture=np.array(platform.machine()))
    print("numerical sampler audit", np.__version__, platform.machine(), args.output)


if __name__ == "__main__":
    main()
