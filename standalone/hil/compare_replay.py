#!/usr/bin/env python3
"""PC-only native parity assessment. Never copy host truth to the estimator."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from standalone.hil.recording import read_packets


def compare(reference, outputs, position_atol_m=1e-7, rotation_atol=1e-9):
    maximum = {k: 0. for k in ("dr_position", "local_position", "global_position",
                               "dr_rotation", "local_rotation", "global_rotation")}
    squared = {k: 0. for k in ("dr", "local", "global")}
    samples = 0
    first_mismatch = None
    for a, b in itertools.zip_longest(read_packets(reference), read_packets(outputs)):
        if a is None or b is None or abs(a["timestamp"] - b["timestamp"]) > 1e-9:
            raise ValueError("reference/output sample loss or time misalignment")
        for key in maximum:
            difference = float(np.max(np.abs(a[key] - b[key])))
            if not np.isfinite(difference):
                raise ValueError("nonfinite estimator result")
            maximum[key] = max(maximum[key], difference)
            limit = position_atol_m if key.endswith("position") else rotation_atol
            if difference > limit and first_mismatch is None:
                first_mismatch = dict(timestamp=a["timestamp"], field=key, max_abs_difference=difference)
        for label in squared:
            squared[label] += float(np.sum((b[label + "_position"] - a["truth_position"])**2))
        samples += 1
    if not samples:
        raise ValueError("no samples to compare")
    return dict(native_parity_passed=first_mismatch is None, samples=samples, first_mismatch=first_mismatch,
                maximum_component_differences=maximum,
                tolerances=dict(position_m=position_atol_m, rotation_matrix=rotation_atol),
                rmse_3d_m={k: float(np.sqrt(v / samples)) for k, v in squared.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--outputs", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    if args.report.exists():
        p.error("fresh report path required")
    report = compare(args.reference, args.outputs)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not report["native_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
