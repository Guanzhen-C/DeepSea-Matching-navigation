#!/usr/bin/env python3
"""PC-only aligned truth scoring for cross-architecture HIL outputs.

Do not require bitwise CPU/GPU equality across architectures. Report all tail
statistics, and explicitly keep full-route acceptance separate from a short
replay. No metric or truth is fed back into either estimator.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from standalone.hil.recording import read_packets


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(rmse_m=float(np.sqrt(np.mean(a*a))), median_m=float(np.median(a)),
                p95_m=float(np.percentile(a, 95)), p99_m=float(np.percentile(a, 99)),
                max_m=float(a.max()), final_m=float(a[-1]))


def score(reference, outputs):
    errors = {k: [] for k in ("dr", "local", "global")}
    native_errors = {k: [] for k in errors}
    stamps = []
    maximum_position_difference = {k: 0. for k in errors}
    for native, board in itertools.zip_longest(read_packets(reference), read_packets(outputs)):
        if native is None or board is None or abs(native["timestamp"] - board["timestamp"]) > 1e-9:
            raise ValueError("missing or misaligned native/board sample")
        for name in errors:
            truth = native["truth_position"]
            position = board[name + "_position"]
            if not np.isfinite(position).all():
                raise ValueError("nonfinite board output")
            errors[name].append(np.linalg.norm(position - truth))
            native_errors[name].append(np.linalg.norm(native[name + "_position"] - truth))
            maximum_position_difference[name] = max(maximum_position_difference[name],
                float(np.linalg.norm(native[name + "_position"] - position)))
        stamps.append(native["timestamp"])
    if not stamps or (len(stamps) > 1 and not np.allclose(np.diff(stamps), .2, atol=1e-8, rtol=0.)):
        raise ValueError("empty or non-5-Hz output stream")
    board_stats = {k: stats(v) for k, v in errors.items()}
    native_stats = {k: stats(v) for k, v in native_errors.items()}
    return dict(samples=len(stamps), end_timestamp_s=stamps[-1], board=board_stats, native=native_stats,
                maximum_board_to_native_position_difference_m=maximum_position_difference,
                RMSE_P95_relative_changes={k: {metric: board_stats[k][metric] / max(native_stats[k][metric], 1e-12) - 1.
                    for metric in ("rmse_m", "p95_m")} for k in errors},
                full_route_acceptance=False,
                interpretation="aligned cross-architecture measurements only; confirm mission completion and resource/latency evidence separately")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        p.error("fresh report path required")
    report = score(args.reference, args.outputs)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
