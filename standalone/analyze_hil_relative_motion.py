#!/usr/bin/env python3
"""PC-only fixed-time SE(3) relative motion audit of completed HIL trials.

Compare T_truth(i)^-1 T_truth(j) with T_est(i)^-1 T_est(j), without fitting
either trajectory or feeding truth to an estimator. A constant world-frame
gauge cancels; changing drift, scale and heading errors do not. Global output
is reported separately: map corrections can change its short-time motion and
must not be interpreted as the frontend odometry supplied to the backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from standalone.hil.score_board_results import stats
from standalone.plot_hil_results import load_run


def relative_errors(positions, rotations, lag):
    """All overlapping fixed-lag windows; channel zero is the PC-only truth."""
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if (positions.ndim != 3 or positions.shape[1:] != (4, 3)
            or rotations.shape != positions.shape[:2] + (3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise ValueError("finite truth/DR/Local/Global poses are required")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or not 0 < lag < len(positions):
        raise ValueError("relative-motion lag must be a positive integer within the trajectory")
    initial_transpose = np.swapaxes(rotations[:-lag], -1, -2)
    delta_translation = np.einsum("ncij,ncj->nci", initial_transpose, positions[lag:] - positions[:-lag])
    delta_rotation = initial_transpose @ rotations[lag:]
    translation = np.linalg.norm(delta_translation[:, 1:] - delta_translation[:, :1], axis=2)
    rotation_error = np.swapaxes(delta_rotation[:, :1], -1, -2) @ delta_rotation[:, 1:]
    cosine = np.clip((np.trace(rotation_error, axis1=-2, axis2=-1) - 1.) / 2., -1., 1.)
    return translation, np.rad2deg(np.arccos(cosine))


def summarize(values, unit):
    return {key.removesuffix("_m") + "_" + unit: value for key, value in stats(values).items()}


def analyze(run, windows_s=(10., 30., 60.)):
    timestamps = np.asarray(run["timestamps"], dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("at least two finite sensor timestamps required")
    differences = np.diff(timestamps)
    cadence = float(differences[0])
    if cadence <= 0. or not np.allclose(differences, cadence, rtol=0., atol=1e-8):
        raise ValueError("relative-motion scoring cannot bridge a sensor-clock gap")
    windows = tuple(float(w) for w in windows_s)
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("nonempty distinct relative-motion windows required")
    result = {}
    for duration in windows:
        if not np.isfinite(duration) or duration <= 0.:
            raise ValueError("relative-motion duration must be positive and finite")
        lag = int(round(duration / cadence))
        if not np.isclose(lag * cadence, duration, rtol=0., atol=1e-8) or not 0 < lag < len(timestamps):
            raise ValueError("relative-motion window must match sensor epochs and fit within the trial")
        translation, rotation = relative_errors(run["positions"], run["rotations"], lag)
        window_arc = run["distance"][lag:] - run["distance"][:-lag]
        channels = {}
        for index, name in enumerate(("dr", "local", "global")):
            maximum = int(np.argmax(translation[:, index]))
            channels[name] = dict(translation_m=summarize(translation[:, index], "m"),
                rotation_deg=summarize(rotation[:, index], "deg"),
                worst_translation_window=dict(start_time_s=float(timestamps[maximum]),
                    end_time_s=float(timestamps[maximum + lag]),
                    start_arc_m=float(run["distance"][maximum]),
                    end_arc_m=float(run["distance"][maximum + lag])))
        result[f"{duration:g}s"] = dict(duration_s=duration, lag_samples=lag,
            overlapping_window_count=len(translation),
            truth_window_arc_m=dict(min=float(window_arc.min()), median=float(np.median(window_arc)), max=float(window_arc.max())),
            channels=channels)
    return dict(metric="SE3 fixed-time relative pose error", windows=result,
        truth_evaluated_only_on_PC=True, trajectory_fit_applied=False,
        interpolated_epochs=False, overlapping_windows_are_not_independent_trials=True,
        global_motion_includes_map_corrections_not_frontend_odometry=True,
        full_route_acceptance=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON report path")
    parser.add_argument("--windows-s", type=float, nargs="+", default=[10., 30., 60.])
    parser.add_argument("--require-full-route", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh report path required")
    root = args.run.resolve()
    run = load_run(root, include_rotations=True, require_full_route=args.require_full_route)
    report = analyze(run, args.windows_s)
    report.update(profile=run["manifest"]["profile"], full_route=run["full_route"],
        samples=len(run["timestamps"]), distance_m=float(run["distance"][-1]),
        analysis_source_sha256=file_sha256(Path(__file__).resolve()),
        source_mode=run["manifest"]["mode"], live_Isaac=run["manifest"].get("live_Isaac", False),
        absolute_position_errors_m={name: stats(run["errors"][:, index])
                                   for index, name in enumerate(("dr", "local", "global"))},
        input_sha256={str(path): file_sha256(path) for path in (
            root / "launch_manifest.json", run["reference"], run["outputs"])})
    # Exclusive creation prevents an earlier experiment report being replaced.
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
