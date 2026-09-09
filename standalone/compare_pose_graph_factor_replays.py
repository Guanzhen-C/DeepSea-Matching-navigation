#!/usr/bin/env python3
"""Compare NumPy, full-window GTSAM, and fixed-lag factor replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("numpy_batch", type=Path)
    parser.add_argument("gtsam_full_window", type=Path)
    parser.add_argument("gtsam_fixed_lag", type=Path)
    parser.add_argument("factor_stream", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _pose_array(payload: dict[str, Any], label: str) -> np.ndarray:
    poses = np.asarray(payload.get("final_poses", []), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1:] != (3,) or poses.shape[0] == 0:
        raise ValueError(f"{label} does not contain an N x 3 pose array")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label} contains non-finite poses")
    return poses


def _pose_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("replayed pose arrays have different shapes")
    translation = np.linalg.norm(first[:, :2] - second[:, :2], axis=1)
    yaw = np.abs(
        (first[:, 2] - second[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    )
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "translation_max_m": float(np.max(translation)),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw**2)))),
        "yaw_max_deg": float(np.rad2deg(np.max(yaw))),
    }


def _solver_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "backend",
        "poses",
        "edges",
        "priors",
        "optimization_runs",
        "initial_cost",
        "final_cost",
        "finite",
        "maximum_absolute_delta_from_logged_online_pose",
        "fixed_lag_active_variables",
        "fixed_lag_marginalized_variables",
        "fixed_lag_global_only_factors",
    )
    return {field: payload.get(field) for field in fields}


def compare_replays(
    numpy_batch: dict[str, Any],
    gtsam_full_window: dict[str, Any],
    gtsam_fixed_lag: dict[str, Any],
    factor_stream: dict[str, Any],
) -> dict[str, Any]:
    """Build a solver-equivalence report from one final factor snapshot."""

    replay_payloads = {
        "numpy_batch": numpy_batch,
        "gtsam_full_window": gtsam_full_window,
        "gtsam_fixed_lag": gtsam_fixed_lag,
    }
    poses = {
        name: _pose_array(payload, name)
        for name, payload in replay_payloads.items()
    }
    common_counts = {
        field: {name: int(payload.get(field, -1)) for name, payload in replay_payloads.items()}
        for field in ("poses", "edges", "priors")
    }
    counts_identical = all(
        len(set(values.values())) == 1 for values in common_counts.values()
    )
    all_finite = all(bool(payload.get("finite")) for payload in replay_payloads.values())

    full_vs_numpy = _pose_difference(
        poses["gtsam_full_window"], poses["numpy_batch"]
    )
    numpy_cost = float(numpy_batch["final_cost"])
    full_cost = float(gtsam_full_window["final_cost"])
    cost_delta = abs(full_cost - numpy_cost)
    cost_limit = max(0.02, 1.0e-5 * max(abs(numpy_cost), 1.0))
    checks = {
        "factor_counts_identical": counts_identical,
        "all_replays_finite": all_finite,
        "full_window_translation_max_le_0p02m": (
            full_vs_numpy["translation_max_m"] <= 0.02
        ),
        "full_window_yaw_max_le_0p02deg": (
            full_vs_numpy["yaw_max_deg"] <= 0.02
        ),
        "full_window_cost_equivalent": cost_delta <= cost_limit,
        "full_window_retains_all_variables": (
            int(gtsam_full_window.get("fixed_lag_active_variables", 0))
            == poses["gtsam_full_window"].shape[0]
            and int(gtsam_full_window.get("fixed_lag_marginalized_variables", -1))
            == 0
        ),
        "fixed_lag_actually_marginalizes": (
            int(gtsam_fixed_lag.get("fixed_lag_active_variables", 0)) > 0
            and int(gtsam_fixed_lag.get("fixed_lag_marginalized_variables", 0)) > 0
        ),
    }

    differences: dict[str, Any] = {
        "gtsam_full_window_vs_numpy_batch": full_vs_numpy,
        "gtsam_fixed_lag_vs_gtsam_full_window": _pose_difference(
            poses["gtsam_fixed_lag"], poses["gtsam_full_window"]
        ),
    }
    logged_online = np.asarray(
        factor_stream.get("final_online_poses", []), dtype=np.float64
    )
    if logged_online.shape == poses["gtsam_fixed_lag"].shape:
        differences["gtsam_fixed_lag_rebuild_vs_logged_online"] = (
            _pose_difference(poses["gtsam_fixed_lag"], logged_online)
        )
    logged_global = np.asarray(
        factor_stream.get("full_history_diagnostic_poses", []),
        dtype=np.float64,
    )
    if logged_global.shape == poses["gtsam_full_window"].shape:
        differences["gtsam_full_window_rebuild_vs_logged_global"] = (
            _pose_difference(poses["gtsam_full_window"], logged_global)
        )
        differences["numpy_batch_vs_logged_global"] = _pose_difference(
            poses["numpy_batch"], logged_global
        )

    return {
        "schema": "oceansim.pose_graph_factor_replay_comparison.v1",
        "overall_verdict": "pass" if all(checks.values()) else "fail",
        "scope": (
            "solver comparison on the same final factor snapshot; this is "
            "not an exact replay of online factor arrival or in-place factor "
            "replacement timing"
        ),
        "checks": checks,
        "factor_counts": common_counts,
        "solver_summaries": {
            name: _solver_summary(payload)
            for name, payload in replay_payloads.items()
        },
        "full_window_cost_comparison": {
            "numpy_batch": numpy_cost,
            "gtsam_full_window": full_cost,
            "absolute_delta": cost_delta,
            "acceptance_limit": cost_limit,
        },
        "pose_differences": differences,
    }


def _markdown(report: dict[str, Any]) -> str:
    difference = report["pose_differences"][
        "gtsam_full_window_vs_numpy_batch"
    ]
    cost = report["full_window_cost_comparison"]
    lines = [
        "# Pose-graph factor replay comparison",
        "",
        f"- Verdict: `{report['overall_verdict']}`",
        f"- Scope: {report['scope']}",
        "",
        "## Full-window solver equivalence",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Translation RMSE | {difference['translation_rmse_m']:.9g} m |",
        f"| Maximum translation difference | {difference['translation_max_m']:.9g} m |",
        f"| Yaw RMSE | {difference['yaw_rmse_deg']:.9g} deg |",
        f"| Maximum yaw difference | {difference['yaw_max_deg']:.9g} deg |",
        f"| NumPy final cost | {cost['numpy_batch']:.9g} |",
        f"| GTSAM full-window final cost | {cost['gtsam_full_window']:.9g} |",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(passed).lower()}`"
        for name, passed in report["checks"].items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _arguments()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    report = compare_replays(
        load(args.numpy_batch),
        load(args.gtsam_full_window),
        load(args.gtsam_fixed_lag),
        load(args.factor_stream),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["overall_verdict"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
