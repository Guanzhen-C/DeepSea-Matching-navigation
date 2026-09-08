#!/usr/bin/env python3
"""Validate and durably archive one completed backend A/B simulation run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--command", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_related(
    metrics_path: Path,
    payload: dict[str, Any],
    payload_key: str,
    sibling_suffix: str,
    *,
    required: bool,
) -> Path | None:
    recorded = payload.get(payload_key)
    candidates = []
    if recorded:
        candidates.append(Path(recorded))
    candidates.append(
        metrics_path.with_name(f"{metrics_path.stem}{sibling_suffix}")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(
            f"missing {payload_key}: checked "
            + ", ".join(str(path) for path in candidates)
        )
    return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    metrics_path = args.metrics.resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("run_status") != "completed":
        raise RuntimeError(
            f"refusing to archive non-completed run: {payload.get('run_status')!r}"
        )
    if payload.get("completed_requested_route") is not True:
        raise RuntimeError("run did not complete the requested route")

    position_csv = _resolve_related(
        metrics_path,
        payload,
        "position_error_csv",
        "_position_errors.csv",
        required=True,
    )
    progress = _resolve_related(
        metrics_path,
        payload,
        "progress_file",
        "_progress.json",
        required=False,
    )
    factor_log = _resolve_related(
        metrics_path,
        payload,
        "pose_graph_factor_log",
        "_pose_graph_factors.json",
        required=False,
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        "metrics": output / f"{args.stem}.json",
        "position_errors": output / f"{args.stem}_position_errors.csv",
        "progress": output / f"{args.stem}_progress.json",
        "pose_graph_factors": output / f"{args.stem}_pose_graph_factors.json",
        "manifest": output / f"{args.stem}_manifest.json",
    }
    material_destinations = [
        destinations["metrics"],
        destinations["position_errors"],
        destinations["manifest"],
    ]
    if progress is not None:
        material_destinations.append(destinations["progress"])
    if factor_log is not None:
        material_destinations.append(destinations["pose_graph_factors"])
    existing = [path for path in material_destinations if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "archive targets already exist: "
            + ", ".join(str(path) for path in existing)
        )

    assert position_csv is not None
    shutil.copy2(position_csv, destinations["position_errors"])
    if progress is not None:
        shutil.copy2(progress, destinations["progress"])
    if factor_log is not None:
        shutil.copy2(factor_log, destinations["pose_graph_factors"])

    archived_metrics = dict(payload)
    archived_metrics["position_error_csv"] = destinations[
        "position_errors"
    ].name
    if factor_log is not None:
        archived_metrics["pose_graph_factor_log"] = destinations[
            "pose_graph_factors"
        ].name
    archived_metrics["archive_source_metrics"] = str(metrics_path)
    archived_metrics["archive_source_revision"] = args.source_revision
    _write_json_atomic(destinations["metrics"], archived_metrics)

    artifact_paths = {
        "metrics": destinations["metrics"],
        "position_errors": destinations["position_errors"],
    }
    if progress is not None:
        artifact_paths["progress"] = destinations["progress"]
    if factor_log is not None:
        artifact_paths["pose_graph_factors"] = destinations[
            "pose_graph_factors"
        ]
    manifest = {
        "schema": "oceansim.backend_ab_run_archive.v1",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "stem": args.stem,
        "source_revision": args.source_revision,
        "source_metrics": str(metrics_path),
        "source_metrics_sha256": _sha256(metrics_path),
        "launch_command": args.command,
        "run_status": payload["run_status"],
        "termination_reason": payload.get("termination_reason"),
        "completed_requested_route": payload["completed_requested_route"],
        "survey_travelled_distance_m": payload.get(
            "survey_travelled_distance_m"
        ),
        "artifacts": {
            name: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    _write_json_atomic(destinations["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
