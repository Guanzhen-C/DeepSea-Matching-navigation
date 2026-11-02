#!/usr/bin/env python3
"""Reproduce interpolation baselines on untouched 1 m Jenkinson bathymetry.

This is a controlled degradation experiment, not a learned SR evaluation.
All selected patches have complete measured support; invalid cells are never
filled. Patch selection and evaluation are deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "standalone"))
from prepare_public_terrain_datasets import _raster_info, _read_raster, _valid_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/project_reports/2026阶段研究/evidence/sr_interpolation_baseline.json")
    args = parser.parse_args()
    source = ROOT / "data/underwater_terrain_datasets/jenkinson_lake_usgs_2023/raw/JenkinsonLake_July_2023_all_1m_bathymetry_NAD83_NAVD88.tif"
    info = _raster_info(source)
    raster = _read_raster(info)
    valid = _valid_mask(raster, info.nodata)
    size, border = 128, 8
    candidates = []
    for row in range(0, raster.shape[0] - size + 1, size):
        for column in range(0, raster.shape[1] - size + 1, size):
            mask = valid[row:row + size, column:column + size]
            if not np.all(mask):
                continue
            tile = np.asarray(raster[row:row + size, column:column + size], dtype=np.float64)
            candidates.append((float(np.std(tile)), row, column))
    if len(candidates) < 4:
        raise RuntimeError("insufficient complete nonoverlapping measured patches")
    candidates.sort()
    selected = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(12, len(candidates)), dtype=int)]
    methods = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    rows = []
    pooled = {}
    for scale in (2, 4):
        for method, interpolation in methods.items():
            errors, slope_errors, consistency_errors = [], [], []
            for relief, row, column in selected:
                target = np.asarray(raster[row:row + size, column:column + size], dtype=np.float64)
                low = target.reshape(size // scale, scale, size // scale, scale).mean(axis=(1, 3))
                prediction = cv2.resize(low, (size, size), interpolation=interpolation)
                cut = (slice(border, -border), slice(border, -border))
                residual = (prediction - target)[cut]
                target_gradient = np.gradient(target, info.dx, abs(info.dy))
                predicted_gradient = np.gradient(prediction, info.dx, abs(info.dy))
                true_slope = np.degrees(np.arctan(np.hypot(*target_gradient)))
                predicted_slope = np.degrees(np.arctan(np.hypot(*predicted_gradient)))
                slope_delta = (predicted_slope - true_slope)[cut]
                reconstructed_low = prediction.reshape(size // scale, scale, size // scale, scale).mean(axis=(1, 3))
                low_border = border // scale
                consistency = (reconstructed_low - low)[low_border:-low_border, low_border:-low_border]
                errors.append(residual.reshape(-1))
                slope_errors.append(slope_delta.reshape(-1))
                consistency_errors.append(consistency.reshape(-1))
                rows.append({"scale": scale, "method": method, "row": row, "column": column, "relief_std_m": relief, "evaluated_pixels": int(residual.size), "rmse_m": float(np.sqrt(np.mean(residual ** 2))), "mae_m": float(np.mean(np.abs(residual))), "slope_rmse_deg": float(np.sqrt(np.mean(slope_delta ** 2)))})
            e = np.concatenate(errors)
            s = np.concatenate(slope_errors)
            c = np.concatenate(consistency_errors)
            pooled[f"x{scale}_{method}"] = {"patch_count": len(selected), "evaluated_pixels": int(e.size), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "mae_m": float(np.mean(np.abs(e))), "p95_absolute_error_m": float(np.percentile(np.abs(e), 95)), "maximum_absolute_error_m": float(np.max(np.abs(e))), "slope_rmse_deg": float(np.sqrt(np.mean(s ** 2))), "downsampling_consistency_rmse_m": float(np.sqrt(np.mean(c ** 2)))}
    output = {"recorded_on": "2026-10-02", "experiment_type": "controlled_block_mean_degradation_interpolation_only", "trained_model": False, "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "native_resolution_m": info.dx, "tile_size_pixels": size, "native_footprint_m": size * info.dx, "evaluation_border_excluded_pixels": border, "validity": "every source cell in every selected patch is measured and finite", "selection": "sort all nonoverlapping full-valid 128x128 patches by height standard deviation; select 12 evenly spaced indices", "eligible_patches": len(candidates), "degradation": "nonoverlapping s-by-s arithmetic block mean, no added noise", "registration": "OpenCV resize with block-centre alignment convention", "independent_survey": False, "statistical_independence_claim": False, "selected_patches": [{"row": r, "column": c, "height_std_m": v} for v, r, c in selected], "pooled_metrics": pooled, "per_patch_metrics": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pooled, ensure_ascii=False, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
