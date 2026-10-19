"""Radiometric contracts shared by acoustic map matchers and particle scoring."""
from __future__ import annotations

import numpy as np


def uses_relative_power(metadata: dict) -> bool:
    return (metadata.get("radiometry") or {}).get("encoding") == (
        "relative_linear_power_from_documented_db"
    )


def log_backscatter(
    values: np.ndarray, floor: float, *, relative_power: bool = False,
) -> np.ndarray:
    """Log with a gain-invariant floor for uncalibrated relative power.

    Legacy display-normalized maps retain their existing absolute floor. A
    dB-derived power map has arbitrary constant gain, so an absolute 0.01
    threshold would flatten otherwise identical patches at different gains.
    The relative floor is a fraction of the positive median on this support.
    """
    values = np.asarray(values, dtype=np.float64)
    threshold = float(floor)
    if relative_power:
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size:
            threshold *= float(np.median(positive))
        threshold = max(threshold, np.finfo(np.float64).tiny)
    return np.log(np.clip(values, threshold, None))
