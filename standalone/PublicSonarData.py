"""Typed public sonar measurements: display RGB is never called linear echo.

No resizing, automatic clean labels, histogram matching, or source writes.
The physical branch predicts log-amplitude residuals; positive multiplicative
updates retain each original complex phase. RGB stays an image-domain task.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from InspectPublicSonarSignals import inspect_jsf


@dataclass(frozen=True)
class PublicSonarSample:
    sample_id: str
    domain: str
    values: np.ndarray
    provenance: dict[str, Any]

    def __post_init__(self):
        a = self.values
        if self.domain == "display_rgb":
            if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
                raise ValueError("display RGB must preserve native uint8 HWC pixels")
        elif self.domain == "envelope":
            if a.ndim != 2 or np.iscomplexobj(a) or (a < 0).any():
                raise ValueError("envelope must contain nonnegative native amplitudes")
        elif self.domain == "complex_field":
            if a.ndim != 2 or not np.iscomplexobj(a):
                raise ValueError("complex sonar field must preserve native complex values")
        else:
            raise ValueError("unreviewed measurement domain")
        if not np.isfinite(a).all() or min(a.shape[:2]) < 1:
            raise ValueError("empty or nonfinite public sonar sample")
        a.flags.writeable = False


def load_display(path, *, sample_id, source, group, license):
    path = Path(path)
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L", "P"):
            raise ValueError("do not silently downconvert high-bit-depth or complex images")
        native = np.array(image.convert("RGB"))
    return PublicSonarSample(sample_id, "display_rgb", native,
        dict(path=str(path.resolve()), source=source, group_id=group, license=license,
             native_shape=list(native.shape), resized=False, original_noise_unknown=True,
             is_linear_echo=False, is_clean_truth=False, domain_id=0))


def load_jsf_envelopes(path, *, source, group, license):
    """Decode only reviewed message-80 format 0; preserve every trace's scale.

    Stored envelope words are unsigned 16-bit. The per-trace block exponent
    is applied as 2**(-N), not normalized independently per ping. Instrument
    units are not calibrated seafloor reflectivity or clean ground truth.
    """
    path = Path(path)
    if path.stat().st_size > 128*1024**2:
        raise ValueError("use an incremental JSF reader for larger files")
    payload = path.read_bytes()
    report = inspect_jsf(payload)
    if not report["every_trace_is_envelope"]:
        raise ValueError("mixed/non-envelope JSF requires an explicit separate decoder")
    groups = {}
    for trace in report["traces"]:
        key = (trace["subsystem"], trace["channel"], trace["samples"])
        offset = trace["file_offset"]+16+240
        words = np.frombuffer(payload, dtype="<u2", count=trace["samples"], offset=offset)
        amplitude = np.ldexp(words.astype(np.float64), -trace["weighting_factor"])
        groups.setdefault(key, []).append((amplitude, trace))
    samples = []
    for (subsystem, channel, count), rows in sorted(groups.items()):
        values = np.stack([row[0] for row in rows])
        identity = f"{path.stem}:sub{subsystem}:ch{channel}:n{count}"
        samples.append(PublicSonarSample(identity, "envelope", values,
            dict(path=str(path.resolve()), source=source, group_id=group, license=license,
                 domain_id=1, native_shape=list(values.shape), resized=False,
                 is_linear_echo=True, is_clean_truth=False, original_noise_unknown=True,
                 sample_order="original trace order within subsystem/channel/count",
                 per_ping_scaling="unsigned16 * 2**(-weighting_factor)",
                 units="block-scaled instrument units; not calibrated reflectivity",
                 trace_metadata=[row[1] for row in rows], specification=report["specification"])))
    if sum(len(s.values) for s in samples) != report["trace_count"]:
        raise ValueError("not every original trace was retained")
    return samples


def load_complex_npy(path, *, sample_id, source, group, license):
    a = np.load(path, allow_pickle=False)
    return PublicSonarSample(sample_id, "complex_field", a,
        dict(path=str(Path(path).resolve()), source=source, group_id=group, license=license,
             native_shape=list(a.shape), resized=False, is_clean_truth=False,
             original_noise_unknown=True, domain_id=2, phase_preserved_on_input=True))


def encode(sample, *, amplitude_scale=1.):
    """Return native 3CHW network values and explicit physical support.

    A fixed scale must be declared/fitted on training sources only. No input
    clipping or per-test-image distribution fitting. Zeros remain protected.
    """
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0:
        raise ValueError("positive fixed training/instrument amplitude scale required")
    if sample.domain == "display_rgb":
        q = sample.values.transpose(2, 0, 1).astype(np.float32)/255.
        return q, np.ones(q.shape[1:], dtype=bool)
    amplitude = np.abs(sample.values) if sample.domain == "complex_field" else sample.values
    support = amplitude > 0
    q = np.zeros(amplitude.shape, np.float64)
    q[support] = np.log(amplitude[support]/amplitude_scale)
    if not np.isfinite(q).all():
        raise ValueError("physical amplitude encoding is nonfinite")
    return np.repeat(q[None].astype(np.float32), 3, axis=0), support


def decode(sample, prediction, *, amplitude_scale=1.):
    q, support = encode(sample, amplitude_scale=amplitude_scale)
    p = np.asarray(prediction, np.float64)
    if p.shape != q.shape or not np.isfinite(p).all():
        raise ValueError("prediction must preserve native 3CHW measurement grid")
    if sample.domain == "display_rgb":
        # This is display RGB restoration, never inferred linear reflectivity.
        return np.rint(np.clip(p.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)
    delta = (p-q.astype(np.float64)).mean(axis=0)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        gain = np.exp(delta)
    if not np.isfinite(gain).all() or (gain <= 0).any():
        raise ValueError("invalid physical multiplicative amplitude update")
    gain[~support] = 1.
    result = sample.values*gain
    if not np.isfinite(result).all():
        raise ValueError("physical restored values are nonfinite")
    return result


def validate_splits(records):
    """Disallow source-group or exact decoded-pixel leakage across splits."""
    group_split, pixel_split = {}, {}
    for row in records:
        split, group = row["split"], row["group_id"]
        if group in group_split and group_split[group] != split:
            raise ValueError("same acquisition/source group crosses training/validation boundary")
        group_split[group] = split
        signature = row.get("decoded_pixel_sha256")
        if signature:
            if signature in pixel_split and pixel_split[signature] != split:
                raise ValueError("exact decoded pixels cross training/validation boundary")
            pixel_split[signature] = split
    return dict(groups=len(group_split), exact_pixel_signatures=len(pixel_split),
                acquisition_group_leakage_detected=False, exact_pixel_leakage_detected=False,
                near_duplicate_or_scene_independence_not_proven=True)
