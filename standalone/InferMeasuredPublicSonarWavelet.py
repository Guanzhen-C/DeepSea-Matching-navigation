"""Read-only measured pre-mosaic envelopes, using the public envelope head.

This is a transfer experiment, not verified removal of real interference.
No RGB-to-amplitude inversion, geometry resizing, private upload, MAT writes,
or claim of recovering unknown clean phase. Scalar gains and full native
reconstruction are saved separately from all original measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import resource
import shutil
import struct
import time
import zlib

import numpy as np
from PIL import Image
import torch

from PublicSonarWavelet import PublicSonarWavelet
from PublicSonarEnvelopeModel import PublicSonarEnvelopeModel
from PublicEnvelopeCoherentModel import PublicEnvelopeCoherentModel,envelope_model_from_state
from EvaluatePublicSonarWavelet import select_checkpoint
from caris_orangecopper import caris_orangecopper


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def guard(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def release_pages(*arrays):
    """Drop only this process's mapped pages after flushing its own outputs.

    A memmap alone does not bound RSS after a full sweep. MADV_DONTNEED does
    not delete measurements or change their bytes; later reads fault them in.
    """
    for array in arrays:
        if not isinstance(array, np.memmap):
            raise ValueError("only explicit file-backed measurement arrays may be released")
        if array.flags.writeable:
            array.flush()
        array._mmap.madvise(mmap.MADV_DONTNEED)


def save(root, report):
    pending = root / "run_summary.tmp"
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pending.replace(root / "run_summary.json")


def tiles(shape, core=512, halo=96):
    if len(shape) != 2 or min(shape) < 1 or core < 4 or core % 4 or halo < 0 or halo % 4:
        raise ValueError("native two-dimensional grid and factor-four aligned tiles required")
    h, w = shape
    for y in range(0, h, core):
        for x in range(0, w, core):
            # Let the model apply its own end padding, exactly as for a whole
            # frame. Do not invent reflected outer measurements for a tile.
            yield (y, x, min(h, y + core), min(w, x + core),
                   max(0, y - halo), max(0, x - halo),
                   min(h, y + core + halo), min(w, x + core + halo))


@torch.inference_mode()
def delta(model, amplitude, device, *, origin=None):
    a = np.asarray(amplitude, np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("finite nonnegative native envelopes required")
    q = np.zeros(a.shape, np.float32)
    positive = a > 0
    q[positive] = np.log(a[positive]).astype(np.float32)
    tensor = torch.from_numpy(q.copy())[None, None].to(device)
    if isinstance(model, PublicSonarEnvelopeModel):
        if origin is None:
            raise ValueError("whole-beam log origin is required for tiled normalized inference")
        reference = torch.tensor(origin, device=device, dtype=tensor.dtype).reshape(1,1,1,1)
        residual = model.residual(tensor, torch.from_numpy(positive)[None,None].to(device), origin=reference)[0,0].cpu().numpy()
    else:
        if origin is not None:
            raise ValueError("legacy envelope inputs must not be silently normalized")
        tensor = tensor.repeat(1,3,1,1)
        residual = (model(tensor, 1) - tensor).mean(1)[0].cpu().numpy()
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite model log-amplitude residual")
    return residual


def whole_log_origin(amplitude):
    total, count = 0., 0
    for start in range(0,len(amplitude),32):
        data=np.asarray(amplitude[start:start+32],np.float64)
        if not np.isfinite(data).all() or (data<0).any():
            raise ValueError("invalid measured envelope for global log origin")
        values=data[data>0]
        total+=float(np.log(values).sum(dtype=np.float64));count+=len(values)
    return total/count if count else 0.


def supported_delta(amplitude, residual, limits, columns, sound_speed, sampling_frequency):
    a, residual = np.asarray(amplitude), np.asarray(residual)
    limits = np.asarray(limits, np.float64)
    if a.shape != residual.shape or limits.shape != (len(a), 2):
        raise ValueError("native envelope, residual and physical-record limits must match")
    if not np.isfinite(limits).all() or (limits[:, 0] <= 0).any() or (limits[:, 1] <= limits[:, 0]).any():
        raise ValueError("reviewed supplied bottom and first-target ranges required")
    ranges = (np.asarray(columns) + 1) * sound_speed / (2 * sampling_frequency)
    u = np.clip((ranges[None] - limits[:, :1]) / (limits[:, 1:] - limits[:, :1]), 0, 1)
    support = u * u * (3 - 2 * u)
    result = (residual * support).astype(np.float32)
    result[a <= 1e-12] = 0
    return result


def mosaic_rows(envelopes, field, start, stop):
    """Original eight simultaneous beams, four-row spatial overlap."""
    count, bins = envelopes.shape[-2:]
    rows = np.arange(start, stop)
    record, beam = rows // 4, rows % 4
    result = np.zeros((len(rows), 2 * bins), np.float64)
    for side in range(2):
        values = np.zeros((len(rows), bins), np.float64)
        for valid, offset, old in ((record < count, 0, 0), (record > 0, 4, 1)):
            a = np.asarray(envelopes[side, beam[valid] + offset, record[valid] - old])
            if field is not None:
                a = a * np.exp(np.asarray(field[side, beam[valid] + offset, record[valid] - old], np.float64))
            values[valid] += a
        result[:, side * bins:(side + 1) * bins] = values[:, ::-1] if side == 0 else values
    return result


def chunk(stream, name, data):
    stream.write(struct.pack(">I", len(data)) + name + data)
    stream.write(struct.pack(">I", zlib.crc32(name + data) & 0xffffffff))


def rgb_rows(values, palette):
    values = np.asarray(values)
    index = np.minimum((np.clip(values, 0, 1) * 256).astype(np.uint16), 255)
    return palette[index]


def native_png(values, path, palette):
    h, w = values.shape
    compressor = zlib.compressobj(3)
    with Path(path).open("xb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        chunk(stream, b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        chunk(stream, b"tEXt", b"Description\x00Public-trained envelope-head transfer experiment, not quality accepted; native grid; fixed original CARIS amber display0..1")
        for row in range(h - 1, -1, -1):
            encoded = compressor.compress(b"\x00" + rgb_rows(values[row], palette).tobytes())
            if encoded:
                chunk(stream, b"IDAT", encoded)
            if row % 256 == 0 and isinstance(values, np.memmap):
                release_pages(values)
        chunk(stream, b"IDAT", compressor.flush())
        chunk(stream, b"IEND", b"")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--reference-mosaic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-family", choices=("legacy", "normalized_envelope"), default="legacy")
    args = parser.parse_args()
    training = json.loads((args.training / "run_summary.json").read_text())
    normalized = args.model_family == "normalized_envelope"
    if training["steps_completed"] != training["steps_requested"] or not training["public_only"]:
        raise ValueError("completed public-only training required")
    envelope_steps = training["steps_completed"] if normalized else training["training_steps_by_domain"]["envelope"]
    if envelope_steps <= 0:
        raise ValueError("envelope head was not trained")
    if normalized:
        if training["status"] != "completed_envelope_training_real_interference_removal_unverified":
            raise ValueError("normalized envelope training is not complete")
        best = min(training["validation_history"],key=lambda m:(m["validation_selection_score"],m["step"]))
        choices=[c for c in training["checkpoints"] if c["step"]==best["step"]]
        if len(choices)!=1:raise ValueError("selected envelope validation checkpoint is missing")
        choice=choices[0]
        if sha(Path(__file__).with_name("PublicSonarEnvelopeModel.py")) != training["source_sha256"]["PublicSonarEnvelopeModel.py"]:
            raise ValueError("normalized envelope model definition changed since training")
        if training['model_family']==PublicEnvelopeCoherentModel.family:
            if sha(Path(__file__).with_name('PublicEnvelopeCoherentModel.py'))!=training['source_sha256']['PublicEnvelopeCoherentModel.py']:
                raise ValueError('directional noise-field definition changed since training')
    else:
        choice = select_checkpoint(training)[1]
    checkpoint = args.training / choice["file"]
    if sha(checkpoint) != choice["sha256"]:
        raise ValueError("training checkpoint hash changed")
    if sha(Path(__file__).with_name("PublicSonarWavelet.py")) != training["source_sha256"]["PublicSonarWavelet.py"]:
        raise ValueError("frozen trained forward graph changed")
    cache_ledger = json.loads((args.cache / "run_summary.json").read_text())
    if not cache_ledger["original_all_native_samples_reproduced_exactly"]:
        raise ValueError("MAT-envelope cache has not reproduced the original mosaic")
    guards = cache_ledger["source_guards"]
    if any(guard(path) != expected for path, expected in guards.items()):
        raise ValueError("original MATs differ from the previously verified extraction")
    source = args.cache / "原始MAT逐束包络_float64.npy"
    amplitudes = np.load(source, mmap_mode="r")
    limits = np.load(args.cache / "bottom_first_ranges.npy")
    reference = np.load(args.reference_mosaic, mmap_mode="r")
    if amplitudes.ndim != 4 or amplitudes.shape[:2] != (2, 8):
        raise ValueError("two sides and all eight simultaneous beams required")
    count, bins = amplitudes.shape[-2:]
    if limits.shape != (2, count, 2) or reference.shape != (4 * count + 4, 2 * bins):
        raise ValueError("native cache, depth limits and reference grid mismatch")
    parts = cache_ledger["parts"]
    pars = parts[0]["parameters"]
    if any(part["parameters"] != pars for part in parts):
        raise ValueError("source instrument parameters vary across parts")
    if shutil.disk_usage(args.cache).free < amplitudes.size * 4 + reference.size * 8 + 4 * 1024**3:
        raise ValueError("insufficient disk headroom for full native outputs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    trained_auxiliary_count=training.get('total_auxiliary_sources',0) if training.get('complex_magnitude_auxiliary_trained',False) else 0
    report = dict(status="preflight", quality_accepted=False, real_clean_truth=False,
        private_raw_data_uploaded=False, raw_MAT_modified=False, native_shape=list(amplitudes.shape),
        native_measurements=int(amplitudes.size), processed_measurements=0, completed_tiles=0,
        checkpoint=choice, checkpoint_selection="validation only, before measured inference",
        envelope_training_steps=envelope_steps, model_family=args.model_family,
        trained_model_family=training.get('model_family'),
        public_complex_magnitude_auxiliary_sources=trained_auxiliary_count,
        complex_phase_features_used=training.get('complex_phase_used_as_model_feature',False),
        complex_head_trained=training.get('complex_head_trained',False),
        source=str(source.resolve()), source_guard_before=guard(source), original_MAT_guards=guards,
        amplitude_scale=1., instrument_unit_alignment_calibrated=False,
        global_unit_offset_normalization=normalized,
        scope="public envelope-head transfer to measured pre-mosaic envelopes; not image-domain inversion",
        limitation=(("One primary public survey with date splits plus public complex-image magnitude auxiliary sources of unknown acquisition independence; no independent-site proof or true clean reference"
                     if trained_auxiliary_count else "One public survey with acquisition-date splits; no independent-site proof or true clean reference")
                    if normalized else "Only four short public envelope groups were used; no independent envelope-domain validation or real clean target"),
        formula="A_candidate=A_original*exp(supported_predicted_log_delta)",
        phase_claim="Positive gains could preserve original complex phase; no clean phase recovery or MAT writeback performed",
        core=512, halo=max(96,training.get('inference_halo',96)), inference_pixels_resampled=False,
        mapped_page_release="flush own outputs then process-local MADV_DONTNEED, no source deletion or global cache drop",
        preallocated_output_is_not_completion_evidence=True)
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in ("InferMeasuredPublicSonarWavelet.py", "PublicSonarWavelet.py", "PublicSonarEnvelopeModel.py", "PublicEnvelopeCoherentModel.py", "EvaluatePublicSonarWavelet.py", "caris_orangecopper.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    report["source_code_sha256"] = {p.name: sha(p) for p in snapshot.iterdir()}
    save(root, report)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.4, device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if normalized:
        model=envelope_model_from_state(state).to(device).eval().requires_grad_(False)
        if report['halo']<getattr(model,'required_halo',96):
            raise ValueError('declared native halo is too short for the trained noise field')
    else:
        if tuple(state["domain_names"]) != PublicSonarWavelet.domains:
            raise ValueError("checkpoint measurement-domain interpretation changed")
        model = PublicSonarWavelet(width=state["width"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    try:
        report["source_sha256_before"] = sha(source)
        origins = [[None]*8 for _ in range(2)]
        if normalized:
            for side in range(2):
                for beam in range(8):
                    origins[side][beam]=whole_log_origin(amplitudes[side,beam])
                    release_pages(amplitudes)
            report["whole_beam_log_origins"] = origins
            report["origin_is_one_noisy_input_scalar_not_per_ping_normalization"] = True
        image = np.array(amplitudes[0, 0, 137:342, 9001:9734])
        expected = delta(model, image, device, origin=origins[0][0])
        tiled = np.empty_like(expected)
        for y, x, end_y, end_x, top, left, bottom, right in tiles(image.shape, core=128, halo=report['halo']):
            predicted = delta(model, image[top:bottom, left:right], device, origin=origins[0][0])
            tiled[y:end_y, x:end_x] = predicted[y-top:end_y-top, x-left:end_x-left]
        if not np.allclose(expected, tiled, rtol=2e-4, atol=2e-5):
            raise ValueError("actual trained whole/tile preflight failed")
        report["whole_vs_tiled_preflight"] = dict(passed=True, max_abs_error=float(np.abs(expected-tiled).max()),
            shape=list(image.shape), scope="complete native test rectangle including outer edges and factor-four alignment")
        # Independent verification of every cache sample's original overlap
        # result, before using the cache for learned modification.
        for start in range(0, len(reference), 32):
            end = min(start + 32, len(reference))
            np.testing.assert_array_equal(mosaic_rows(amplitudes, None, start, end), reference[start:end])
            if start % 256 == 0:
                release_pages(amplitudes, reference)
        release_pages(amplitudes, reference)
        report["entire_original_mosaic_reproduced_before_inference"] = True
        field = np.lib.format.open_memmap(root / "per_beam_supported_log_delta_float32.npy", "w+", np.float32, amplitudes.shape)
        plan = list(tiles((count, bins),halo=report['halo']))
        report.update(status="full_native_pre_mosaic_envelope_inference", total_tiles=16 * len(plan))
        save(root, report)
        last = time.monotonic()
        for side in range(2):
            for beam in range(8):
                for y, x, end_y, end_x, top, left, bottom, right in plan:
                    predicted = delta(model, amplitudes[side, beam, top:bottom, left:right], device, origin=origins[side][beam])
                    residual = predicted[y-top:end_y-top, x-left:end_x-left]
                    a = amplitudes[side, beam, y:end_y, x:end_x]
                    result = supported_delta(a, residual, limits[side, y:end_y], np.arange(x, end_x), pars["c"], pars["BeamSigFs"])
                    field[side, beam, y:end_y, x:end_x] = result
                    report["processed_measurements"] += result.size
                    report["completed_tiles"] += 1
                    if report["completed_tiles"] % 32 == 0:
                        release_pages(amplitudes, field)
                    if time.monotonic() - last > 5:
                        field.flush()
                        report["elapsed_s"] = time.monotonic() - started
                        save(root, report)
                        print(json.dumps({k: report[k] for k in ("status", "completed_tiles", "total_tiles", "processed_measurements", "elapsed_s")}), flush=True)
                        last = time.monotonic()
        field.flush()
        release_pages(amplitudes, field)
        if report["processed_measurements"] != amplitudes.size:
            raise ValueError("not every original envelope sample was processed")
        report["status"] = "reconstructing_complete_native_mosaic"
        save(root, report)
        candidate = np.lib.format.open_memmap(root / "SideScanImag_public_envelope_candidate_float64.npy", "w+", np.float64, reference.shape)
        for start in range(0, len(candidate), 32):
            end = min(start + 32, len(candidate))
            reconstructed = mosaic_rows(amplitudes, field, start, end)
            if not np.isfinite(reconstructed).all() or (reconstructed < 0).any():
                raise ValueError("invalid reconstructed physical-envelope candidate")
            candidate[start:end] = reconstructed
            if start % 256 == 0:
                release_pages(amplitudes, field, candidate)
        candidate.flush()
        release_pages(amplitudes, field, candidate)
        report["status"] = "rendering_full_native_candidate_not_quality_acceptance"
        save(root, report)
        palette = (caris_orangecopper(256) * 255).astype(np.uint8)
        native_png(candidate, root / "实测原始包络_公开数据小波模型_全部原采样像素_未验收.png", palette)
        # Full-voyage display ONLY. Numeric results and the PNG above retain
        # every measured sample; a 9:16 preview does not invent resolution.
        rows = np.rint(np.linspace(0, len(candidate) - 1, 3840)).astype(int)
        columns = np.rint(np.linspace(0, bins * 2 - 1, 2160)).astype(int)
        np.savez(root / "portrait_display_indices.npz", rows=rows, columns=columns)
        for name, array in (("原始实测_完整竖版.png", reference), ("包络分支候选_完整竖版_未验收.png", candidate)):
            preview = rgb_rows(np.asarray(array[np.ix_(rows, columns)])[::-1], palette)
            Image.fromarray(preview).save(root / name)
        for side, start_y, start_x, label in ((0, 1728, 9728, "Port"), (1, 768, 9216, "Stbd")):
            a = np.asarray(amplitudes[side, :, start_y:start_y+256, start_x:start_x+1024])
            d = np.asarray(field[side, :, start_y:start_y+256, start_x:start_x+1024], np.float64)
            comparison = np.concatenate((a.mean(0), (a*np.exp(d)).mean(0)), axis=1)
            Image.fromarray(rgb_rows(comparison, palette)).save(root / (label + "_强条纹原尺寸区域_原始与候选.png"))
        report["source_sha256_after"] = sha(source)
        if report["source_sha256_after"] != report["source_sha256_before"] or guard(source) != report["source_guard_before"]:
            raise ValueError("source envelope cache changed")
        if any(guard(path) != expected for path, expected in guards.items()):
            raise ValueError("source MAT guards changed")
        report.update(status="completed_full_native_envelope_transfer_candidate_not_quality_acceptance",
            output_shape=list(candidate.shape), original_sources_unchanged=True,
            full_native_numeric_complete=True, full_native_PNG_complete=True,
            preview_only_spatial_sampling=True,
            output_sha256={p.name: sha(p) for p in root.iterdir() if p.suffix in (".npy", ".png")})
    except Exception as exc:
        report.update(status="failed_preserved_artifacts_no_automatic_restart", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_s=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        save(root, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("original_MAT_guards", "output_sha256")}), flush=True)


if __name__ == "__main__":
    main()
