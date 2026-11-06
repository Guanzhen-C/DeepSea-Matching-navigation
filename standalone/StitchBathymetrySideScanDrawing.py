"""Join consecutive Data files within one voyage, not a geographic mosaic.

The stored noncoherent side-scan strips contain four trailing rows per file.
Join these to the next file's first four rows using the ORIGINAL amplitude-sum
rule. Depth grids keep their original slant-range axes and receive only the
accumulated beam-ordinal offset; no new cross-file interpolation is introduced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

if __package__:
    from .caris_orangecopper import caris_orangecopper
    from .Bathymetry_SonarDefaultColormap3D import Bathymetry_SonarDefaultColormap3D
else:
    from caris_orangecopper import caris_orangecopper
    from Bathymetry_SonarDefaultColormap3D import Bathymetry_SonarDefaultColormap3D


def boundary_info(previous_ids, following_ids):
    previous, following = previous_ids[-1], following_ids[0]
    delta_us = int((following[1]-previous[1])*1_000_000 + following[2]-previous[2])
    if int(following[0])-int(previous[0]) != 1 or delta_us <= 0:
        raise ValueError("Data files are not consecutive forward-time pings")
    return {"previous_PingNum": int(previous[0]), "following_PingNum": int(following[0]),
            "time_difference_microseconds": delta_us}


def retained_duplicate_headers(ids, count):
    if ids.shape != (count, 3) or np.any(np.diff(ids[:, 0]) < 0):
        raise ValueError("inconsistent or reverse-order ping identities")
    time_us = ids[:, 1]*1_000_000 + ids[:, 2]
    if np.any(np.diff(time_us) < 0):
        raise ValueError("reverse-order timestamps")
    duplicated = np.flatnonzero(np.diff(ids[:, 0]) == 0)
    result = []
    for index in duplicated:
        if not np.array_equal(ids[index], ids[index+1]):
            raise ValueError("same PingNum has conflicting timestamps")
        result.append({"PingNum": int(ids[index, 0]),
                       "input_record_positions_1based": [int(index+1), int(index+2)],
                       "policy": "preserve the supplied identical-header records; do not deduplicate"})
    return result


def stitch_sidescan(arrays, counts, target, *, block_rows=128):
    if not arrays or len(arrays) != len(counts) or block_rows < 1:
        raise ValueError("invalid strip inputs")
    columns = arrays[0].shape[1]
    for array, count in zip(arrays, counts):
        if count < 1 or array.shape != (4*count+4, columns) or array.dtype != np.float64:
            raise ValueError("expected complete float64 eight/four overlap strips")
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float64,
                                    shape=(4*sum(counts)+4, columns))
    offset, joins = 0, []
    for index, (array, count) in enumerate(zip(arrays, counts)):
        # Previous trailing four rows + new first four rows. Do not duplicate
        # the trailing rows or reset the original sum at a file boundary.
        tail = out[offset:offset+4].copy() if index else None
        for row in range(0, len(array), block_rows):
            out[offset+row:offset+row+len(array[row:row+block_rows])] = array[row:row+block_rows]
            if row % (block_rows*4) == 0:
                out.flush()
        if index:
            out[offset:offset+4] += tail
            joins.append(offset)
        offset += 4*count
        out.flush()
    return {"shape": list(out.shape), "join_start_rows_0based": joins,
            "four_tail_rows_fused_per_join": 4, "dtype": "float64"}


def load_parts(entries):
    parts = []
    for entry in entries:
        directory = Path(entry["output"])
        summary = json.loads((directory / "run_summary.json").read_text())
        settings = summary["matlab_settings"]
        if (settings["SelectedUlaNo"] != [1, 2] or settings["PingBeamOverlapFlag"] != 1
                or settings["PingBeamCoherentOverlapFlag"] != 0
                or settings["PingOverlapBeamNum"] != 4
                or settings["PingBeamNonCoherentOverlapFactor"] != 1):
            raise ValueError("joining saved strips requires the original default noncoherent settings")
        count = summary["processed_pings"]
        ids = np.load(directory / "PingNum_PingS_PingUs.npy")
        duplicated = retained_duplicate_headers(ids, count)
        parts.append({"label": entry["label"], "directory": directory,
                      "summary": summary, "count": count, "ids": ids,
                      "range_axis": np.load(directory / "BeamXlis.npy"),
                      "retained_duplicate_headers": duplicated})
    if not parts:
        raise ValueError("no completed input strips")
    for part in parts[1:]:
        if part["summary"]["parameters"] != parts[0]["summary"]["parameters"]:
            raise ValueError("sonar processing parameters differ between Data files")
        if part["summary"]["matlab_settings"] != parts[0]["summary"]["matlab_settings"]:
            raise ValueError("MATLAB processing settings differ between Data files")
        if not np.array_equal(part["range_axis"], parts[0]["range_axis"]):
            raise ValueError("side-scan range axes differ")
        source_a = Path(part["summary"]["source"]).parent.parent
        source_b = Path(parts[0]["summary"]["source"]).parent.parent
        if source_a != source_b:
            raise ValueError("do not combine separate voyages")
    for previous, following in zip(parts, parts[1:]):
        boundary_info(previous["ids"], following["ids"])
    return parts


def render_session(label, parts, directory, joins, *, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.font_manager import FontProperties
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    total = sum(part["count"] for part in parts)
    omitted = [record["PingNum"] for part in parts
               for record in part["summary"].get("explicitly_skipped_records", [])]
    data_names = [Path(part["summary"]["source"]).name for part in parts]
    title = label + "：" + " + ".join(data_names) + f"（{total} 条记录）"
    if omitted:
        title += "，已跳过 " + ",".join(map(str, omitted)) + " 号"
    images = []
    for kind in ("侧扫", "测深"):
        fig, ax = plt.subplots(figsize=(6.4, 8.), constrained_layout=True)
        if kind == "侧扫":
            data = np.load(directory / "SideScanImag.npy", mmap_mode="r")
            row_step = max(1, math.ceil(len(data)/4800))
            col_step = max(1, math.ceil(data.shape[1]/3200))
            texture = np.array(data[::row_step, ::col_step], copy=True)
            x = parts[0]["range_axis"]
            ax.imshow(texture, origin="lower", aspect="auto", interpolation="nearest",
                      extent=(x[0], x[-1], 1, len(data)), vmin=0, vmax=1,
                      cmap=ListedColormap(caris_orangecopper(256)))
        else:
            offset = 0
            xmin, xmax = math.inf, -math.inf
            for part in parts:
                folder = part["directory"]
                z = np.load(folder / "GridZlisOutput.npy", mmap_mode="r")
                x = np.load(folder / "GridXlisInput.npy")
                y = np.load(folder / "GridYlisInput.npy")
                row_step = max(1, math.ceil(len(z)/2400))
                col_step = max(1, math.ceil(z.shape[1]/3200))
                texture = np.array(z[::row_step, ::col_step], copy=True)
                image = ax.imshow(texture, origin="lower", aspect="auto", interpolation="nearest",
                                  extent=(x[0], x[-1], y[0]+offset, y[-1]+offset),
                                  vmin=-60, vmax=-10,
                                  cmap=ListedColormap(Bathymetry_SonarDefaultColormap3D(256)[::-1]))
                xmin, xmax = min(xmin, x[0]), max(xmax, x[-1])
                offset += 4*part["count"]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(1, 4*total)
            fig.colorbar(image, ax=ax, label="相对深度 / m")
        for join_index, row in enumerate(joins):
            ax.axhline(row+.5, color="white", linestyle="--", linewidth=.6, alpha=.8)
            ax.text(.02, row+.5, data_names[join_index]+" → "+data_names[join_index+1], transform=ax.get_yaxis_transform(),
                    color="white", fontsize=8, va="bottom",
                    bbox={"facecolor": "black", "alpha": .35, "edgecolor": "none"})
        ax.set_title(title + f"\n连续{kind}图（非地理配准地图）", fontsize=10)
        ax.set_xlabel("回波斜距 / m")
        ax.set_ylabel("连续波束序号")
        path = directory / f"{label}_连续{kind}.png"
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        images.append(str(path))
    return images


def stitch_session(label, entries, output, *, dpi=600):
    parts = load_parts(entries)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = [part["count"] for part in parts]
    arrays = [np.load(part["directory"] / "SideScanImag.npy", mmap_mode="r") for part in parts]
    print(f"[Stitch] {label}: joining {counts} ping", flush=True)
    side_stats = stitch_sidescan(arrays, counts, output / "SideScanImag.npy")
    ids = np.concatenate([part["ids"] for part in parts])
    np.save(output / "PingNum_PingS_PingUs.npy", ids)
    np.save(output / "BeamXlis.npy", parts[0]["range_axis"])
    np.save(output / "BeamYlis.npy", np.arange(1, side_stats["shape"][0]+1))
    boundaries = []
    depth_parts = []
    offset = 0
    for index, part in enumerate(parts):
        depth_parts.append({"directory": str(part["directory"]), "beam_ordinal_offset": offset,
                            "original_grid_x": str(part["directory"] / "GridXlisInput.npy"),
                            "original_grid_y": str(part["directory"] / "GridYlisInput.npy"),
                            "original_grid_z": str(part["directory"] / "GridZlisOutput.npy")})
        if index:
            boundaries.append(dict(boundary_info(parts[index-1]["ids"], part["ids"]),
                                   joined_first_row_1based=offset+1))
        offset += 4*part["count"]
    images = render_session(label, parts, output, side_stats["join_start_rows_0based"], dpi=dpi)
    result = {"label": label, "processed_pings": len(ids), "source_parts": [str(part["directory"]) for part in parts],
              "unique_identity_count": len(np.unique(ids, axis=0)),
              "retained_duplicate_headers": [{"part": str(part["directory"]), **record}
                                               for part in parts for record in part["retained_duplicate_headers"]],
              "side_scan": side_stats, "boundaries": boundaries, "depth_parts": depth_parts,
              "depth_join_method": "Original grids at their original range coordinates, plus cumulative beam ordinal offsets. No new cross-file interpolation or resampling.",
              "existing_skipped_records": [record for part in parts for record in part["summary"].get("explicitly_skipped_records", [])],
              "additional_skipped_records": [], "GPS_used": False, "georeferenced_map": False,
              "images": images, "export_dpi": dpi, "elapsed_wall_s": time.monotonic()-started,
              "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    (output / "stitch_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def group_completed_voyages(entries):
    """Group by the actual source voyage, never by morning/afternoon alone."""
    grouped = defaultdict(list)
    for entry in entries:
        if not entry["status"].startswith("completed"):
            raise ValueError("cannot stitch a failed or incomplete Data group")
        source_voyage = Path(entry["source"]).resolve().parent.parent
        grouped[source_voyage].append(entry)
    if not grouped:
        raise ValueError("no completed voyages")
    output, used = [], set()
    for voyage, parts in sorted(grouped.items()):
        parts = sorted(parts, key=lambda item: Path(item["source"]).name)
        label = parts[0]["label"].rsplit("_", 1)[0]
        if label in used:
            label += "_" + voyage.name
        used.add(label)
        output.append((label, parts))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("dpi must be positive")
    manifest = json.loads(args.batch_manifest.read_text())
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sessions = []
    for label, entries in group_completed_voyages(manifest["groups"]):
        sessions.append(stitch_session(label, entries, output / label, dpi=args.dpi))
    report = {"source_batch_manifest": str(args.batch_manifest.resolve()),
              "source_manifest_sha256": hashlib.sha256(args.batch_manifest.read_bytes()).hexdigest(),
              "status": "completed", "sessions": sessions, "image_count": 2*len(sessions),
              "total_pings": sum(session["processed_pings"] for session in sessions),
              "georeferenced_map": False, "separate_source_voyages": True}
    (output / "stitched_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
