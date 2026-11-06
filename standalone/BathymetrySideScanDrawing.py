"""Reproduce the supplied BathymetrySideScanDrawing.m, without SLAM or GPS.

All selection, summation, coordinate and clipping constants follow the MATLAB
source. HDF5 is read one ping at a time; complete float64 outputs are disk-backed.
Only the displayed texture is sampled to fit the figure canvas. Its axes remain
the original slant range and beam ordinal, NOT a georeferenced map.

SciPy LinearNDInterpolator implements linear/no-extrapolation interpolation;
duplicate XY samples are averaged as in MATLAB scatteredInterpolant. Different
Delaunay implementations can choose different triangles in degenerate cases;
this is not a claim of bitwise MATLAB agreement or identical figure layout.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator

if __package__:
    from .caris_orangecopper import caris_orangecopper
    from .Bathymetry_SonarDefaultColormap3D import Bathymetry_SonarDefaultColormap3D
else:
    from caris_orangecopper import caris_orangecopper
    from Bathymetry_SonarDefaultColormap3D import Bathymetry_SonarDefaultColormap3D


class PingFile:
    """Read the supplied MATLAB v7.3 cell vector in MATLAB axis order."""

    def __init__(self, path):
        import h5py
        self.path = Path(path)
        self.file = h5py.File(self.path, "r", rdcc_nbytes=1024 * 1024)
        try:
            self.cells = self.file[self.path.stem + "Data"]
            if self.cells.ndim != 2 or 1 not in self.cells.shape:
                raise ValueError("expected a MATLAB cell vector")
            self.refs = self.cells[()].ravel()
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def __len__(self):
        return len(self.refs)

    def record(self, index):
        return self.file[self.refs[index]]

    def identity(self, index):
        group = self.record(index)
        return tuple(int(group[key][0, 0]) for key in ("PingNum", "PingS", "PingUs"))

    def beam_sum(self, index, side):
        group = self.record(index)
        data = group[side + "BeamformingData"]
        if data.ndim != 3 or data.shape[0] < 2 or data.shape[2] != 8:
            raise ValueError("expected MATLAB [8, range samples, >=2 ULA] data")
        summed = np.zeros((8, data.shape[1]), dtype=np.complex128)
        # SelectedUlaNo=[1 2]; sum COMPLEX subarray outputs before abs().
        selected = data[:2, :, :]
        for ula in (0, 1):
            raw = selected[ula].T
            if raw.dtype.fields and {"real", "imag"}.issubset(raw.dtype.fields):
                raw = raw["real"].astype(np.float64) + 1j * raw["imag"]
            summed += np.asarray(raw, dtype=np.complex128)
        return summed

    def depth(self, index):
        group = self.record(index)
        return group["TargetAngleInfo"][()].T, group["TargetDelayInfo"][()].T


def sidescan_step(beam_sum, previous=0, *, overlap=True, coherent=False,
                  factor=1., last=False):
    """Original eight-row/four-row overlap logic, including first/final edges."""
    beams = np.asarray(beam_sum)
    if beams.ndim != 2 or beams.shape[0] != 8:
        raise ValueError("expected eight beam rows")
    if not overlap:
        return np.abs(beams[2:6]), 0
    current = beams.copy() if coherent else np.abs(beams)
    current[:4] += previous
    if not coherent:
        current[:4] *= factor
    next_previous = current[4:8].copy()
    return np.abs(current if last else current[:4]), next_previous


def depth_points(angles, delays, ping_index, side, sound_speed):
    """MATLAB beams 3:6, angle [-60,20], install=30, depth (-60,-10)."""
    angles, delays = np.asarray(angles), np.asarray(delays)
    if angles.shape != delays.shape or angles.ndim != 2 or angles.shape[0] != 8:
        raise ValueError("expected matching [8, samples] angle and delay arrays")
    if side not in ("Port", "Stbd"):
        raise ValueError("unknown side")
    blocks = []
    for beam_offset, beam in enumerate((2, 3, 4, 5)):
        angle = angles[beam]
        valid_angle = (angle >= -60.) & (angle <= 20.)
        grazing = 30. - angle[valid_angle]
        slant = sound_speed / 2. * delays[beam, valid_angle]
        x = (-1. if side == "Port" else 1.) * slant * np.cos(grazing / 180. * np.pi)
        z = -slant * np.sin(grazing / 180. * np.pi)
        keep = ~np.isnan(z) & (z > -60.) & (z < -10.)
        y = np.full(int(keep.sum()), ping_index * 4 + beam_offset + 1., dtype=np.float64)
        blocks.append(np.column_stack((x[keep], y, z[keep])))
    return np.concatenate(blocks)


def matlab_colon(start, step, stop):
    if not all(np.isfinite([start, step, stop])) or step <= 0 or stop < start:
        raise ValueError("invalid colon range")
    count_float = (stop - start) / step
    count = int(np.floor(count_float + 8 * np.finfo(float).eps * max(1., abs(count_float))))
    return start + np.arange(count + 1, dtype=np.float64) * step


def select_records(readers, source_count, skip_ping_numbers=()):
    """Explicit operator-selected omission only; never catch/read past errors."""
    requested = set(skip_ping_numbers)
    if any(isinstance(ping, bool) or not isinstance(ping, (int, np.integer)) or ping < 0
           for ping in requested):
        raise ValueError("skip PingNum values must be nonnegative integers")
    indices, identities, skipped = [], [], []
    for index in range(source_count):
        identity = readers[0].identity(index)
        if any(reader.identity(index) != identity for reader in readers[1:]):
            raise ValueError(f"ping ID or timestamp differs at cell {index}")
        if identity[0] in requested:
            skipped.append({"PingNum": identity[0], "PingS": identity[1], "PingUs": identity[2],
                            "source_cell_1based": index + 1, "source_index_0based": index,
                            "scope": "both sides, bathymetry and sidescan",
                            "reason": "explicit operator request; no echo or depth values fabricated"})
        else:
            indices.append(index)
            identities.append(identity)
    matched = [record["PingNum"] for record in skipped]
    if requested != set(matched):
        raise ValueError(f"requested skip PingNum not found in selected input: {sorted(requested-set(matched))}")
    if len(matched) != len(set(matched)):
        raise ValueError("a requested skip PingNum identifies multiple records")
    if not indices:
        raise ValueError("no pings remain after explicit omissions")
    return indices, identities, skipped


def averaged_samples(points):
    """Match scatteredInterpolant's averaging of duplicate XY locations."""
    xy, inverse, counts = np.unique(points[:, :2], axis=0, return_inverse=True, return_counts=True)
    values = np.bincount(inverse, weights=points[:, 2]) / counts
    return xy, values


def write_interpolated_grid(points, output, *, block_rows=32):
    if len(points) < 3:
        raise ValueError("not enough depth points after the MATLAB filters")
    x = matlab_colon(float(points[:, 0].min()), .1, float(points[:, 0].max()))
    y = matlab_colon(float(points[:, 1].min()), .2, float(points[:, 1].max()))
    np.save(output / "GridXlisInput.npy", x)
    np.save(output / "GridYlisInput.npy", y)
    print(f"[Python replica] linear interpolation: {len(points):,} points, {len(y):,} x {len(x):,} grid", flush=True)
    xy, values = averaged_samples(points)
    interpolator = LinearNDInterpolator(xy, values, fill_value=np.nan, rescale=False)
    grid = np.lib.format.open_memmap(output / "GridZlisOutput.npy", mode="w+",
                                    dtype=np.float64, shape=(len(y), len(x)))
    finite_count = 0
    for row in range(0, len(y), block_rows):
        xx, yy = np.meshgrid(x, y[row:row + block_rows])
        block = interpolator(xx, yy)
        grid[row:row + len(block)] = block
        finite_count += int(np.isfinite(block).sum())
        if row % 3200 == 0:
            grid.flush()
            print(f"[Python replica] grid rows {min(row + len(block), len(y)):,}/{len(y):,}", flush=True)
    grid.flush()
    return {"shape": list(grid.shape), "valid_cells": finite_count,
            "unique_interpolation_points": len(xy), "dtype": "float64"}


def render(output, *, dpi, overlap, coherent, snapshot_count, skipped_pings=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.font_manager import FontProperties
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    images = []
    strides = {}
    side = np.load(output / "SideScanImag.npy", mmap_mode="r")
    sx, sy = np.load(output / "BeamXlis.npy"), np.load(output / "BeamYlis.npy")
    if overlap:
        name = "标准分辨率波束拼接相干模式" if coherent else "标准分辨率波束拼接非相干模式"
        title = "标准分辨率波束拼接模式：线性阵列编号=[1 2]\n图像拼接波束号=[1 2 3 4 5 6 7 8]，波束重叠数=4"
    else:
        name = "标准分辨率模式"
        title = "标准分辨率模式：线性阵列编号=[1 2]，图像拼接波束号=[3 4 5 6]"
    for key, data, x, y, limits, cmap, title, name in [
        ("sidescan", side, sx, sy, (0., 1.), caris_orangecopper(256), title, name),
        ("bathymetry", np.load(output / "GridZlisOutput.npy", mmap_mode="r"),
         np.load(output / "GridXlisInput.npy"), np.load(output / "GridYlisInput.npy"),
         (-60., -10.), Bathymetry_SonarDefaultColormap3D(256)[::-1],
         "标准分辨率模式测深图：图像拼接波束号=[3 4 5 6]",
         f"均值法测深快拍数{snapshot_count:g},垂直航迹向网格0.1米,波束向网格0.2米"),
    ]:
        row_step = max(1, math.ceil(data.shape[0] / 2400))
        col_step = max(1, math.ceil(data.shape[1] / 3200))
        # Sampling is ONLY the on-screen raster, not the saved numerical data.
        texture = np.array(data[::row_step, ::col_step], copy=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        plotted = ax.imshow(texture, origin="lower", aspect="auto", interpolation="nearest",
                            extent=(x[0], x[-1], y[0], y[-1]),
                            vmin=limits[0], vmax=limits[1], cmap=ListedColormap(cmap))
        ax.set_xlabel("回波距离(m)")
        ax.set_ylabel("波束序号")
        if skipped_pings:
            title += "\n已跳过 PingNum=" + ",".join(map(str, skipped_pings)) + "（未补值）"
        ax.set_title(title, fontsize=9)
        if key == "bathymetry":
            fig.colorbar(plotted, ax=ax)
        path = output / (name + ".png")
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        images.append(str(path))
        strides[key] = [row_step, col_step]
    return images, strides


def run(data_directory, output, *, max_pings=0, overlap=True, coherent=False,
        factor=1., dpi=1200, skip_ping_numbers=()):
    import h5py
    import scipy
    data_directory, output = Path(data_directory).resolve(), Path(output).resolve()
    if output == data_directory or data_directory in output.parents:
        raise ValueError("output must be outside the original data directory")
    if max_pings < 0 or not np.isfinite(factor) or dpi <= 0:
        raise ValueError("invalid run arguments")
    filenames = ["SysPara.mat", "PortNormalModeBeamformingTotal.mat",
                 "StbdNormalModeBeamformingTotal.mat", "PortNormalModeDepthTotal.mat",
                 "StbdNormalModeDepthTotal.mat"]
    input_stats = {name: (data_directory / name).stat() for name in filenames}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with h5py.File(data_directory / "SysPara.mat", "r") as f:
        parameters = {k: float(f["SysPara"][k][0, 0]) for k in ("c", "BeamSigFs", "DoaEstProSampleNum")}
    with contextlib.ExitStack() as stack:
        readers = [stack.enter_context(PingFile(data_directory / name)) for name in filenames[1:]]
        lengths = [len(reader) for reader in readers]
        if len(set(lengths)) != 1 or not lengths[0]:
            raise ValueError("four sonar cell vectors must have equal, nonzero length")
        source_count = min(lengths[0], max_pings) if max_pings else lengths[0]
        indices, identities, skipped = select_records(readers, source_count, skip_ping_numbers)
        count = len(indices)
        if skipped:
            (output / "explicitly_skipped_pings.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2) + "\n")
            print("[Python replica] explicitly omitted PingNum: " + ",".join(str(p["PingNum"]) for p in skipped), flush=True)
        first = [readers[i].beam_sum(indices[0], side) for i, side in enumerate(("Port", "Stbd"))]
        bins = first[0].shape[1]
        if first[1].shape != first[0].shape:
            raise ValueError("port/starboard beam shapes differ")
        rows = 4 * count + (4 if overlap else 0)
        mosaic = np.lib.format.open_memmap(output / "SideScanImag.npy", mode="w+",
                                          dtype=np.float64, shape=(rows, 2 * bins))
        previous = [0, 0]
        offset = 0
        for index, source_index in enumerate(indices):
            blocks = []
            for i, side in enumerate(("Port", "Stbd")):
                beams = first[i] if index == 0 else readers[i].beam_sum(source_index, side)
                block, previous[i] = sidescan_step(beams, previous[i], overlap=overlap,
                                                 coherent=coherent, factor=factor, last=index == count - 1)
                blocks.append(block)
            size = len(blocks[0])
            mosaic[offset:offset + size, :bins] = blocks[0][:, ::-1]
            mosaic[offset:offset + size, bins:] = blocks[1]
            offset += size
            if (index + 1) % 100 == 0 or index == count - 1:
                mosaic.flush()
                print(f"[Python replica] side scan {index + 1}/{count} pings", flush=True)
        assert offset == rows
        mosaic.flush()
        del mosaic, first, blocks, block, beams
        beam_x = np.r_[np.arange(-bins, 0), np.arange(1, bins + 1)] / parameters["BeamSigFs"] * (parameters["c"] / 2.)
        np.save(output / "BeamXlis.npy", beam_x)
        np.save(output / "BeamYlis.npy", np.arange(1, rows + 1))
        np.save(output / "PingNum_PingS_PingUs.npy", np.asarray(identities, dtype=np.int64))
        np.save(output / "SourceCellIndices1Based.npy", np.asarray(indices, dtype=np.int64) + 1)
        sides = []
        for reader, side in zip(readers[2:], ("Port", "Stbd")):
            blocks = [depth_points(*reader.depth(source_index), index, side, parameters["c"])
                      for index, source_index in enumerate(indices)]
            sides.append(np.concatenate(blocks))
            del blocks
        points = np.concatenate(sides)
        del sides
    np.save(output / "DepthTotalValidXYZ.npy", points)
    grid_stats = write_interpolated_grid(points, output)
    point_count = len(points)
    del points
    images, strides = render(output, dpi=dpi, overlap=overlap, coherent=coherent,
                             snapshot_count=parameters["DoaEstProSampleNum"],
                             skipped_pings=[p["PingNum"] for p in skipped])
    unchanged = all((data_directory / name).stat().st_size == st.st_size
                    and (data_directory / name).stat().st_mtime_ns == st.st_mtime_ns
                    for name, st in input_stats.items())
    if not unchanged:
        raise RuntimeError("an input file changed during the run")
    source_folder = data_directory.parents[2]
    matlab_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_folder.glob("*.m")}
    report = {
        "source": str(data_directory), "source_matlab_sha256": matlab_hashes,
        "processed_pings": count, "available_pings": lengths[0], "parameters": parameters,
        "source_pings_in_requested_range": source_count,
        "requested_skip_ping_numbers": sorted(set(skip_ping_numbers)), "explicitly_skipped_records": skipped,
        "skip_policy": ("Omit only requested records from all four streams; unchanged four-beam overlap and ordinal coordinates on the remaining sequence. Original timestamps retained; no fabricated samples."
                        if skipped else "none; strict default"),
        "matlab_settings": {"SelectedUlaNo": [1, 2], "PingBeamOverlapFlag": int(overlap),
                            "PingBeamCoherentOverlapFlag": int(coherent),
                            "PingBeamNonCoherentOverlapFactor": factor, "PingOverlapBeamNum": 4,
                            "depth_selected_beams": [3, 4, 5, 6], "depth_angle_gate_deg": [-60, 20],
                            "depth_open_interval_m": [-60, -10], "ArrayInstallAngle": 30,
                            "DisXRes": .1, "DisYRes": .2},
        "sidescan_shape": [rows, 2 * bins], "depth_points": point_count, "depth_grid": grid_stats,
        "echo_and_geometry_storage_dtype": "float64", "original_inputs_size_and_mtime_unchanged": unchanged,
        "render_only_row_column_strides": strides, "images": images, "export_dpi": dpi,
        "ping_number_gaps_preserved_without_algorithm_change": True,
        "GPS_used": False, "SLAM_used": False, "georeferenced_map": False,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__},
        "elapsed_wall_s": time.monotonic() - started,
        "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "equivalence_limits": ["No MATLAB runtime was executed for an exact numerical oracle.",
                               "SciPy/MATLAB Delaunay tie breaking and figure rendering may differ.",
                               "Full-resolution numeric outputs are preserved; PNG textures are sampled for bounded rendering.",
                               "Native MATLAB .fig serialization is not reproduced; numeric arrays and PNG are saved."],
    }
    (output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-pings", type=int, default=0, help="0: reproduce the complete selected Data directory")
    parser.add_argument("--ping-beam-overlap-flag", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ping-beam-coherent-overlap-flag", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ping-beam-noncoherent-overlap-factor", type=float, default=1.)
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--skip-ping", type=int, action="append", default=[],
                        help="explicit PingNum to omit from both products/sides; default skips nothing")
    args = parser.parse_args()
    run(args.data_directory, args.output_directory, max_pings=args.max_pings,
        overlap=bool(args.ping_beam_overlap_flag), coherent=bool(args.ping_beam_coherent_overlap_flag),
        factor=args.ping_beam_noncoherent_overlap_factor, dpi=args.dpi,
        skip_ping_numbers=args.skip_ping)


if __name__ == "__main__":
    main()
