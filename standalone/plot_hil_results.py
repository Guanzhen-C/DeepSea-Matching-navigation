#!/usr/bin/env python3
"""Verified Jetson online trajectories, errors and measured-terrain figures.

Truth is read only on the PC. Never replace board outputs with the native PC
reference, align trajectories after the fact, or silently hide missing samples.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from standalone.hil.recording import read_packets
from standalone.hil.score_board_results import stats
from standalone.terrain_plot_mesh import measured_surface


def load_run(directory, *, require_full_route=False, include_rotations=False):
    manifest = json.loads((directory / "launch_manifest.json").read_text())
    source = Path(manifest["recording"]) if manifest.get("recording") else directory / "native_reference"
    recorded = json.loads((source / "recording_summary.json").read_text())
    metrics = json.loads((source / "native_metrics.json").read_text())
    sender = json.loads((directory / "sender/sender_summary.json").read_text())
    receiver = json.loads((directory / "board_artifacts/receiver_summary.json").read_text())
    diagnostics = json.loads((directory / "board_artifacts/estimator_diagnostics.json").read_text())
    if any(item["status"] != "completed" for item in (recorded, sender, receiver)) or metrics["run_status"] != "completed":
        raise ValueError("an experiment component did not finish")
    if sender["events"] != recorded["events"] or receiver["events"] != recorded["events"]:
        raise ValueError("host/board sensor event counts differ")
    for kind, native_key in (("imu", "imu_samples"), ("dvl", "dvl_samples"),
                             ("depth", "depth_samples"), ("sonar", "pings")):
        if recorded["events"][kind] != metrics[native_key]:
            raise ValueError("recorded inputs differ from independent native sensor counters")
    count = recorded["samples"]
    if sender["samples"] != count or sender["skipped_events"] != 0:
        raise ValueError("missing board outputs or skipped input events")
    if diagnostics["graph_diagnostics"]["optimizer_backend"] != "gtsam_isam2_incremental_fixed_lag":
        raise ValueError("not the qualified native GTSAM fixed-lag backend")
    features = diagnostics["feature_diagnostics"]
    if features["active_compute_backend"] != "cuda" or features["cuda_fallbacks"] != 0:
        raise ValueError("front end did not retain the CUDA path")
    full = bool(metrics["completed_requested_route"])
    if require_full_route and not full:
        raise ValueError("short-segment evidence cannot stand in for full-route HIL")
    if manifest.get("live_Isaac") and (recorded.get("delivery") != "live_ROS2_to_Jetson"
            or sender.get("input_source") != "live_Isaac_sensor_stream"):
        raise ValueError("live-source claim lacks actual live delivery evidence")
    positions = np.empty((count, 4, 3))  # truth, DR, Local, online Global
    rotations = np.empty((count, 4, 3, 3)) if include_rotations else None
    native = np.empty((count, 3, 3))
    timestamps = np.empty(count)
    roundtrip = np.empty(count)
    lateness = np.full(count, np.nan)
    reference = source / "host_reference.osh.gz"
    outputs = directory / "sender/returned_estimator_outputs.osh.gz"
    used = 0
    for index, (truth, board) in enumerate(itertools.zip_longest(read_packets(reference), read_packets(outputs))):
        if index >= count or truth is None or board is None or abs(truth["timestamp"] - board["timestamp"]) > 1e-9:
            raise ValueError("missing, excess, or misaligned board output")
        timestamps[index] = board["timestamp"]
        positions[index, 0] = truth["truth_position"]
        if rotations is not None:
            rotation = np.asarray(truth["truth_rotation"], dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError("invalid recorded rotation matrix shape")
            rotations[index, 0] = rotation
        for channel, name in enumerate(("dr", "local", "global")):
            positions[index, channel + 1] = board[name + "_position"]
            native[index, channel] = truth[name + "_position"]
            if rotations is not None:
                rotation = np.asarray(board[name + "_rotation"], dtype=np.float64)
                if rotation.shape != (3, 3):
                    raise ValueError("invalid recorded rotation matrix shape")
                rotations[index, channel + 1] = rotation
        roundtrip[index] = board["transport_and_processing_roundtrip_s"]
        lateness[index] = board.get("arrival_lateness_s", np.nan)
        used += 1
    if used != count or not np.isfinite(positions).all() or not np.isfinite(native).all():
        raise ValueError("incomplete or nonfinite recorded trajectory")
    if rotations is not None:
        # Input integrity, not a navigation acceptance threshold. Never repair
        # or project an invalid orientation onto SO(3) in the scoring path.
        for chunk in np.array_split(rotations, max(1, int(np.ceil(count / 10000)))):
            if (not np.isfinite(chunk).all()
                    or not np.allclose(np.swapaxes(chunk, -1, -2) @ chunk, np.eye(3), rtol=0., atol=1e-6)
                    or not np.allclose(np.linalg.det(chunk), 1., rtol=0., atol=1e-6)):
                raise ValueError("invalid recorded rotation matrix")
    if not np.allclose(np.diff(timestamps), 1. / metrics["ping_rate_hz"], rtol=0., atol=1e-8):
        raise ValueError("output stream contains a sensor-clock gap")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions[:, 0], axis=0), axis=1))]
    if not np.isclose(distance[-1], metrics["survey_travelled_distance_m"], rtol=0., atol=1e-6):
        raise ValueError("truth arc differs from original independent distance report")
    errors = np.linalg.norm(positions[:, 1:] - positions[:, :1], axis=2)
    native_errors = np.linalg.norm(native - positions[:, :1], axis=2)
    return dict(manifest=manifest, source=source, full_route=full, timestamps=timestamps,
                distance=distance, positions=positions, errors=errors, native_errors=native_errors,
                roundtrip=roundtrip, lateness=lateness, diagnostics=diagnostics, sender=sender,
                reference=reference, outputs=outputs, rotations=rotations)


def set_font():
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "pdf.fonttype": 3, "path.simplify": False})


def plot_errors(run, prefix):
    distance = run["distance"] / 1000.
    errors = run["errors"]
    report = {name: stats(errors[:, index]) for index, name in enumerate(("dr", "local", "global"))}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.1]})
    colors = ("#798594", "#ec761e", "#007cba")
    for i, (name, color) in enumerate(zip(("板端 DR", "板端 Local", "板端在线 Global"), colors)):
        top.plot(distance, errors[:, i], color=color, lw=1., label=name, rasterized=True)
    top.set_ylabel("三维位置误差 / m")
    top.legend(loc="upper left", ncol=3)
    s = report["global"]
    top.text(.99, .96, f'Global RMSE {s["rmse_m"]:.3f} m · P95 {s["p95_m"]:.3f} m\n'
             f'最大 {s["max_m"]:.3f} m · 最终 {s["final_m"]:.3f} m',
             transform=top.transAxes, ha="right", va="top",
             bbox=dict(fc="white", ec="#bac7d1", boxstyle="round,pad=.4"))
    bottom.plot(distance, run["native_errors"][:, 2], color="#be7d9d", lw=.8, alpha=.65,
                label="同输入 PC 参考 Global", rasterized=True)
    bottom.plot(distance, errors[:, 2], color=colors[2], lw=1., label="Jetson 实际发布 Global", rasterized=True)
    bottom.axhline(s["p95_m"], color="#48a8d9", lw=.8, ls="--", label="板端 Global P95")
    bottom.set_yscale("symlog", linthresh=1.)
    bottom.set_ylabel("Global 误差 / m（symlog）")
    bottom.set_xlabel("真值轨迹累计航程 / km")
    bottom.legend(loc="upper left", ncol=3)
    for ax in (top, bottom):
        ax.grid(color="#dce3eb", lw=.65)
        ax.set_axisbelow(True)
        ax.set_xlim(0, distance[-1])
    profile = {"lake": "湖底 Chelan", "sea": "海底 Monterey"}[run["manifest"]["profile"]]
    scope = "全航程" if run["full_route"] else "短程验证"
    fig.suptitle(f'Jetson AGX Orin · {profile} {distance[-1]:.3f} km {scope}\n'
                 f'{len(distance):,} 个同步在线样本 · {run["manifest"]["mode"]}', fontsize=18)
    fig.text(.07, .015, "导航处理器在环传感器仿真，非实船试验或控制器闭环；采用板端实际输出，不做事后轨迹对齐。", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .92))
    for extension in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(extension), dpi=180, facecolor="white")
    plt.close(fig)
    return report


def plot_terrain(run, terrain, prefix):
    metadata = json.loads((terrain / "metadata.json").read_text())
    resolution = float(metadata["resolution_m"])
    elevation = np.load(terrain / "elevation_m.npy", mmap_mode="r")
    valid = np.load(terrain / "bathymetry_valid_mask.npy", mmap_mode="r")
    paths = [("真值", run["positions"][:, 0], "#263442", 2.5, (0, (5, 4))),
             ("板端 Local", run["positions"][:, 2], "#ee7921", 1.7, "-"),
             ("板端 Global", run["positions"][:, 3], "#007ac4", 1.1, "-")]
    xy = np.concatenate([p[:, :2] for _, p, *_ in paths])
    margin = max(100., .03 * float(np.ptp(xy, axis=0).max()))
    low, high = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    stride = max(1, int(np.ceil(max(high - low) / resolution / 500.)))
    points, triangles = measured_surface(elevation, valid, resolution,
                                         (low[0], high[0], low[1], high[1]), stride=stride)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_trisurf(points[:, 0] / 1000., points[:, 1] / 1000., points[:, 2], triangles=triangles,
                    color="#c1cdd1", alpha=.66, shade=True, lw=0, zorder=1, rasterized=True)
    for zorder, (name, path, color, width, style) in enumerate(paths, start=3):
        ax.plot(path[:, 0] / 1000., path[:, 1] / 1000., path[:, 2], color=color,
                lw=width, ls=style, label=name, zorder=zorder, rasterized=True)
    zmin = min(float(points[:, 2].min()), float(run["positions"][:, :, 2].min()))
    zmax = max(float(points[:, 2].max()), float(run["positions"][:, :, 2].max()))
    zspan = max(10., zmax - zmin)
    exaggeration = min(12., max(1., float(max(high - low)) / zspan / 3.))
    ax.set_box_aspect(((high - low)[0], (high - low)[1], zspan * exaggeration))
    ax.set_xlim(low[0] / 1000., high[0] / 1000.)
    ax.set_ylim(low[1] / 1000., high[1] / 1000.)
    ax.set_zlim(zmin, zmin + zspan)
    ax.view_init(elev=32, azim=-55)
    ax.set_xlabel("东向 / km")
    ax.set_ylabel("北向 / km")
    ax.set_zlabel("局部高程 / m")
    ax.legend(loc="upper left", ncol=3)
    fig.suptitle(f'实测地形与板端在线定位轨迹 · {run["distance"][-1] / 1000.:.3f} km', fontsize=18)
    fig.text(.06, .045, f'输入地形 {resolution:g} m；仅绘图 LOD {resolution*stride:g} m；垂向显示夸张 {exaggeration:.1f}×。', fontsize=10)
    fig.text(.06, .02, 'NoData 留空；轨迹不平移、不抬高、不贴地；真实接近的三条轨迹会自然重合。', fontsize=10)
    for extension in (".png", ".pdf"):
        fig.savefig(prefix.with_suffix(extension), dpi=180, facecolor="white")
    plt.close(fig)
    return dict(input_resolution_m=resolution, display_lod_m=resolution*stride,
                vertical_display_exaggeration=exaggeration, no_trajectory_coordinate_offsets=True,
                no_interpolation_across_NoData=True, all_online_samples_plotted=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="fresh figure/artifact directory")
    p.add_argument("--terrain", type=Path)
    p.add_argument("--require-full-route", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("fresh output directory required")
    run = load_run(args.run.resolve(), require_full_route=args.require_full_route)
    args.output.mkdir(parents=True)
    set_font()
    report = dict(profile=run["manifest"]["profile"], full_route=run["full_route"],
                  samples=len(run["timestamps"]), distance_m=float(run["distance"][-1]),
                  mode=run["manifest"]["mode"], live_Isaac=run["manifest"].get("live_Isaac", False),
                  online_board_errors=plot_errors(run, args.output / "board_position_errors"),
                  trajectory_alignment_applied=False, world_truth_sent_to_board_after_seed=False)
    if args.terrain:
        report["terrain_plot"] = plot_terrain(run, args.terrain, args.output / "board_3d_trajectories")
    columns = np.column_stack((run["timestamps"], run["distance"], run["positions"].reshape(-1, 12),
                               run["errors"], run["native_errors"], run["roundtrip"], run["lateness"]))
    names = ["timestamp_s", "distance_m"] + [f"{track}_{axis}_m" for track in ("truth", "dr", "local", "global")
             for axis in "xyz"] + [f"{track}_error_m" for track in ("dr", "local", "global")] + [
             f"native_{track}_error_m" for track in ("dr", "local", "global")] + ["roundtrip_s", "arrival_lateness_s"]
    np.savetxt(args.output / "aligned_online_trajectories.csv", columns, delimiter=",", header=",".join(names), comments="")
    report["input_sha256"] = {}
    for name in (run["reference"], run["outputs"]):
        h = hashlib.sha256()
        with name.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        report["input_sha256"][str(name)] = h.hexdigest()
    (args.output / "plot_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
