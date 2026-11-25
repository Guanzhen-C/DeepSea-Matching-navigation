"""Local-only scene inventory. Presence of a map never supplies an original pair."""
import argparse
from collections import Counter
import json
from pathlib import Path

from research.sonar_range_factor_v3.replay import sha
from research.sonar_range_factor_v4.replay import ROOT,SCENES,load_scene
from .replay import write_json

FILES={
    "Monterey":"demo/hil/jetson_20261021/sea_live_5km_canonical_v3/native_reference/native_metrics_pose_graph_factors.json",
    "Chelan":"demo/generalization/final_improved_multiterrain_20261015/chelan_full_route_seed31_confirmed_output_fix_20261016_pose_graph_factors.json",
    "Jenkinson":"demo/generalization/final_improved_multiterrain_20261015/jenkinson_full_route_seed31_confirmed_output_fix_20261016_pose_graph_factors.json",
    "Sheboygan":"demo/generalization/final_improved_multiterrain_20261015/sheboygan_full_route_seed31_confirmed_output_fix_20261016_pose_graph_factors.json",
}


def run():
    old={}
    for region,name in FILES.items():
        f=ROOT/name
        g=json.loads(f.read_text())
        old[region]=dict(representative=name,sha256=sha(f),nodes=len(g["pose_initializations"]),
            kind=dict(Counter(e["kind"] for e in g["edges"])),
            node_timestamps_present=bool(g.get("pose_timestamps_s")),
            any_raw_local_keyframe_vertical_values=bool(g.get("raw_local_keyframe_positions_xyz")),
            original_paired_ranges=bool(g.get("mapped_range_changes")),
            pair_source_tokens=(f.parent/"mapped_range_change_trace.json").is_file(),
            pair_phase_centres_and_acquisition_pressure_linked=False,
            independent_field_truth_verified=False,ready=False)
    qiandao={}
    for name in SCENES:
        g,tr,_truth,hashes,src=load_scene(name)
        obs=g["mapped_range_changes"][0]["observations"]
        qiandao[name]=dict(graph_hashes=hashes,seed=src["source_seed"],range_batches=len(tr),
            original_pair_fields=sorted({"range_a","range_b","origin_a","origin_b","sensor_z_a","sensor_z_b"}&set(obs)),
            tokens_verified_in_replay=True,
            mapped_factor_times_and_node_times=True,absolute_ping_clock_must_bind_with_original_receiver_log=True,
            independent_field_truth_verified=False,simulator_truth_present=True,ready=True)
    root=ROOT/"data/自动化所"
    field=dict(metadata_directory_found=root.is_dir(),
        port_beamformed_mat_files=sum(1 for _ in root.rglob("PortNormalModeBeamformingTotal.mat")),
        starboard_beamformed_mat_files=sum(1 for _ in root.rglob("StbdNormalModeBeamformingTotal.mat")),
        system_config_mat_files=sum(1 for _ in root.rglob("SysPara.mat")),
        GNSS_mat_files=sum(1 for _ in root.rglob("GPS_data.mat")),
        calibrated_surveyed_target_pairs_verified=False,independent_field_truth_verified=False,
        note="native complex beamformed data are documented in stage two; filename counts do not certify synchronization")
    return dict(schema="ocean_range_v5_archive_inventory",code_sha256=sha(Path(__file__)),historical=old,
                qiandao_simulation=qiandao,qiandao_local_field=field)


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error("Choose a new output path")
    write_json(a.output,run())
