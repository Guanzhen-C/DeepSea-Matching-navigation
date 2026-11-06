"""Diagnose the first well-supported historical pair rejected by geometry.

Only recorded received scans and saved estimator-only poses are read. No
terrain, original truth NPZ or true pose is accessible to this diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from SideScanRecordedInputs import RecordedSideScanInputs
from SideScanFeatureMap import SideScanFeatureWindow, pose_matrix, matrix_pose
from SideScanLoopClosure import feature_observations, row_compatible_pairs
from SideScanLoopGeometry import fit_range_plane_loop
from SideScanLearnedFeatures import LoFTRSideScanMatcher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, type=Path)
    p.add_argument('--replay', required=True, type=Path)
    p.add_argument('--estimator-only-baseline', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve diagnostics.')
    trace = json.loads((a.replay/'historical_loop_trace.json').read_text())
    chosen = next((dict(new_timestamp_s=row['timestamp_s'], **c) for row in trace for c in row['candidates']
        if c.get('reason') == 'historical_feature_on_wrong_physical_sonar_side' and c.get('inliers', 0) >= 100), None)
    if chosen is None: raise ValueError('No registered well-supported rejected pair to inspect.')
    estimation = np.load(a.estimator_only_baseline/'estimated_states.npz')
    inputs = RecordedSideScanInputs(a.source_run/'recorded_inputs')
    def build(stamp):
        i = int(np.searchsorted(inputs.timestamps, stamp))
        if inputs.timestamps[i] != stamp: raise ValueError('Exact recorded timestamp required.')
        window = SideScanFeatureWindow(150)
        for j in range(max(0, i-149), i+1):
            pose = np.r_[estimation['positions'][j, 1, :2], estimation['yaw_rad'][j, 1]]
            window.add(inputs.packet(j), pose, j+1, vertical_position_m=estimation['positions'][j, 1, 2])
        q = window.build(pose)
        global_pose = np.r_[estimation['positions'][i, 2, :2], estimation['yaw_rad'][i, 2]]
        return q, global_pose
    old, old_pose = build(chosen['old_timestamp_s'])
    new, new_pose = build(chosen['new_timestamp_s'])
    seed = matrix_pose(np.linalg.inv(pose_matrix(old_pose))@pose_matrix(new_pose))
    reverse = np.cos(seed[2]) < 0
    image = new['image'][::-1, ::-1].copy() if reverse else new['image']
    mask = np.repeat(new['support'], new['scale'], axis=0).astype(np.uint8)*255
    if reverse: mask = mask[::-1, ::-1].copy()
    import torch
    torch.set_num_threads(1)
    pa, pb, _ = LoFTRSideScanMatcher()(old['image'], image,
        np.repeat(old['support'], old['scale'], axis=0).astype(np.uint8)*255, mask)
    if reverse: pb = np.array([image.shape[1]-1, image.shape[0]-1])-pb
    oa, ob = feature_observations(old, pa), feature_observations(new, pb)
    keep = row_compatible_pairs(oa, ob, seed[2])
    oa = {k: v[keep] for k, v in oa.items()}; ob = {k: v[keep] for k, v in ob.items()}
    diagnostic = []
    result = fit_range_plane_loop(oa, ob, seed, new['anchor_vertical_z_m']-old['anchor_vertical_z_m'],
                                 solution_diagnostics=diagnostic)
    a.out.mkdir()
    np.savez_compressed(a.out/'observed_correspondences_only.npz', old_image=old['image'], new_image=new['image'],
        old_pixels=pa[keep], new_pixels=pb[keep], seed_pose2=seed,
        delta_z=new['anchor_vertical_z_m']-old['anchor_vertical_z_m'],
        **{'a_'+k: v for k, v in oa.items()}, **{'b_'+k: v for k, v in ob.items()})
    report = dict(selected_without_truth_error=chosen, seed_pose2=seed.tolist(),
        matches=len(pa), row_consensus=int(keep.sum()), solver_multistart=diagnostic,
        geometry_result={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()},
        true_pose_or_DEM_read=False)
    (a.out/'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
