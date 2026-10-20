import importlib.util
import json
from pathlib import Path

import numpy as np

from oceansim.slam import InertialState
from oceansim.slam.math3d import rotation_from_rpy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('aperture_diagnostic', ROOT/'standalone/diagnose_terrain_aperture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_estimated_frame_composition_preserves_world_xyz():
    points = np.array([[1.,2.,3.],[-2.,1.,0.]])
    old_position = np.array([12.,7.,-20.])
    old_rotation = rotation_from_rpy(.1,.2,.3)
    reference = InertialState(position=np.array([14.,8.,-22.]),
                              rotation=rotation_from_rpy(-.2,.3,.6))
    transformed = module.transform_between_body_frames(points, old_position, old_rotation, reference)
    np.testing.assert_allclose(transformed @ reference.rotation.T + reference.position,
                               points @ old_rotation.T + old_position, atol=1.e-12)


def test_output_source_audit_cannot_select_using_truth(tmp_path):
    def record(t, status, pf_x):
        return dict(timestamp_s=t, truth_evaluation_only={'position':[0.,0.,0.]},
            decision=dict(status=status, cohort_count=1, best_map_pose2=[pf_x,0.,0.],
                          best_last_measurement_observable_rank=3))
    records = [record(30.,'tracking',0.), record(60.,'confirmed',20.)]
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    (tmp_path/'online_poses_5hz.csv').write_text(
        'timestamp_s,travelled_distance_m,truth_x_m,truth_y_m,truth_z_m,dr_x_m,dr_y_m,dr_z_m,local_x_m,local_y_m,local_z_m,global_x_m,global_y_m,global_z_m\n'
        '30,45,0,0,0,0,0,0,0,0,0,10,0,0\n'
        '60,90,0,0,0,0,0,0,0,0,0,5,0,0\n')
    result = module.audit_output_sources(tmp_path)
    # An unconfirmed perfect PF is ignored; a confirmed worse PF is retained.
    assert result['records'][0]['counterfactual_error_xy_m'] == 10.
    assert result['records'][1]['counterfactual_error_xy_m'] == 20.
    assert result['counterfactual']['rmse_xy_m'] > result['recorded_online']['rmse_xy_m']
