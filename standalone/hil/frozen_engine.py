"""Extract the ENTIRE frozen estimator, excluding Kit, sensor generation and truth.

This deliberately fails closed on any unknown source hash. The algorithm's
initialization, all backend closures, and the whole sonar update are copied as
AST nodes, not reimplemented. Only three diagnostic truth_pose2 assignments
become NaN and diagnostic truth arguments become None. A separate native-vs-
extracted replay must still pass before this is accepted as an HIL baseline.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

BASELINES = {
    "lake": ("204d3e24048a8b515616d9305b7516221bcf8fed",
             "649daf3dd3bc5d3997c92065eacc49cf5597c493afc12b99f426e2705efbf2b7"),
    "sea": ("8b5df53cb57f9de2506d78d90b16e33bf5e96510",
            "a11700c8c71aef21881f57f11d5ba322cc82400d16a6cc9b23e5c559c50064cc"),
}
CONSTANTS = {"LOCAL_FRONTEND", "LOCAL_INTENSITY_STATE_UPDATE_ENABLED", "POSE_GRAPH_BACKEND",
             "TERRAIN_PARTICLE_ONLINE_OUTPUT", "MAP_TO_ODOM_SOFT_CORRECTION_ENABLED"}


def _assigned(node, name):
    return isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
        isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store) and x.id == name
        for x in ast.walk(node.targets[0] if isinstance(node, ast.Assign) else node.target))


class _Portable(ast.NodeTransformer):
    def __init__(self):
        self.truth_diagnostics_removed = []

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("isaacsim.oceansim"):
            node.module = node.module.removeprefix("isaacsim.")
        return node

    def visit_Assign(self, node):
        if _assigned(node, "truth_pose2"):
            self.truth_diagnostics_removed.append(node.lineno)
            node.value = ast.parse("np.full(3, np.nan)", mode="eval").body
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg in ("truth_state", "evaluation_truth_state") and isinstance(node.value, ast.Name):
            node.value = ast.Constant(None)
        return self.generic_visit(node)


def source_tree(path: Path, profile: str):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASELINES[profile][1]:
        raise ValueError("unrecognized baseline source; refusing algorithm substitution")
    return ast.parse(source.decode()), hashlib.sha256(source).hexdigest()


def _snapshot_nodes():
    return ast.parse('''
output = {
    "timestamp": float(sim_time), "ping_count": int(ping_count),
    "dr_position": dead_reckoning.state.position.copy(),
    "dr_rotation": dead_reckoning.state.rotation.copy(),
    "local_position": slam.state.position.copy(),
    "local_rotation": slam.state.rotation.copy(),
    "global_position": global_position.copy(), "global_rotation": global_rotation.copy(),
    "terrain_match_attempts": int(terrain_match_attempts),
    "terrain_prior_accepts": int(terrain_prior_accepts),
    "keyframes": len(slam.pose_graph.keyframes),
    "output_policy": TERRAIN_PARTICLE_ONLINE_OUTPUT,
    "feature_backend": slam.feature_odometry.active_compute_backend,
    "feature_cuda_calls": slam.feature_odometry.diagnostics().cuda_ransac_calls,
}
''').body


def build_module(path: Path, profile: str):
    tree, digest = source_tree(path, profile)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body) if _assigned(n, "imu_bias_limit"))
    stop = next(i for i, n in enumerate(main.body) if _assigned(n, "truth_track"))
    initialization = copy.deepcopy(main.body[start:stop])
    candidates = [n for n in ast.walk(main) if isinstance(n, ast.If)
                  and "next_ping" in ast.unparse(n.test) and any(_assigned(b, "terrain_points_sensor") for b in n.body)]
    if len(candidates) != 1:
        raise ValueError("sonar loop extraction is ambiguous")
    body = candidates[0].body
    begin = next(i for i, n in enumerate(body) if _assigned(n, "terrain_points_sensor"))
    end = next(i for i, n in enumerate(body) if _assigned(n, "ping_samples"))
    sonar = copy.deepcopy(body[begin:end])
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import) and all(n.name.split(".")[0] in {
            "argparse", "atexit", "csv", "json", "math", "os", "random", "sys", "time", "traceback", "numpy"
        } for n in node.names):
            imports.append(copy.deepcopy(node))
        elif isinstance(node, ast.ImportFrom) and (node.module in {"__future__", "collections", "dataclasses", "pathlib"}
              or (node.module or "").startswith("isaacsim.oceansim.slam")
              or node.module == "isaacsim.oceansim.sensors.bathymetric_sidescan_profiles"):
            # Do not import the simulated sensor generator on the board.
            if node.module == "isaacsim.oceansim.slam.large_scale_sim":
                continue
            imports.append(copy.deepcopy(node))
    helpers = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in {"parse_args", "_finite_float_or_none", "_chi_square_sample_summary"}]
    constants = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.Assign)
                 and any(_assigned(n, key) for key in CONSTANTS)]
    prefix = ast.parse('''
def create_engine(ARGS, world_model, initial, startup_specific_force):
    if ARGS.survey_pattern != "full-map-route":
        raise ValueError("HIL only supports the frozen full-map mission")
    if ARGS.terrain_replay_directory is not None:
        raise ValueError("host-only truth replay archives must remain on the PC")
    sonar_profile = get_v2_profile({"v2-standard": "standard", "v2-high-resolution": "high"}[ARGS.sonar_profile])
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
''').body[0]
    prefix.body += initialization
    loop = ast.parse('''
ping_count = 0
ping_period = 1.0 / ARGS.ping_hz
sim_time = 0.0
latest_gyro = np.zeros(3)
latest_acceleration = np.mean(startup_specific_force, axis=0)
performance_wall_s = Counter()
output = None
while True:
    event = yield output
    sim_time = event["timestamp"]
    payload = event["payload"]
    if event["kind"] == "imu":
        latest_gyro = payload["gyro"]
        latest_acceleration = payload["acceleration"]
        slam.process_imu(sim_time, latest_gyro, latest_acceleration)
        dead_reckoning.propagate(sim_time, latest_gyro, latest_acceleration)
    elif event["kind"] == "dvl":
        slam.process_dvl(payload["velocity_body"])
        dead_reckoning.update_dvl(payload["velocity_body"])
    elif event["kind"] == "depth":
        slam.process_depth(payload["depth_m"])
        dead_reckoning.update_depth(payload["depth_m"])
    elif event["kind"] == "sonar":
        sonar_ping = restore_acoustic(payload)
        if abs(sonar_ping.timestamp - sim_time) > 1e-9:
            raise ValueError("sonar timestamp mismatch")
        if slam.state.timestamp < sim_time - 1e-12:
            slam.process_imu(sim_time, latest_gyro, latest_acceleration)
            dead_reckoning.propagate(sim_time, latest_gyro, latest_acceleration)
    elif event["kind"] == "finish":
        return {
            "feature_diagnostics": asdict(slam.feature_odometry.diagnostics()),
            "graph_diagnostics": asdict(slam.pose_graph.diagnostics()),
            "graph_factors": slam.pose_graph.factor_stream_payload(),
            "terrain_match_attempts": terrain_match_attempts,
            "terrain_prior_accepts": terrain_prior_accepts,
            "terrain_particle_statistics": terrain_backend_particle_stats,
            "terrain_navigation_output_statistics": terrain_navigation_output_stats,
        }
    else:
        raise ValueError("unknown estimator event")
''').body
    sonar_branch = loop[-1].body[-1].orelse[0].orelse[0].orelse[0]
    if not isinstance(sonar_branch, ast.If) or "sonar" not in ast.unparse(sonar_branch.test):
        raise AssertionError("bad generated loop")
    sonar_branch.body += sonar
    sonar_branch.body += _snapshot_nodes()
    # Only sonar events publish positions, matching the original 5 Hz output.
    loop[-1].body.insert(2, ast.parse("output = None").body[0])
    prefix.body += loop
    module = ast.Module(body=imports + constants + helpers + [prefix], type_ignores=[])
    transform = _Portable()
    module = transform.visit(module)
    ast.fix_missing_locations(module)
    if len(transform.truth_diagnostics_removed) != 3:
        raise ValueError("unexpected diagnostic-truth transformation count")
    forbidden = {"truth", "truth0", "navigation", "simulation_app", "World", "carb", "omni", "Gf"}
    residual = [(n.id, n.lineno) for n in ast.walk(prefix) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load) and n.id in forbidden]
    if residual:
        raise ValueError(f"live simulation dependency in estimator: {residual}")
    return module, {"profile": profile, "baseline_sha": BASELINES[profile][0], "source_sha256": digest,
                    "initialization_lines": [main.body[start].lineno, main.body[stop - 1].end_lineno],
                    "sonar_lines": [body[begin].lineno, body[end - 1].end_lineno],
                    "diagnostic_truth_replacements": transform.truth_diagnostics_removed,
                    "native_parity_validated": False}


def load_engine(path: Path, profile: str, root: Path):
    from standalone.hil.protocol import restore_acoustic
    module, audit = build_module(path, profile)
    namespace = {"ROOT": root, "__file__": str(path), "restore_acoustic": restore_acoustic}
    exec(compile(module, str(path) + "::hil-extracted", "exec"), namespace)
    return namespace, audit
