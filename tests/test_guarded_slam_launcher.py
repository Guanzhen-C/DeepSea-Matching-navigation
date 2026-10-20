from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('guarded_launcher', ROOT/'standalone/run_chelan_guarded.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_admission_uses_available_not_nominal_free_memory():
    assert guard.memory_available_bytes('MemTotal: 33554432 kB\nMemFree: 4096 kB\nMemAvailable: 12582912 kB\n') == 12 * 1024**3


def test_missing_available_memory_cannot_be_assumed_safe():
    with pytest.raises(KeyError):
        guard.memory_available_bytes('MemTotal: 33554432 kB\n')


def test_launcher_is_bounded_and_never_kills_other_jobs():
    source = (ROOT/'standalone/run_chelan_guarded.py').read_text()
    assert 'MemoryMax=7G' in source
    assert 'MemoryHigh=7G' in source
    assert 'MemoryHigh=6G' not in source
    assert 'MemorySwapMax=0' in source
    assert 'available < 12 * GIB' in source
    assert 'gpu_free_mib < 2048' in source
    assert "'/usr/bin/flock', '--nonblock'" in source
    assert 'if processes.returncode == 0:' in source
    assert 'kill(' not in source and 'pkill' not in source
    assert "'source_sha256'" in source


def test_monterey_runs_frozen_code_not_experimental_code():
    wrapper = (ROOT/'standalone/run_monterey_72km_stable.sh').read_text()
    standard = (ROOT/'standalone/run_v2_full_map_slam_isaac.sh').read_text()
    assert '8b5df53cb57f9de2506d78d90b16e33bf5e96510' in wrapper
    assert 'monterey-72km-stable-20260918' in wrapper
    assert '"${STABLE_DIR}/standalone/run_v2_full_map_slam_isaac.sh"' in wrapper
    assert '--terrain-merge-converged-cohorts' not in wrapper
    assert '--terrain-merge-converged-cohorts' not in standard


def test_online_pose_journal_flushes_before_isaac_shutdown():
    source = (ROOT/'standalone/large_scale_slam_isaac_demo.py').read_text()
    assert 'newline="", buffering=1' in source
    closing = source.index('            live_trace_stream.close()')
    assert closing < source.index('        with ARGS.metrics.open("w"')
    assert closing < source.index('        simulation_app.close()', closing)


def test_native_recovery_import_uses_isaacsim_package_namespace():
    source = (ROOT/'standalone/large_scale_slam_isaac_demo.py').read_text()
    assert 'from isaacsim.oceansim.slam.raw_mode_recovery import RawModeRecovery' in source
    assert 'from oceansim.slam.raw_mode_recovery import' not in source
