from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from PrepareSideScanRuntime import source_closure, ENTRYPOINTS


def test_runtime_closure_captures_conditional_imports_and_cycles_without_executing_sources(tmp_path):
    (tmp_path/'main.py').write_text('if False:\n import child\nraise RuntimeError("never execute")\n')
    (tmp_path/'child.py').write_text('from main import anything\nimport json\n')
    (tmp_path/'not_used.py').write_text('unrelated source')
    (tmp_path/'data.npy').write_bytes(b'private data not a Python source')
    assert source_closure(tmp_path, ['main.py']) == ['child.py', 'main.py']


def test_runtime_closure_includes_actual_projected_loop_replay_and_independent_analysis_sources():
    source = Path(__file__).resolve().parents[1]/'standalone'
    names = set(source_closure(source, ENTRYPOINTS))
    assert {'ReplayPureSideScanNavigation.py', 'AnalyzeSideScanLoopReplay.py',
            'ValidateProjectedSideScanFullRoute.py',
            'SideScanHistoricalMatching.py', 'SideScanProjectedSubmap.py',
            'SideScanProjectedMatching.py', 'SideScanPlaneRegistration.py',
            'SideScanLoopGeometry.py', 'WriteSideScanReport.py'} <= names
