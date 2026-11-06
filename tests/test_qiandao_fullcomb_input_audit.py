from pathlib import Path
import hashlib
import os
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
import AuditQiandaoFullCombInputs as audit


def test_linux_process_start_is_actual_past_start_not_current_file_timestamp():
    start = audit.process_start_unix(os.getpid())
    assert 0 <= time.time()-start < 3600


def test_streaming_digest_preserves_input(tmp_path):
    path = tmp_path/'native.bin'
    original = b'measured amplitude\0not RGB'
    path.write_bytes(original)
    assert audit.sha(path) == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


def test_wrong_unit_is_rejected_without_inspecting_other_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['audit', '--experiment', str(tmp_path),
        '--unit', 'unrelated.service', '--pid', '1234'])
    def forbidden(_):
        raise AssertionError('Must not query a unit outside the explicit Qiandao scope.')
    monkeypatch.setattr(audit, 'current_pid', forbidden)
    with pytest.raises(ValueError):
        audit.main()
    assert not (tmp_path/'post_launch_input_source_audit.json').exists()
