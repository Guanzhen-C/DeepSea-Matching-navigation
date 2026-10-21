import gzip
import json

import numpy as np
import pytest

from standalone.hil.protocol import pack
from standalone.hil.recording import LENGTH
from standalone.score_hil_variants import compare


def packets(path, records):
    with gzip.open(path, "wb") as stream:
        for record in records:
            data = pack(record)
            stream.write(LENGTH.pack(len(data)))
            stream.write(data)


def fixture(tmp_path, *, missing=False, wrong_time=False, invalid=False):
    recording = tmp_path / "synthetic_native_fixture"
    recording.mkdir()
    (recording / "recording_summary.json").write_text(json.dumps(dict(status="completed", samples=2, baseline_sha="synthetic_unit_fixture")))
    (recording / "native_metrics.json").write_text(json.dumps(dict(run_status="completed", pings=2, ping_rate_hz=5.)))
    references, estimates = [], []
    for stamp in (0., .2):
        reference = dict(timestamp=stamp, truth_position=np.array([1., 2., 3.]))
        estimate = dict(timestamp=stamp + (.01 if wrong_time else 0.))
        for track in ("dr", "local", "global"):
            reference[track + "_position"] = np.array([1.1, 2., 3.])
            estimate[track + "_position"] = np.array([np.nan if invalid else 1.3, 2., 3.])
        references.append(reference)
        estimates.append(estimate)
    packets(recording / "host_reference.osh.gz", references)
    candidate = tmp_path / "synthetic_candidate.osh.gz"
    packets(candidate, estimates[:1] if missing else estimates)
    return recording, candidate


def test_comparison_retains_original_reference_and_never_aligns_or_claims_acceptance(tmp_path):
    recording, candidate = fixture(tmp_path)
    result = compare(recording, {"candidate_PC": candidate}, control="candidate_PC")
    assert result["metrics"]["original_PC"]["global"]["rmse_m"] == pytest.approx(.1)
    assert result["metrics"]["candidate_PC"]["global"]["rmse_m"] == pytest.approx(.3)
    assert result["maximum_position_difference_from_control_m"]["original_PC"]["global"] == pytest.approx(.2)
    assert not result["full_route_acceptance"]
    assert not result["trajectory_alignment_applied"]


@pytest.mark.parametrize("bad", ["missing", "wrong_time", "invalid"])
def test_comparison_rejects_incomplete_or_invalid_measurements(tmp_path, bad):
    recording, candidate = fixture(tmp_path, **{bad: True})
    with pytest.raises(ValueError):
        compare(recording, {"candidate": candidate})


def test_cannot_replace_original_reference_with_new_variant(tmp_path):
    recording, candidate = fixture(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        compare(recording, {"original_PC": candidate})
