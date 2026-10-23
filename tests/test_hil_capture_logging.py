from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from standalone.hil.capture_native import Instrument, log_metrics_summary


class MustNotSerialize:
    """Stand-in for arbitrarily large report diagnostics, never for state."""


def test_final_log_ignores_large_diagnostics_and_does_not_mutate_report(capsys):
    payload = MustNotSerialize()
    metrics = dict(run_status="completed", pings=243569, global_error_m=4.35,
                   full_map_completed=True, global_cohort_records=payload)
    before = metrics.copy()
    log_metrics_summary(metrics, Path("/saved/native_metrics.json"))
    output = capsys.readouterr().out
    summary = json.loads(output.split(": ", 1)[1])
    assert len(output) < 1500
    assert summary["pings"] == 243569 and summary["global_error_m"] == 4.35
    assert summary["full_metrics_file"] == "/saved/native_metrics.json"
    assert summary["PC_reference_not_board_output"]
    assert summary["complete_report_retained_on_disk"]
    assert "global_cohort_records" not in summary
    assert metrics == before and metrics["global_cohort_records"] is payload


def test_final_log_retains_strict_json(capsys):
    with pytest.raises(ValueError):
        log_metrics_summary(dict(global_error_m=float("nan")), Path("/saved/report.json"))
    assert not capsys.readouterr().out


def test_only_identified_final_print_in_main_is_rewritten():
    tree = ast.parse('''
print("outside", json.dumps(metrics))
def main():
    metrics = {"pings": 3}
    print("ordinary", json.dumps(metrics))
    print("[OceanSim] live metrics: " + json.dumps(metrics, allow_nan=False), flush=True)
    return metrics
''')
    instrument = Instrument(Path("/bundle"))
    result = ast.fix_missing_locations(instrument.visit(tree))
    assert instrument.hooks == ["final_log"]
    main = result.body[1]
    assert ast.unparse(main.body[1]).startswith("print('ordinary'")
    assert ast.unparse(main.body[2]) == "hil_capture_log_metrics(metrics)"
    assert ast.unparse(result.body[0]).startswith("print('outside'")


def test_new_final_log_cannot_traverse_or_change_estimator_outputs():
    tree = ast.parse('''
def main(metrics):
    print("[OceanSim] live metrics: " f"{json.dumps(metrics, allow_nan=False)}", flush=True)
    return metrics
''')
    captured = []
    namespace = dict(hil_capture_log_metrics=captured.append)
    instrument = Instrument(Path("/bundle"))
    exec(compile(ast.fix_missing_locations(instrument.visit(tree)), "log-hook", "exec"), namespace)
    metrics = dict(pings=3, huge_diagnostics=MustNotSerialize())
    assert namespace["main"](metrics) is metrics
    assert captured == [metrics]
    assert instrument.hooks == ["final_log"]
