from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_scenario(monkeypatch):
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    prims = types.ModuleType("isaacsim.core.prims")
    prims.RigidPrim = object
    prims.SingleRigidPrim = object
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", prims)

    prim_utils = types.ModuleType("isaacsim.core.utils.prims")
    prim_utils.get_prim_path = lambda prim: str(prim)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.prims", prim_utils)

    spec = importlib.util.spec_from_file_location(
        "_test_sensor_example_scenario",
        ROOT
        / "isaacsim"
        / "oceansim"
        / "modules"
        / "SensorExample_python"
        / "scenario.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MHL_Sensor_Example_Scenario


class _AcousticSensor:
    def __init__(self):
        self.timestamps = []
        self.closed = False

    def scan(self, timestamp):
        self.timestamps.append(timestamp)

    def close(self):
        self.closed = True


def test_sensor_example_updates_acoustic_sensors_at_ten_hz(monkeypatch):
    Scenario = _load_scenario(monkeypatch)
    scenario = Scenario()
    scenario._running_scenario = True
    scenario._ctrl_mode = "No control"
    scenario._mbes = _AcousticSensor()
    scenario._sidescan = _AcousticSensor()

    for _ in range(10):
        scenario.update_scenario(0.02)

    assert len(scenario._mbes.timestamps) == 2
    assert len(scenario._sidescan.timestamps) == 2
    assert scenario._mbes.timestamps == scenario._sidescan.timestamps


def test_sensor_example_closes_acoustic_visualizations(monkeypatch):
    Scenario = _load_scenario(monkeypatch)
    scenario = Scenario()
    scenario._ctrl_mode = "No control"
    mbes = _AcousticSensor()
    sidescan = _AcousticSensor()
    scenario._mbes = mbes
    scenario._sidescan = sidescan

    scenario.teardown_scenario()

    assert mbes.closed
    assert sidescan.closed
