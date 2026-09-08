import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class _Attribute:
    def __init__(self):
        self.value = None


class _AttributeHandle:
    def __init__(self, attribute):
        self._attribute = attribute

    def set(self, value):
        self._attribute.value = value


class _Node:
    def __init__(self):
        self.attributes = {}

    def get_attribute(self, name):
        return self.attributes.setdefault(name, _Attribute())


def _load_ros_adapters(monkeypatch):
    omni = types.ModuleType("omni")
    omni_graph = types.ModuleType("omni.graph")
    omni_graph_core = types.ModuleType("omni.graph.core")
    omni_graph_core.Controller = types.SimpleNamespace(
        attribute=lambda attribute: _AttributeHandle(attribute)
    )
    omni.graph = omni_graph
    omni_graph.core = omni_graph_core

    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.graph", omni_graph)
    monkeypatch.setitem(sys.modules, "omni.graph.core", omni_graph_core)

    multibeam_parent = types.ModuleType(
        "isaacsim.oceansim.sensors.MultibeamEchosounderSensor"
    )
    multibeam_parent.MultibeamEchosounderSensor = type(
        "MultibeamEchosounderSensor", (), {}
    )
    sidescan_parent = types.ModuleType("isaacsim.oceansim.sensors.SideScanSonarSensor")
    sidescan_parent.SideScanSonarSensor = type("SideScanSonarSensor", (), {})
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.oceansim.sensors.MultibeamEchosounderSensor",
        multibeam_parent,
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.oceansim.sensors.SideScanSonarSensor",
        sidescan_parent,
    )

    modules = []
    for filename in (
        "MultibeamEchosounderSensor_ROS.py",
        "SideScanSonarSensor_ROS.py",
    ):
        module_name = f"_test_{filename[:-3]}"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "isaacsim" / "oceansim" / "sensors" / filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def test_multibeam_ros_adapter_writes_point_cloud(monkeypatch):
    multibeam_module, _ = _load_ros_adapters(monkeypatch)
    adapter = multibeam_module.MultibeamEchosounderSensor_ROS.__new__(
        multibeam_module.MultibeamEchosounderSensor_ROS
    )
    adapter._og_node = _Node()

    points = np.array([[1.0, 2.0, -10.0], [3.0, 4.0, -10.0]], dtype=np.float32)
    ping = types.SimpleNamespace(
        timestamp=12.5,
        valid_points=lambda frame: points,
    )
    adapter._publish_ping(ping)

    attributes = adapter._og_node.attributes
    assert attributes["inputs:timeStamp"].value == 12.5
    assert attributes["inputs:bufferSize"].value == points.nbytes
    np.testing.assert_array_equal(attributes["inputs:data"].value, points)


def test_sidescan_ros_adapter_writes_mono8_waterfall(monkeypatch):
    _, sidescan_module = _load_ros_adapters(monkeypatch)
    adapter = sidescan_module.SideScanSonarSensor_ROS.__new__(
        sidescan_module.SideScanSonarSensor_ROS
    )
    adapter._og_node = _Node()
    waterfall = np.array([[0, 64, 255], [8, 16, 32]], dtype=np.uint8)
    adapter.get_waterfall = lambda: waterfall

    adapter._publish_ping(types.SimpleNamespace(timestamp=7.25))

    attributes = adapter._og_node.attributes
    assert attributes["inputs:timeStamp"].value == 7.25
    assert attributes["inputs:width"].value == 3
    assert attributes["inputs:height"].value == 2
    assert attributes["inputs:bufferSize"].value == waterfall.size
    np.testing.assert_array_equal(
        attributes["inputs:data"].value,
        waterfall.reshape(-1),
    )
