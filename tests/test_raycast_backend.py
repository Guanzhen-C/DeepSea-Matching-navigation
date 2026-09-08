from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_physx_backend(monkeypatch):
    class Float3:
        def __init__(self, *values):
            self.values = tuple(float(value) for value in values)

        def __getitem__(self, index):
            return self.values[index]

    class PlaneQuery:
        def __init__(self):
            self.origins = []

        def raycast_closest(self, origin, direction, max_range):
            origin_values = np.asarray(origin.values)
            direction_values = np.asarray(direction.values)
            self.origins.append(origin_values.copy())
            distance = (-10.0 - origin_values[2]) / direction_values[2]
            return {
                "hit": 0.0 <= distance <= max_range,
                "distance": distance,
                "normal": [0.0, 0.0, 1.0],
                "collision": "/World/Seabed",
            }

    query = PlaneQuery()
    carb = types.ModuleType("carb")
    carb.Float3 = Float3
    carb.log_warn = lambda _message: None
    monkeypatch.setitem(sys.modules, "carb", carb)

    package_names = [
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.utils",
        "isaacsim.oceansim",
        "isaacsim.oceansim.sensors",
        "isaacsim.oceansim.utils",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    semantics = types.ModuleType("isaacsim.core.utils.semantics")
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.semantics", semantics)
    sys.modules["isaacsim.core.utils"].semantics = semantics

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_usd = types.ModuleType("omni.usd")
    omni_physx = types.ModuleType("omni.physx")
    omni_physx.get_physx_scene_query_interface = lambda: query
    omni.usd = omni_usd
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    monkeypatch.setitem(sys.modules, "omni.physx", omni_physx)

    acoustic_name = "isaacsim.oceansim.utils.acoustic_sonar"
    acoustic_spec = importlib.util.spec_from_file_location(
        acoustic_name,
        ROOT / "isaacsim/oceansim/utils/acoustic_sonar.py",
    )
    acoustic = importlib.util.module_from_spec(acoustic_spec)
    monkeypatch.setitem(sys.modules, acoustic_name, acoustic)
    acoustic_spec.loader.exec_module(acoustic)

    backend_name = "isaacsim.oceansim.sensors.raycast_backend"
    backend_spec = importlib.util.spec_from_file_location(
        backend_name,
        ROOT / "isaacsim/oceansim/sensors/raycast_backend.py",
    )
    backend_module = importlib.util.module_from_spec(backend_spec)
    monkeypatch.setitem(sys.modules, backend_name, backend_module)
    backend_spec.loader.exec_module(backend_module)
    return backend_module.PhysXRaycastBackend, query


def test_physx_backend_accepts_shared_and_per_ray_origins(monkeypatch):
    Backend, query = _load_physx_backend(monkeypatch)
    backend = Backend(reflectivity_resolver=lambda _hit: 0.8)
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    origins = np.array([[-0.525, 0.0, 0.0], [0.525, 0.0, -2.0]])

    result = backend.cast(origins, directions, 20.0)

    np.testing.assert_allclose(query.origins, origins)
    np.testing.assert_allclose(result.distances, [10.0, 8.0])
    np.testing.assert_allclose(result.positions_world[:, 0], origins[:, 0])
    np.testing.assert_allclose(result.positions_world[:, 2], -10.0)
    np.testing.assert_allclose(result.ray_origins_world, origins)

    query.origins.clear()
    shared = np.array([1.0, 2.0, 0.0])
    legacy_result = backend.cast(shared, directions, 20.0)
    np.testing.assert_allclose(query.origins, [shared, shared])
    np.testing.assert_allclose(
        legacy_result.ray_origins_world, [shared, shared]
    )
