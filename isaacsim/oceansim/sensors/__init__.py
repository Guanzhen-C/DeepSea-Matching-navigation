"""OceanSim sensor exports loaded on first use.

Avoiding eager Isaac/Kit imports keeps numerical and Warp-only backends
testable outside a running SimulationApp.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "BarometerSensor": (".BarometerSensor", "BarometerSensor"),
    "BathymetricSideScanV2Profile": (
        ".bathymetric_sidescan_profiles",
        "BathymetricSideScanV2Profile",
    ),
    "BathymetricSideScanSonarSensor": (
        ".BathymetricSideScanSonarSensor",
        "BathymetricSideScanSonarSensor",
    ),
    "DVLsensor": (".DVLsensor", "DVLsensor"),
    "ImagingSonarSensor": (".ImagingSonarSensor", "ImagingSonarSensor"),
    "MultibeamEchosounderSensor": (
        ".MultibeamEchosounderSensor",
        "MultibeamEchosounderSensor",
    ),
    "SideScanSonarSensor": (".SideScanSonarSensor", "SideScanSonarSensor"),
    "STANDARD_V2_PROFILE": (
        ".bathymetric_sidescan_profiles",
        "STANDARD_V2_PROFILE",
    ),
    "HIGH_V2_PROFILE": (
        ".bathymetric_sidescan_profiles",
        "HIGH_V2_PROFILE",
    ),
    "get_v2_profile": (".bathymetric_sidescan_profiles", "get_v2_profile"),
    "UW_Camera": (".UW_Camera", "UW_Camera"),
    "WarpMeshRaycastBackend": (
        ".warp_raycast_backend",
        "WarpMeshRaycastBackend",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [*_LAZY_EXPORTS]
