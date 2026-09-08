"""OceanSim package.

Isaac/Kit modules are loaded lazily so dependency-free utilities such as the
SLAM estimator can also be imported by ROS tools and unit tests outside Kit.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "BarometerSensor": (".sensors.BarometerSensor", "BarometerSensor"),
    "BathymetricSideScanSonarSensor": (
        ".sensors.BathymetricSideScanSonarSensor",
        "BathymetricSideScanSonarSensor",
    ),
    "DVLsensor": (".sensors.DVLsensor", "DVLsensor"),
    "ImagingSonarSensor": (
        ".sensors.ImagingSonarSensor",
        "ImagingSonarSensor",
    ),
    "MultibeamEchosounderSensor": (
        ".sensors.MultibeamEchosounderSensor",
        "MultibeamEchosounderSensor",
    ),
    "SideScanSonarSensor": (
        ".sensors.SideScanSonarSensor",
        "SideScanSonarSensor",
    ),
    "UW_Camera": (".sensors.UW_Camera", "UW_Camera"),
    "WaterSurface": (".watersurface.WaterSurface", "WaterSurface"),
    "UWCam_KittiWriter": (".writers.UWCam_KittiWriter", "UWCam_KittiWriter"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "sensors",
    "slam",
    "utils",
    "watersurface",
    "writers",
    *_LAZY_EXPORTS,
]
