from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _load_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_sensor_classes(monkeypatch):
    class FakeBaseSensor:
        def __init__(
            self,
            prim_path,
            name,
            position=None,
            translation=None,
            orientation=None,
            scale=None,
            visible=None,
        ):
            self.prim_path = prim_path
            self.name = name
            self._position = np.zeros(3) if position is None else np.asarray(position)
            if translation is not None:
                self._position = np.asarray(translation)
            self._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        def initialize(self, physics_sim_view=None):
            return None

        def get_world_pose(self):
            return self._position, self._orientation

    class Timeline:
        @staticmethod
        def get_current_time():
            return 12.5

    package_names = [
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.utils",
        "isaacsim.oceansim",
        "isaacsim.oceansim.sensors",
        "isaacsim.oceansim.utils",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    base_sensor_module = types.ModuleType("isaacsim.core.api.sensors")
    base_sensor_module.BaseSensor = FakeBaseSensor
    monkeypatch.setitem(sys.modules, "isaacsim.core.api.sensors", base_sensor_module)

    rotations_module = types.ModuleType("isaacsim.core.utils.rotations")
    rotations_module.quat_to_rot_matrix = lambda quaternion: np.eye(3)
    monkeypatch.setitem(
        sys.modules, "isaacsim.core.utils.rotations", rotations_module
    )

    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: Timeline()
    omni_module = types.ModuleType("omni")
    omni_module.__path__ = []
    omni_module.timeline = timeline_module
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline_module)

    backend_module = types.ModuleType(
        "isaacsim.oceansim.sensors.raycast_backend"
    )
    backend_module.PhysXRaycastBackend = object
    monkeypatch.setitem(
        sys.modules, "isaacsim.oceansim.sensors.raycast_backend", backend_module
    )

    core = _load_file(
        "isaacsim.oceansim.utils.acoustic_sonar",
        ROOT / "isaacsim/oceansim/utils/acoustic_sonar.py",
    )
    _load_file(
        "isaacsim.oceansim.utils.acoustic_visualization",
        ROOT / "isaacsim/oceansim/utils/acoustic_visualization.py",
    )
    mbes_module = _load_file(
        "isaacsim.oceansim.sensors.MultibeamEchosounderSensor",
        ROOT / "isaacsim/oceansim/sensors/MultibeamEchosounderSensor.py",
    )
    sidescan_module = _load_file(
        "isaacsim.oceansim.sensors.SideScanSonarSensor",
        ROOT / "isaacsim/oceansim/sensors/SideScanSonarSensor.py",
    )
    integrated_module = _load_file(
        "isaacsim.oceansim.sensors.BathymetricSideScanSonarSensor",
        ROOT
        / "isaacsim/oceansim/sensors/BathymetricSideScanSonarSensor.py",
    )
    return (
        core,
        mbes_module.MultibeamEchosounderSensor,
        sidescan_module.SideScanSonarSensor,
        integrated_module.BathymetricSideScanSonarSensor,
    )


class HorizontalPlaneBackend:
    def __init__(self, raycast_batch_type, plane_z=-10.0):
        self._batch_type = raycast_batch_type
        self._plane_z = float(plane_z)
        self.calls = 0
        self.last_directions = None
        self.last_origins = None

    def cast(self, origin_world, directions_world, max_range):
        self.calls += 1
        directions = np.asarray(directions_world, dtype=np.float64)
        origin = np.asarray(origin_world, dtype=np.float64)
        if origin.shape == (3,):
            origins = np.broadcast_to(origin, directions.shape).copy()
        else:
            origins = origin.reshape(directions.shape)
        self.last_origins = origins.copy()
        self.last_directions = directions.copy()
        distances = (self._plane_z - origins[:, 2]) / directions[:, 2]
        hit = (
            np.isfinite(distances)
            & (distances >= 0.0)
            & (distances <= max_range)
        )
        positions = origins + directions * distances[:, None]
        positions[~hit] = np.nan
        normals = np.tile([0.0, 0.0, 1.0], (directions.shape[0], 1))
        return self._batch_type(
            hit_mask=hit,
            distances=np.where(hit, distances, np.nan),
            positions_world=positions,
            normals_world=normals,
            reflectivities=np.ones(directions.shape[0]),
            prim_paths=tuple("/World/seafloor" if value else "" for value in hit),
            ray_origins_world=origins,
        )


def test_vectorized_beam_footprints_match_scalar_reference(monkeypatch):
    _, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)
    line = np.array(
        [0.0, 0.8, 0.0, 0.8, 0.2, 0.0, 0.4, 0.0],
        dtype=np.float64,
    )
    source = np.array([-1, 17, -1, 23, 29, -1, 31, -1], dtype=np.int64)

    for radius in (0, 1, 3):
        expected = np.zeros_like(line)
        expected_source = np.full(source.shape, -1, dtype=np.int64)
        if radius <= 0:
            expected = line.copy()
            expected_source = source.copy()
        else:
            for centre in np.flatnonzero(source >= 0):
                for offset in range(-radius, radius + 1):
                    index = centre + offset
                    if index < 0 or index >= line.size:
                        continue
                    weight = 1.0 - abs(offset) / float(radius + 1)
                    contribution = line[centre] * weight
                    if contribution > expected[index]:
                        expected[index] = contribution
                        expected_source[index] = source[centre]

        actual, actual_source = IntegratedSensor._spread_beam_footprints(
            line, source, radius
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_source, expected_source)


def test_multibeam_sensor_produces_bathymetric_swath(monkeypatch):
    core, MultibeamSensor, _, _ = _load_sensor_classes(monkeypatch)
    backend = HorizontalPlaneBackend(core.RaycastBatch)
    sensor = MultibeamSensor(
        prim_path="/World/robot/mbes",
        num_beams=3,
        swath_width_deg=120.0,
        min_range=0.5,
        max_range=30.0,
        range_noise_std=0.0,
        attenuation=0.0,
        raycast_backend=backend,
    )

    ping = sensor.scan()

    np.testing.assert_allclose(ping.ranges, [20.0, 10.0, 20.0])
    np.testing.assert_allclose(ping.points_world[:, 2], -10.0)
    np.testing.assert_allclose(
        ping.two_way_travel_times, 2.0 * ping.ranges / 1500.0
    )
    assert ping.timestamp == 12.5
    assert sensor.get_pointcloud().shape == (3, 3)


def test_sidescan_sensor_builds_conventional_waterfall(monkeypatch):
    core, _, SideScanSensor, _ = _load_sensor_classes(monkeypatch)
    backend = HorizontalPlaneBackend(core.RaycastBatch)
    sensor = SideScanSensor(
        prim_path="/World/robot/sidescan",
        range_bins=32,
        waterfall_lines=4,
        rays_per_side=16,
        tilt_deg=45.0,
        vertical_beam_width_deg=40.0,
        min_range=0.5,
        max_range=30.0,
        range_noise_std=0.0,
        multiplicative_noise_std=0.0,
        additive_noise_std=0.0,
        attenuation=0.0,
        raycast_backend=backend,
    )

    ping = sensor.scan()

    assert np.count_nonzero(ping.port) > 0
    np.testing.assert_allclose(ping.port, ping.starboard)
    np.testing.assert_array_equal(ping.display_line[:32], ping.display_line[32:][::-1])
    waterfall = sensor.get_waterfall()
    assert waterfall.shape == (4, 64)
    np.testing.assert_array_equal(waterfall[-1], ping.display_line)
    assert sensor.get_waterfall(valid_only=True).shape == (1, 64)


def test_integrated_sensor_uses_one_cast_and_preserves_pixel_lineage(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)
    backend = HorizontalPlaneBackend(core.RaycastBatch)
    sensor = IntegratedSensor(
        prim_path="/World/robot/integrated_sonar",
        num_beams=8,
        swath_width_deg=120.0,
        imaging_rays_per_side=32,
        range_bins=40,
        waterfall_lines=4,
        beam_footprint_bins=0,
        min_range=0.5,
        max_range=30.0,
        range_noise_std=0.0,
        multiplicative_noise_std=0.0,
        additive_noise_std=0.0,
        attenuation=0.0,
        raycast_backend=backend,
    )

    ping = sensor.scan(timestamp=3.25)

    assert backend.calls == 1
    assert ping.timestamp == 3.25
    assert ping.bathymetry.timestamp == ping.side_scan.timestamp
    assert ping.bathymetry.frame_id == ping.side_scan.frame_id
    np.testing.assert_allclose(ping.origin_world, np.zeros(3))
    assert ping.shared_ray_count == backend.last_directions.shape[0]
    assert ping.shared_ray_count == 64
    assert ping.products_share_raw_returns
    np.testing.assert_array_equal(
        ping.imaging.source_ray_indices, np.arange(ping.shared_ray_count)
    )
    np.testing.assert_allclose(
        ping.shared_ray_directions[ping.bathymetry_source_ray_indices],
        ping.bathymetry.beam_directions,
        atol=1.0e-14,
    )
    points, backscatter, beam_indices = ping.valid_registered_samples()
    assert points.shape == (8, 3)
    assert backscatter.shape == (8,)
    np.testing.assert_array_equal(beam_indices, np.arange(8))
    imaging_points, imaging_backscatter, source_rays = (
        ping.valid_imaging_samples()
    )
    assert imaging_points.shape == (64, 3)
    assert imaging_backscatter.shape == (64,)
    assert source_rays.shape == (64,)
    assert np.unique(source_rays).size == source_rays.size
    np.testing.assert_allclose(
        ping.shared_ray_directions[source_rays],
        ping.imaging.beam_directions,
        atol=1.0e-14,
    )
    raw_indices = ping.bathymetry_source_ray_indices
    np.testing.assert_array_equal(
        ping.bathymetry.ranges, ping.imaging.ranges[raw_indices]
    )
    np.testing.assert_array_equal(
        ping.bathymetry.intensities, ping.imaging.intensities[raw_indices]
    )
    np.testing.assert_array_equal(
        ping.bathymetry.points_sensor,
        ping.imaging.points_sensor[raw_indices],
    )

    for line, source in (
        (ping.side_scan.port, ping.port_source_imaging_indices),
        (ping.side_scan.starboard, ping.starboard_source_imaging_indices),
    ):
        populated = source >= 0
        np.testing.assert_allclose(
            line[populated], ping.imaging.intensities[source[populated]]
        )
    port_sources = ping.port_source_beam_indices[
        ping.port_source_beam_indices >= 0
    ]
    starboard_sources = ping.starboard_source_beam_indices[
        ping.starboard_source_beam_indices >= 0
    ]
    assert np.all(ping.bathymetry.beam_directions[port_sources, 1] >= 0.0)
    assert np.all(ping.bathymetry.beam_directions[starboard_sources, 1] < 0.0)
    assert sensor.get_waterfall(valid_only=True).shape == (1, 80)

    ping.side_scan.timestamp += 0.01
    with pytest.raises(ValueError, match="share one timestamp"):
        core.BathymetricSideScanPing(
            bathymetry=ping.bathymetry,
            side_scan=ping.side_scan,
            port_source_beam_indices=ping.port_source_beam_indices,
            starboard_source_beam_indices=(
                ping.starboard_source_beam_indices
            ),
        )


def test_integrated_sensor_emits_multibeam_sidescan_rows(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)
    backend = HorizontalPlaneBackend(core.RaycastBatch)
    sensor = IntegratedSensor(
        prim_path="/World/robot/v2_sonar",
        num_beams=8,
        swath_width_deg=170.0,
        imaging_rays_per_side=16,
        along_track_beams_per_side=8,
        along_track_beam_spacing_deg=0.14,
        range_bins=40,
        waterfall_lines=20,
        waterfall_decimation=2,
        beam_footprint_bins=0,
        min_range=0.0,
        max_range=250.0,
        range_noise_std=0.0,
        multiplicative_noise_std=0.0,
        additive_noise_std=0.0,
        attenuation=0.0,
        random_seed=4,
        raycast_backend=backend,
        viewport=True,
    )

    ping = sensor.scan(timestamp=4.0)

    assert backend.calls == 1
    assert ping.side_scan.port_beams.shape == (8, 40)
    assert ping.side_scan.starboard_beams.shape == (8, 40)
    np.testing.assert_allclose(
        ping.side_scan.port,
        np.max(ping.side_scan.port_beams, axis=0),
    )
    np.testing.assert_allclose(
        ping.side_scan.starboard,
        np.max(ping.side_scan.starboard_beams, axis=0),
    )
    assert np.all(np.abs(ping.imaging.beam_directions[:, 0]) > 0.0)
    raw = ping.side_scan.as_raw_sonar_image_dict()
    assert raw["image"]["beam_count"] == 16
    assert raw["samples_per_beam"] == 40
    # One physical ping contributes all eight along-track image rows; native
    # 40-bin sides are max-pooled by two only for the live waterfall.
    assert sensor.get_waterfall(valid_only=True).shape == (8, 40)
    processed = sensor.get_processed_waterfall(valid_only=True)
    assert processed.shape == (8, 40)
    assert np.count_nonzero(processed) > np.count_nonzero(
        sensor.get_waterfall(valid_only=True)
    )
    # Slant correction leaves only a narrow filtered trackline, rather than a
    # water-column band proportional to vehicle altitude.
    centre = processed.shape[1] // 2
    assert np.count_nonzero(processed[:, centre - 3 : centre + 3]) > 0


def test_processed_view_does_not_change_raw_measurements(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)

    def make_sensor(viewport):
        return IntegratedSensor(
            prim_path=f"/World/robot/integrated_{viewport}",
            num_beams=8,
            swath_width_deg=120.0,
            imaging_rays_per_side=32,
            range_bins=80,
            waterfall_lines=8,
            min_range=0.5,
            max_range=30.0,
            range_noise_std=0.02,
            multiplicative_noise_std=0.04,
            additive_noise_std=0.008,
            attenuation=0.003,
            random_seed=31,
            raycast_backend=HorizontalPlaneBackend(core.RaycastBatch),
            viewport=viewport,
        )

    raw_only = make_sensor(False)
    with_processed_view = make_sensor(True)
    for timestamp in (1.0, 2.0):
        raw_ping = raw_only.scan(timestamp=timestamp)
        viewed_ping = with_processed_view.scan(timestamp=timestamp)
        np.testing.assert_array_equal(
            raw_ping.bathymetry.ranges, viewed_ping.bathymetry.ranges
        )
        np.testing.assert_array_equal(
            raw_ping.imaging.intensities, viewed_ping.imaging.intensities
        )
        np.testing.assert_array_equal(
            raw_ping.side_scan.display_line,
            viewed_ping.side_scan.display_line,
        )
        np.testing.assert_array_equal(
            raw_ping.port_source_imaging_indices,
            viewed_ping.port_source_imaging_indices,
        )


def test_integrated_sensor_can_throttle_only_visualization(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)

    def make_sensor(name):
        return IntegratedSensor(
            prim_path=f"/World/robot/{name}",
            num_beams=8,
            swath_width_deg=120.0,
            imaging_rays_per_side=32,
            range_bins=80,
            waterfall_lines=16,
            min_range=0.5,
            max_range=30.0,
            range_noise_std=0.02,
            multiplicative_noise_std=0.04,
            additive_noise_std=0.008,
            attenuation=0.003,
            random_seed=31,
            raycast_backend=HorizontalPlaneBackend(core.RaycastBatch),
            viewport=True,
        )

    throttled = make_sensor("throttled")
    reference = make_sensor("reference")
    mbes_updates = []
    sidescan_updates = []
    throttled._update_visualization = lambda *args: mbes_updates.append(args)
    throttled._update_sidescan_visualization = (
        lambda *args: sidescan_updates.append(args)
    )

    skipped = throttled.scan(timestamp=1.0, update_visualization=False)
    reference_skipped = reference.scan(timestamp=1.0)
    assert mbes_updates == []
    assert sidescan_updates == []
    assert throttled.get_processed_waterfall(valid_only=True).shape[0] == 0

    presented = throttled.scan(timestamp=2.0, update_visualization=True)
    reference_presented = reference.scan(timestamp=2.0)
    assert len(mbes_updates) == len(sidescan_updates) == 1
    assert throttled.get_processed_waterfall(valid_only=True).shape[0] == 1

    for actual, expected in (
        (skipped, reference_skipped),
        (presented, reference_presented),
    ):
        np.testing.assert_array_equal(
            actual.bathymetry.ranges, expected.bathymetry.ranges
        )
        np.testing.assert_array_equal(
            actual.imaging.intensities, expected.imaging.intensities
        )
        np.testing.assert_array_equal(
            actual.side_scan.display_line, expected.side_scan.display_line
        )

    # The raw slant-range waterfall remains a full-rate sensor/ROS product.
    assert throttled.get_waterfall(valid_only=True).shape[0] == 2


def test_integrated_sensor_uses_parallel_spatial_row_origins(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)
    backend = HorizontalPlaneBackend(core.RaycastBatch)
    rays_per_side = 16
    rows_per_side = 8
    spacing_m = 0.15
    sensor = IntegratedSensor(
        prim_path="/World/robot/v2_parallel_sonar",
        translation=np.array([10.0, 20.0, 0.0]),
        num_beams=8,
        swath_width_deg=120.0,
        imaging_rays_per_side=rays_per_side,
        along_track_beams_per_side=rows_per_side,
        along_track_origin_spacing_m=spacing_m,
        range_bins=40,
        waterfall_lines=20,
        beam_footprint_bins=0,
        min_range=0.0,
        max_range=30.0,
        range_noise_std=0.0,
        multiplicative_noise_std=0.0,
        additive_noise_std=0.0,
        attenuation=0.0,
        raycast_backend=backend,
    )

    ping = sensor.scan(timestamp=4.0)

    assert backend.calls == 1
    assert ping.timestamp == 4.0
    imaging_source = ping.imaging.source_ray_indices
    np.testing.assert_allclose(
        ping.shared_ray_origins_sensor[imaging_source],
        ping.imaging.ray_origins_sensor,
    )
    np.testing.assert_allclose(
        ping.shared_ray_origins_world[imaging_source],
        ping.imaging.ray_origins_world,
    )
    np.testing.assert_allclose(
        backend.last_origins, ping.shared_ray_origins_world
    )
    bathymetry_source = ping.bathymetry_source_ray_indices
    np.testing.assert_array_equal(
        ping.bathymetry.ranges,
        ping.imaging.ranges[bathymetry_source],
    )
    np.testing.assert_array_equal(
        ping.bathymetry.intensities,
        ping.imaging.intensities[bathymetry_source],
    )
    np.testing.assert_array_equal(
        ping.bathymetry.points_sensor,
        ping.imaging.points_sensor[bathymetry_source],
    )
    assert ping.along_track_coverage_m == pytest.approx(1.2)
    assert np.any(np.abs(ping.bathymetry.points_sensor[:, 0]) > 0.0)

    samples_per_side = rows_per_side * rays_per_side
    port_origins = ping.imaging.ray_origins_sensor[:samples_per_side].reshape(
        rows_per_side, rays_per_side, 3
    )
    row_centres = port_origins[:, 0, 0]
    np.testing.assert_allclose(np.diff(row_centres), spacing_m)
    assert row_centres[0] - 0.5 * spacing_m == pytest.approx(-0.6)
    assert row_centres[-1] + 0.5 * spacing_m == pytest.approx(0.6)

    port_directions = ping.imaging.beam_directions[:samples_per_side].reshape(
        rows_per_side, rays_per_side, 3
    )
    for row in range(1, rows_per_side):
        np.testing.assert_allclose(port_directions[row], port_directions[0])
    np.testing.assert_array_equal(ping.imaging.beam_directions[:, 0], 0.0)
    # A horizontal plane and zero local-X direction preserve each virtual
    # phase centre's X coordinate at every returned range.
    np.testing.assert_allclose(
        ping.imaging.points_sensor[:, 0],
        ping.imaging.ray_origins_sensor[:, 0],
    )


def test_integrated_geometry_and_intensity_share_the_same_noise_draw(monkeypatch):
    core, _, _, IntegratedSensor = _load_sensor_classes(monkeypatch)
    integrated = IntegratedSensor(
        prim_path="/World/robot/integrated",
        num_beams=8,
        imaging_rays_per_side=16,
        range_bins=32,
        min_range=0.5,
        max_range=30.0,
        range_noise_std=0.05,
        multiplicative_noise_std=0.2,
        additive_noise_std=0.1,
        random_seed=77,
        raycast_backend=HorizontalPlaneBackend(core.RaycastBatch),
    )

    for timestamp in (1.0, 2.0, 3.0):
        integrated_ping = integrated.scan(timestamp=timestamp)
        source = integrated_ping.bathymetry_source_ray_indices
        np.testing.assert_array_equal(
            integrated_ping.bathymetry.ranges,
            integrated_ping.imaging.ranges[source],
        )
        np.testing.assert_array_equal(
            integrated_ping.bathymetry.intensities,
            integrated_ping.imaging.intensities[source],
        )
        np.testing.assert_array_equal(
            integrated_ping.bathymetry.hit_mask,
            integrated_ping.imaging.hit_mask[source],
        )
