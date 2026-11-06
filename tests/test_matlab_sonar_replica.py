from pathlib import Path

import numpy as np
import pytest

from standalone.BathymetrySideScanDrawing import (
    PingFile, averaged_samples, depth_points, matlab_colon, run,
    select_records, sidescan_step, write_interpolated_grid,
)
from standalone.Bathymetry_SonarDefaultColormap3D import Bathymetry_SonarDefaultColormap3D
from standalone.caris_orangecopper import caris_orangecopper


def test_palette_control_points_and_original_fixed_bathymetry_length():
    cp = caris_orangecopper(7)
    np.testing.assert_allclose(cp, [[0, 0, 0], [.22, .1, .04], [.48, .24, .08],
                                   [.74, .38, .1], [.92, .56, .16], [1, .78, .32], [1, .94, .7]])
    np.testing.assert_allclose(caris_orangecopper(1)[0], [1, .94, .7])
    cmap = Bathymetry_SonarDefaultColormap3D(7)
    assert cmap.shape == (256, 3)  # Original MATLAB ignores n.
    np.testing.assert_array_equal(cmap[[0, -1]], [[1, .2, 0], [0, 0, .6]])


@pytest.mark.parametrize("n", [0, -1, 1.2, True])
def test_copper_rejects_invalid_lengths(n):
    with pytest.raises(ValueError):
        caris_orangecopper(n)


def literal_matlab_side_scan(pings, overlap, coherent, factor):
    previous = 0
    out = []
    for i, source in enumerate(pings):
        selected = list(range(8)) if overlap else [2, 3, 4, 5]
        beam = np.zeros((len(selected), source.shape[1]), dtype=complex)
        for ula in [0, 1]:
            beam = beam + source[selected, :, ula]
        if not overlap:
            out.append(abs(beam))
            continue
        if not coherent:
            beam = abs(beam)
        beam[:4] = beam[:4] + previous
        if not coherent:
            beam[:4] = beam[:4] * factor
        previous = beam[4:8].copy()
        out.append(abs(beam if i == len(pings)-1 else beam[:4]))
    return np.concatenate(out)


@pytest.mark.parametrize("overlap,coherent,factor", [(False, False, 1), (True, False, 1),
                                                   (True, False, .5), (True, True, 1)])
@pytest.mark.parametrize("count", [1, 3])
def test_streaming_summation_matches_literal_reference(overlap, coherent, factor, count):
    rng = np.random.default_rng(72)
    pings = rng.normal(size=(count, 8, 9, 4)) + 1j*rng.normal(size=(count, 8, 9, 4))
    expected = literal_matlab_side_scan(pings, overlap, coherent, factor)
    previous = 0
    actual = []
    for i, ping in enumerate(pings):
        block, previous = sidescan_step(ping[:, :, 0] + ping[:, :, 1], previous,
                                       overlap=overlap, coherent=coherent, factor=factor, last=i == count-1)
        actual.append(block)
    np.testing.assert_array_equal(np.concatenate(actual), expected)


def test_depth_filters_exact_selected_beams_open_depth_limits_and_beam_ordinal():
    angle = np.full((8, 3), -60.)
    delay = np.tile(np.array([10., 20., 60.]) * 2. / 1480., (8, 1))
    port = depth_points(angle, delay, 2, "Port", 1480.)
    starboard = depth_points(angle, delay, 2, "Stbd", 1480.)
    assert port.shape == (4, 3)
    np.testing.assert_array_equal(port[:, 1], [9, 10, 11, 12])
    np.testing.assert_allclose(port[:, 2], -20.)
    np.testing.assert_array_equal(port[:, 0], -starboard[:, 0])
    angle[2:6] = 21
    assert depth_points(angle, delay, 2, "Port", 1480.).shape == (0, 3)


def test_depth_angle_upper_endpoint_is_included():
    angle = np.full((8, 1), 20.)
    delay = np.full((8, 1), 200./1480.)
    points = depth_points(angle, delay, 0, "Stbd", 1480.)
    assert len(points) == 4
    np.testing.assert_allclose(points[:, 2], -100*np.sin(10/180*np.pi))


def test_colon_and_duplicate_averaging():
    np.testing.assert_allclose(matlab_colon(0, .2, 1), [0, .2, .4, .6, .8, 1])
    np.testing.assert_allclose(matlab_colon(.1, .2, .8), [.1, .3, .5, .7])
    xy, z = averaged_samples(np.array([[0, 0, -10], [1, 0, -20], [0, 0, -30]], dtype=float))
    np.testing.assert_array_equal(xy, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(z, [-20, -20])


def test_linear_grid_matches_plane_and_does_not_extrapolate(tmp_path):
    points = np.array([[0, 0, -30], [1, 0, -29], [0, 1, -28]], dtype=float)
    stats = write_interpolated_grid(points, tmp_path, block_rows=2)
    z = np.load(tmp_path / "GridZlisOutput.npy")
    x, y = np.load(tmp_path / "GridXlisInput.npy"), np.load(tmp_path / "GridYlisInput.npy")
    xx, yy = np.meshgrid(x, y)
    np.testing.assert_allclose(z[np.isfinite(z)], (-30+xx+2*yy)[np.isfinite(z)], atol=1e-12)
    assert np.isnan(z[-1, -1])
    assert stats["shape"] == [6, 11]


def fixture_mat_files(directory, count=3):
    h5py = pytest.importorskip("h5py")
    directory.mkdir()
    rng = np.random.default_rng(8)
    beam = rng.normal(size=(count, 8, 6, 4)) + 1j*rng.normal(size=(count, 8, 6, 4))
    with h5py.File(directory / "SysPara.mat", "w") as f:
        group = f.create_group("SysPara")
        for name, value in {"c": 1480., "BeamSigFs": 37650., "DoaEstProSampleNum": 72.}.items():
            group.create_dataset(name, data=[[value]])
    for side in ("Port", "Stbd"):
        for kind in ("Beamforming", "Depth"):
            path = directory / (side+"NormalMode"+kind+"Total.mat")
            with h5py.File(path, "w") as f:
                refs = f.create_dataset(path.stem+"Data", (count, 1), dtype=h5py.ref_dtype)
                for i in range(count):
                    group = f.create_group(f"ref{i}")
                    for name, value in {"PingNum": 10+i, "PingS": 1234567890, "PingUs": 100+i*330000}.items():
                        group.create_dataset(name, data=[[float(value)]])
                    if kind == "Beamforming":
                        raw = np.empty((4, 6, 8), dtype=[("real", "<f8"), ("imag", "<f8")])
                        raw["real"] = beam[i].real.transpose(2, 1, 0)
                        raw["imag"] = beam[i].imag.transpose(2, 1, 0)
                        group.create_dataset(side+"BeamformingData", data=raw)
                    else:
                        angles = np.tile(np.linspace(-50., 10., 6), (8, 1))
                        delays = np.full_like(angles, 90./1480.)
                        group.create_dataset("TargetAngleInfo", data=angles.T)
                        group.create_dataset("TargetDelayInfo", data=delays.T)
                    refs[i, 0] = group.ref
    return beam


def test_hdf5_axis_order_and_complex_ula_selection(tmp_path):
    directory = tmp_path / "input"
    beam = fixture_mat_files(directory)
    with PingFile(directory / "PortNormalModeBeamformingTotal.mat") as f:
        assert len(f) == 3
        assert f.identity(0) == (10, 1234567890, 100)
        np.testing.assert_array_equal(f.beam_sum(1, "Port"), beam[1, :, :, 0]+beam[1, :, :, 1])


def test_complete_replica_outputs_keep_original_coordinates_and_inputs(tmp_path):
    directory = tmp_path / "input"
    beams = fixture_mat_files(directory)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    output = tmp_path / "output"
    report = run(directory, output, dpi=40)
    expected = literal_matlab_side_scan(beams, True, False, 1.)
    actual = np.load(output / "SideScanImag.npy")
    np.testing.assert_array_equal(actual[:, :6], expected[:, ::-1])
    np.testing.assert_array_equal(actual[:, 6:], expected)
    np.testing.assert_array_equal(np.load(output / "BeamYlis.npy"), np.arange(1, 17))
    np.testing.assert_allclose(np.load(output / "BeamXlis.npy"), np.r_[np.arange(-6, 0), np.arange(1, 7)]/37650*740)
    assert report["processed_pings"] == 3
    assert report["GPS_used"] is False and report["SLAM_used"] is False
    assert all(Path(path).stat().st_size > 1000 for path in report["images"])
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with pytest.raises(FileExistsError):
        run(directory, output, dpi=40)


def test_output_inside_original_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        run(tmp_path, tmp_path / "output")


class HeaderReader:
    def __init__(self, ids=(10, 11, 12)):
        self.ids = ids

    def identity(self, index):
        return self.ids[index], 1234567890, index * 330000


def test_explicit_skip_tracks_original_cell_and_keeps_real_timestamps():
    indices, identities, skipped = select_records([HeaderReader() for _ in range(4)], 3, [11])
    assert indices == [0, 2]
    assert identities == [(10, 1234567890, 0), (12, 1234567890, 660000)]
    assert skipped[0]["PingNum"] == 11 and skipped[0]["source_cell_1based"] == 2


@pytest.mark.parametrize("skip", [[99], [10, 11, 12], [-1], [True]])
def test_explicit_skip_rejects_unknown_all_or_invalid_ids(skip):
    with pytest.raises(ValueError):
        select_records([HeaderReader() for _ in range(4)], 3, skip)


def test_skip_does_not_hide_four_stream_header_mismatch():
    with pytest.raises(ValueError, match="timestamp differs"):
        select_records([HeaderReader(), HeaderReader((10, 99, 12))], 3, [11])


def test_skip_never_reads_selected_bad_record_and_synchronizes_both_products(tmp_path, monkeypatch):
    directory = tmp_path / "input"
    beams = fixture_mat_files(directory)
    beam_method, depth_method = PingFile.beam_sum, PingFile.depth

    def guarded_beam(self, index, side):
        if index == 1:
            raise OSError("must not attempt the explicitly omitted record")
        return beam_method(self, index, side)

    def guarded_depth(self, index):
        if index == 1:
            raise OSError("depth must omit the same physical ping")
        return depth_method(self, index)

    monkeypatch.setattr(PingFile, "beam_sum", guarded_beam)
    monkeypatch.setattr(PingFile, "depth", guarded_depth)
    output = tmp_path / "output"
    report = run(directory, output, dpi=40, skip_ping_numbers=[11])
    expected = literal_matlab_side_scan(beams[[0, 2]], True, False, 1.)
    actual = np.load(output / "SideScanImag.npy")
    np.testing.assert_array_equal(actual[:, :6], expected[:, ::-1])
    np.testing.assert_array_equal(actual[:, 6:], expected)
    np.testing.assert_array_equal(np.load(output / "SourceCellIndices1Based.npy"), [1, 3])
    np.testing.assert_array_equal(np.load(output / "PingNum_PingS_PingUs.npy")[:, 0], [10, 12])
    assert report["processed_pings"] == 2 and report["available_pings"] == 3
    assert report["explicitly_skipped_records"][0]["PingNum"] == 11


def test_other_read_errors_still_abort_when_one_ping_is_explicitly_skipped(tmp_path, monkeypatch):
    directory = tmp_path / "input"
    fixture_mat_files(directory)
    original = PingFile.beam_sum

    def bad_other_ping(self, index, side):
        if index == 2:
            raise OSError("another unexpected read error")
        return original(self, index, side)

    monkeypatch.setattr(PingFile, "beam_sum", bad_other_ping)
    output = tmp_path / "output"
    with pytest.raises(OSError, match="unexpected read error"):
        run(directory, output, dpi=40, skip_ping_numbers=[11])
    assert not (output / "run_summary.json").exists()
