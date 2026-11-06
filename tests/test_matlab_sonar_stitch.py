import numpy as np
import pytest

from standalone.BathymetrySideScanDrawing import sidescan_step
from standalone.StitchBathymetrySideScanDrawing import boundary_info, group_completed_voyages, retained_duplicate_headers, stitch_sidescan


def assemble(pings):
    previous, blocks = 0, []
    for i, ping in enumerate(pings):
        block, previous = sidescan_step(ping, previous, last=i == len(pings)-1)
        blocks.append(block)
    return np.concatenate(blocks)


@pytest.mark.parametrize("counts", [(1, 1), (2, 3), (1, 4, 2)])
def test_join_is_exactly_the_original_full_sequence_overlap_without_duplicate_tail(counts, tmp_path):
    rng = np.random.default_rng(530)
    pings = rng.normal(size=(sum(counts), 8, 7)) + 1j*rng.normal(size=(sum(counts), 8, 7))
    boundaries = np.cumsum([0]+list(counts))
    arrays = [assemble(pings[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]
    originals = [array.copy() for array in arrays]
    path = tmp_path / "joined.npy"
    summary = stitch_sidescan(arrays, counts, path, block_rows=3)
    np.testing.assert_array_equal(np.load(path), assemble(pings))
    assert summary["shape"] == [4*sum(counts)+4, 7]
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def test_incomplete_strip_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        stitch_sidescan([np.zeros((9, 4))], [1], tmp_path / "bad.npy")


def test_boundary_retains_real_time_and_consecutive_ping_numbers():
    result = boundary_info(np.array([[6146, 100, 755258]]), np.array([[6147, 101, 85258]]))
    assert result["time_difference_microseconds"] == 330000
    assert result["previous_PingNum"] == 6146 and result["following_PingNum"] == 6147


@pytest.mark.parametrize("following", [[[1, 200, 0]], [[6148, 101, 0]], [[6147, 99, 0]]])
def test_separate_or_nonconsecutive_voyages_are_not_silently_joined(following):
    with pytest.raises(ValueError, match="consecutive"):
        boundary_info(np.array([[6146, 100, 755258]]), np.array(following))


def test_identical_header_duplicate_is_preserved_and_reported_without_deduplication():
    ids = np.array([[6665, 100, 0], [6666, 100, 330000], [6666, 100, 330000], [6667, 100, 660000]])
    original = ids.copy()
    result = retained_duplicate_headers(ids, 4)
    assert result[0]["PingNum"] == 6666
    assert result[0]["input_record_positions_1based"] == [2, 3]
    np.testing.assert_array_equal(ids, original)


@pytest.mark.parametrize("ids", [np.array([[2, 100, 0], [1, 100, 330000]]),
                                 np.array([[2, 100, 0], [2, 100, 330000]])])
def test_reversed_or_conflicting_headers_are_still_rejected(ids):
    with pytest.raises(ValueError):
        retained_duplicate_headers(ids, 2)


def voyage_entries(root, session, label):
    return [{"source": str(root/session/"数据"/name), "label": label+"_"+name,
             "status": "completed", "output": str(root/"outputs"/session/name)}
            for name in ("Data02", "Data01")]


def test_a_single_afternoon_return_voyage_does_not_require_a_morning_run(tmp_path):
    entries = voyage_entries(tmp_path, "20260924_下午返航", "下午返航")
    grouped = group_completed_voyages(entries)
    assert len(grouped) == 1 and grouped[0][0] == "下午返航"
    assert [entry["label"] for entry in grouped[0][1]] == ["下午返航_Data01", "下午返航_Data02"]


def test_previous_morning_and_departure_sessions_remain_separate(tmp_path):
    entries = voyage_entries(tmp_path, "20260925_上午返航", "上午返航")
    entries += voyage_entries(tmp_path, "20260925_下午出发", "下午出发")
    grouped = group_completed_voyages(entries)
    assert [label for label, _ in grouped] == ["上午返航", "下午出发"]


def test_same_time_of_day_on_different_dates_is_never_mixed(tmp_path):
    entries = voyage_entries(tmp_path, "20260924_下午返航", "下午返航")
    entries += voyage_entries(tmp_path, "20260925_下午返航", "下午返航")
    grouped = group_completed_voyages(entries)
    assert len(grouped) == 2
    assert len({label for label, _ in grouped}) == 2
    assert all(len(parts) == 2 for _, parts in grouped)


def test_failed_group_cannot_be_silently_included_in_stitching(tmp_path):
    entries = voyage_entries(tmp_path, "20260924_下午返航", "下午返航")
    entries[0]["status"] = "failed"
    with pytest.raises(ValueError, match="incomplete"):
        group_completed_voyages(entries)
