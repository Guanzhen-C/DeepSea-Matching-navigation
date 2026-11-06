from pathlib import Path
import sys
import hashlib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from NativeSonarPaletteStream import NativePaletteWriter, decoded_index_rows
from ExportPinnedFullSamplePortrait import export, indexed_info
from ExportNativeSonarPortraitPNG import row_repetitions


def source_png(root, width=7, height=4):
    palette = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    a = np.arange(width*height, dtype=np.uint8).reshape(height, width)
    path = root/'native.png'
    writer = NativePaletteWriter(path, width, height, palette)
    writer.append(a)
    writer.finish()
    return path, a, palette


def test_preserves_every_source_index_and_matches_waterfall_vertical_order(tmp_path):
    source, values, palette = source_png(tmp_path)
    before = source.read_bytes()
    r = export(source, tmp_path/'out', hashlib.sha256(before).hexdigest())
    width, height = r['png']['width_height']
    counts = row_repetitions(len(values), height)
    expected = np.concatenate([np.repeat(values[-1-i][None], n, axis=0) for i, n in enumerate(counts)])
    actual = np.stack(list(decoded_index_rows(Path(r['png']['file']), palette)))
    np.testing.assert_array_equal(actual, expected)
    assert width == 7 and height == 13
    assert r['all_source_rows_and_columns_retained'] and not r['measured_resolution_increased']
    assert source.read_bytes() == before


def test_PDF_keeps_exact_original_indexed_stream_and_uses_only_display_flip(tmp_path):
    source, _, _ = source_png(tmp_path)
    info = indexed_info(source)
    r = export(source, tmp_path/'out')
    data = Path(r['pdf']['file']).read_bytes()
    assert b'/ColorSpace [/Indexed /DeviceRGB 255' in data
    assert b'/Predictor 15 /Colors 1 /BitsPerComponent 8 /Columns 7' in data
    assert b'756 0 0 -1344 0 1344 cm' in data
    assert r['pdf']['embedded_idat_SHA256'] == info['idat_SHA256']


def test_wrong_pinned_checksum_rejected_before_output_created(tmp_path):
    source, _, _ = source_png(tmp_path)
    with pytest.raises(ValueError):
        export(source, tmp_path/'out', '0'*64)
    assert not (tmp_path/'out').exists()


def test_existing_result_never_overwritten(tmp_path):
    source, _, _ = source_png(tmp_path)
    out = tmp_path/'out'
    out.mkdir()
    with pytest.raises(FileExistsError):
        export(source, out)


def test_already_tall_native_source_does_not_discard_or_repeat_rows(tmp_path):
    source, _, _ = source_png(tmp_path, width=3, height=8)
    r = export(source, tmp_path/'out')
    assert r['png']['width_height'] == [3, 8]
    assert r['row_repetitions_min_max'] == [1, 1]
