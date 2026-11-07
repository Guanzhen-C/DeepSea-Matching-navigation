from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone'))
from EnhanceEdgeSonarShotwell import (
    DEFAULT_REFERENCE, SOURCE_HASHES, clean_environment, portrait_pdf,
    replace_palette,
)
from ExportPinnedFullSamplePortrait import indexed_info, sha
from NativeSonarPaletteStream import NativePaletteWriter, decoded_index_rows


def fixture_png(path):
    palette = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    data = np.arange(91, dtype=np.uint8).reshape(7, 13)
    writer = NativePaletteWriter(path, 13, 7, palette)
    writer.append(data)
    writer.finish()
    return palette, data


def test_only_palette_changes_every_native_index_preserved(tmp_path):
    source, result = tmp_path / 'source.png', tmp_path / 'result.png'
    palette, data = fixture_png(source)
    source_sha = sha(source)
    modified = np.minimum(palette.astype(np.uint16) * 2, 255).astype(np.uint8)
    report = replace_palette(source, result, modified.tobytes())
    assert report['idat_SHA256'] == indexed_info(source)['idat_SHA256']
    assert (report['width'], report['height']) == (13, 7)
    assert np.array_equal(np.stack(list(decoded_index_rows(result, modified))), data)
    assert sha(source) == source_sha
    with Image.open(result) as img:
        assert np.array_equal(np.asarray(img.convert('RGB')), modified[data])


def test_keep_previous_output_and_reject_wrong_palette(tmp_path):
    source, result = tmp_path / 'source.png', tmp_path / 'result.png'
    palette, _ = fixture_png(source)
    with pytest.raises(ValueError, match='256'):
        replace_palette(source, result, b'not a palette')
    assert not result.exists()
    result.write_bytes(b'previous user result')
    with pytest.raises(FileExistsError):
        replace_palette(source, result, palette.tobytes())
    assert result.read_bytes() == b'previous user result'


def test_corrupt_png_is_not_enhanced(tmp_path):
    source, result = tmp_path / 'source.png', tmp_path / 'result.png'
    palette, _ = fixture_png(source)
    payload = bytearray(source.read_bytes())
    payload[-5] ^= 1
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        replace_palette(source, result, palette.tobytes())
    assert not result.exists()


def test_portrait_pdf_preserves_enhanced_palette_and_compressed_samples(tmp_path):
    source, output = tmp_path / 'source.png', tmp_path / 'enhanced.pdf'
    palette, _ = fixture_png(source)
    info = indexed_info(source)
    report = portrait_pdf(source, output, info)
    payload = output.read_bytes()
    assert report['embedded_IDAT_SHA256'] == info['idat_SHA256']
    assert report['embedded_dimensions'] == [13, 7]
    assert palette.tobytes().hex().encode() in payload
    assert b'/Width 13 /Height 7' in payload
    assert b'/Interpolate false' in payload
    assert b'756 0 0 -1344 0 1344 cm' in payload
    assert b'Display-only enhancement' in payload
    with pytest.raises(FileExistsError):
        portrait_pdf(source, output, info)


def run_real_engine(tmp_path, analysis):
    binary = DEFAULT_REFERENCE / 'build_palette_adapter/shotwell-enhance-palette'
    if not binary.exists():
        pytest.skip('Optional real upstream source engine not built on this machine')
    for name, expected in SOURCE_HASHES.items():
        assert sha(DEFAULT_REFERENCE / name) == expected
    manifest = json.loads((binary.parent / 'build_manifest.json').read_text())
    assert sha(binary) == manifest['binary_SHA256']
    Image.fromarray(analysis).save(tmp_path / 'analysis.png')
    palette = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    Image.fromarray(palette.reshape(16, 16, 3)).save(tmp_path / 'palette.png')
    subprocess.run([str(binary), str(tmp_path / 'analysis.png'),
                    str(tmp_path / 'palette.png'), str(tmp_path / 'output.png'),
                    str(tmp_path / 'parameters.json')], env=clean_environment(), check=True)
    params = json.loads((tmp_path / 'parameters.json').read_text())
    with Image.open(tmp_path / 'output.png') as img:
        result = np.asarray(img).reshape(256, 3)
    return params, result


def test_real_shotwell_engine_standard_contrast_branch(tmp_path):
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    params, result = run_real_engine(tmp_path, ramp.reshape(16, 16, 3))
    assert params['black_point'] == 5
    assert params['white_point'] == 249
    assert params['shadow_parameter'] == 0
    assert params['actual_Gee_transformation_order'] == list(range(8))
    assert np.array_equal(result[0], [0, 0, 0])
    assert np.array_equal(result[-1], [255, 255, 255])
    assert (np.diff(result[:, 0].astype(int)) >= 0).all()


def test_real_shotwell_engine_shadow_branch_and_black_preservation(tmp_path):
    values = (np.arange(256) % 81).astype(np.uint8)
    analysis = np.repeat(values[:, None], 3, axis=1).reshape(16, 16, 3)
    params, result = run_real_engine(tmp_path, analysis)
    assert params['black_point'] == 0
    assert 0 < params['white_point'] < 81
    assert 0 < params['shadow_parameter'] <= 12.8
    assert np.array_equal(result[0], [0, 0, 0])
    assert result[30, 0] > 30
    assert (np.diff(result[:, 0].astype(int)) >= 0).all()
