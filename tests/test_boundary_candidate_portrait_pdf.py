from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'standalone'))
from NativeSonarPaletteStream import NativePaletteWriter
from PackageBoundaryCandidatePortraitPDF import package


def test_candidate_label_original_palette_pixels_and_flip(tmp_path):
    source = tmp_path/'source.png'
    palette = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    w = NativePaletteWriter(source, 7, 4, palette)
    w.append(np.arange(28, dtype=np.uint8).reshape(4, 7))
    w.finish()
    output = tmp_path/'candidate.pdf'
    report = package(source, output)
    assert report['compressed_pixels_and_palette_exact']
    assert report['embedded_original_dimensions'] == [7, 4]
    assert not report['adopted_as_default']
    assert not report['quality_accepted']
    b = output.read_bytes()
    assert b'756 0 0 -1344 0 1344 cm' in b
    assert b'Not the pinned baseline' in b
    assert b'/Colors 1' in b and b'/Interpolate false' in b
    with pytest.raises(FileExistsError):
        package(source, output)
