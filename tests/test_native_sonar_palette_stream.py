from pathlib import Path
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'standalone'))
from NativeSonarPaletteStream import amplitude_indices,NativePaletteWriter,decoded_index_rows
import NativeSonarPaletteStream as png_stream
from RecordedSubbandNativeStream import overlap_block,overlap_tail,supported_complex_subband
from SonarRecordedSubbandHypothesis import complex_bandpass
from caris_orangecopper import caris_orangecopper


def test_indexed_PNG_is_exact_original_RGB_and_streaming_roundtrip(tmp_path):
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    values=np.linspace(0,1.3,7*129).reshape(7,129);indices=amplitude_indices(values)
    path=tmp_path/'native.png';writer=NativePaletteWriter(path,129,7,palette)
    writer.append(indices[:3]);writer.append(indices[3:]);result=writer.finish()
    np.testing.assert_array_equal(np.stack(list(decoded_index_rows(path,palette))),indices)
    np.testing.assert_array_equal(np.asarray(Image.open(path).convert('RGB')),palette[indices])
    assert result['index_SHA256']==hashlib.sha256(indices.tobytes()).hexdigest()
    with pytest.raises(FileExistsError):NativePaletteWriter(path,129,7,palette)
    with pytest.raises(ValueError):list(decoded_index_rows(path,palette[::-1]))


def test_indexed_writer_rejects_incomplete_or_excess_rows(tmp_path):
    palette=np.zeros((256,3),np.uint8);writer=NativePaletteWriter(tmp_path/'unfinished.png',8,3,palette)
    writer.append(np.zeros((2,8),np.uint8))
    with pytest.raises(ValueError):writer.finish()
    with pytest.raises(ValueError):writer.append(np.zeros((2,8),np.uint8))
    writer.close()
    assert writer.incomplete.exists() and not writer.path.exists()


def test_native_stream_rejects_corrupted_CRC_or_truncated_file(tmp_path):
    palette=(caris_orangecopper(256)*255).astype(np.uint8)
    path=tmp_path/'good.png';writer=NativePaletteWriter(path,31,7,palette)
    writer.append(np.arange(217,dtype=np.uint8).reshape(7,31));writer.finish()
    content=bytearray(path.read_bytes());content[48]^=1
    corrupted=tmp_path/'corrupt.png';corrupted.write_bytes(content)
    with pytest.raises(ValueError,match='CRC'):list(decoded_index_rows(corrupted,palette))
    truncated=tmp_path/'truncated.png';truncated.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError):list(decoded_index_rows(truncated,palette))


def test_disk_reserve_stops_writing_before_exhaustion_and_preserves_incomplete(tmp_path,monkeypatch):
    palette=np.zeros((256,3),np.uint8);writer=NativePaletteWriter(tmp_path/'reserved.png',128,4,palette)
    writer.stream.flush();before=writer.incomplete.stat().st_size
    monkeypatch.setattr(png_stream.shutil,'disk_usage',lambda path:SimpleNamespace(free=1))
    with pytest.raises(OSError,match='disk reserve'):writer.append(np.zeros((1,128),np.uint8))
    writer.close()
    assert writer.incomplete.stat().st_size==before and not writer.path.exists()


def test_chunked_overlap_matches_independent_eight_beam_formula_including_tail():
    rng=np.random.default_rng(581);a=rng.random((2,17,8,193));pieces=[];previous=None
    for start in range(0,17,4):
        rows,previous=overlap_block(a[:,start:start+4],previous);pieces.append(rows)
    pieces.append(overlap_tail(previous));actual=np.concatenate(pieces)
    expected=[]
    for r in range(18):
        for b in range(4):
            sides=[]
            for side in range(2):
                row=(a[side,r,b] if r<17 else np.zeros(193))+(a[side,r-1,b+4] if r else 0)
                sides.append(row[::-1] if side==0 else row)
            expected.append(np.concatenate(sides))
    np.testing.assert_array_equal(actual,np.asarray(expected))


def test_supported_complex_filter_preserves_water_numeric_zeros_and_FIR_edges():
    rng=np.random.default_rng(582);z=rng.normal(size=(3,8,2048))+1j*rng.normal(size=(3,8,2048))
    z[:,:,900:904]=0;original=z.copy();limits=np.tile([8.,12.],(3,1))
    h=complex_bandpass(37650.,[-15000.,-3000.]);out,counts=supported_complex_subband(z,h,limits,1480.,37650.)
    ranges=(np.arange(2048)+1)*1480./(2*37650.)
    np.testing.assert_array_equal(out[:,:,ranges<=8],z[:,:,ranges<=8])
    np.testing.assert_array_equal(out[:,:,:129],z[:,:,:129])
    np.testing.assert_array_equal(out[:,:,-129:],z[:,:,-129:])
    np.testing.assert_array_equal(out[:,:,900:904],z[:,:,900:904])
    np.testing.assert_array_equal(z,original)
    assert counts['water_protected_values']>0 and (abs(out-z)>1e-5).any()
