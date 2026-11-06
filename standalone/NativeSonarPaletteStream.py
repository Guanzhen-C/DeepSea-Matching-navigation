"""Lossless native CARIS palette indices, streamed without full-image decode.

Indexed PNG stores exactly the same 256 RGB display colours as the established
renderer. It does not store float acoustic amplitudes or add measured resolution.
"""
from pathlib import Path
import hashlib
import shutil
import struct
import zlib

import numpy as np


def amplitude_indices(values):
    a=np.asarray(values)
    if not np.isfinite(a).all() or (a<0).any():raise ValueError('finite nonnegative native amplitudes required')
    return np.minimum((np.clip(a,0,1)*256).astype(np.uint16),255).astype(np.uint8)


class NativePaletteWriter:
    def __init__(self,path,width,height,palette,*,reserve_bytes=0):
        self.path=Path(path);self.incomplete=self.path.with_suffix('.png.incomplete')
        if self.path.exists() or self.incomplete.exists():raise FileExistsError('preserve existing native image')
        self.palette=np.asarray(palette,np.uint8)
        if width<1 or height<1 or self.palette.shape!=(256,3) or reserve_bytes<0:
            raise ValueError('native dimensions, original 256-colour palette and nonnegative reserve required')
        self.width=width;self.height=height;self.rows=0;self.reserve=reserve_bytes
        self.stream=self.incomplete.open('xb');self.compressor=zlib.compressobj(6)
        self.index_digest=hashlib.sha256();self.RGB_digest=hashlib.sha256()
        self.stream.write(b'\x89PNG\r\n\x1a\n')
        self.chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,3,0,0,0))
        self.chunk(b'PLTE',self.palette.tobytes())
        self.chunk(b'tEXt',b'Description\x00Complete original native sample grid; original fixed CARIS amber display; indexed colours are lossless RGB encoding, not float amplitudes; experimental signal processing, not accepted quality')

    def chunk(self,tag,data):
        if shutil.disk_usage(self.path.parent).free-len(data)-12<self.reserve:
            raise OSError('stop before violating declared disk reserve; incomplete native image preserved')
        self.stream.write(struct.pack('>I',len(data))+tag+data)
        self.stream.write(struct.pack('>I',zlib.crc32(tag+data)&0xffffffff))

    def append(self,indices):
        a=np.asarray(indices)
        if a.dtype!=np.uint8 or a.ndim!=2 or a.shape[1]!=self.width or self.rows+len(a)>self.height:
            raise ValueError('complete correctly ordered native uint8 index rows required')
        for row in a:
            self.index_digest.update(row.tobytes());self.RGB_digest.update(self.palette[row].tobytes())
            data=self.compressor.compress(b'\x00'+row.tobytes())
            if data:self.chunk(b'IDAT',data)
            self.rows+=1

    def finish(self):
        if self.rows!=self.height:raise ValueError('native image coverage incomplete')
        self.chunk(b'IDAT',self.compressor.flush());self.chunk(b'IEND',b'')
        self.stream.close();self.incomplete.rename(self.path)
        return dict(native_width=self.width,native_height=self.height,all_native_rows_written=self.rows,
                    index_SHA256=self.index_digest.hexdigest(),RGB_SHA256=self.RGB_digest.hexdigest(),
                    palette_SHA256=hashlib.sha256(self.palette.tobytes()).hexdigest(),
                    indexed_colours_preserve_exact_defined_RGB=True,float_acoustic_amplitudes_stored_in_PNG=False)

    def close(self):
        if not self.stream.closed:self.stream.close()


def decoded_index_rows(path,expected_palette):
    palette=np.asarray(expected_palette,np.uint8);decoder=zlib.decompressobj()
    pending=bytearray();width=height=None;produced=0;have_palette=False;ended=False
    with Path(path).open('rb') as stream:
        if stream.read(8)!=b'\x89PNG\r\n\x1a\n':raise ValueError('invalid PNG signature')
        while True:
            header=stream.read(8)
            if len(header)!=8:raise ValueError('truncated indexed PNG')
            length,tag=struct.unpack('>I4s',header)
            if length>16*1024**2:raise ValueError('unexpectedly large native PNG chunk')
            payload=stream.read(length);crc=stream.read(4)
            if len(payload)!=length or len(crc)!=4 or struct.unpack('>I',crc)[0]!=(zlib.crc32(tag+payload)&0xffffffff):
                raise ValueError('PNG CRC or payload differs')
            if tag==b'IHDR':
                width,height,bits,kind,compression,method,interlace=struct.unpack('>IIBBBBB',payload)
                if bits!=8 or kind!=3 or compression or method or interlace or width<1 or height<1:
                    raise ValueError('reviewed native indexed PNG format required')
            elif tag==b'PLTE':
                if payload!=palette.tobytes():raise ValueError('original palette differs')
                have_palette=True
            elif tag==b'IDAT':
                if width is None or not have_palette:raise ValueError('native header/palette missing')
                data=payload
                while True:
                    capacity=width+1-len(pending)
                    part=decoder.decompress(data,capacity);data=decoder.unconsumed_tail;pending.extend(part)
                    if len(pending)==width+1:
                        if pending[0]!=0:raise ValueError('unexpected indexed row predictor')
                        produced+=1
                        if produced>height:raise ValueError('excess native PNG rows')
                        yield np.frombuffer(pending,np.uint8,offset=1).copy()
                        pending.clear()
                    if not data and len(part)<capacity:break
            elif tag==b'IEND':
                if pending or produced!=height or not decoder.eof or decoder.unused_data or stream.read(1):
                    raise ValueError('native indexed PNG coverage/termination differs')
                ended=True;break
    if not ended:raise ValueError('incomplete native image')
