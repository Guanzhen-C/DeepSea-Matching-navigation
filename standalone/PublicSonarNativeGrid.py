"""Bounded native-grid tiling and fixed-palette numeric display, no SciPy."""
from pathlib import Path
import mmap
import struct
import zlib
import numpy as np


def tiles(shape,core=512,halo=96):
    if len(shape)!=2 or min(shape)<1 or core<4 or core%4 or halo<0 or halo%4:
        raise ValueError('native grid and factor-four aligned contexts required')
    h,w=shape
    for y in range(0,h,core):
        for x in range(0,w,core):
            yield y,x,min(h,y+core),min(w,x+core),max(0,y-halo),max(0,x-halo),min(h,y+core+halo),min(w,x+core+halo)


def release_pages(*arrays):
    for array in arrays:
        if not isinstance(array,np.memmap):raise ValueError('explicit file-backed array required')
        if array.flags.writeable:array.flush()
        array._mmap.madvise(mmap.MADV_DONTNEED)


def rgb_rows(values,palette):
    index=np.minimum((np.clip(np.asarray(values),0,1)*256).astype(np.uint16),255)
    return palette[index]


def native_png(values,path,palette):
    h,w=values.shape;compressor=zlib.compressobj(3)
    def chunk(stream,tag,data):
        stream.write(struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff))
    with Path(path).open('xb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n')
        chunk(stream,b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))
        chunk(stream,b'tEXt',b'Description\x00Complete native public sonar grid, original fixed CARIS amber palette; log limits recorded; no true clean reference or quality acceptance')
        for row in range(h-1,-1,-1):
            data=compressor.compress(b'\x00'+rgb_rows(values[row],palette).tobytes())
            if data:chunk(stream,b'IDAT',data)
        chunk(stream,b'IDAT',compressor.flush());chunk(stream,b'IEND',b'')
