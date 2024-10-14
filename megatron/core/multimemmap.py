import mmap
import json
import numpy as np
from numpy.lib.format import dtype_to_descr, descr_to_dtype

_PREFIX = b'\x99MULTI'
_METALENTYPE = np.uint32
_METALENSIZE = np.dtype(_METALENTYPE).itemsize
_HEADER_OFFSET = len(_PREFIX) + _METALENSIZE


def get_single_arr_metadata(arr, offset):
    return {
        'dtype': dtype_to_descr(arr.dtype),
        'shape': list(arr.shape),
        'size': arr.nbytes,
        'offset': offset,
    }


def get_arr_metadata(data, init_offset=0):
    offset = init_offset
    metadata = []
    for arr in data:
        metadata.append(
            get_single_arr_metadata(arr, offset)
            )
        offset += metadata[-1]['size']
    return metadata
    
    
def _adjust_offset(metadata, extra_offset):
    for i in range(len(metadata)):
        metadata[i]['offset'] += extra_offset
    return metadata
    
def adjust_offset(metadata):
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    meta_bytes_len = len(metadata_bytes)
    meta_offset = meta_bytes_len
    for arrmeta in metadata:
        meta_offset += max(len(str(meta_offset + arrmeta['offset'])), 0) * 2
    # lendigits = len(str(len(metadata_bytes)))
    # meta_offset = len(metadata_bytes) + len(metadata) * lendigits
    newmetadata = _adjust_offset(metadata, meta_offset)
    newmetadata_bytes = json.dumps(newmetadata).encode('utf-8')
    assert len(newmetadata_bytes) <= meta_offset
    return newmetadata, _HEADER_OFFSET + meta_offset

def write(file, data: list[np.ndarray]):
    metadata = get_arr_metadata(
        data, _HEADER_OFFSET)
    metadata, data_start_index = adjust_offset(metadata)
    metadata_bytes = json.dumps(metadata).encode('utf-8')
    metadata_bytes = metadata_bytes.ljust(data_start_index - _HEADER_OFFSET)
    with open(file, 'wb') as f:
        f.write(_PREFIX)
        f.write(np.array([len(metadata_bytes)], dtype=_METALENTYPE).tobytes())
        f.write(metadata_bytes)
        for arr in data:
            f.write(arr.tobytes())
            
            
def read_metadata(file):
    with open(file, 'rb') as f:
        prefix = f.read(len(_PREFIX))
        assert prefix == _PREFIX
        meta_len = np.frombuffer(f.read(_METALENSIZE), dtype=_METALENTYPE)[0]
        return json.loads(f.read(meta_len).decode('utf-8'))


class MuiltiMemMap:
    
    def __init__(self, file):
        self.file = file
        self.metadata = read_metadata(self.file)
        self.fp = open(file, 'rb')
        self.mmap = mmap.mmap(
            self.fp.fileno(), 0,
            #flags=mmap.MAP_DENYWRITE,
            access=mmap.ACCESS_READ,
            )
            
    def get_arrays(self):
        arrays = []
        for array_meta in self.metadata:
            arrays.append(np.ndarray.__new__(
                np.ndarray,
                tuple(array_meta['shape']),
                dtype=array_meta['dtype'],
                buffer=self.mmap,
                offset=array_meta['offset'],
                ))
        return arrays
            
    def __del__(self):
        self.mmap.close()
        self.fp.close()
        
        
if __name__ == '__main__':
    arr1 = np.zeros((10,), dtype=np.int32)
    arr2 = np.zeros((10,), dtype=np.int64)
    arr3 = np.zeros((10,), dtype=np.int32)
    data = [arr1, arr2, arr3]
    metadata = get_arr_metadata(data, _HEADER_OFFSET)
    metadata, data_start_index = adjust_offset(metadata)
    write('test.dat', data)
