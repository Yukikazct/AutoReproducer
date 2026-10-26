"""Read NumPy artifact headers without loading data, pickle or author modules."""
import ast
import io
from pathlib import PurePosixPath
import stat
import zipfile


MAX_ARRAY_HEADER = 65536
MAX_ARCHIVE_ARRAYS = 64


def _npy_header(stream):
    magic = stream.read(8)
    if magic[:6] != b'\x93NUMPY' or magic[6:] not in {b'\x01\x00', b'\x02\x00', b'\x03\x00'}:
        raise ValueError('Unsupported NumPy array header')
    size_bytes = 2 if magic[6] == 1 else 4
    encoded_size = stream.read(size_bytes)
    if len(encoded_size) != size_bytes:
        raise ValueError('Truncated NumPy array header')
    size = int.from_bytes(encoded_size, 'little')
    if not 1 <= size <= MAX_ARRAY_HEADER:
        raise ValueError('Array header exceeds observation budget')
    raw = stream.read(size)
    if len(raw) != size or not raw.endswith(b'\n'):
        raise ValueError('Truncated NumPy array header')
    header = ast.literal_eval(raw.decode('utf-8' if magic[6] == 3 else 'latin1').strip())
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Invalid NumPy array schema')
    shape = header['shape']
    if (not isinstance(shape, tuple) or len(shape) > 32
            or any(type(item) is not int or not 0 <= item < 2 ** 63 for item in shape)
            or type(header['fortran_order']) is not bool):
        raise ValueError('Invalid NumPy array dimensions')
    descriptor = header['descr']
    if isinstance(descriptor, str) and 0 < len(descriptor) <= 512:
        dtype = descriptor
    elif isinstance(descriptor, list) and len(descriptor) <= 128:
        dtype = 'structured (field values not read)'
    else:
        raise ValueError('Unsupported NumPy dtype header')
    elements = 1
    for size in shape:
        elements *= size
    return {'shape': list(shape), 'dtype': dtype, 'elements': elements,
            'fortran_order': header['fortran_order'], 'data_read': False,
            'scope': 'declared_header_only; payload and sample correspondence unverified'}


def numpy_artifact_metadata(content, suffix):
    """Only describe header declarations; matching shapes are not matched rows."""
    if suffix == '.npy':
        return {'array': _npy_header(io.BytesIO(content))}
    if suffix != '.npz':
        raise ValueError('Unsupported array artifact')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_ARCHIVE_ARRAYS:
            raise ValueError('Array archive exceeds member budget')
        arrays = {}
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            if (not name.endswith('.npy') or path.is_absolute() or len(path.parts) != 1
                    or '\\' in name or ':' in name or name in {'.npy', '..npy'}
                    or len(name) > 255 or member.flag_bits & 1
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Array archive contains an unsupported member')
            key = name[:-4]
            if key in arrays:
                raise ValueError('Duplicate array names in archive')
            with archive.open(member) as stream:
                arrays[key] = _npy_header(stream)
        return {'array_count': len(arrays), 'arrays': arrays,
                'leading_dimensions': {key: value['shape'][0] if value['shape'] else None
                                       for key, value in arrays.items()},
                'sample_correspondence': 'unknown; headers do not establish row identity'}
