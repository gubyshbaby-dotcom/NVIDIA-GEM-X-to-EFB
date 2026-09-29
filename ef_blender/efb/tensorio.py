"""Reading torch.save and numpy files with neither torch nor numpy. No bpy.

GEM-X leaves its answer in preprocess/hpe_results.pt, written by torch.save, and Blender
ships no torch. The file is a zip: one pickle (data.pkl) describing the object tree, plus
one raw little-endian blob per tensor storage (data/0, data/1, ...). This reads both.

The pickle is never trusted. A stock Unpickler imports and calls whatever a file names,
which is how a .pt file runs code; this one resolves a fixed list of names - the
containers, torch's tensor rebuild functions and storage types, numpy's array
reconstruction - and turns everything else into an inert Opaque that only remembers what
it was called. Nothing outside the list is imported, so opening a file cannot run it.

  read_torch(path)  -> the object torch.save was given, tensors as Tensor
  read_npz(path)    -> {name: Tensor}
  read_npy(data)    -> Tensor
"""

from __future__ import annotations

import ast
import collections
import io
import math
import pickle
import struct
import zipfile

__all__ = ["Tensor", "Opaque", "read_torch", "read_npz", "read_npy", "TensorIOError",
           "walk"]


class TensorIOError(ValueError):
    pass


_STORAGES = {
    "FloatStorage": "f", "DoubleStorage": "d", "HalfStorage": "e",
    "BFloat16Storage": "bf16", "LongStorage": "q", "IntStorage": "i",
    "ShortStorage": "h", "CharStorage": "b", "ByteStorage": "B", "BoolStorage": "?",
}

_NUMPY_CODES = {
    "f4": "f", "f8": "d", "f2": "e", "i8": "q", "i4": "i", "i2": "h", "i1": "b",
    "u8": "Q", "u4": "I", "u2": "H", "u1": "B", "b1": "?",
}


class Tensor:
    """A dense array held as a flat C-order list of python numbers."""

    __slots__ = ("shape", "data", "dtype")

    def __init__(self, shape, data, dtype="f"):
        self.shape = tuple(int(s) for s in shape)
        self.data = list(data)
        self.dtype = dtype
        if len(self.data) != self.numel:
            raise TensorIOError("tensor of shape %s holds %d values"
                                % (self.shape, len(self.data)))

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self):
        return self.shape[0] if self.shape else 1

    def __repr__(self):
        return "Tensor(shape=%s, dtype=%s)" % (self.shape, self.dtype)

    def item(self):
        if self.numel != 1:
            raise TensorIOError("item() on a tensor of shape %s" % (self.shape,))
        return self.data[0]

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape[shape.index(-1)] = self.numel // known if known else 0
        return Tensor(shape, self.data, self.dtype)

    def rows(self, width=None):
        """The last axis as rows: a (L, k) tensor gives L tuples of k."""
        width = width or (self.shape[-1] if self.shape else 1)
        d = self.data
        return [tuple(d[i:i + width]) for i in range(0, len(d), width)]

    def tolist(self):
        if not self.shape:
            return self.data[0]

        def build(offset, dims):
            if len(dims) == 1:
                return self.data[offset:offset + dims[0]]
            step = 1
            for s in dims[1:]:
                step *= s
            return [build(offset + i * step, dims[1:]) for i in range(dims[0])]

        return build(0, list(self.shape))


class Opaque:
    """Anything the reader will not construct. Keeps its name, its arguments and any state
    it was handed, and does nothing else."""

    def __init__(self, name, args=(), kwargs=None):
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}
        self.state = None
        self.items = {}
        self.list = []

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        self.items[key] = value

    def append(self, value):
        self.list.append(value)

    def extend(self, values):
        self.list.extend(values)

    def fields(self) -> dict:
        """Whatever mapping this stood for: items a dict subclass was filled with, else
        the attribute dict its state carried."""
        if self.items:
            return self.items
        state = self.state
        if isinstance(state, tuple) and state and isinstance(state[0], dict):
            state = state[0]
        return state if isinstance(state, dict) else {}

    def __repr__(self):
        return "Opaque(%s)" % self.name


def _opaque_type(module, name):
    full = "%s.%s" % (module, name)

    class _Stub:
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            return Opaque(full, args, kwargs)

    _Stub.__name__ = str(name)
    return _Stub


def _decode(raw: bytes, code: str, count: int, little=True):
    end = "<" if little else ">"
    if code == "bf16":
        shorts = struct.unpack(end + "%dH" % count, raw[:2 * count])
        return [struct.unpack("<f", struct.pack("<I", s << 16))[0] for s in shorts]
    if code == "?":
        return [b != 0 for b in raw[:count]]
    size = struct.calcsize(code)
    return list(struct.unpack(end + "%d%s" % (count, code), raw[:size * count]))


def _itemsize(code):
    return 2 if code == "bf16" else struct.calcsize(code)


class _Storage:
    __slots__ = ("code", "key", "loader", "_values")

    def __init__(self, code, key, loader):
        self.code, self.key, self.loader = code, key, loader
        self._values = None

    def values(self):
        if self._values is None:
            raw = self.loader(self.key)
            self._values = _decode(raw, self.code, len(raw) // _itemsize(self.code),
                                   self.loader.little)
        return self._values


class _StorageType:
    def __init__(self, name):
        self.name = name
        self.code = _STORAGES[name]


def _gather(storage, offset, size, stride):
    """Strided view -> flat C-order list."""
    values = storage.values()
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    if not size:
        return [values[offset]]
    out = []

    def walk_dim(base, d):
        if d == len(size) - 1:
            s = stride[d]
            out.extend(values[base + i * s] for i in range(size[d]))
            return
        for i in range(size[d]):
            walk_dim(base + i * stride[d], d + 1)

    if 0 not in size:
        walk_dim(int(offset), 0)
    return out


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    if not isinstance(storage, _Storage):
        raise TensorIOError("tensor without a storage")
    return Tensor(size, _gather(storage, storage_offset, size, stride), storage.code)


def _rebuild_parameter(data, *rest):
    return data


def _rebuild_from_type(func, new_type, args, state):
    return func(*args)


def _ordered_dict(*args):
    return collections.OrderedDict(*args)


class _NumpyDtype:
    def __init__(self, spec, *rest):
        self.spec = str(spec)
        self.little = True

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) > 1 and state[1] in "<>|=":
            self.little = state[1] != ">"


class _NumpyArray:
    """What numpy.core.multiarray._reconstruct hands back; the payload arrives in
    __setstate__ as (version, shape, dtype, fortran, raw bytes)."""

    def __init__(self, *args):
        self.tensor = None

    def __setstate__(self, state):
        _v, shape, dtype, fortran, raw = state[:5]
        code = _NUMPY_CODES.get(dtype.spec.lstrip("<>|="))
        if code is None or not isinstance(raw, (bytes, bytearray)):
            self.tensor = Opaque("numpy.ndarray[%s]" % dtype.spec)
            return
        n = 1
        for s in shape:
            n *= s
        values = _decode(bytes(raw), code, n, dtype.little)
        if fortran and len(shape) > 1:
            values = _c_order(values, shape)
        self.tensor = Tensor(shape, values, code)


def _numpy_reconstruct(cls, *args):
    return _NumpyArray()


def _numpy_scalar(dtype, raw=b""):
    code = _NUMPY_CODES.get(dtype.spec.lstrip("<>|="))
    if code is None:
        return Opaque("numpy.scalar[%s]" % dtype.spec)
    return _decode(bytes(raw), code, 1, dtype.little)[0]


def _c_order(values, shape):
    """Fortran-order flat list -> C-order flat list."""
    shape = list(shape)
    strides, step = [], 1
    for s in shape:
        strides.append(step)
        step *= s
    out = []

    def walk_dim(base, d):
        if d == len(shape):
            out.append(values[base])
            return
        for i in range(shape[d]):
            walk_dim(base + i * strides[d], d + 1)

    walk_dim(0, 0)
    return out


def _codecs_encode(text, encoding="latin1", *rest):
    """How a protocol 2 pickle spells a bytes object: _codecs.encode(str, "latin1")."""
    return str(text).encode(encoding if encoding in ("latin1", "latin-1") else "latin1")


_SAFE = {
    ("_codecs", "encode"): _codecs_encode,
    ("collections", "OrderedDict"): _ordered_dict,
    ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
    ("builtins", "slice"): slice, ("builtins", "complex"): complex,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type,
    ("torch._tensor", "_rebuild_from_type"): _rebuild_from_type,
    ("torch", "Size"): tuple,
    ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct,
    ("numpy.core.multiarray", "scalar"): _numpy_scalar,
    ("numpy._core.multiarray", "scalar"): _numpy_scalar,
    ("numpy", "dtype"): _NumpyDtype,
    ("numpy", "ndarray"): _NumpyArray,
}


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, fh, loader):
        super().__init__(fh)
        self.loader = loader

    def find_class(self, module, name):
        if module == "torch" and name in _STORAGES:
            return _StorageType(name)
        fn = _SAFE.get((module, name))
        if fn is not None:
            return fn
        return _opaque_type(module, name)

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
            raise TensorIOError("unsupported persistent id %r" % (pid,))
        _tag, storage_type, key = pid[:3]
        code = storage_type.code if isinstance(storage_type, _StorageType) else "B"
        return _Storage(code, str(key), self.loader)


class _ZipLoader:
    def __init__(self, zf, prefix):
        self.zf, self.prefix = zf, prefix
        self.little = True
        order = prefix + "byteorder"
        if order in zf.namelist():
            self.little = zf.read(order).strip() != b"big"

    def __call__(self, key):
        return self.zf.read(self.prefix + "data/" + key)


def _finish(obj):
    """Swap the numpy placeholders for the Tensors they decoded."""
    if isinstance(obj, _NumpyArray):
        return obj.tensor
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _finish(obj[k])
        return obj
    if isinstance(obj, list):
        return [_finish(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_finish(v) for v in obj)
    return obj


def read_torch(path):
    """What torch.load would return, with tensors as Tensor. Zip-format files only, which
    is everything torch.save has written since torch 1.6."""
    if not zipfile.is_zipfile(path):
        raise TensorIOError("%s is not a zip-format torch file (torch >= 1.6 writes "
                            "those; re-save it with torch.save)" % path)
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise TensorIOError("%s has no data.pkl - not a torch.save file" % path)
        name = min(pkl, key=len)
        prefix = name[:-len("data.pkl")]
        loader = _ZipLoader(zf, prefix)
        # Tensors are gathered as they are rebuilt, so every storage is read before the
        # zip closes.
        return _finish(_SafeUnpickler(io.BytesIO(zf.read(name)), loader).load())


def read_npy(data: bytes) -> Tensor:
    if data[:6] != b"\x93NUMPY":
        raise TensorIOError("not a .npy payload")
    major = data[6]
    if major == 1:
        hlen = struct.unpack("<H", data[8:10])[0]
        start = 10
    else:
        hlen = struct.unpack("<I", data[8:12])[0]
        start = 12
    header = ast.literal_eval(data[start:start + hlen].decode("latin1"))
    descr, fortran, shape = header["descr"], header["fortran_order"], header["shape"]
    if not isinstance(descr, str):
        raise TensorIOError("structured .npy arrays are not supported")
    code = _NUMPY_CODES.get(descr.lstrip("<>|="))
    if code is None:
        raise TensorIOError("unsupported .npy dtype %r" % descr)
    n = 1
    for s in shape:
        n *= s
    values = _decode(data[start + hlen:], code, n, descr[0] != ">")
    if fortran and len(shape) > 1:
        values = _c_order(values, shape)
    return Tensor(shape, values, code)


def read_npz(path) -> dict:
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            try:
                out[name[:-4]] = read_npy(zf.read(name))
            except TensorIOError:
                continue
    return out


def walk(obj, prefix=""):
    """(dotted path, value) for every leaf of a nested dict/list tree. For telling a user
    what a file actually holds when it is not what we expected."""
    if isinstance(obj, Opaque) and obj.fields():
        obj = obj.fields()
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, "%s.%s" % (prefix, k) if prefix else str(k))
    elif isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (int, float)):
        for i, v in enumerate(obj):
            yield from walk(v, "%s[%d]" % (prefix, i))
    else:
        yield prefix, obj


def is_finite(values) -> bool:
    return all(math.isfinite(v) for v in values)
