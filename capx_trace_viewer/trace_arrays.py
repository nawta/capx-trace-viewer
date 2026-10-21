"""Array previews and images of the trace viewer (docs/trace-format.md section 9.2).

capx_trace_viewer/trace_api.py opens an ``.npy``/``.npz`` file below the trace root (a descriptor
checked by ``capx_trace_viewer.reader.Dir``) and calls ``describe_file`` (``/array``) or
``render_file_png`` (``/array-image``) on that descriptor. Rules:

- ``np.load`` is never used (so no pickle): npy headers are parsed by hand, and only
  numeric and bool dtypes are accepted.
- Only what the answer needs is read: the npy header and the elements of the preview
  slice (``/array``, at most PREVIEW_READ_BYTES of data per array), or one 2-D plane
  (``/array-image``). Statistics are not recomputed: the recorder computed them at
  record time and stored them in the event (section 9.2), so ``stats`` is null here.
- A preview is the nested-list form of a leading slice with at most MAX_PREVIEW_VALUES
  values and MAX_PREVIEW_LISTS lists; the slice is shrunk until both fit. An empty array
  has no values but can nest many lists (shape ``(4,) * 8 + (0,)`` has 87,381), so its
  preview is built without numpy and capped the same way. Arrays with more than MAX_NDIM
  dimensions are refused. A listing (no ``?name=``) also stops previewing once its
  previews hold LISTING_PREVIEW_ITEMS values and lists together, which bounds the JSON.
- ``.npz``: before ``zipfile`` parses the central directory, its size and entry count
  are checked from the end record. Members must be stored or deflated (Python inflates
  bzip2 and LZMA members without an output limit, so they are refused) and not
  encrypted, and a member's declared size must match its npy header. Members are read in
  bounded chunks; every byte read or decompressed counts against
  MAX_ARRAY_DECOMPRESSED_BYTES per request. Any error while decompressing
  (``zlib.error``, a bad CRC, a truncated member) becomes ArrayError, which the
  endpoints answer with HTTP 400.
- A bit-packed mask (npz members ``<name>__packbits`` or ``<name>__packed`` plus
  ``<name>__shape``, as the trace recorder writes them) is presented as one
  bool array ``<name>`` with ``packed: true``; only the bits a preview or a plane needs
  are unpacked. Without ``?name=`` every member is listed from its header, previews stop
  after LISTING_PREVIEW_BYTES (``preview_omitted: "listing_budget"``).
- Images (``image`` in the ``/array`` answer says how an array is drawn): a 2-D array is
  one image (one mask for bool); a 3-D array whose last axis has size 1 is one image with
  a channel axis (one mask for bool: segmentation masks have shape (H, W, 1)); any other
  3-D bool array is a stack of masks (``index`` picks axis 0); a 3-D uint8 array with 3
  or 4 channels is an RGB(A) image; any other numeric 3-D array is a stack (``index``
  picks axis 0). Images are at most MAX_IMAGE_SIDE pixels per side.
- Section 2: nothing here prints or warns. No reduction runs on an empty array, min/max
  only over finite values, float work under ``np.errstate``.

Related modules: capx_trace_viewer/trace_api.py (the endpoints and the concurrency limit),
capx_trace_viewer/reader.py (``Dir``, ``pread_upto``), tests/test_trace_arrays.py.
"""

from __future__ import annotations

import io
import math
import os
import re
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from capx_trace_viewer import reader

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_ARRAY_FILE_BYTES = 64 * 1024 * 1024
# Bytes one request may read or decompress (also the largest array accepted).
MAX_ARRAY_DECOMPRESSED_BYTES = 256 * 1024 * 1024
# A service call's decoded outputs share one npz: SAM3 alone stores up to 50 bit-packed
# masks there (two members each, <name>__packbits and <name>__shape).
MAX_NPZ_MEMBERS = 4096
MAX_CENTRAL_DIRECTORY_BYTES = MAX_NPZ_MEMBERS * 512
MAX_PREVIEW_VALUES = 4096
# Lists one preview may nest (see preview_lists). An empty array has no values but can
# still nest many lists, and so can an array with many axes of size 1.
MAX_PREVIEW_LISTS = 4096
# Dimensions of an array the viewer accepts (numpy 1.x arrays have at most 32). An npy
# header may name thousands, and so deep a nested preview cannot be encoded as JSON.
MAX_NDIM = 32
# Data bytes one preview may read (the leading slice is shrunk until it fits).
PREVIEW_READ_BYTES = 1024 * 1024
# Preview data bytes of one /array answer that lists every array of a file, and the
# values plus lists those previews may hold together (this bounds the size of the JSON).
LISTING_PREVIEW_BYTES = 16 * 1024 * 1024
LISTING_PREVIEW_ITEMS = 256 * 1024
MAX_IMAGE_SIDE = 1024
READ_CHUNK = 1024 * 1024

# Bit-packed bool arrays in an npz: <name>__packbits (the trace recorder) or
# <name>__packed, each with <name>__shape.
PACKED_SUFFIXES = ("__packbits", "__packed")
_NPY_MAGIC = b"\x93NUMPY"
_NPY_HEADER_RE = re.compile(
    r"\{\s*'descr':\s*'(?P<descr>[<>|=]?(?:b1|[iu][1248]|f[248]))',\s*"
    r"'fortran_order':\s*(?P<fortran>True|False),\s*"
    r"'shape':\s*\((?P<shape>(?:\s*\d+\s*,)*(?:\s*\d+\s*)?)\),?\s*\}\s*"
)
_EOCD = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_ZIP_METHODS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_STACK_KINDS = ("mask_stack", "stack")


class ArrayError(ValueError):
    """The file is not an array file the viewer accepts (message is shown to the user)."""


class NameRequired(ArrayError):
    """``/array-image`` of a file with several arrays needs ``?name=``."""


def finite_json(value: Any) -> Any:
    """Replace non-finite floats by strings so the JSON is valid for browsers."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite_json(v) for v in value]
    return value


class Budget:
    """Bytes one request may still read or decompress."""

    __slots__ = ("left",)

    def __init__(self, limit: int | None = None) -> None:
        self.left = MAX_ARRAY_DECOMPRESSED_BYTES if limit is None else int(limit)

    def take(self, nbytes: int) -> None:
        if nbytes > self.left:
            raise ArrayError("the array data is larger than the viewer limit")
        self.left -= nbytes


class ListingBudget:
    """What the previews of one listing (``/array`` without ``?name=``) may still use: data
    bytes read (LISTING_PREVIEW_BYTES) and values plus lists (LISTING_PREVIEW_ITEMS)."""

    __slots__ = ("bytes_left", "items_left")

    def __init__(self) -> None:
        self.bytes_left = LISTING_PREVIEW_BYTES
        self.items_left = LISTING_PREVIEW_ITEMS

    def take(self, nbytes: int, items: int) -> bool:
        """Reserve one preview; False (and nothing taken) when it does not fit."""
        if nbytes > self.bytes_left or items > self.items_left:
            return False
        self.bytes_left -= nbytes
        self.items_left -= items
        return True


# ---------------------------------------------------------------------------
# .npy headers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NpyHeader:
    dtype: np.dtype
    shape: tuple[int, ...]
    fortran: bool
    offset: int  # bytes before the data

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.count * self.dtype.itemsize


def parse_npy_header(read: Callable[[int], bytes]) -> NpyHeader:
    """Header of an .npy stream; ``read(n)`` returns the next ``n`` bytes (fewer at EOF)."""
    head = read(10)
    if len(head) < 10 or not head.startswith(_NPY_MAGIC):
        raise ArrayError("not an .npy file")
    major = head[6]
    if major == 1:
        header_len = int.from_bytes(head[8:10], "little")
        start = 10
    elif major in (2, 3):
        more = read(2)
        if len(more) < 2:
            raise ArrayError("truncated .npy header")
        header_len = int.from_bytes(head[8:10] + more, "little")
        start = 12
    else:
        raise ArrayError(f"unsupported .npy version {major}")
    if header_len > 65536:
        raise ArrayError("bad .npy header length")
    raw = read(header_len)
    if len(raw) < header_len:
        raise ArrayError("bad .npy header length")
    match = _NPY_HEADER_RE.fullmatch(raw.decode("latin-1"))
    if match is None:
        raise ArrayError("unsupported .npy header (numeric and bool dtypes only)")
    dtype = np.dtype(match.group("descr"))
    shape = tuple(int(x) for x in match.group("shape").replace(" ", "").split(",") if x)
    if len(shape) > MAX_NDIM:
        raise ArrayError(f"arrays with more than {MAX_NDIM} dimensions are not supported")
    header = NpyHeader(dtype, shape, match.group("fortran") == "True", start + header_len)
    if header.nbytes > MAX_ARRAY_DECOMPRESSED_BYTES:
        raise ArrayError("array is larger than the viewer limit")
    return header


# ---------------------------------------------------------------------------
# Array sources: an .npy file, an npz member, a bit-packed mask
# ---------------------------------------------------------------------------


class RawArray:
    """An .npy file or one npz member: its header plus reads of data byte ranges."""

    packed = False

    def __init__(self, name: str, header: NpyHeader, read_range: Callable[[int, int, Budget], bytes]) -> None:
        self.name = name
        self.header = header
        self._read_range = read_range

    @property
    def shape(self) -> tuple[int, ...]:
        return self.header.shape

    @property
    def dtype(self) -> np.dtype:
        return self.header.dtype

    @property
    def fortran(self) -> bool:
        return self.header.fortran

    def elements(self, start: int, end: int, budget: Budget) -> np.ndarray:
        """Flat elements ``[start, end)`` in storage order (a read-only 1-D array)."""
        if end <= start:
            return np.zeros(0, dtype=self.dtype)
        size = self.dtype.itemsize
        data = self._read_range(start * size, end * size, budget)
        return np.frombuffer(data, dtype=self.dtype, count=end - start)


class PackedMask:
    """A bool array stored as ``np.packbits(mask.ravel())`` plus its shape."""

    packed = True
    fortran = False

    def __init__(self, name: str, shape: tuple[int, ...], bits: RawArray) -> None:
        self.name = name
        self.shape = shape
        self.bits = bits
        self.dtype = np.dtype(np.bool_)

    def elements(self, start: int, end: int, budget: Budget) -> np.ndarray:
        """Flat mask elements ``[start, end)`` (C order), unpacking only those bits."""
        if end <= start:
            return np.zeros(0, dtype=np.bool_)
        first, last = start // 8, (end + 7) // 8
        raw = self.bits.elements(first, last, budget)
        budget.take((last - first) * 8)
        bits = np.unpackbits(raw)
        offset = start - first * 8
        return bits[offset : offset + (end - start)].view(np.bool_)


Source = RawArray | PackedMask


def _read_stream(stream: Any, size: int, budget: Budget, *, keep: bool = True) -> bytes:
    """Up to ``size`` bytes of a (decompressing) stream in bounded chunks, each counted."""
    chunks: list[bytes] = []
    while size > 0:
        chunk = stream.read(min(READ_CHUNK, size))
        if not chunk:
            break
        budget.take(len(chunk))
        size -= len(chunk)
        if keep:
            chunks.append(chunk)
    return b"".join(chunks)


def _check_zip_directory(fd: int, size: int) -> None:
    """Refuse archives whose central directory would cost zipfile much memory.

    Finds the end record the way zipfile does (the last 22 bytes when they are an end
    record without a comment, else the last end-record signature in the final 64 KiB).
    """
    if size < 22:
        raise ArrayError("cannot read the .npz file (not a zip archive)")
    window = (1 << 16) + 22
    tail_len = min(size, window + 20)
    tail = reader.pread_upto(fd, tail_len, size - tail_len)
    if len(tail) != tail_len:
        raise ArrayError("cannot read the .npz file (truncated)")
    if tail[-22:-18] == _EOCD and tail[-2:] == b"\0\0":
        pos = tail_len - 22
    else:
        pos = tail.rfind(_EOCD, max(0, tail_len - window))
        if pos < 0 or pos + 22 > tail_len:
            raise ArrayError("cannot read the .npz file (not a zip archive)")
    _sig, _disk, _cd_disk, _n_disk, n_total, cd_size, cd_offset, _comment = struct.unpack(
        "<4s4H2LH", tail[pos : pos + 22]
    )
    zip64 = pos >= 20 and tail[pos - 20 : pos - 16] == _ZIP64_LOCATOR
    if zip64 or n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise ArrayError("ZIP64 .npz files are not supported")
    if n_total > MAX_NPZ_MEMBERS:
        raise ArrayError("too many arrays in the .npz file")
    if cd_size > MAX_CENTRAL_DIRECTORY_BYTES:
        raise ArrayError("the .npz directory is larger than the viewer limit")


class ArrayFile:
    """The arrays of one ``.npy`` or ``.npz`` file, read from an open descriptor.

    The caller owns ``fd`` and closes it after ``close()``.
    """

    def __init__(self, fd: int, st: os.stat_result, filename: str) -> None:
        self.fd = fd
        self.size = st.st_size
        self.members: dict[str, zipfile.ZipInfo] = {}
        self._fh: Any = None
        self._zip: zipfile.ZipFile | None = None
        if self.size > MAX_ARRAY_FILE_BYTES:
            raise ArrayError(f"file is larger than {MAX_ARRAY_FILE_BYTES // (1024 * 1024)} MB")
        lower = filename.lower()
        if lower.endswith(".npy"):
            self.kind = "npy"
        elif lower.endswith(".npz"):
            self.kind = "npz"
            try:
                self._open_zip()
            except BaseException:
                self.close()
                raise
        else:
            raise ArrayError("not an .npy or .npz file")

    def __enter__(self) -> ArrayFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        zf, self._zip = self._zip, None
        fh, self._fh = self._fh, None
        try:
            if zf is not None:
                zf.close()
        finally:
            if fh is not None:
                fh.close()  # closefd=False: the descriptor stays open for the caller

    # -- npz ------------------------------------------------------------------
    def _open_zip(self) -> None:
        _check_zip_directory(self.fd, self.size)
        self._fh = os.fdopen(self.fd, "rb", closefd=False)
        self._zip = zipfile.ZipFile(self._fh)
        infos = self._zip.infolist()
        if len(infos) > MAX_NPZ_MEMBERS:
            raise ArrayError("too many arrays in the .npz file")
        for info in infos:
            name = info.filename
            if not name.endswith(".npy") or "/" in name or name[:-4] in self.members:
                raise ArrayError("unexpected member in the .npz file")
            self.members[name[:-4]] = info

    def _member(self, name: str, budget: Budget) -> RawArray:
        info = self.members[name]
        zf = self._zip
        assert zf is not None
        if info.flag_bits & 0x1:
            raise ArrayError("encrypted .npz members are not supported")
        if info.compress_type not in _ZIP_METHODS:
            raise ArrayError("unsupported compression in the .npz file (only stored and deflated members)")
        with zf.open(info) as stream:
            header = parse_npy_header(lambda n: _read_stream(stream, n, budget))
        if info.file_size != header.offset + header.nbytes:
            raise ArrayError("an .npz member's size does not match its .npy header")

        def read_range(start: int, end: int, budget: Budget) -> bytes:
            with zf.open(info) as stream:
                _read_stream(stream, header.offset + start, budget, keep=False)
                data = _read_stream(stream, end - start, budget)
            if len(data) != end - start:
                raise ArrayError("truncated .npz member")
            return data

        return RawArray(name, header, read_range)

    def _packed(self, base: str, bits: RawArray, shape_src: RawArray, budget: Budget) -> PackedMask | None:
        """The mask of a packbits member and its shape member, or None if they do not match."""
        if bits.dtype != np.uint8 or len(bits.shape) != 1:
            return None
        if shape_src.dtype.kind not in "iu" or len(shape_src.shape) != 1 or shape_src.shape[0] > 8:
            return None
        values = shape_src.elements(0, shape_src.shape[0], budget).tolist()
        shape = tuple(int(v) for v in values)
        if any(v < 0 for v in shape):
            return None
        # The packed member passed the size limit in its header, so the mask has at most
        # 8 x that many elements; drawing more than the request budget fails in render_png.
        if bits.shape[0] != (math.prod(shape) + 7) // 8:
            return None
        return PackedMask(base, shape, bits)

    # -- npy ------------------------------------------------------------------
    def _npy(self) -> RawArray:
        pos = 0

        def read(n: int) -> bytes:
            nonlocal pos
            data = reader.pread_upto(self.fd, n, pos)
            pos += len(data)
            return data

        header = parse_npy_header(read)
        if header.offset + header.nbytes > self.size:
            raise ArrayError("truncated .npy data")

        def read_range(start: int, end: int, budget: Budget) -> bytes:
            budget.take(end - start)
            data = reader.pread_upto(self.fd, end - start, header.offset + start)
            if len(data) != end - start:
                raise ArrayError("truncated .npy data")
            return data

        return RawArray("array", header, read_range)

    # -- selection --------------------------------------------------------------
    def all_arrays(self, budget: Budget) -> dict[str, Source]:
        """Every array of the file; bit-packed masks are merged into one bool array."""
        if self.kind == "npy":
            return {"array": self._npy()}
        raw = {name: self._member(name, budget) for name in self.members}
        out: dict[str, Source] = {}
        merged: set[str] = set()
        for key, src in raw.items():
            suffix = next((s for s in PACKED_SUFFIXES if key.endswith(s)), None)
            if suffix is None:
                continue
            base = key[: -len(suffix)]
            shape_src = raw.get(base + "__shape")
            if shape_src is None or base in raw or base in out:
                continue
            mask = self._packed(base, src, shape_src, budget)
            if mask is not None:
                out[base] = mask
                merged.update((key, base + "__shape"))
        for key, src in raw.items():
            if key not in merged:
                out[key] = src
        return out

    def select(self, name: str, budget: Budget) -> dict[str, Source] | None:
        """``{name: array}`` for ``?name=``, reading only that array's members; None if absent.

        ``name`` may be a member, a packed member (``a0__packbits``: its mask) or the base
        name of a packed mask (``a0``).
        """
        if self.kind == "npy":
            return {name: self._npy()} if name == "array" else None
        members = self.members
        for suffix in PACKED_SUFFIXES:
            if name.endswith(suffix) and name in members:
                base = name[: -len(suffix)]
                bits = self._member(name, budget)
                if base + "__shape" in members and base not in members:
                    mask = self._packed(base, bits, self._member(base + "__shape", budget), budget)
                    if mask is not None:
                        return {name: mask}
                return {name: bits}
        if name in members:
            return {name: self._member(name, budget)}
        for suffix in PACKED_SUFFIXES:
            if name + suffix in members and name + "__shape" in members:
                mask = self._packed(
                    name, self._member(name + suffix, budget), self._member(name + "__shape", budget), budget
                )
                return {name: mask} if mask is not None else None
        return None


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def _element_strides(shape: tuple[int, ...], fortran: bool) -> list[int]:
    strides = [1] * len(shape)
    if fortran:
        for i in range(1, len(shape)):
            strides[i] = strides[i - 1] * shape[i - 1]
    else:
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def preview_lists(pshape: tuple[int, ...] | list[int]) -> int:
    """Lists in the nested-list form of an array of shape ``pshape`` (0 for a scalar).

    Shape (a, b, c) nests 1 + a + a*b lists. The count stops at the first axis of size 0:
    shape (4, 0) is ``[[], [], [], []]`` (5 lists), shape (0, 4) is ``[]`` (1 list).
    """
    total, level = 0, 1
    for n in pshape:
        total += level
        level *= int(n)
        if level == 0:
            break
    return total


def preview_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shape of the leading slice shown as a preview: at most MAX_PREVIEW_VALUES values and
    MAX_PREVIEW_LISTS lists.

    Caps per axis (4 on leading axes, 16 on the last two, 64 for 1-D), then the first
    axis that is still larger than 1 is halved until both counts fit. Every step makes an
    axis of size >= 2 strictly smaller, so the loop always ends; once every axis is 0 or
    1, the slice holds at most one value and MAX_NDIM lists.
    """
    ndim = len(shape)
    if ndim == 0:
        return ()
    caps = [4] * ndim
    caps[-1] = 64 if ndim == 1 else 16
    if ndim >= 2:
        caps[-2] = 16
    pshape = [min(int(n), c) for n, c in zip(shape, caps, strict=True)]
    while math.prod(pshape) > MAX_PREVIEW_VALUES or preview_lists(pshape) > MAX_PREVIEW_LISTS:
        axis = next((i for i, n in enumerate(pshape) if n > 1), None)
        if axis is None:
            break
        pshape[axis] = max(1, pshape[axis] // 2)
    return tuple(pshape)


def _empty_preview(pshape: tuple[int, ...]) -> list[Any]:
    """The nested lists of an empty preview slice (``pshape`` has an axis of size 0).

    Built directly, never with numpy's ``tolist()``: the lists stop at the first axis of
    size 0, and preview_shape has already capped their number. Sibling lists share one
    object; JSON encodes each occurrence.
    """
    zero = pshape.index(0)
    item: list[Any] = []
    for n in reversed(pshape[:zero]):
        item = [item] * n
    return item


def _slice_index(shape: tuple[int, ...], pshape: tuple[int, ...], fortran: bool) -> np.ndarray:
    """Storage positions of the leading slice ``pshape`` of an array of ``shape``, as an
    array of shape ``pshape`` (built axis by axis: ``np.indices`` would need one more
    dimension than numpy allows for a 32-D array)."""
    ndim = len(shape)
    index = np.zeros(pshape, dtype=np.intp)
    for axis, (p, stride) in enumerate(zip(pshape, _element_strides(shape, fortran), strict=True)):
        along = [1] * ndim
        along[axis] = p
        index += (np.arange(p, dtype=np.intp) * stride).reshape(along)
    return index


def _preview_plan(src: Source) -> tuple[tuple[int, ...], int, int]:
    """(preview shape, elements to read from the start, bytes those elements take).

    Shrinks the preview until its data (from the first element to the last one the
    slice needs) fits in PREVIEW_READ_BYTES, halving the axis that spans the most data.
    """
    shape = tuple(int(n) for n in src.shape)
    pshape = list(preview_shape(shape))
    strides = _element_strides(shape, src.fortran)

    def cost(ps: list[int]) -> tuple[int, int]:
        needed = sum((p - 1) * s for p, s in zip(ps, strides, strict=True)) + 1
        nbytes = (needed + 7) // 8 if src.packed else needed * src.dtype.itemsize
        return needed, nbytes

    needed, nbytes = cost(pshape)
    while nbytes > PREVIEW_READ_BYTES:
        candidates = [i for i, p in enumerate(pshape) if p > 1]
        if not candidates:
            break
        axis = max(candidates, key=lambda i: (pshape[i] - 1) * strides[i])
        pshape[axis] = max(1, pshape[axis] // 2)
        needed, nbytes = cost(pshape)
    return tuple(pshape), needed, nbytes


def image_layout(shape: tuple[int, ...], dtype: np.dtype) -> dict[str, Any] | None:
    """How ``/array-image`` draws an array, or None if it cannot be drawn.

    ``kind``: ``mask`` (bool) or ``gray`` (other dtypes) for a 2-D array or a 3-D array
    whose last axis has size 1 (H x W with a channel axis, e.g. a segmentation mask of
    shape (H, W, 1)); ``mask_stack`` (any other 3-D bool array: masks along axis 0);
    ``rgb`` (uint8 with 3 or 4 channels); ``stack`` (other numeric 3-D: images along axis
    0). ``count`` is the number of images (``index`` of ``/array-image`` picks one),
    ``height`` and ``width`` their size.
    """
    if dtype.kind not in "biuf":
        return None
    is_bool = dtype == np.bool_
    if len(shape) == 2:
        kind, count, height, width = ("mask" if is_bool else "gray"), 1, shape[0], shape[1]
    elif len(shape) == 3:
        if shape[2] == 1:
            kind, count, height, width = ("mask" if is_bool else "gray"), 1, shape[0], shape[1]
        elif is_bool:
            kind, count, height, width = "mask_stack", shape[0], shape[1], shape[2]
        elif dtype == np.uint8 and shape[2] in (3, 4):
            kind, count, height, width = "rgb", 1, shape[0], shape[1]
        else:
            kind, count, height, width = "stack", shape[0], shape[1], shape[2]
    else:
        return None
    if count == 0 or height == 0 or width == 0:
        return None
    return {"kind": kind, "count": int(count), "height": int(height), "width": int(width)}


def describe(src: Source, budget: Budget, listing: ListingBudget | None = None) -> dict[str, Any]:
    """The ``/array`` answer for one array: shape, dtype, preview, mask and image info.

    ``listing`` is the preview budget shared by all arrays of a listing; once a preview
    does not fit in it, that array gets ``preview: null`` and ``preview_omitted:
    "listing_budget"``.
    """
    shape = tuple(int(n) for n in src.shape)
    dtype = src.dtype
    info: dict[str, Any] = {
        "shape": list(shape),
        "dtype": str(dtype),
        "stats": None,  # recorded in the event at capture time; not recomputed here
        "preview": None,
        "preview_shape": None,
        "truncated": True,
        "is_mask": bool(dtype == np.bool_ and len(shape) in (2, 3)),
        "packed": src.packed,
        "image": image_layout(shape, dtype),
    }
    if math.prod(shape) == 0:
        # No values and nothing to read: only the (capped) empty lists of the slice.
        pshape = preview_shape(shape)
        if listing is not None and not listing.take(0, preview_lists(pshape)):
            info["preview_omitted"] = "listing_budget"
            return info
        info.update(preview=_empty_preview(pshape), preview_shape=list(pshape), truncated=pshape != shape)
        return info
    pshape, needed, nbytes = _preview_plan(src)
    if listing is not None and not listing.take(nbytes, math.prod(pshape) + preview_lists(pshape)):
        info["preview_omitted"] = "listing_budget"
        return info
    flat = src.elements(0, needed, budget)
    if not shape:
        info.update(preview=finite_json(flat[0].item()), preview_shape=[], truncated=False)
        return info
    part = flat[_slice_index(shape, pshape, src.fortran)]
    info.update(
        preview=finite_json(part.tolist()),
        preview_shape=list(pshape),
        truncated=tuple(pshape) != shape,
    )
    return info


def describe_file(fd: int, st: os.stat_result, filename: str, name: str | None) -> dict[str, dict[str, Any]] | None:
    """The ``arrays`` of the ``/array`` answer; None when ``name`` names no array."""
    budget = Budget()
    with _converted_errors(filename), ArrayFile(fd, st, filename) as af:
        if name is not None:
            selected = af.select(name, budget)
            if selected is None:
                return None
            return {key: describe(src, budget) for key, src in selected.items()}
        listing = ListingBudget()
        arrays = af.all_arrays(budget)
        return {key: describe(src, budget, listing) for key, src in arrays.items()}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _plane(src: Source, layout: dict[str, Any], index: int, budget: Budget) -> np.ndarray:
    """The 2-D image (or H x W x C for RGB) that ``layout`` and ``index`` select."""
    kind, height, width = layout["kind"], layout["height"], layout["width"]
    shape = tuple(int(n) for n in src.shape)
    if src.fortran:
        arr = src.elements(0, math.prod(shape), budget).reshape(shape, order="F")
        if kind in _STACK_KINDS:
            return arr[index]
        if len(shape) == 3 and kind in ("gray", "mask"):
            return arr[:, :, 0]  # (H, W, 1)
        return arr
    channels = shape[2] if kind == "rgb" else 1
    start = index * height * width if kind in _STACK_KINDS else 0
    flat = src.elements(start, start + height * width * channels, budget)
    if kind == "rgb":
        return flat.reshape(height, width, channels)
    return flat.reshape(height, width)


def render_png(src: Source, index: int, budget: Budget) -> bytes:
    """PNG of a mask, a depth or other numeric image, an RGB image, or one image of a stack."""
    from PIL import Image  # imported here: only this endpoint needs Pillow

    layout = image_layout(tuple(int(n) for n in src.shape), src.dtype)
    if layout is None:
        raise ArrayError("only 2-D arrays (or stacks of them) can be shown as images")
    if layout["kind"] in _STACK_KINDS:
        if not 0 <= index < layout["count"]:
            raise ArrayError(f"index must be in [0, {layout['count']})")
    else:
        index = 0  # a single image: index is ignored, as before
    plane = _plane(src, layout, index, budget)
    step = max(1, math.ceil(max(plane.shape[0], plane.shape[1]) / MAX_IMAGE_SIDE))
    plane = plane[::step, ::step]
    if layout["kind"] == "rgb":
        img = np.ascontiguousarray(plane)
    elif plane.dtype == np.bool_:
        img = np.where(plane, np.uint8(255), np.uint8(0))
    else:
        values = plane.astype(np.float64)
        with np.errstate(all="ignore"):
            valid = np.isfinite(values)
            img = np.zeros(values.shape, dtype=np.uint8)
            if valid.any():
                finite = values[valid]
                lo, hi = float(finite.min()), float(finite.max())
                span = hi - lo
                if math.isfinite(span):
                    offset = finite - lo
                else:
                    # hi - lo overflows near the float64 limits; halved values keep it finite
                    offset, span = finite / 2.0 - lo / 2.0, hi / 2.0 - lo / 2.0
                if span > 0.0:
                    # Divide by the span before multiplying by 255; 255 / span overflows for a subnormal span.
                    img[valid] = np.clip(offset / span * 255.0, 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="png")
    return buf.getvalue()


def render_file_png(fd: int, st: os.stat_result, filename: str, name: str | None, index: int) -> bytes | None:
    """The ``/array-image`` PNG; None when ``name`` names no array.

    Raises NameRequired when the file holds several arrays and ``name`` is None.
    """
    budget = Budget()
    with _converted_errors(filename), ArrayFile(fd, st, filename) as af:
        if name is None:
            arrays = af.all_arrays(budget)
            if len(arrays) != 1:
                raise NameRequired("name is required for a file with several arrays")
            src = next(iter(arrays.values()))
        else:
            selected = af.select(name, budget)
            if selected is None:
                return None
            src = next(iter(selected.values()))
        return render_png(src, index, budget)


class _converted_errors:
    """Turn any error while parsing or decompressing into ArrayError (HTTP 400)."""

    def __init__(self, filename: str) -> None:
        self.what = ".npz file" if filename.lower().endswith(".npz") else "array file"

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is None or isinstance(exc, ArrayError) or not isinstance(exc, Exception):
            return False
        raise ArrayError(f"cannot read the {self.what} ({type(exc).__name__})") from None
