"""Tests of capx_trace_viewer/trace_arrays.py (docs/trace-format.md section 9.2).

Covers array endpoint regressions: previews that always end (a 6-D array
used to loop forever), reads bounded by what the answer needs (zip bombs with lying
member sizes, bzip2/LZMA members, damaged deflate data, central directories that are too
large, many bit-packed masks), no statistics recomputed over whole arrays, and 3-D bool
arrays drawn as stacks of masks. Also the parsing edge cases that moved here from
test_trace_api.py (packed members, Fortran order, big-endian, empty and scalar arrays).

previews are bounded in lists as well as values (an empty array of shape
(4,) * 8 + (0,) used to build 87,381 lists, (4,) * 15 + (0,) about 1.4e9), arrays with
more than 32 dimensions are refused, a listing's previews share a budget of values and
lists, and a bool array of shape (H, W, 1) is one H x W mask (it was drawn as H masks one
pixel wide).
"""

from __future__ import annotations

import io
import itertools
import json
import math
import os
import struct
import threading
import warnings
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from capx_trace_viewer import reader
from capx_trace_viewer import trace_arrays
from capx_trace_viewer.trace_arrays import ArrayError, ArrayFile, Budget
from trace_viewer_testlib import damaged_npz_bytes


@contextmanager
def opened(path: Path) -> Iterator[tuple[int, os.stat_result]]:
    fd = os.open(path, os.O_RDONLY)
    try:
        yield fd, os.fstat(fd)
    finally:
        os.close(fd)


def describe(path: Path, name: str | None = None) -> dict[str, Any] | None:
    with opened(path) as (fd, st):
        return trace_arrays.describe_file(fd, st, path.name, name)


def png(path: Path, name: str | None = None, index: int = 0) -> np.ndarray:
    with opened(path) as (fd, st):
        data = trace_arrays.render_file_png(fd, st, path.name, name, index)
    assert data is not None
    return np.asarray(Image.open(io.BytesIO(data)))


def full_array(path: Path, name: str) -> np.ndarray:
    """The whole array ``name`` as the endpoints read it (for comparisons)."""
    with opened(path) as (fd, st), ArrayFile(fd, st, path.name) as af:
        budget = Budget()
        src = af.select(name, budget)[name]
        flat = src.elements(0, math.prod(src.shape), budget)
        return flat.reshape(src.shape, order="F" if src.fortran else "C")


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def count_lists(value: Any) -> int:
    """Lists in a nested-list preview, counted independently of trace_arrays."""
    return 1 + sum(count_lists(v) for v in value) if isinstance(value, list) else 0


def lists_of_shape(shape: tuple[int, ...]) -> int:
    """Lists in the nested-list form of an array of ``shape`` (1 + a + a*b for (a, b, c))."""
    total, level = 0, 1
    for n in shape:
        total += level
        level *= n
        if level == 0:
            break
    return total


def npy_with_shape(shape: tuple[int, ...], descr: str = "|u1") -> bytes:
    """An npy v1 file whose header names ``shape`` (numpy 1.x cannot save more than 32
    dimensions); the data is zeros."""
    dims = "".join(f"{n}," for n in shape)
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}"
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    count = math.prod(shape)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1") + b"\0" * count


def run_with_watchdog(fn: Any, timeout: float = 20.0) -> Any:
    """Run ``fn`` in a thread; fail (instead of hanging) if it does not return."""
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "did not finish (endless loop?)"
    if "error" in result:
        raise result["error"]
    return result["value"]


# ---------------------------------------------------------------------------
# previews always end
# ---------------------------------------------------------------------------


def test_six_d_array_with_a_size_one_leading_axis_gets_a_preview(tmp_path: Path) -> None:
    """(1, 4, 4, 4, 16, 16) uint8 used to loop forever: the halved axis was already 1."""
    arr = np.arange(np.prod((1, 4, 4, 4, 16, 16)), dtype=np.uint64).astype(np.uint8).reshape(1, 4, 4, 4, 16, 16)
    path = tmp_path / "six.npy"
    np.save(path, arr)
    info = run_with_watchdog(lambda: describe(path))["array"]
    assert info["preview_shape"] == [1, 1, 4, 4, 16, 16] and info["truncated"]
    assert np.array_equal(np.asarray(info["preview"], dtype=np.uint8), arr[:1, :1, :4, :4, :16, :16])


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5, 6, 7])
def test_preview_shape_fits_and_ends_for_every_shape(ndim: int) -> None:
    sizes = (0, 1, 2, 3, 5, 16, 17, 64, 100, 4097)
    shapes = list(itertools.islice(itertools.product(sizes, repeat=ndim), 3000))

    def check() -> None:
        for shape in shapes:
            pshape = trace_arrays.preview_shape(shape)
            assert len(pshape) == ndim
            assert all(0 <= p <= n for p, n in zip(pshape, shape, strict=True)), (shape, pshape)
            assert math.prod(pshape) <= trace_arrays.MAX_PREVIEW_VALUES, (shape, pshape)

    run_with_watchdog(check)


def test_preview_reads_only_the_leading_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No whole-array pass: a 16 MB npy is answered from its header and first rows."""
    arr = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
    path = tmp_path / "big.npy"
    np.save(path, arr)
    read: list[int] = []
    real = reader.pread_upto

    def counting(fd: int, size: int, offset: int) -> bytes:
        data = real(fd, size, offset)
        read.append(len(data))
        return data

    monkeypatch.setattr(reader, "pread_upto", counting)
    info = describe(path)["array"]
    assert info["stats"] is None  # recorded in the event at capture time, not recomputed
    assert np.array_equal(np.asarray(info["preview"], dtype=np.float32), arr[:16, :16])
    assert sum(read) < 200_000, sum(read)


def test_preview_of_a_huge_stack_is_shrunk_to_the_read_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_arrays, "PREVIEW_READ_BYTES", 4096)
    arr = np.arange(4 * 64 * 64, dtype=np.float64).reshape(4, 64, 64)
    path = tmp_path / "stack.npy"
    np.save(path, arr)
    info = describe(path)["array"]
    pshape = tuple(info["preview_shape"])
    assert pshape[0] == 1 and info["truncated"]
    assert np.array_equal(np.asarray(info["preview"]), arr[tuple(slice(0, p) for p in pshape)])


def test_fortran_order_preview_and_image(tmp_path: Path) -> None:
    arr = np.asfortranarray(np.arange(40 * 30, dtype=np.int32).reshape(40, 30))
    path = tmp_path / "fortran.npy"
    np.save(path, arr)
    info = describe(path)["array"]
    assert np.array_equal(np.asarray(info["preview"]), arr[:16, :16])
    assert np.array_equal(full_array(path, "array"), arr)
    assert png(path).shape == (40, 30)


# ---------------------------------------------------------------------------
# Previews also bound list count; empty arrays never use tolist().
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(4,) * 8 + (0,), (4,) * 10 + (0,), (2,) * 18 + (0,), (16, 16, 16, 0), (3, 0, 5), (0, 4), (4, 0)],
)
def test_empty_array_previews_are_bounded(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """The regression case: np.empty((4,) * 8 + (0,)) gave a 128-byte file whose preview held
    87,381 lists; (4,) * 15 + (0,) would have built about 1.4e9 of them."""
    path = tmp_path / "empty.npy"
    np.save(path, np.empty(shape, dtype=np.uint8))
    info = run_with_watchdog(lambda: describe(path))["array"]
    pshape = tuple(info["preview_shape"])
    assert count_lists(info["preview"]) == lists_of_shape(pshape) <= trace_arrays.MAX_PREVIEW_LISTS
    assert len(pshape) == len(shape) and all(0 <= p <= n for p, n in zip(pshape, shape, strict=True))
    assert info["truncated"] == (pshape != shape) and info["shape"] == list(shape)
    # the leading slice of the empty array (nested lists end at its first axis of size 0)
    assert np.asarray(info["preview"]).shape == pshape[: pshape.index(0) + 1]
    assert len(json.dumps(info)) < 64 * 1024


def test_preview_shape_of_huge_empty_arrays_is_bounded_without_building_it() -> None:
    for shape in ((4,) * 15 + (0,), (4,) * 31 + (0,), (1,) * 31 + (0,), (0,) + (4,) * 31):
        pshape = trace_arrays.preview_shape(shape)
        assert lists_of_shape(pshape) <= trace_arrays.MAX_PREVIEW_LISTS, (shape, pshape)


@pytest.mark.parametrize("fortran", [False, True])
def test_previews_of_arrays_with_many_axes_are_bounded(tmp_path: Path, fortran: bool) -> None:
    """(4,) * 6 + (1,) * 26 has 4096 values but would nest over 100,000 lists; numpy's
    np.indices could not index a 32-D array at all."""
    arr = (np.arange(4**6) % 251).astype(np.int16).reshape((4,) * 6 + (1,) * 26)
    path = tmp_path / "deep.npy"
    np.save(path, np.asfortranarray(arr) if fortran else arr)
    info = describe(path)["array"]
    pshape = tuple(info["preview_shape"])
    assert lists_of_shape(pshape) <= trace_arrays.MAX_PREVIEW_LISTS and count_lists(info["preview"]) == lists_of_shape(pshape)
    assert np.array_equal(np.asarray(info["preview"]), arr[tuple(slice(0, p) for p in pshape)])


def test_arrays_with_more_than_32_dimensions_are_refused(tmp_path: Path) -> None:
    """An npy header may name thousands of dimensions; an empty one would have given a
    preview nested too deep for JSON."""
    for shape in ((1,) * 33, (1,) * 2000 + (0,)):
        path = tmp_path / "deep.npy"
        path.write_bytes(npy_with_shape(shape))
        with pytest.raises(ArrayError, match="more than 32 dimensions"):
            describe(path)
    path.write_bytes(npy_with_shape((1,) * 32))  # 32 is accepted
    assert describe(path)["array"]["preview"] is not None


def test_listing_previews_share_a_budget_of_values_and_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it, 4096 members of 4096 values each would make a JSON answer of about
    100 MB while reading only 16 MB of data."""
    path = tmp_path / "many.npz"
    np.savez(path, **{f"x{i}": np.full((4, 4, 16, 16), i, dtype=np.uint8) for i in range(8)})
    items = 4096 + lists_of_shape((4, 4, 16, 16))
    monkeypatch.setattr(trace_arrays, "LISTING_PREVIEW_ITEMS", 3 * items)
    listing = describe(path)
    with_preview = [k for k, a in listing.items() if a["preview"] is not None]
    omitted = [k for k, a in listing.items() if a.get("preview_omitted") == "listing_budget"]
    assert len(with_preview) == 3 and len(omitted) == 5
    assert describe(path, "x7")["x7"]["preview"][0][0][0][0] == 7  # ?name= always previews


# ---------------------------------------------------------------------------
# bounded reads of npz members
# ---------------------------------------------------------------------------


def _lying_member_zip(path: Path, method: int, expand_bytes: int, arr: np.ndarray) -> None:
    """An npz whose only member declares the size of ``arr``'s npy while its compressed
    stream expands much further (a decompression bomb)."""
    npy = npy_bytes(arr)
    comp = zipfile._get_compressor(method)  # noqa: SLF001 - the same compressor zipfile uses
    parts = [comp.compress(npy)]
    chunk = b"\0" * (1 << 20)
    left = expand_bytes
    while left > 0:
        parts.append(comp.compress(chunk[: min(left, len(chunk))]))
        left -= len(chunk)
    parts.append(comp.flush())
    data = b"".join(parts)
    name = b"a.npy"
    crc = zlib.crc32(npy) & 0xFFFFFFFF
    version = 63 if method == zipfile.ZIP_LZMA else (46 if method == zipfile.ZIP_BZIP2 else 20)
    local = struct.pack("<IHHHHHIIIHH", 0x04034B50, version, 0, method, 0, 0x21, crc, len(data), len(npy), len(name), 0) + name
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII", 0x02014B50, version, version, 0, method, 0, 0x21, crc, len(data), len(npy),
        len(name), 0, 0, 0, 0, 0, 0,
    ) + name
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local) + len(data), 0)
    path.write_bytes(local + data + central + eocd)


class _CountingDecompressor:
    """Wraps zlib.decompressobj to count what zipfile inflates."""

    total = 0

    def __init__(self, wbits: int) -> None:
        self.inner = _REAL_DECOMPRESSOBJ(wbits)

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        out = self.inner.decompress(data, max_length)
        _CountingDecompressor.total += len(out)
        return out

    def flush(self, *args: Any) -> bytes:
        out = self.inner.flush(*args)
        _CountingDecompressor.total += len(out)
        return out

    @property
    def unconsumed_tail(self) -> bytes:
        return self.inner.unconsumed_tail

    @property
    def eof(self) -> bool:
        return self.inner.eof


_REAL_DECOMPRESSOBJ = zlib.decompressobj


def test_zip_bomb_with_a_lying_member_size_is_not_inflated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The member declares 4x4 float64 (256 bytes of data) but its deflate stream holds
    64 MB more. Only the declared bytes may be inflated, for the preview and the image."""
    arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    path = tmp_path / "bomb.npz"
    _lying_member_zip(path, zipfile.ZIP_DEFLATED, 64 << 20, arr)
    monkeypatch.setattr(zlib, "decompressobj", _CountingDecompressor)
    _CountingDecompressor.total = 0
    info = describe(path)["a"]
    assert np.array_equal(np.asarray(info["preview"]), arr)
    listing_inflated = _CountingDecompressor.total
    _CountingDecompressor.total = 0
    image = png(path, "a")
    assert image.shape == (4, 4)
    assert listing_inflated < 64 * 1024 and _CountingDecompressor.total < 64 * 1024, (
        listing_inflated, _CountingDecompressor.total)


@pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_bzip2_and_lzma_members_are_refused(tmp_path: Path, method: int) -> None:
    """Python inflates these without an output limit, so they are never opened."""
    path = tmp_path / "bomb.npz"
    _lying_member_zip(path, method, 8 << 20, np.zeros(16))
    for call in (lambda: describe(path), lambda: describe(path, "a"), lambda: png(path, "a")):
        with pytest.raises(ArrayError, match="unsupported compression"):
            call()


def _damaged_npz(tmp_path: Path) -> Path:
    """An np.savez_compressed file with one byte flipped so that inflating raises zlib.error."""
    path = tmp_path / "damaged.npz"
    path.write_bytes(damaged_npz_bytes())
    return path


def test_damaged_deflate_data_is_an_array_error(tmp_path: Path) -> None:
    path = _damaged_npz(tmp_path)
    with pytest.raises(ArrayError, match=r"cannot read the \.npz file \(error\)"):
        png(path, "a0")
    for call in (lambda: describe(path), lambda: describe(path, "a0")):
        try:
            call()  # a preview may end before the damaged byte
        except ArrayError:
            pass


def test_central_directory_checks_happen_before_zipfile_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "four.npz"
    np.savez(path, a=np.zeros(3), b=np.ones(3), c=np.arange(3), d=np.arange(4))
    assert set(describe(path)) == {"a", "b", "c", "d"}
    monkeypatch.setattr(trace_arrays, "MAX_NPZ_MEMBERS", 3)
    with pytest.raises(ArrayError, match="too many arrays"):
        describe(path)
    monkeypatch.setattr(trace_arrays, "MAX_NPZ_MEMBERS", 4096)
    monkeypatch.setattr(trace_arrays, "MAX_CENTRAL_DIRECTORY_BYTES", 100)
    with pytest.raises(ArrayError, match="directory is larger"):
        describe(path)
    monkeypatch.setattr(trace_arrays, "MAX_CENTRAL_DIRECTORY_BYTES", 2 << 20)

    def zip_class_must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("zipfile parsed a directory that the end record refuses")

    data = path.read_bytes()
    eocd = data.rfind(b"PK\x05\x06")
    for label, patched in (
        ("entries", data[: eocd + 10] + struct.pack("<H", 0xFFFF) + data[eocd + 12:]),
        ("locator", data[:eocd] + b"PK\x06\x07" + b"\0" * 16 + data[eocd:]),
    ):
        bad = tmp_path / f"zip64_{label}.npz"
        bad.write_bytes(patched)
        monkeypatch.setattr(trace_arrays.zipfile, "ZipFile", zip_class_must_not_run)
        with pytest.raises(ArrayError, match="ZIP64"):
            describe(bad)
        monkeypatch.undo()


def test_many_packed_masks_are_listed_without_unpacking_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """20 masks of 1024 x 1024 unpack to 20 MB; the listing reads headers and a few bits."""
    masks = {}
    for i in range(20):
        m = np.zeros((1024, 1024), dtype=bool)
        m[i::20, :] = True
        masks[f"a{i}__packbits"] = np.packbits(m.ravel())
        masks[f"a{i}__shape"] = np.array(m.shape)
    path = tmp_path / "masks.npz"
    np.savez_compressed(path, **masks)
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_DECOMPRESSED_BYTES", 4 << 20)
    listing = describe(path)
    assert set(listing) == {f"a{i}" for i in range(20)}
    assert all(a["packed"] and a["is_mask"] and a["shape"] == [1024, 1024] for a in listing.values())
    assert listing["a3"]["preview"][3][:4] == [True, True, True, True] and listing["a3"]["preview"][4][0] is False
    image = png(path, "a7")  # one mask: 1 MB unpacked, within the 4 MB request budget
    assert image.shape == (1024, 1024) and image[7].min() == 255 and image[8].max() == 0
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_DECOMPRESSED_BYTES", 512 << 10)
    with pytest.raises(ArrayError, match="larger than the viewer limit"):
        png(path, "a7")


def test_listing_previews_stop_at_the_listing_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "many.npz"
    np.savez(path, **{f"x{i}": np.full((64, 64), i, dtype=np.float64) for i in range(8)})
    monkeypatch.setattr(trace_arrays, "LISTING_PREVIEW_BYTES", 3 * 16 * 64 * 8)
    listing = describe(path)
    with_preview = [k for k, a in listing.items() if a["preview"] is not None]
    omitted = [k for k, a in listing.items() if a.get("preview_omitted") == "listing_budget"]
    assert len(with_preview) == 3 and len(omitted) == 5
    assert all(listing[k]["shape"] == [64, 64] for k in omitted)
    assert describe(path, "x7")["x7"]["preview"][0][0] == 7.0  # ?name= always previews


def test_name_selects_only_that_arrays_members(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    masks: dict[str, np.ndarray] = {}
    for i in range(50):
        m = np.zeros((32, 40), dtype=bool)
        m[i % 32, :] = True
        masks[f"a{i}__packbits"] = np.packbits(m.ravel())
        masks[f"a{i}__shape"] = np.array(m.shape)
    path = tmp_path / "sam3.npz"
    np.savez_compressed(path, **masks)
    opened_members: list[str] = []
    real = ArrayFile._member

    def spy(self: ArrayFile, name: str, budget: Budget) -> Any:
        opened_members.append(name)
        return real(self, name, budget)

    monkeypatch.setattr(ArrayFile, "_member", spy)
    one = describe(path, "a49__packbits")
    assert set(one) == {"a49__packbits"} and one["a49__packbits"]["shape"] == [32, 40]
    assert sorted(opened_members) == ["a49__packbits", "a49__shape"]
    opened_members.clear()
    assert set(describe(path, "a3")) == {"a3"} and sorted(opened_members) == ["a3__packbits", "a3__shape"]
    assert describe(path, "nope") is None
    assert describe(path, "a3__shape")["a3__shape"]["preview"] == [32, 40]  # a member by its own name


# ---------------------------------------------------------------------------
# 3-D arrays in /array-image
# ---------------------------------------------------------------------------


def test_bool_array_with_a_last_axis_of_one_is_one_mask(tmp_path: Path) -> None:
    """A (H, W, 1) array is the shape of segmentation masks (plan_grasp accepts
    it); it was drawn as H masks one pixel wide. It is one H x W mask now."""
    arr = np.array([[[True], [False]], [[False], [True]]])
    path = tmp_path / "hw1.npy"
    np.save(path, arr)
    info = describe(path)["array"]
    assert info["image"] == {"kind": "mask", "count": 1, "height": 2, "width": 2} and info["is_mask"]
    assert png(path).tolist() == [[255, 0], [0, 255]]
    assert png(path, index=5).tolist() == [[255, 0], [0, 255]]  # one image: index is ignored
    seg = np.zeros((480, 640, 1), dtype=np.int32)
    seg[100:200, 50:150] = 3
    for label, mask in (("c", seg == 3), ("f", np.asfortranarray(seg == 3))):
        path = tmp_path / f"mask_{label}.npy"
        np.save(path, mask)
        assert describe(path)["array"]["image"] == {"kind": "mask", "count": 1, "height": 480, "width": 640}
        image = png(path)
        assert image.shape == (480, 640) and np.array_equal(image == 255, seg[:, :, 0] == 3), label
    packed = tmp_path / "packed.npz"
    mask = seg == 3
    np.savez(packed, a0__packbits=np.packbits(mask.ravel()), a0__shape=np.array(mask.shape))
    assert describe(packed)["a0"]["image"] == {"kind": "mask", "count": 1, "height": 480, "width": 640}
    assert np.array_equal(png(packed, "a0") == 255, mask[:, :, 0])


def test_bool_arrays_with_a_last_axis_over_one_stay_stacks(tmp_path: Path) -> None:
    arr = np.zeros((2, 3, 2), dtype=bool)
    arr[1, 2, 0] = True
    path = tmp_path / "stack.npy"
    np.save(path, arr)
    assert describe(path)["array"]["image"] == {"kind": "mask_stack", "count": 2, "height": 3, "width": 2}
    assert png(path, index=1).tolist() == [[0, 0], [0, 0], [255, 0]]
    with pytest.raises(ArrayError, match=r"index must be in \[0, 2\)"):
        png(path, index=2)


@pytest.mark.parametrize(
    ("arr", "layout", "index", "shape"),
    [
        (np.zeros((6, 8), dtype=bool), {"kind": "mask", "count": 1, "height": 6, "width": 8}, 0, (6, 8)),
        (np.linspace(0, 1, 48, dtype=np.float32).reshape(6, 8), {"kind": "gray", "count": 1, "height": 6, "width": 8}, 0, (6, 8)),
        (np.linspace(0, 1, 48, dtype=np.float32).reshape(6, 8, 1), {"kind": "gray", "count": 1, "height": 6, "width": 8}, 0, (6, 8)),
        (np.zeros((6, 8, 3), dtype=np.uint8), {"kind": "rgb", "count": 1, "height": 6, "width": 8}, 0, (6, 8, 3)),
        (np.zeros((6, 8, 4), dtype=np.uint8), {"kind": "rgb", "count": 1, "height": 6, "width": 8}, 0, (6, 8, 4)),
        (np.arange(3 * 6 * 8, dtype=np.float64).reshape(3, 6, 8), {"kind": "stack", "count": 3, "height": 6, "width": 8}, 2, (6, 8)),
        (np.zeros((3, 6, 8), dtype=bool), {"kind": "mask_stack", "count": 3, "height": 6, "width": 8}, 1, (6, 8)),
        (np.zeros((6, 8, 1), dtype=bool), {"kind": "mask", "count": 1, "height": 6, "width": 8}, 0, (6, 8)),
        (np.zeros((6, 8, 1), dtype=np.uint8), {"kind": "gray", "count": 1, "height": 6, "width": 8}, 0, (6, 8)),
        (np.zeros((6, 8, 3), dtype=np.int16), {"kind": "stack", "count": 6, "height": 8, "width": 3}, 5, (8, 3)),
    ],
)
def test_image_layouts(tmp_path: Path, arr: np.ndarray, layout: dict[str, Any], index: int, shape: tuple[int, ...]) -> None:
    path = tmp_path / "a.npy"
    np.save(path, arr)
    assert describe(path)["array"]["image"] == layout
    assert png(path, index=index).shape == shape


def test_stack_plane_values(tmp_path: Path) -> None:
    arr = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    path = tmp_path / "s.npy"
    np.save(path, arr)
    plane = png(path, index=1).astype(np.int64)
    expected = np.clip((arr[1] - arr[1].min()) * (255.0 / (arr[1].max() - arr[1].min())), 0, 255).astype(np.uint8)
    assert np.array_equal(plane, expected)


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.array([[-1e308, 1e308], [0.0, 1.0]]), [[0, 255], [127, 127]]),  # hi - lo overflows float64
        (np.array([[np.finfo(np.float64).min, np.finfo(np.float64).max]]), [[0, 255]]),
        (np.array([[np.nan, -np.inf], [2.0, 4.0]]), [[0, 0], [0, 255]]),
        (np.array([[np.iinfo(np.int64).min, np.iinfo(np.int64).max]], dtype=np.int64), [[0, 255]]),
        (np.full((2, 2), 7.0), [[0, 0], [0, 0]]),
        # For a subnormal span, 255 / span overflowed to inf and drew the middle value white.
        (np.array([[1e-310, 2e-310, 3e-310]]), [[0, 127, 255]]),
        (np.array([[0.0, 5e-324, 1e-323]]), [[0, 127, 255]]),
    ],
)
def test_gray_images_scale_every_finite_range(tmp_path: Path, arr: np.ndarray, expected: list[list[int]], capfd: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "a.npy"
    np.save(path, arr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert png(path).tolist() == expected
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("arr", [np.zeros(5), np.zeros((2, 2, 2, 2)), np.zeros((0, 4)), np.zeros((3, 0, 4), dtype=bool)])
def test_arrays_that_are_not_images(tmp_path: Path, arr: np.ndarray) -> None:
    path = tmp_path / "a.npy"
    np.save(path, arr)
    assert describe(path)["array"]["image"] is None
    with pytest.raises(ArrayError, match="only 2-D arrays"):
        png(path)


# ---------------------------------------------------------------------------
# parsing edge cases (moved from test_trace_api.py)
# ---------------------------------------------------------------------------


def test_packed_mask_members(tmp_path: Path) -> None:
    mask = np.zeros((3, 5, 7), dtype=bool)
    mask[1, 2, 3] = True
    good = tmp_path / "good.npz"
    np.savez(good, m__packed=np.packbits(mask.ravel()), m__shape=np.array(mask.shape, dtype=np.int64), s=np.ones(2))
    listing = describe(good)
    assert set(listing) == {"m", "s"} and listing["m"]["packed"] is True
    assert np.array_equal(full_array(good, "m"), mask)
    packbits = tmp_path / "packbits.npz"  # the recorder's naming
    np.savez(packbits, a0__packbits=np.packbits(mask.ravel()), a0__shape=np.array(mask.shape))
    assert set(describe(packbits)) == {"a0"}
    assert np.array_equal(full_array(packbits, "a0__packbits"), mask)
    bad = tmp_path / "bad.npz"  # shape does not match the packed size: both members shown raw
    np.savez(bad, m__packed=np.packbits(mask.ravel()), m__shape=np.array([100, 100], dtype=np.int64))
    assert set(describe(bad)) == {"m__packed", "m__shape"}
    assert describe(bad, "m") is None


@pytest.mark.parametrize(
    ("arr", "check"),
    [
        (np.zeros((0, 4), dtype=np.float32), lambda a: a["preview"] == [] and not a["truncated"]),
        (np.array([[np.nan, np.inf], [-np.inf, 1e308]]), lambda a: a["preview"] == [["nan", "inf"], ["-inf", 1e308]]),
        (np.full((5, 5), np.nan, dtype=np.float32), lambda a: a["preview"][0][0] == "nan" and a["stats"] is None),
        (np.arange(6, dtype=">f4"), lambda a: a["preview"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (np.float64(2.5), lambda a: a["preview"] == 2.5 and a["shape"] == [] and a["preview_shape"] == []),
        (np.array(True), lambda a: a["preview"] is True),
    ],
)
def test_edge_cases_never_warn(tmp_path: Path, arr: np.ndarray, check: Any, capfd: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "a.npy"
    np.save(path, arr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        info = describe(path)["array"]
        if info["image"] is not None:
            png(path)
    assert check(info), info
    assert capfd.readouterr() == ("", "")


def test_truncated_npy_and_member_sizes_that_disagree(tmp_path: Path) -> None:
    data = npy_bytes(np.arange(100, dtype=np.int64))
    short = tmp_path / "short.npy"
    short.write_bytes(data[:-8])
    with pytest.raises(ArrayError, match="truncated"):
        describe(short)
    # an npz member whose declared size is not header + data
    path = tmp_path / "odd.npz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.npy", data + b"extra bytes")
    with pytest.raises(ArrayError, match="size does not match"):
        describe(path)


def test_arrays_larger_than_the_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.npy"
    np.save(path, np.zeros((64, 48), dtype=np.float32))
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_FILE_BYTES", 100)
    with pytest.raises(ArrayError, match="file is larger"):
        describe(path)
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_FILE_BYTES", 64 << 20)
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_DECOMPRESSED_BYTES", 50)
    with pytest.raises(ArrayError, match="larger than the viewer limit"):
        describe(path)
