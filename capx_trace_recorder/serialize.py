"""Argument and return summaries, long texts and the units table.

Implements docs/trace-format.md section 5.3. ``summarize()`` runs on the calling
thread (arguments before the call, the return value before the wrapper
returns). It never evaluates or consumes anything the program did not: no
``repr()``, no iteration of generators, no attribute access on arbitrary
objects. Arrays are copied at call time within the array budget; the writer
thread later saves the copies (``npy``) and computes statistics over finite
values (``array_stats``), so nothing here can emit a numpy warning.

Summaries are plain JSON-able dicts. Values that need a file are returned as
``Pending*`` objects next to the summary; each one holds a reference to the
summary node it fills in when the writer has written the file.

Related: ``capx_trace_recorder/store.py`` (``BuildContext`` writes pending files),
``capx_trace_recorder/hooks/api.py`` and ``capx_trace_recorder/hooks/services.py`` (callers).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
from typing import Any

from capx_trace_recorder.config import INLINE_TEXT_LIMIT, MASK_VALUE, SECRET_KEYS
from capx_trace_recorder.timeout import is_trial_alarm

MAX_DEPTH = 4
MAX_ITEMS = 50
MAX_NODES = 2000
INLINE_ARRAY_SIZE = 64
# Arrays beyond the array budget keep only shape, dtype and a hash computed on
# the calling thread. Hashing is bounded so that it never costs more than a few
# milliseconds there: an array of at most HASH_FULL_BYTES is hashed whole
# (``sha256``); a larger one only over its first HASH_FULL_BYTES bytes in C order
# (``sha256_prefix`` with ``sha256_prefix_bytes``); and one summary hashes at most
# HASH_BUDGET_BYTES in total, after which ``sha256_skipped: "hash_budget"``.
HASH_FULL_BYTES = 1024 * 1024
HASH_BUDGET_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pending files (filled in by the writer)
# ---------------------------------------------------------------------------


class PendingArray:
    """A numpy array copied at call time, saved later as ``npy``."""

    __slots__ = ("array", "node", "label")

    def __init__(self, array: Any, node: dict[str, Any], label: str) -> None:
        self.array = array
        self.node = node
        self.label = label

    @property
    def nbytes(self) -> int:
        return int(getattr(self.array, "nbytes", 0) or 0)


class PendingImage:
    """An image (PIL copy or uint8 array) saved later as PNG."""

    __slots__ = ("image", "node", "label")

    def __init__(self, image: Any, node: dict[str, Any], label: str) -> None:
        self.image = image
        self.node = node
        self.label = label

    @property
    def nbytes(self) -> int:
        nbytes = getattr(self.image, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        size = getattr(self.image, "size", (0, 0))
        try:
            return int(size[0]) * int(size[1]) * 4
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            return 0


class PendingText:
    """A long string saved later as ``txt.gz``."""

    __slots__ = ("text", "node", "label")

    def __init__(self, text: str, node: dict[str, Any], label: str) -> None:
        self.text = text
        self.node = node
        self.label = label

    @property
    def nbytes(self) -> int:
        return len(self.text)


Pending = PendingArray | PendingImage | PendingText


class ArrayBudget:
    """Bytes of array copies allowed for one call, charged to the episode."""

    __slots__ = ("call_left", "episode", "exceeded")

    def __init__(self, call_limit: int, episode: Any | None) -> None:
        self.call_left = int(call_limit)
        self.episode = episode  # object with ``array_bytes_left`` (EpisodeRecorder) or None
        self.exceeded = 0

    def take(self, nbytes: int) -> bool:
        if nbytes > self.call_left:
            self.exceeded += 1
            return False
        if self.episode is not None:
            if not self.episode.take_array_budget(nbytes):
                self.exceeded += 1
                return False
        self.call_left -= nbytes
        return True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def type_name(value: Any) -> str:
    """``module.QualName`` of the value's type, without touching the value."""
    tp = type(value)
    module = getattr(tp, "__module__", "") or ""
    qual = getattr(tp, "__qualname__", None) or getattr(tp, "__name__", "?")
    if module in ("builtins", ""):
        return str(qual)
    return f"{module}.{qual}"


def float_value(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def is_secret_name(name: Any) -> bool:
    return isinstance(name, str) and name.lower() in SECRET_KEYS


def text_value(text: str, pending: list[Pending], label: str = "text") -> str | dict[str, Any]:
    """Inline up to 4000 characters, else a preview plus a pending ``txt.gz``."""
    if len(text) <= INLINE_TEXT_LIMIT:
        return text
    node: dict[str, Any] = {
        "truncated": True,
        "preview": text[:INLINE_TEXT_LIMIT],
        "length": len(text),
    }
    pending.append(PendingText(text, node, label))
    return node


def _np():
    return sys.modules.get("numpy")


def _pil_image_class():
    mod = sys.modules.get("PIL.Image")
    return getattr(mod, "Image", None) if mod is not None else None


def _torch_tensor_class():
    mod = sys.modules.get("torch")
    return getattr(mod, "Tensor", None) if mod is not None else None


def json_safe(value: Any, _depth: int = 0) -> Any:
    """Convert parsed JSON / numpy scalars into strict JSON values."""
    if _depth > 64:
        return "<too deep>"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float_value(float(value))
    np = _np()
    if np is not None and isinstance(value, np.generic):
        try:
            return json_safe(value.item(), _depth + 1)
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            return type_name(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in dict.items(value)}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth + 1) for v in value]
    return type_name(value)


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


class _Walker:
    __slots__ = ("pending", "budget", "nodes", "label", "image_args", "hash_left")

    def __init__(self, pending: list[Pending], budget: ArrayBudget | None, label: str,
                 image_args: bool) -> None:
        self.pending = pending
        self.budget = budget
        self.nodes = 0
        self.label = label
        self.image_args = image_args
        self.hash_left = HASH_BUDGET_BYTES

    def walk(self, value: Any, depth: int) -> Any:
        self.nodes += 1
        if value is None or isinstance(value, bool):
            return bool(value) if value is not None else None
        if type(value) is int:
            return value
        np = _np()
        if np is not None and isinstance(value, np.ndarray):
            return self.array(value)
        if np is not None and isinstance(value, np.generic):
            try:
                item = value.item()
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                return {"type": type_name(value)}
            if isinstance(item, float):
                return float_value(item)
            if isinstance(item, (bool, int, str)) or item is None:
                return item
            return {"type": type_name(value)}
        # Base-class methods are called explicitly so that an override of
        # __float__ / __int__ / __str__ in a subclass never runs.
        if isinstance(value, float):
            return float_value(float.__float__(value))
        if isinstance(value, int):  # int subclasses (IntEnum etc.)
            return int.__int__(value)
        if isinstance(value, str):
            return text_value(str.__str__(value), self.pending, "text")
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                length = len(value)
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                length = None
            return {"type": type_name(value), "length": length}
        pil = _pil_image_class()
        if pil is not None and isinstance(value, pil):
            return self.pil_image(value)
        tensor = _torch_tensor_class()
        if tensor is not None and isinstance(value, tensor):
            return self.torch_tensor(value)
        if depth >= MAX_DEPTH:
            return {"type": type_name(value), "truncated": "depth"}
        tp = type(value)
        if tp is dict or (isinstance(value, dict) and tp.__module__ == "collections"):
            return self.mapping(value, depth)
        if tp is list or tp is tuple:
            return self.sequence(value, depth, tp.__name__)
        if isinstance(value, tuple) and tuple in tp.__bases__ and hasattr(tp, "_fields"):
            return self.namedtuple(value, depth)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self.dataclass(value, depth)
        return {"type": type_name(value)}

    # -- containers ---------------------------------------------------------
    def mapping(self, value: dict, depth: int) -> dict[str, Any]:
        items: dict[str, Any] = {}
        length = dict.__len__(value)
        shown = 0
        for i, (key, item) in enumerate(dict.items(value)):
            if shown >= MAX_ITEMS or self.nodes >= MAX_NODES:
                break
            if isinstance(key, str):
                key_s = key
            elif key is None or isinstance(key, (bool, int, float)):
                key_s = str(key)
            else:
                key_s = f"<{type_name(key)}#{i}>"
            if key_s in items:
                key_s = f"{key_s}#{i}"
            items[key_s] = MASK_VALUE if is_secret_name(key) else self.walk(item, depth + 1)
            shown += 1
        out: dict[str, Any] = {"type": type_name(value), "length": length, "items": items}
        if shown < length:
            out["truncated"] = length - shown
        return out

    def sequence(self, value: list | tuple, depth: int, kind: str) -> dict[str, Any]:
        length = len(value)
        items = []
        for item in (list.__iter__(value) if kind == "list" else tuple.__iter__(value)):
            if len(items) >= MAX_ITEMS or self.nodes >= MAX_NODES:
                break
            items.append(self.walk(item, depth + 1))
        out: dict[str, Any] = {"type": kind, "length": length, "items": items}
        if len(items) < length:
            out["truncated"] = length - len(items)
        return out

    def namedtuple(self, value: tuple, depth: int) -> dict[str, Any]:
        fields = getattr(type(value), "_fields", ())
        items: dict[str, Any] = {}
        for name, item in zip(fields, tuple.__iter__(value)):
            if len(items) >= MAX_ITEMS or self.nodes >= MAX_NODES:
                break
            items[str(name)] = MASK_VALUE if is_secret_name(name) else self.walk(item, depth + 1)
        return {"type": type_name(value), "fields": items}

    def dataclass(self, value: Any, depth: int) -> dict[str, Any]:
        items: dict[str, Any] = {}
        try:
            fields = dataclasses.fields(value)
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            return {"type": type_name(value)}
        for f in fields:
            if len(items) >= MAX_ITEMS or self.nodes >= MAX_NODES:
                break
            if is_secret_name(f.name):
                items[f.name] = MASK_VALUE
                continue
            try:
                item = object.__getattribute__(value, f.name)
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                items[f.name] = {"type": "<unavailable>"}
                continue
            items[f.name] = self.walk(item, depth + 1)
        return {"type": type_name(value), "dataclass": True, "fields": items}

    # -- arrays and images --------------------------------------------------
    def array(self, arr: Any) -> dict[str, Any]:
        np = _np()
        node: dict[str, Any] = {
            "type": "ndarray",
            "shape": [int(s) for s in arr.shape],
            "dtype": str(arr.dtype),
        }
        kind = arr.dtype.kind
        if kind in ("O", "V") or arr.dtype.fields is not None:
            node["values_omitted"] = "object_or_structured_dtype"
            return node
        size = int(arr.size)
        if size <= INLINE_ARRAY_SIZE:
            try:
                node["values"] = _inline_values(arr.tolist())
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                node["values_omitted"] = "tolist_failed"
            return node
        nbytes = int(arr.nbytes)
        if self.image_args and _looks_like_image(arr):
            # an image a service received: stored as PNG (lossless), not npy
            if self.budget is None or self.budget.take(nbytes):
                copy = np.array(arr, copy=True, order="C")
                node["image"] = {"label": self.label}
                self.pending.append(PendingImage(copy, node["image"], self.label))
                return node
        if self.budget is None or self.budget.take(nbytes):
            copy = np.array(arr, copy=True, order="C")
            self.pending.append(PendingArray(copy, node, self.label))
            return node
        node["not_saved"] = "array_budget"
        self.bounded_hash(arr, nbytes, node)
        return node

    def bounded_hash(self, arr: Any, nbytes: int, node: dict[str, Any]) -> None:
        """A hash of an array that is not saved, within the per-summary hash budget."""
        np = _np()
        node["sha256"] = None
        n_bytes = min(nbytes, HASH_FULL_BYTES)
        if n_bytes > self.hash_left:
            node["sha256_skipped"] = "hash_budget"
            return
        self.hash_left -= n_bytes
        try:
            if nbytes <= HASH_FULL_BYTES:
                data = np.ascontiguousarray(arr).tobytes()
                node["sha256"] = hashlib.sha256(data).hexdigest()
            else:
                n_items = max(1, HASH_FULL_BYTES // max(1, int(arr.itemsize)))
                data = np.ascontiguousarray(arr.flat[:n_items]).tobytes()
                node["sha256_prefix"] = hashlib.sha256(data).hexdigest()
                node["sha256_prefix_bytes"] = len(data)
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            node["sha256_skipped"] = "hash_failed"

    def pil_image(self, img: Any) -> dict[str, Any]:
        node: dict[str, Any] = {"type": "PIL.Image"}
        try:
            size = img.size
            node["size"] = [int(size[0]), int(size[1])]
            node["mode"] = str(img.mode)
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            return node
        core = getattr(img, "_im", None)
        loaded = core is not None and type(core).__name__ != "DeferredError"
        if not loaded:
            node["image_not_loaded"] = True
            return node
        nbytes = node["size"][0] * node["size"][1] * max(1, len(node["mode"]))
        if self.budget is None or self.budget.take(nbytes):
            try:
                node["image"] = {"label": self.label}
                self.pending.append(PendingImage(img.copy(), node["image"], self.label))
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                node.pop("image", None)
                node["image_not_saved"] = "copy_failed"
        else:
            node["image_not_saved"] = "array_budget"
        return node

    def torch_tensor(self, tensor: Any) -> dict[str, Any]:
        node: dict[str, Any] = {"type": "torch.Tensor"}
        try:
            node["shape"] = [int(s) for s in tensor.shape]
            node["dtype"] = str(tensor.dtype)
            node["device"] = str(tensor.device)
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            pass
        return node


def _looks_like_image(arr: Any) -> bool:
    try:
        return (
            str(arr.dtype) == "uint8"
            and arr.ndim == 3
            and arr.shape[2] in (3, 4)
            and arr.shape[0] >= 8
            and arr.shape[1] >= 8
        )
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return False


def _inline_values(values: Any) -> Any:
    if isinstance(values, list):
        return [_inline_values(v) for v in values]
    if isinstance(values, float):
        return float_value(values)
    if isinstance(values, complex):
        return [float_value(values.real), float_value(values.imag)]
    if isinstance(values, bytes):
        return {"type": "bytes", "length": len(values)}
    return values


def summarize(value: Any, pending: list[Pending], budget: ArrayBudget | None = None, *,
              label: str = "value", image_args: bool = False) -> Any:
    """Summarise one value (section 5.3). Appends files to write to ``pending``."""
    return _Walker(pending, budget, label, image_args).walk(value, 0)


def summarize_arguments(bound: dict[str, Any], pending: list[Pending],
                        budget: ArrayBudget | None = None, *, label: str = "value",
                        image_args: bool = False) -> dict[str, Any]:
    """Summarise named arguments; secret parameter names are masked."""
    walker = _Walker(pending, budget, label, image_args)
    out: dict[str, Any] = {}
    for name, value in bound.items():
        if is_secret_name(name):
            out[name] = MASK_VALUE
        else:
            out[name] = walker.walk(value, 0)
        if walker.nodes >= MAX_NODES:
            out["<truncated>"] = "node_limit"
            break
    return out


def pending_bytes(pending: list[Pending]) -> int:
    total = 0
    seen: set[int] = set()
    for item in pending:
        obj = getattr(item, "array", None)
        if obj is None:
            obj = getattr(item, "image", None)
        if obj is None:
            obj = getattr(item, "text", None)
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        total += item.nbytes
    return total


def pending_binary_bytes(pending: list[Pending]) -> int:
    """Bytes of the array and image copies in ``pending`` (what a queue_full drop releases)."""
    return pending_bytes([p for p in pending if not isinstance(p, PendingText)])


def split_pending(pending: list[Pending]) -> tuple[list[PendingText], list[dict[str, Any]]]:
    """``(texts to keep, nodes of dropped copies)`` for a queue_full drop.

    Long texts are text, not binary data, so they are written even when the
    writer queue is full; array and image copies are released at once and their
    summary nodes say ``written: false, reason: queue_full``.
    """
    kept = [p for p in pending if isinstance(p, PendingText)]
    dropped = [p.node for p in pending if not isinstance(p, PendingText)]
    return kept, dropped


_SAME_AFTER: Any = object()


def pending_builders(pending: list[Pending], after: Any = None, *, drop_after: Any = _SAME_AFTER,
                     extra_bytes: int = 0) -> tuple[Any, Any, int]:
    """``(builder, drop_builder, nbytes)`` that write ``pending`` on the writer.

    ``drop_builder`` holds no array or image copy (only the texts to keep and the
    nodes to mark), so ``TraceStore.submit`` frees the copies when it drops the
    item for the queue budget. ``after(ctx)`` runs in the builder, and in the
    drop builder too unless ``drop_after`` is given; whatever the drop builder
    runs must hold no large data. ``extra_bytes`` are further copies ``after``
    holds (a state snapshot, see ``hooks/api.state_parts``); they are counted in
    ``nbytes`` and ``drop_after`` must not hold them.
    """
    kept, dropped = split_pending(pending)
    on_drop = after if drop_after is _SAME_AFTER else drop_after

    def builder(ctx: Any) -> None:
        ctx.fill_pending(pending)
        if after is not None:
            after(ctx)

    def drop_builder(ctx: Any) -> None:
        ctx.fill_pending(kept)
        ctx.fill_dropped(dropped)
        if on_drop is not None:
            on_drop(ctx)

    return builder, drop_builder, pending_binary_bytes(pending) + int(extra_bytes or 0)


# ---------------------------------------------------------------------------
# Writer-side statistics (never on the trial thread)
# ---------------------------------------------------------------------------


def array_stats(arr: Any) -> dict[str, Any] | None:
    """min / max / mean over finite values for numeric and bool dtypes.

    Returns ``None`` for other dtypes. The values are ``None`` when the array
    has no finite element. Floating-point errors are silenced with
    ``np.errstate`` (thread-local in numpy); empty reductions are never run.
    """
    np = _np()
    if np is None:
        return None
    kind = arr.dtype.kind
    if kind not in ("b", "i", "u", "f"):
        return None
    with np.errstate(all="ignore"):
        flat = arr.reshape(-1)
        if kind == "f":
            finite = flat[np.isfinite(flat)]
            n_nonfinite = int(flat.size - finite.size)
        else:
            finite = flat
            n_nonfinite = 0
        if finite.size == 0:
            return {"min": None, "max": None, "mean": None, "n_finite": 0, "n_nonfinite": n_nonfinite}
        if kind == "b":
            mn, mx = bool(finite.min()), bool(finite.max())
            mean = float(finite.mean(dtype=np.float64))
        else:
            mn = finite.min().item()
            mx = finite.max().item()
            mean = float(finite.mean(dtype=np.float64))
        return {
            "min": float_value(mn) if isinstance(mn, float) else mn,
            "max": float_value(mx) if isinstance(mx, float) else mx,
            "mean": float_value(mean),
            "n_finite": int(finite.size),
            "n_nonfinite": n_nonfinite,
        }


# ---------------------------------------------------------------------------
# Units and coordinate frames (only where the API code states them)
# ---------------------------------------------------------------------------

_M = {"unit": "m"}
_QUAT = {"unit": "unit quaternion, order (w, x, y, z)"}

# Classes whose docstrings state "position: (3,) XYZ in meters" and
# "quaternion_wxyz: (4,) WXYZ unit quaternion" without naming a frame.
_FRANKA_POSE_CLASSES = (
    "FrankaControlApi",
    "FrankaControlPrivilegedApi",
    "FrankaControlNutAssemblyPrivilegedApi",
    "FrankaControlNutAssemblyVisualApi",
    "FrankaControlSpillWipeApi",
    "FrankaControlSpillWipePrivilegedApi",
)
_BBOX_WORLD_CLASSES = (
    "FrankaControlPrivilegedApi",
    "FrankaControlSpillWipePrivilegedApi",
)
_SKILL_LIBRARY_CLASSES = (
    "FrankaControlApiReducedSkillLibrary",
)

UNITS: dict[tuple[str, str], dict[str, Any]] = {}
for _cls in _FRANKA_POSE_CLASSES:
    UNITS[(_cls, "get_object_pose")] = {
        "returns": {
            "position": dict(_M, frame=None),
            "quaternion_wxyz": dict(_QUAT, frame=None),
            "bbox_extent": dict(_M, frame="world" if _cls in _BBOX_WORLD_CLASSES else None),
        },
    }
    UNITS[(_cls, "sample_grasp_pose")] = {
        "returns": {"position": dict(_M, frame=None), "quaternion_wxyz": dict(_QUAT, frame=None)},
    }
    UNITS[(_cls, "goto_pose")] = {
        "args": {
            "position": dict(_M, frame=None),
            "quaternion_wxyz": dict(_QUAT, frame=None),
            "z_approach": dict(_M, frame=None),
        },
    }
UNITS[("FrankaHandoverApi", "get_object_pose")] = {
    "returns": {
        "position": dict(_M, frame="robot0 base"),
        "quaternion_wxyz": dict(_QUAT, frame="robot0 base"),
        "bbox_extent": dict(_M, frame=None),
    },
}
UNITS[("FrankaHandoverApi", "goto_pose_arm0")] = {
    "args": {
        "position": dict(_M, frame="robot0 base"),
        "quaternion_wxyz": dict(_QUAT, frame="robot0 base"),
        "z_approach": dict(_M, frame=None),
    },
}
UNITS[("FrankaHandoverApi", "goto_pose_arm1")] = {
    "args": {
        "position": dict(_M, frame="robot0 base"),
        "quaternion_wxyz": dict(_QUAT, frame="robot0 base"),
        "z_approach": dict(_M, frame=None),
    },
}
for _cls in _SKILL_LIBRARY_CLASSES:
    UNITS[(_cls, "depth_to_point_cloud")] = {
        "args": {"depth_img": dict(_M, frame=None)},
        "returns": {"points": dict(_M, frame="camera")},
    }
    UNITS[(_cls, "mask_to_world_points")] = {"returns": {"points": dict(_M, frame="world")}}
    UNITS[(_cls, "pixel_to_world_point")] = {
        "args": {"u": {"unit": "pixel (column)", "frame": "image"},
                 "v": {"unit": "pixel (row)", "frame": "image"}},
        "returns": {"point": {"unit": None, "frame": "world"}},
    }
    UNITS[(_cls, "interpolate_segment")] = {"args": {"step": dict(_M, frame=None)}}


def units_for(api_class: str, name: str) -> dict[str, Any] | None:
    """Units and frames stated by the API's own docstring, else ``None``."""
    return UNITS.get((api_class, name))


__all__ = [
    "ArrayBudget",
    "PendingArray",
    "PendingImage",
    "PendingText",
    "summarize",
    "summarize_arguments",
    "pending_bytes",
    "pending_binary_bytes",
    "split_pending",
    "pending_builders",
    "text_value",
    "json_safe",
    "float_value",
    "type_name",
    "array_stats",
    "units_for",
    "is_secret_name",
]
