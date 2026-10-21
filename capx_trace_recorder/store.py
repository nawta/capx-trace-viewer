"""Trace store: one daemon writer thread per process (docs/trace-format.md, section 4).

Producer side (any thread): ``EpisodeRecorder.emit()`` assigns ``seq``,
``event_id`` and clocks inside a per-episode critical section and puts the item
into a ``queue.SimpleQueue`` (never blocks). Everything a builder needs is an
immutable snapshot taken by the caller (request bodies are ``str``/``bytes``,
arrays are copies taken at call time).

Writer side (only the writer thread): items are processed in FIFO order. For
each item the builder writes its files first (tmp name, then ``os.replace``),
then the event line is appended with a single ``os.write`` call; the byte count
is checked and a short write is truncated back. The writer checks seq
continuity, keeps the heartbeat file fresh (every 10 s), enforces the disk
floor, computes ``trace_complete`` and ``gaps`` at ``episode_end``, and rewrites
``episode.json`` (tmp + rename) on every change of status. It catches
``BaseException`` around each item and around its loop, so
``threading.excepthook`` never prints. The recorder never prints, warns or logs.

Queue budget (``CAPX_TRACE_MAX_QUEUE_MB``): an item that carries a copy of
binary data (frames, arrays, images) says how many bytes it holds (``nbytes``)
and gives a light ``drop_builder`` that holds none of it. When the queued bytes
would go over the budget, ``submit`` swaps the builder for the drop builder, so
the copies are freed at once, and the writer counts one ``queue_full`` gap per
dropped item. Model request and response bodies are text: they are kept
(``keep_payload``) and only the images in them are not written; such an item
counts one ``queue_full`` gap only when an image was actually left out.

Writer state per sink lives on the sink itself (``_writer_state``) and is cut
down to a small record once ``episode_end`` is written, so an ended episode can
still take late events (``late: true``) without the store keeping its tables.
Only open logs are visited by the heartbeat. The disk-low stop writes its final
marker to the open logs, to the log of the item that noticed it and to the
run's unscoped log (``_stop_all_for_disk``). Lines the run's unscoped log could
not take are recorded next to it in ``unscoped.<pid>.status.json``.

Exception handlers: code that can run on the trial thread re-raises
``TimeoutError`` (the trial alarm, section 2); handlers marked ``timeout-ok:
writer thread`` run only on the writer thread, which never receives the alarm
(``tests/recorder/test_timeout_rule.py`` checks every handler).

Related: ``capx_trace_recorder/context.py`` (episode registry), ``capx_trace_recorder/payload.py``,
``capx_trace_recorder/segments.py``, ``capx_trace_recorder/serialize.py`` (pending files).
The read-only reader is ``capx_trace_viewer.reader``.
"""

from __future__ import annotations

import atexit
import gzip
import hashlib
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from capx_trace_recorder import SCHEMA_VERSION, TRACER_VERSION
from capx_trace_recorder.config import (
    ATEXIT_WAIT_S,
    HEARTBEAT_INTERVAL_S,
    MAX_EVENT_LINE_BYTES,
    TraceConfig,
)
from capx_trace_recorder.ids import hostname, new_event_id, utc_iso

PID = os.getpid()
DISK_CHECK_INTERVAL_S = 5.0
# ``drop_builder`` default: on a queue_full drop keep the builder (it runs with
# ``ctx.binary_dropped`` set and writes placeholders for the binary files)
KEEP_BUILDER: Any = object()

# Indirections so tests can simulate short writes and low disk space.
_os_write = os.write
_disk_free = lambda path: shutil.disk_usage(path).free  # noqa: E731


RUN_JSON_ATTEMPTS = 3


def _now() -> float:
    return time.time()


def _write_all(fd: int, data: bytes) -> None:
    """Write all of ``data`` (a short write is continued); raises OSError when a
    write makes no progress."""
    view = memoryview(data)
    while view:
        n = _os_write(fd, view)
        if n <= 0:
            raise OSError("short write: no progress")
        view = view[n:]


def _unlink_quietly(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:  # timeout-ok: writer thread
        pass


# ---------------------------------------------------------------------------
# Queue items
# ---------------------------------------------------------------------------


class Item:
    """One unit of work for the writer.

    ``nbytes`` is the size of the binary copies the builder holds. ``drop_builder``
    replaces ``builder`` when the item goes over the queue budget (``KEEP_BUILDER``:
    keep the builder); ``keep_payload`` items (model bodies) are never swapped.
    """

    __slots__ = ("kind", "sink", "event", "builder", "nbytes", "binary_dropped", "done",
                 "path", "line", "drop_builder", "keep_payload")

    def __init__(self, kind: str, sink: Any = None, event: dict[str, Any] | None = None,
                 builder: Callable[["BuildContext"], None] | None = None, nbytes: int = 0,
                 drop_builder: Any = KEEP_BUILDER, keep_payload: bool = False) -> None:
        self.kind = kind
        self.sink = sink
        self.event = event
        self.builder = builder
        self.nbytes = int(nbytes or 0)
        self.binary_dropped = False
        self.done: threading.Event | None = None
        self.path: Path | None = None
        self.line: bytes | None = None
        self.drop_builder = drop_builder
        self.keep_payload = keep_payload


# ---------------------------------------------------------------------------
# Producer-side sinks
# ---------------------------------------------------------------------------


class EventSink:
    """Base of ``EpisodeRecorder`` and ``UnscopedLog``: seq + queueing."""

    is_unscoped = False

    def __init__(self, store: "TraceStore", run_id: str, run_dir: Path, base_dir: Path) -> None:
        self.store = store
        self.run_id = run_id
        self.run_dir = run_dir
        self.base_dir = base_dir
        self.episode_id: str | None = None
        self._lock = threading.RLock()
        self._seq = 0
        self.dead = False
        self.ended = False
        self.sim_epoch = 0
        self._writer_state: "WriterEpisode | None" = None  # set and used by the writer only

    # sim clock (overridden by EpisodeRecorder)
    def clock(self) -> tuple[int | None, float | None]:
        return None, None

    def emit(self, event_type: str, *, status: str = "info", data: dict[str, Any] | None = None,
             parent: str | None = None, refs: dict[str, Any] | None = None,
             builder: Callable[["BuildContext"], None] | None = None,
             event_id: str | None = None, nbytes: int = 0,
             sim: tuple[int | None, float | None] | None = None,
             writer_generated: bool = False, drop_builder: Any = KEEP_BUILDER,
             keep_payload: bool = False, recorder_owned: bool = False) -> str:
        """Queue one event; returns its ``event_id``. Never blocks; raises only
        ``TimeoutError`` (the trial alarm) and non-``Exception`` exceptions.

        ``writer_generated`` marks the writer's own diagnostics: they are never
        ``late`` even when they follow ``episode_end``. ``recorder_owned`` permits a
        recorder-scheduled event such as ``trial_videos`` after the end without marking
        it late. ``nbytes``,
        ``drop_builder`` and ``keep_payload``: see ``Item`` and ``TraceStore.submit``.
        """
        event_id = event_id or new_event_id()
        if self.dead:
            return event_id
        try:
            sim_step, sim_time = self.clock() if sim is None else sim
        except TimeoutError:  # the trial alarm (section 2)
            raise
        except Exception:  # noqa: BLE001
            sim_step, sim_time = None, None
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "event_id": event_id,
            "parent_event_id": parent,
            "seq": 0,
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "sim_step": sim_step,
            "sim_time": sim_time,
            "sim_epoch": self.sim_epoch,
            "event_type": event_type,
            "status": status,
            "late": False,
            "refs": refs or {},
            "artifacts": [],
            "data": data if data is not None else {},
        }
        item = Item("event", self, event, builder, nbytes, drop_builder, keep_payload)
        with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            if self.ended and event_type != "episode_end" and not (writer_generated or recorder_owned):
                event["late"] = True
            if not self.store.submit(item):
                self.dead = True
        return event_id

    def next_seq_for_writer(self) -> int:
        """Seq for a line the writer appends directly (disk-low marker)."""
        with self._lock:
            self._seq += 1
            return self._seq

    def internal(self, builder: Callable[["BuildContext"], None], nbytes: int = 0,
                 drop_builder: Any = KEEP_BUILDER) -> None:
        """Run ``builder`` on the writer with this sink's state (no event line)."""
        if self.dead:
            return
        if not self.store.submit(Item("internal", self, None, builder, nbytes, drop_builder)):
            self.dead = True

    def note_gap(self, code: str) -> None:
        """Count one gap of ``code`` in this log (no event line)."""

        def builder(ctx: "BuildContext", code: str = code) -> None:
            ctx.wep.add_gap(code, 1, None)

        self.internal(builder)


class UnscopedLog(EventSink):
    """``<run>/unscoped.<pid>.jsonl``: events that belong to no episode."""

    is_unscoped = True

    def __init__(self, store: "TraceStore", run_id: str, run_dir: Path) -> None:
        super().__init__(store, run_id, run_dir, run_dir / f"unscoped.{os.getpid()}.files")
        self.events_path = run_dir / f"unscoped.{os.getpid()}.jsonl"


class EpisodeRecorder(EventSink):
    """Producer-side state of one episode (see ``capx_trace_recorder/context.py``)."""

    def __init__(self, store: "TraceStore", run_id: str, episode_id: str, run_dir: Path,
                 source: str) -> None:
        super().__init__(store, run_id, run_dir, run_dir / episode_id)
        self.episode_id = episode_id
        self.events_path = self.base_dir / "events.jsonl"
        self.source = source
        self.env: Any = None  # CodeExecutionEnvBase (or None until bound)
        self.low_env: Any = None
        self.sim_thread: int | None = None
        self._clock_cache: tuple[int | None, float | None] = (None, None)
        self.trial_thread: int | None = None
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.resets_seen = 0
        self.alarm_seen = False
        self.n_code_blocks = 0
        self.n_logical_calls = 0
        self.n_http_attempts = 0
        self.turn_index = 0
        self.overhead_trial = 0.0
        self.overhead_all = 0.0
        self._array_used = 0
        self.array_limit = store.config.array_episode_bytes
        self.config = store.config
        self.multi_turn_prompt: Any = None
        self.config_flags: dict[str, Any] = {}
        self.current_role: Any = None  # role context for pool threads (section 7.3)
        self.codegen_requests = 0
        self.task_text_emitted = False
        self.last_codegen_response: tuple[str | None, Any] = (None, None)
        self.vdm_responses: list[tuple[str, Any]] = []
        self.frame_tap: Any = None
        self.web_session_id: str | None = None
        self.web_session: Any = None  # the web UI Session (its ``env`` binds the env, H13)
        self.pending_user_feedback: list[str] = []
        self.hooks_failed: list[dict[str, Any]] = []
        self.misc: dict[str, Any] = {}
        self.released = False
        from capx_trace_recorder.segments import SegmentRegistry

        self.segments = SegmentRegistry()

    # -- clocks -------------------------------------------------------------
    def clock(self) -> tuple[int | None, float | None]:
        if self.released:
            return None, None  # late events after the release carry no clock (section 4.4)
        low = self.low_env
        if low is None:
            return self._clock_cache
        ident = threading.get_ident()
        if self.sim_thread is None or ident == self.sim_thread:
            from capx_trace_recorder.state_probe import sim_clock

            self._clock_cache = sim_clock(low)
        return self._clock_cache

    # -- budgets and overhead -----------------------------------------------
    def take_array_budget(self, nbytes: int) -> bool:
        with self._lock:
            if self._array_used + nbytes > self.array_limit:
                return False
            self._array_used += nbytes
            return True

    def count(self, name: str, delta: int = 1) -> int:
        """Increment a counter attribute atomically (pool threads share episodes)."""
        with self._lock:
            value = getattr(self, name) + delta
            setattr(self, name, value)
            return value

    def add_overhead(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.overhead_all += seconds
        if threading.get_ident() == self.trial_thread:
            self.overhead_trial += seconds

    def release_references(self) -> None:
        """Drop what the episode no longer needs once ``episode_end`` is written.

        The ended recorder stays reachable (env binding tombstones, contexts of
        threads still running) so late events can be recorded, but it must not
        keep the env, its frame buffer, the segment texts or the web session
        alive. Late events after this point record no clock, frames or state.
        """
        from capx_trace_recorder.segments import SegmentRegistry

        self.released = True  # first: clock() gives no clock from here on (a producer may be reading it)
        self.segments = SegmentRegistry()
        self.vdm_responses = []
        self.env = None
        self.low_env = None
        self.sim_thread = None
        self._clock_cache = (None, None)
        self.frame_tap = None
        self.current_role = None
        self.misc = {}
        self.last_codegen_response = (None, None)
        self.multi_turn_prompt = None
        self.pending_user_feedback = []
        self.web_session = None


# ---------------------------------------------------------------------------
# Writer-side state
# ---------------------------------------------------------------------------


class WriterEpisode:
    """Writer-only state of one sink (episode or unscoped log)."""

    def __init__(self, sink: EventSink) -> None:
        self.sink = sink
        self.base_dir: Path = sink.base_dir
        self.events_path: Path = sink.events_path  # type: ignore[attr-defined]
        self.fd: int | None = None
        self.size = 0
        self.expected_seq = 1
        self.opened = False
        self.start_written = False
        self.end_written = False
        self.end_processed = False
        self.summary: dict[str, Any] = {}
        self.gaps: dict[str, dict[str, Any]] = {}
        self.warned_codes: set[str] = set()
        self.dropped_events = 0
        self.late_events = 0
        self.n_warnings = 0
        self.n_events = 0
        self.last_seq = 0
        self.lost_after_stop = 0
        self.made_dirs: set[str] = set()
        self.written_files: set[str] = set()
        self.failed_files: dict[str, str] = {}
        # raw-pixel sha256 -> every harness image with those pixels, oldest first
        # (two renders of an unchanged scene have the same pixels; see hooks/http.py)
        self.pixel_index: dict[str, list[dict[str, Any]]] = {}
        self.video_reads: list[dict[str, Any]] = []
        self.video_encodes: dict[str, dict[str, Any]] = {}
        self.video_copy_pending = False
        self.links: dict[str, Any] = {}
        self.last_heartbeat = 0.0
        self.closed_for_good = False
        # queue_full drops per kind of item (tells a VDM video source why it is unknown)
        self.dropped_kinds: dict[str, int] = {}
        # unscoped logs: lines that could not be written (kept outside the file too,
        # in unscoped.<pid>.status.json), how many of them a written warning line
        # already reported, and whether such a warning is queued
        self.lines_not_written = 0
        self.lines_reported = 0
        self.failure_warning_queued = False

    @property
    def is_unscoped(self) -> bool:
        return self.sink.is_unscoped

    def add_gap(self, code: str, count: int = 1, first_seq: int | None = None) -> None:
        gap = self.gaps.get(code)
        if gap is None:
            self.gaps[code] = {"code": code, "count": int(count), "first_seq": first_seq}
        else:
            gap["count"] += int(count)
            if gap["first_seq"] is None:
                gap["first_seq"] = first_seq

    def gap_list(self) -> list[dict[str, Any]]:
        return [dict(g) for g in self.gaps.values()]

    def gap_total(self) -> int:
        return sum(int(g["count"]) for g in self.gaps.values())


class BuildContext:
    """What a builder may do on the writer thread."""

    def __init__(self, store: "TraceStore", wep: WriterEpisode, item: Item) -> None:
        self.store = store
        self.wep = wep
        self.item = item
        self.event = item.event
        self.binary_dropped = item.binary_dropped
        self.binary_skipped = 0  # binary files not written because of the queue budget
        self.base_dir = wep.base_dir
        try:
            self._resolved_base_dir = self.base_dir.resolve(strict=False)
        except TimeoutError:
            raise
        except (RuntimeError, ValueError, OSError):
            self._resolved_base_dir = None
        # artifact paths are relative to the episode dir, or to the run dir for
        # unscoped events (prefixed with the unscoped files directory name)
        self.rel_prefix = (wep.base_dir.name + "/") if wep.is_unscoped else ""

    @property
    def data(self) -> dict[str, Any]:
        return self.event["data"] if self.event is not None else {}

    # -- files --------------------------------------------------------------
    def _add_artifact(self, art: dict[str, Any]) -> dict[str, Any]:
        if self.event is not None:
            self.event.setdefault("artifacts", []).append(art)
        return art

    def write_file(self, relpath: str, content: bytes | Callable[[], bytes], *, label: str,
                   kind: str, mime: str, binary: bool, extra: dict[str, Any] | None = None,
                   dedupe: bool = False, add: bool = True) -> dict[str, Any]:
        """Write ``base_dir/relpath`` (tmp + rename); returns the artifact reference."""
        art: dict[str, Any] = {
            "path": self.rel_prefix + relpath,
            "sha256": None,
            "kind": kind,
            "mime": mime,
            "bytes": None,
            "label": label,
            "written": False,
        }
        if extra:
            art.update(extra)
        parts = relpath.split("/") if isinstance(relpath, str) else []
        target = self.base_dir / relpath if parts else self.base_dir
        try:
            safe = (bool(parts) and all(part not in ("", ".", "..") for part in parts)
                    and not Path(relpath).is_absolute()
                    and self._resolved_base_dir is not None
                    and target.resolve(strict=False).is_relative_to(self._resolved_base_dir))
        except TimeoutError:
            raise
        except (RuntimeError, ValueError, OSError):
            safe = False
        if not safe:
            art["reason"] = "unsafe_path"
            self.wep.add_gap("unsafe_path", 1, self.event.get("seq") if self.event else None)
            return self._add_artifact(art) if add else art
        if binary and self.binary_dropped:
            # the gap is counted once per dropped item by TraceStore._handle
            art["reason"] = "queue_full"
            self.binary_skipped += 1
            self.wep.failed_files[str(self.base_dir / relpath)] = "queue_full"
            return self._add_artifact(art) if add else art
        ok, reason = self.store.disk_allows(binary, trigger=self.wep.sink)
        if not ok:
            art["reason"] = reason
            self.wep.add_gap(reason or "disk_low", 1, self.event.get("seq") if self.event else None)
            self.store.warn_once(self.wep, reason or "disk_low",
                                 "files skipped: free disk space below CAPX_TRACE_MIN_FREE_GB")
            return self._add_artifact(art) if add else art
        try:
            data = content() if callable(content) else content
        except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
            art["reason"] = f"encode_failed: {type(exc).__name__}"
            self.wep.add_gap("write_failed", 1, self.event.get("seq") if self.event else None)
            return self._add_artifact(art) if add else art
        art["bytes"] = len(data)
        art["sha256"] = hashlib.sha256(data).hexdigest()
        key = str(target)
        if dedupe and (key in self.wep.written_files or target.exists()):
            self.wep.written_files.add(key)
            art["written"] = True
            return self._add_artifact(art) if add else art
        ok, reason = self.store.atomic_write(target, data, self.wep)
        if ok:
            self.wep.written_files.add(key)
            art["written"] = True
        else:
            art["reason"] = reason
            self.wep.failed_files[key] = reason or "write_failed"
            self.wep.add_gap("write_failed", 1, self.event.get("seq") if self.event else None)
            self.store.warn_once(self.wep, "write_failed", f"file write failed: {reason}")
        return self._add_artifact(art) if add else art

    def write_asset(self, content: bytes, ext: str, *, label: str, kind: str, mime: str,
                    binary: bool = True, extra: dict[str, Any] | None = None,
                    add: bool = True) -> dict[str, Any]:
        """Content-addressed ``assets/<sha[:2]>/<sha>.<ext>``."""
        if binary and self.binary_dropped:
            return self.write_file("assets/unwritten." + ext, b"", label=label, kind=kind,
                                   mime=mime, binary=True, extra=extra, add=add)
        sha = hashlib.sha256(content).hexdigest()
        rel = f"assets/{sha[:2]}/{sha}.{ext}"
        return self.write_file(rel, content, label=label, kind=kind, mime=mime, binary=binary,
                               extra=extra, dedupe=True, add=add)

    def text_asset(self, text: str, label: str = "text", add: bool = True) -> dict[str, Any]:
        raw = text.encode("utf-8", "surrogatepass")
        blob = gzip.compress(raw, mtime=0)
        return self.write_asset(blob, "txt.gz", label=label, kind="text", mime="application/gzip",
                                binary=False, extra={"text_sha256": hashlib.sha256(raw).hexdigest(),
                                                     "length": len(text)}, add=add)

    def json_asset(self, obj: Any, label: str, add: bool = True) -> dict[str, Any]:
        raw = json.dumps(obj, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        blob = gzip.compress(raw, mtime=0)
        return self.write_asset(blob, "json.gz", label=label, kind="json", mime="application/gzip",
                                binary=False, add=add)

    def npy_asset(self, arr: Any, label: str, add: bool = True) -> dict[str, Any]:
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return self.write_asset(buf.getvalue(), "npy", label=label, kind="array",
                                mime="application/octet-stream", binary=True,
                                extra={"shape": [int(s) for s in arr.shape], "dtype": str(arr.dtype)},
                                add=add)

    def npz_asset(self, arrays: dict[str, Any], label: str, add: bool = True) -> dict[str, Any]:
        import numpy as np

        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self.write_asset(buf.getvalue(), "npz", label=label, kind="arrays",
                                mime="application/octet-stream", binary=True,
                                extra={"names": sorted(arrays)}, add=add)

    def png_asset(self, image: Any, label: str, add: bool = True,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
        png, width, height = encode_png(image)
        more = {"width": width, "height": height}
        if extra:
            more.update(extra)
        return self.write_asset(png, "png", label=label, kind="image", mime="image/png",
                                binary=True, extra=more, add=add)

    def fill_pending(self, pending: list[Any]) -> None:
        """Write the files of a summary and fill in its nodes."""
        from capx_trace_recorder.serialize import PendingArray, PendingImage, PendingText, array_stats

        for item in pending:
            try:
                if isinstance(item, PendingText):
                    art = self.text_asset(item.text, label="text", add=False)
                    item.node["asset"] = art["path"]
                    item.node["sha256"] = art.get("text_sha256")
                    item.node["written"] = art["written"]
                    if not art["written"]:
                        item.node["reason"] = art.get("reason")
                elif isinstance(item, PendingImage):
                    art = self.png_asset(item.image, label=item.label, add=True)
                    item.node.update({"path": art["path"], "sha256": art["sha256"],
                                      "written": art["written"], "width": art.get("width"),
                                      "height": art.get("height")})
                    if not art["written"]:
                        item.node["reason"] = art.get("reason")
                    arr = image_array(item.image)
                    if arr is not None:
                        from capx_trace_recorder.payload import raw_pixel_sha256

                        pix = raw_pixel_sha256(arr)
                        item.node["pixel_sha256"] = pix
                        if pix and self.event is not None:
                            self.wep.pixel_index.setdefault(pix, []).append({
                                "kind": item.label,
                                "event_id": self.event.get("event_id"),
                                "sim_step": self.event.get("sim_step"),
                                "wall_time": self.event.get("wall_time"),
                            })
                elif isinstance(item, PendingArray):
                    stats = array_stats(item.array)
                    if stats is not None:
                        item.node["stats"] = stats
                    art = self.npy_asset(item.array, label=item.label, add=True)
                    item.node["path"] = art["path"]
                    item.node["written"] = art["written"]
                    if not art["written"]:
                        item.node["reason"] = art.get("reason")
            except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
                item.node["written"] = False
                item.node["reason"] = f"encode_failed: {type(exc).__name__}"
                self.wep.add_gap("write_failed", 1, self.event.get("seq") if self.event else None)

    def fill_dropped(self, nodes: list[dict[str, Any]]) -> None:
        """Mark the summary nodes of copies released by a queue_full drop."""
        for node in nodes:
            node["written"] = False
            node["reason"] = "queue_full"

    def note_dropped(self, kind: str) -> None:
        """Remember that an item of ``kind`` was dropped for the queue budget."""
        self.wep.dropped_kinds[kind] = self.wep.dropped_kinds.get(kind, 0) + 1

    def warn(self, code: str, message: str, related_event_id: str | None = None,
             count_gap: bool = True) -> None:
        if count_gap:
            self.wep.add_gap(code, 1, self.event.get("seq") if self.event else None)
        self.store.warn_once(self.wep, code, message, related_event_id)


def image_array(image: Any) -> Any:
    """uint8 pixel array of a PIL image or an array (None otherwise)."""
    import numpy as np

    if isinstance(image, np.ndarray):
        return image
    try:
        return np.asarray(image)
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None


def encode_png(image: Any) -> tuple[bytes, int | None, int | None]:
    from PIL import Image

    import numpy as np

    if isinstance(image, np.ndarray):
        arr = np.ascontiguousarray(image)
        if arr.dtype != np.uint8:
            raise TypeError(f"not uint8: {arr.dtype}")
        img = Image.fromarray(arr)
    else:
        img = image
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    width, height = img.size
    return buf.getvalue(), int(width), int(height)


def encode_jpeg(arr: Any, quality: int) -> bytes:
    from PIL import Image

    import numpy as np

    a = np.ascontiguousarray(arr)
    if a.dtype != np.uint8:
        raise TypeError(f"not uint8: {a.dtype}")
    if a.ndim == 3 and a.shape[2] == 4:
        a = np.ascontiguousarray(a[:, :, :3])
    img = Image.fromarray(a)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Process info (writer thread, cached)
# ---------------------------------------------------------------------------

_PROCESS_INFO: dict[str, Any] | None = None
_PROCESS_INFO_CAPX_SEEN = False


def _git_commit(repo: Path) -> str | None:
    try:
        git = repo / ".git"
        if git.is_file():
            text = git.read_text().strip()
            if text.startswith("gitdir:"):
                git = (repo / text.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(":", 1)[1].strip()
        ref_path = git / ref
        if ref_path.is_file():
            return ref_path.read_text().strip() or None
        common = git
        commondir = git / "commondir"
        if commondir.is_file():
            common = (git / commondir.read_text().strip()).resolve()
            if (common / ref).is_file():
                return (common / ref).read_text().strip() or None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.strip().split(" ")
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None
    return None


def _process_start_time(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        rest = stat[stat.rfind(")") + 2:].split()
        start_ticks = int(rest[19])
        btime = None
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return None


def process_info() -> dict[str, Any]:
    """Host, python, git commit, process start (writer thread only: it reads files)."""
    global _PROCESS_INFO, _PROCESS_INFO_CAPX_SEEN
    if (_PROCESS_INFO is None or _PROCESS_INFO.get("pid") != os.getpid()
            or (not _PROCESS_INFO_CAPX_SEEN and getattr(sys.modules.get("capx"), "__file__", None))):
        from capx_trace_recorder.config import capx_checkout_root

        _PROCESS_INFO_CAPX_SEEN = bool(getattr(sys.modules.get("capx"), "__file__", None))

        _PROCESS_INFO = {
            "pid": os.getpid(),
            "host": hostname(),
            "python": sys.version.split()[0],
            "git_commit": _git_commit(root) if (root := capx_checkout_root()) else None,
            "process_start_time": _process_start_time(os.getpid()),
        }
    return dict(_PROCESS_INFO)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class TraceStore:
    """Process singleton owning the queue and the writer thread."""

    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self.pid = os.getpid()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._bytes_lock = threading.Lock()
        self._queued_bytes = 0
        self._in_flight = 0  # items queued or being processed (for flush())
        self._writer: threading.Thread | None = None
        self._writer_failed = False
        self._atexit_registered = False
        self._stopping = False
        # writer-only
        self._open_weps: dict[int, WriterEpisode] = {}  # opened logs whose end is not processed
        self._disk_state = "ok"
        self._disk_checked = 0.0
        self._free_bytes: int | None = None
        self._tmp_counter = 0
        self._run_json_done: set[str] = set()
        self._run_json_failures: dict[str, int] = {}
        self.stats = {"queue_full": 0, "items": 0, "internal_errors": 0, "lost_after_stop": 0,
                      "run_write_failed": 0}

    # -- producer side --------------------------------------------------------
    def writer_alive(self) -> bool:
        w = self._writer
        return w is not None and w.is_alive()

    @property
    def stopped(self) -> bool:
        """True once free disk space fell below half the floor (all writing stopped)."""
        return self._disk_state == "stopped"

    def _ensure_writer(self) -> bool:
        w = self._writer
        if w is not None:
            return w.is_alive()
        if self._writer_failed or self._stopping:
            return False
        with self._start_lock:
            if self._writer is None and not self._writer_failed:
                thread = threading.Thread(target=self._writer_main, name="capx-trace-writer",
                                          daemon=True)
                if not self._atexit_registered:
                    atexit.register(self._atexit)
                    self._atexit_registered = True
                # set before start(): an alarm inside start() must not lead to a
                # second writer (FIFO order); a thread that never started shows as dead
                self._writer = thread
                try:
                    thread.start()
                except TimeoutError:  # the trial alarm (section 2)
                    raise
                except Exception:  # noqa: BLE001
                    self._writer_failed = True
                    return False
        w = self._writer
        return w is not None and w.is_alive()

    def submit(self, item: Item) -> bool:
        """Queue ``item`` (never blocks).

        Over the byte budget the item is dropped centrally: ``binary_dropped`` is
        set, its builder is replaced by its ``drop_builder`` (which holds no binary
        copy, so the copies are freed now) and the writer counts one
        ``queue_full`` gap. ``keep_payload`` items keep their builder and stay
        counted: model bodies are text and are always written (their images are not).
        """
        if not self._ensure_writer():
            return False
        with self._bytes_lock:
            if self._disk_state == "stopped" and item.kind in ("event", "internal"):
                # the writer drops it anyway (all writing stopped): free the copies now
                item.builder = None
                item.nbytes = 0
            elif item.nbytes:
                if self._queued_bytes + item.nbytes > self.config.max_queue_bytes:
                    item.binary_dropped = True
                    self.stats["queue_full"] += 1
                    if item.keep_payload:
                        self._queued_bytes += item.nbytes
                    else:
                        item.nbytes = 0
                        if item.drop_builder is not KEEP_BUILDER:
                            item.builder = item.drop_builder
                else:
                    self._queued_bytes += item.nbytes
            item.drop_builder = None
            self._in_flight += 1
            # counted and queued under the lock, so the writer never subtracts an
            # item's bytes before they were added
            self._queue.put(item)
        return True

    def queued_bytes(self) -> int:
        return self._queued_bytes

    def append_run_line(self, run_dir: Path, name: str, line: bytes | Callable[[], bytes]) -> bool:
        """Append ``line`` to the run-level file ``run_dir/name`` on the writer.

        ``line`` may be a callable that the writer calls to make the bytes (H16
        redacts and serialises the runner record there, not on the caller's thread).
        """
        item = Item("run_line")
        item.path = run_dir / name
        item.line = line  # type: ignore[assignment]
        return self.submit(item)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until the queue is empty, including items the writer queued
        while processing (its own warnings). For tests; never on a trial thread."""
        deadline = time.monotonic() + timeout
        while True:
            if not self.writer_alive():
                return False
            item = Item("barrier")
            item.done = threading.Event()
            with self._bytes_lock:
                self._in_flight += 1
                self._queue.put(item)
            if not item.done.wait(max(0.0, deadline - time.monotonic())):
                return False
            with self._bytes_lock:
                idle = self._in_flight == 0
            if idle:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def shutdown(self, timeout: float = ATEXIT_WAIT_S) -> None:
        if self._stopping:
            return
        self._stopping = True
        w = self._writer
        if w is None or not w.is_alive():
            return
        item = Item("stop")
        item.done = threading.Event()
        self._queue.put(item)
        w.join(timeout)

    def _atexit(self) -> None:
        try:
            self.shutdown(ATEXIT_WAIT_S)
        except TimeoutError:  # section 2: re-raised (KeyboardInterrupt is not caught)
            raise
        except Exception:  # noqa: BLE001
            pass

    # -- writer thread --------------------------------------------------------
    def _writer_main(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    item = None
                if item is not None:
                    if item.kind == "stop":
                        try:
                            self._heartbeat(force=True)
                        except BaseException:  # noqa: BLE001 - timeout-ok: writer thread
                            pass
                        if item.done is not None:
                            item.done.set()
                        return
                    try:
                        self._handle(item)
                    except BaseException as exc:  # noqa: BLE001 - timeout-ok: writer thread
                        self.stats["internal_errors"] += 1
                        try:
                            self._internal_error(item, exc)
                        except BaseException:  # noqa: BLE001 - timeout-ok: writer thread
                            pass
                    finally:
                        with self._bytes_lock:
                            if item.nbytes:
                                self._queued_bytes -= item.nbytes
                            self._in_flight -= 1
                        if item.done is not None:
                            item.done.set()
                    self.stats["items"] += 1
                try:
                    self._heartbeat()
                except BaseException:  # noqa: BLE001 - timeout-ok: writer thread
                    pass
        except BaseException:  # noqa: BLE001 - timeout-ok: writer thread; the thread ends quietly, emit() sees it
            return

    def _internal_error(self, item: Item, exc: BaseException) -> None:
        sink = item.sink
        if sink is None or self._disk_state == "stopped":
            return
        wep = self._wep(sink)
        seq = item.event.get("seq") if item.event else None
        wep.add_gap("internal_error", 1, seq)
        self.warn_once(wep, "internal_error",
                       f"writer failed on an item: {type(exc).__name__}", None)
        if wep.end_processed and not wep.is_unscoped and not wep.closed_for_good:
            self._write_episode_json(wep)  # the warning above may be a repeat that writes no line

    def _wep(self, sink: EventSink) -> WriterEpisode:
        """The writer state of ``sink``, kept on the sink (it lives as long as the sink)."""
        wep = sink.__dict__.get("_writer_state")
        if wep is None or wep.sink is not sink:
            wep = WriterEpisode(sink)
            sink._writer_state = wep
        return wep

    def _handle(self, item: Item) -> None:
        if item.kind == "barrier":
            return
        if item.kind == "run_line":
            self._append_run_line(item)
            return
        sink = item.sink
        if sink is None:
            return
        self.disk_allows(False, trigger=item)  # refresh the disk state (at most every 5 s)
        if self._disk_state == "stopped":
            self._drop_after_stop(item, sink.__dict__.get("_writer_state"))
            return
        wep = self._wep(sink)
        if wep.closed_for_good:
            self._drop_after_stop(item, wep)
            return
        gaps_before = wep.gap_total()  # before this item's own queue_full gap
        if item.binary_dropped and not item.keep_payload:
            # one gap per dropped item, whatever it carried (section 4.4)
            self._queue_full_gap(wep, item)
        if item.kind == "internal":
            if item.builder is not None:
                item.builder(BuildContext(self, wep, item))
            if wep.end_processed and not wep.is_unscoped and wep.gap_total() != gaps_before:
                # a gap counted after episode_end (note_gap, or this item was dropped for the
                # queue budget): episode.json must show it, even when no warning line follows
                self._write_episode_json(wep)
            return
        if item.kind != "event" or item.event is None:
            return
        event = item.event
        seq = int(event.get("seq") or 0)
        if seq > wep.expected_seq:
            missing = seq - wep.expected_seq
            wep.dropped_events += missing
            wep.add_gap("dropped_events", missing, wep.expected_seq)
            self.warn_once(wep, "dropped_events",
                           f"{missing} event(s) missing before seq {seq}", None)
        wep.expected_seq = max(wep.expected_seq, seq + 1)
        if not wep.opened:
            self._open(wep)
        etype = event.get("event_type")
        if etype == "episode_start":
            earlier = wep.summary
            wep.summary = self._summary_from_start(event)
            # a hook installed between open_episode() and this line (bootstrap
            # queues it as an internal item) must not be lost
            if earlier.get("hooks_installed_during_episode"):
                wep.summary["hooks_installed_during_episode"] = earlier["hooks_installed_during_episode"]
            tracer = event.get("data", {}).get("tracer") or {}
            failed = tracer.get("hooks_failed") if isinstance(tracer, dict) else None
            if failed:
                wep.add_gap("hook_install_failed", len(failed), seq)
        ctx = BuildContext(self, wep, item)
        if item.builder is not None:
            try:
                item.builder(ctx)
            except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
                from capx_trace_recorder.payload import put_redacted

                put_redacted(event["data"], "recorder_error", f"{type(exc).__name__}: {exc}", 500)
                wep.add_gap("internal_error", 1, seq)
                self.warn_once(wep, "internal_error",
                               f"builder failed for {etype}: {type(exc).__name__}", event.get("event_id"))
        if item.binary_dropped and item.keep_payload and ctx.binary_skipped:
            # a model body is always written; only the images in it were not
            self._queue_full_gap(wep, item)
        if self._disk_state == "stopped" or wep.closed_for_good:
            # the disk-low stop happened inside the builder (write_file): its final
            # marker is already the last line, so this older line is not appended
            self._drop_after_stop(item, wep)
            return
        if event.get("late"):
            wep.late_events += 1
            wep.add_gap("late_event", 1, seq)
        if etype == "trace_warning":
            wep.n_warnings += 1
            code = event.get("data", {}).get("code")
            if code and event.get("data", {}).get("counts_as_gap", True) and not event["data"].get("writer_generated"):
                wep.add_gap(str(code), 1, seq)
        self._resolve_frame_refs(wep, event)
        if etype == "episode_end":
            data = event["data"]
            gaps = wep.gap_list()
            data["gaps"] = gaps
            data["dropped_events"] = wep.dropped_events
            data["late_events"] = wep.late_events
            data["n_warnings"] = wep.n_warnings
            data["trace_complete"] = not gaps
        line = self._encode_line(ctx, event)
        ok = self._append_line(wep, line, event)
        if ok:
            wep.n_events += 1
            wep.last_seq = seq
            if etype == "trial_videos":
                wep.video_copy_pending = False
        if etype == "episode_start" and not wep.is_unscoped:
            wep.start_written = ok
            self._write_episode_json(wep)
            self._heartbeat_one(wep, force=True)
        elif etype == "episode_end" and not wep.is_unscoped:
            wep.end_written = ok
            wep.end_processed = True  # no more heartbeats, even if the line failed
            self._finish_summary(wep, event)
            self._write_episode_json(wep)
            self._close_fd(wep)
            self._release(wep)
        elif wep.end_written and not wep.is_unscoped:
            # late event or late warning: episode.json must show the gap
            self._write_episode_json(wep)
            self._close_fd(wep)

    def _queue_full_gap(self, wep: WriterEpisode, item: Item) -> None:
        """One ``queue_full`` gap for an item whose binary data was dropped."""
        seq = item.event.get("seq") if item.event else None
        wep.add_gap("queue_full", 1, seq)
        kind = item.event.get("event_type") if item.event else "internal"
        wep.dropped_kinds[str(kind)] = wep.dropped_kinds.get(str(kind), 0) + 1
        self.warn_once(wep, "queue_full",
                       "binary data dropped: writer queue above CAPX_TRACE_MAX_QUEUE_MB",
                       item.event.get("event_id") if item.event else None)

    def _drop_after_stop(self, item: Item, wep: WriterEpisode | None) -> None:
        """An item that arrives after all writing stopped (disk low, section 3).

        Nothing is written. ``episode_end`` still releases the episode's
        references (env, frame buffer, segments), which the experiment keeps using.
        """
        self.stats["lost_after_stop"] += 1
        if wep is not None:
            wep.lost_after_stop += 1
        event = item.event
        if item.kind == "event" and event is not None and event.get("event_type") == "episode_end":
            sink = item.sink
            if wep is not None:
                wep.end_processed = True
                self._release(wep)
            elif sink is not None:
                self._release_sink(sink)

    # -- episode.json ---------------------------------------------------------
    def _summary_from_start(self, event: dict[str, Any]) -> dict[str, Any]:
        data = event.get("data", {})
        keys = ("source", "task", "condition", "seed", "attempt_index", "runner_invocation",
                "condition_label", "models", "config_flags", "env")
        summary = {k: data.get(k) for k in keys}
        summary["started_wall_time"] = event.get("wall_time")
        summary["pid"] = event.get("pid")
        return summary

    def _finish_summary(self, wep: WriterEpisode, event: dict[str, Any]) -> None:
        data = event.get("data", {})
        wep.summary["ended_wall_time"] = event.get("wall_time")
        for key in ("end_reason", "duration_s", "alarm_seen", "n_code_blocks", "n_model_calls",
                    "n_http_attempts", "tracer_overhead_s"):
            wep.summary[key] = data.get(key)
        hs = data.get("harness_summary")
        if isinstance(hs, dict):
            wep.summary["harness_summary"] = {
                k: hs.get(k) for k in ("reward", "task_completed", "sandbox_rc", "success",
                                       "num_code_blocks", "num_regenerations", "num_finishes")
            }
        wep.summary["exception"] = data.get("exception")

    def _write_episode_json(self, wep: WriterEpisode) -> bool:
        """Rewrite ``episode.json``; a failure becomes a ``write_failed`` gap.

        The warning it queues makes the writer rewrite the file once more (for
        an ended episode), so a later successful write shows the gap.
        """
        sink = wep.sink
        gaps = wep.gap_list()
        doc = {
            "schema_version": SCHEMA_VERSION,
            "tracer_version": TRACER_VERSION,
            "run_id": sink.run_id,
            "episode_id": sink.episode_id,
            "status": "ended" if wep.end_written else "running",
            "episode_end_written": wep.end_written,
            "trace_complete": (not gaps) if wep.end_written else None,
            "video_copy_pending": wep.video_copy_pending,
            "gaps": gaps,
            "dropped_events": wep.dropped_events,
            "late_events": wep.late_events,
            "n_warnings": wep.n_warnings,
            "n_events": wep.n_events,
            "last_seq": wep.last_seq,
            "events_bytes": wep.size,
            "heartbeat_interval_s": HEARTBEAT_INTERVAL_S,
            "updated_wall_time": time.time(),
            "host": process_info().get("host"),
        }
        doc.update(wep.summary)
        raw = json.dumps(doc, ensure_ascii=True, indent=1, allow_nan=False,
                         default=lambda o: str(o)).encode()
        ok, reason = self.atomic_write(wep.base_dir / "episode.json", raw, wep, count_disk=False)
        if not ok:
            wep.add_gap("write_failed", 1, None)
            self.warn_once(wep, "write_failed", f"episode.json write failed: {reason}",
                           key="write_failed:episode.json")
        return ok

    def update_summary(self, wep: WriterEpisode, fields: dict[str, Any]) -> None:
        wep.summary.update(fields)
        if wep.start_written:
            self._write_episode_json(wep)

    # -- files ----------------------------------------------------------------
    def _open(self, wep: WriterEpisode) -> None:
        wep.opened = True
        if not wep.end_processed:
            self._open_weps[id(wep.sink)] = wep
        try:
            os.makedirs(wep.events_path.parent, exist_ok=True)
        except OSError:  # timeout-ok: writer thread
            pass
        self._ensure_run_json(wep.sink.run_dir, wep.sink.run_id)

    def _release(self, wep: WriterEpisode) -> None:
        """Cut an ended episode's writer state down to a small record.

        Kept: seq, counters, gaps and the summary (late events rewrite
        ``episode.json``). Dropped: the lookup tables and the file sets (a late
        frame reference falls back to ``os.path.exists``), and the sink's
        references to the env, frames, segments and web session.
        """
        wep.pixel_index = {}
        wep.video_reads = []
        wep.video_encodes = {}
        wep.links = {}
        wep.written_files = set()
        wep.made_dirs = set()
        wep.failed_files = {}
        self._open_weps.pop(id(wep.sink), None)
        self._release_sink(wep.sink)

    def _release_sink(self, sink: EventSink) -> None:
        release = getattr(sink, "release_references", None)
        if release is None:
            return
        try:
            release()
        except Exception:  # noqa: BLE001 - timeout-ok: writer thread
            pass

    def _ensure_run_json(self, run_dir: Path, run_id: str) -> None:
        """Create ``run.json`` once (``O_EXCL``, never replaced).

        Every byte is written (a short write is continued); a write that makes no
        progress or fails removes the partial file, records ``write_failed`` in the
        run's unscoped log and is tried again for the run's next item, at most
        ``RUN_JSON_ATTEMPTS`` times.
        """
        key = str(run_dir)
        if key in self._run_json_done:
            return
        failures = self._run_json_failures.get(key, 0)
        if failures >= RUN_JSON_ATTEMPTS:
            return
        path = run_dir / "run.json"
        try:
            os.makedirs(run_dir, exist_ok=True)
            if path.exists():
                self._run_json_done.add(key)
                return
            info = process_info()
            doc = {
                "schema_version": SCHEMA_VERSION,
                "tracer_version": TRACER_VERSION,
                "run_id": run_id,
                "created_wall_time": time.time(),
                "created_at": utc_iso(),
                "created_by": {"pid": info.get("pid"), "host": info.get("host"),
                               "python": info.get("python"), "git_commit": info.get("git_commit")},
                "trace_dir": str(self.config.trace_dir),
                "runner_run_dir": os.environ.get("CAPX_TRACE_RUNNER_RUN_DIR") or None,
            }
            raw = json.dumps(doc, ensure_ascii=True, indent=1).encode()
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                _write_all(fd, raw)
            except BaseException:  # timeout-ok: writer thread (re-raised below)
                os.close(fd)
                fd = -1
                _unlink_quietly(path)  # this process created it: a partial run.json is removed
                raise
            finally:
                if fd >= 0:
                    os.close(fd)
            self._run_json_done.add(key)
        except FileExistsError:
            self._run_json_done.add(key)
        except OSError as exc:  # timeout-ok: writer thread
            self._run_json_failures[key] = failures + 1
            self._run_write_failed(path, f"{type(exc).__name__}: {exc}")

    def atomic_write(self, target: Path, data: bytes, wep: WriterEpisode | None = None,
                     count_disk: bool = True) -> tuple[bool, str | None]:
        """Write ``target`` via a tmp file in the same directory and ``os.replace``."""
        parent = target.parent
        tmp: Path | None = None
        try:
            key = str(parent)
            if wep is None or key not in wep.made_dirs:
                os.makedirs(parent, exist_ok=True)
                if wep is not None and not wep.end_processed:
                    wep.made_dirs.add(key)
            self._tmp_counter += 1
            tmp = parent / f".{target.name}.tmp.{os.getpid()}.{self._tmp_counter}"
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                _write_all(fd, data)
            finally:
                os.close(fd)
            os.replace(tmp, target)
            return True, None
        except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
            try:
                if tmp is not None and os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:  # noqa: BLE001 - timeout-ok: writer thread
                pass
            return False, f"{type(exc).__name__}: {exc}"[:200]

    def _encode_line(self, ctx: BuildContext, event: dict[str, Any]) -> bytes:
        from capx_trace_recorder.serialize import json_safe

        try:
            text = json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            event["data"] = json_safe(event.get("data"))
            event["refs"] = json_safe(event.get("refs"))
            event["artifacts"] = json_safe(event.get("artifacts"))
            text = json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                              default=lambda o: f"<{type(o).__name__}>")
        raw = (text + "\n").encode("ascii")
        if len(raw) <= MAX_EVENT_LINE_BYTES:
            return raw
        # move the data into an asset and keep the small scalar fields inline
        data = event.get("data", {})
        art = ctx.json_asset(json_safe(data), label="payload", add=True)
        small = {k: v for k, v in data.items()
                 if v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200)}
        small["oversize"] = True
        small["data_asset"] = art["path"]
        small["data_asset_written"] = art["written"]
        event["data"] = small
        text = json.dumps(event, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                          default=lambda o: f"<{type(o).__name__}>")
        return (text + "\n").encode("ascii")

    def _append_line(self, wep: WriterEpisode, line: bytes, event: dict[str, Any] | None = None) -> bool:
        """One ``write`` call; on a short write or error truncate back.

        A failure is a ``write_failed`` gap. For an episode it reaches
        ``episode.json``; a run's unscoped log has no such file, so its failures
        are handled by ``_unscoped_line_failed``.
        """
        if wep.fd is None and wep.closed_for_good:
            return False  # after the disk-low marker nothing is appended
        try:
            if wep.fd is None:
                os.makedirs(wep.events_path.parent, exist_ok=True)
                wep.fd = os.open(str(wep.events_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                wep.size = os.fstat(wep.fd).st_size
        except OSError as exc:  # timeout-ok: writer thread
            reason = f"cannot open {wep.events_path.name}: {exc}"[:200]
            if wep.is_unscoped:
                self._unscoped_line_failed(wep, reason, event)
            else:
                wep.add_gap("write_failed", 1, None)
                self.warn_once(wep, "write_failed", reason, None)
            return False
        before = wep.size
        try:
            n = _os_write(wep.fd, line)
        except OSError:  # timeout-ok: writer thread
            n = -1
        if n != len(line):
            try:
                os.ftruncate(wep.fd, before)
            except OSError:  # timeout-ok: writer thread
                pass
            reason = f"short or failed write to {wep.events_path.name}"
            if wep.is_unscoped:
                self._unscoped_line_failed(wep, reason, event)
            else:
                wep.add_gap("write_failed", 1, None)
                self.warn_once(wep, "write_failed", reason, None)
            return False
        wep.size = before + n
        if wep.is_unscoped:
            self._unscoped_line_written(wep, event)
        return True

    @staticmethod
    def _is_loss_report(event: dict[str, Any] | None) -> bool:
        data = event.get("data") if isinstance(event, dict) else None
        return isinstance(data, dict) and bool(data.get("unscoped_log_failure"))

    def _unscoped_line_failed(self, wep: WriterEpisode, reason: str, event: dict[str, Any] | None) -> None:
        """A line of a run's ``unscoped.<pid>.jsonl`` could not be written.

        The log has no ``episode.json``, so the failure is kept outside it: counted
        in ``stats["run_write_failed"]`` and written to ``unscoped.<pid>.status.json``
        next to it (tmp + rename; ``lines_not_written``, ``last_reason``). When the
        log takes a line again, a ``write_failed`` warning with the count follows
        (``_unscoped_line_written``).
        """
        self.stats["run_write_failed"] += 1
        wep.lines_not_written += 1
        wep.add_gap("write_failed", 1, None)  # in memory only: the log has no episode.json
        if self._is_loss_report(event):
            wep.failure_warning_queued = False  # the report itself was lost: a later line brings a new one
        name = wep.events_path.name
        status = {
            "schema_version": SCHEMA_VERSION,
            "run_id": wep.sink.run_id,
            "file": name,
            "pid": os.getpid(),
            "lines_not_written": wep.lines_not_written,
            "last_reason": reason,
            "updated_wall_time": time.time(),
        }
        status_path = wep.events_path.with_name(name[: -len(".jsonl")] + ".status.json")
        self.atomic_write(status_path, json.dumps(status, ensure_ascii=True, indent=1).encode(), None,
                          count_disk=False)

    def _unscoped_line_written(self, wep: WriterEpisode, event: dict[str, Any] | None) -> None:
        """After a line of an unscoped log was written: report lines it lost before.

        A ``write_failed`` warning (``count``: lines not written so far, filled in
        when it is written) is queued once the log takes lines again, so a log
        that recovered says how many of its lines are missing.
        """
        if self._is_loss_report(event):
            wep.failure_warning_queued = False
            wep.lines_reported = int(event["data"].get("count") or 0)
        if wep.lines_not_written <= wep.lines_reported or wep.failure_warning_queued:
            return
        wep.failure_warning_queued = True
        name = wep.events_path.name

        def builder(ctx: BuildContext) -> None:
            ctx.data["count"] = ctx.wep.lines_not_written
            ctx.data["message"] = f"{ctx.wep.lines_not_written} line(s) of {name} could not be written"

        try:
            wep.sink.emit("trace_warning", status="warning", builder=builder, writer_generated=True,
                          data={"code": "write_failed", "file": name, "count": None, "message": None,
                                "unscoped_log_failure": True, "writer_generated": True})
        except Exception:  # noqa: BLE001 - timeout-ok: writer thread
            wep.failure_warning_queued = False

    def _close_fd(self, wep: WriterEpisode) -> None:
        if wep.fd is not None:
            try:
                os.close(wep.fd)
            except OSError:  # timeout-ok: writer thread
                pass
            wep.fd = None

    def _append_run_line(self, item: Item) -> None:
        path = item.path
        if path is None:
            return
        # the runner's parent process may write nothing else: it must notice a disk stop too
        self.disk_allows(False, trigger=path.parent)
        if self._disk_state == "stopped":
            self.stats["lost_after_stop"] += 1
            return
        line = item.line
        if callable(line):
            try:
                line = line()
            except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
                self._run_write_failed(path, f"line not built: {type(exc).__name__}")
                return
        line = line or b""
        self._ensure_run_json(path.parent, path.parent.name)
        try:
            os.makedirs(path.parent, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                before = os.fstat(fd).st_size
                n = _os_write(fd, line)
                if n != len(line):
                    os.ftruncate(fd, before)
                    self._run_write_failed(path, "short write")
            finally:
                os.close(fd)
        except OSError as exc:  # timeout-ok: writer thread
            self._run_write_failed(path, f"{type(exc).__name__}: {exc}")

    def _run_write_failed(self, path: Path, reason: str) -> None:
        """A run-level file (``runner_records.jsonl``, ``run.json``) could not be written.

        Recorded as a ``write_failed`` warning in the run's ``unscoped.<pid>.jsonl``
        (a run-level file the viewer reads), since no episode owns these files.
        """
        self.stats["run_write_failed"] += 1
        try:
            from capx_trace_recorder.context import unscoped_log

            log = unscoped_log(self, path.parent.name)
            log.emit("trace_warning", status="warning",
                     data={"code": "write_failed", "file": path.name,
                           "message": f"{path.name}: {reason}"[:300], "writer_generated": True},
                     writer_generated=True)
        except Exception:  # noqa: BLE001 - timeout-ok: writer thread
            pass

    # -- disk -----------------------------------------------------------------
    def disk_allows(self, binary: bool, trigger: Any = None) -> tuple[bool, str | None]:
        """Whether a file may be written now (refreshes the free space every 5 s).

        ``trigger`` is what is being written: the item in hand, a builder's sink, or
        a run dir (run-level line). When this call notices that all writing must
        stop, the final marker also goes to that item's log and run (see
        ``_stop_all_for_disk``).
        """
        if self._disk_state == "stopped":
            return False, "disk_low"  # the stop is final for this process (section 3)
        now = time.monotonic()
        if now - self._disk_checked >= DISK_CHECK_INTERVAL_S or self._free_bytes is None:
            self._disk_checked = now
            path = self.config.trace_dir
            while not path.exists() and path != path.parent:
                path = path.parent
            try:
                self._free_bytes = int(_disk_free(str(path)))
            except Exception:  # noqa: BLE001 - timeout-ok: writer thread
                self._free_bytes = None
            floor = self.config.min_free_bytes
            if self._free_bytes is not None and floor > 0:
                if self._free_bytes < floor / 2:
                    self._disk_state = "stopped"
                    self._stop_all_for_disk(trigger)
                elif self._free_bytes < floor:
                    self._disk_state = "low"
                else:
                    self._disk_state = "ok"
        if self._disk_state == "stopped":
            return False, "disk_low"
        if self._disk_state == "low" and binary:
            return False, "disk_low"
        return True, None

    def _stop_all_for_disk(self, trigger: Any = None) -> None:
        """Write one final ``disk_low`` marker, then stop all writing (section 3).

        The marker goes to every open log, to the log of the item that noticed the
        stop (an episode whose first item that was gets its directory, its
        ``events.jsonl`` with the marker, and an ``episode.json`` with the
        ``disk_low`` gap and the fields of its ``episode_start``), and to the
        run-level ``unscoped.<pid>.jsonl`` of every run concerned, so that a stop
        between two episodes is visible and the episodes that follow are known to
        be missing. An ended episode has a complete trace and keeps it (its files
        are not touched).
        """
        message = ("free disk space below half of CAPX_TRACE_MIN_FREE_GB; recording stopped: "
                   "later events and episodes of this process are not recorded")
        runs: dict[str, Path] = {}
        for wep in list(self._open_weps.values()):
            if wep.closed_for_good or not wep.opened or wep.end_processed:
                continue
            self._write_final_marker(wep, message)
            runs[wep.sink.run_id] = wep.sink.run_dir
        sink = trigger.sink if isinstance(trigger, Item) else (
            trigger if isinstance(trigger, EventSink) else None)
        if sink is not None:
            wep = self._wep(sink)
            if not wep.closed_for_good and not wep.end_processed:
                event = trigger.event if isinstance(trigger, Item) else None
                if (not wep.opened and not wep.is_unscoped and isinstance(event, dict)
                        and event.get("event_type") == "episode_start"):
                    wep.summary = self._summary_from_start(event)
                self._write_final_marker(wep, message)
            runs[sink.run_id] = sink.run_dir
        elif isinstance(trigger, Path):  # a run-level line (runner_records.jsonl)
            runs[trigger.name] = trigger
        for run_id in runs:
            try:
                from capx_trace_recorder.context import unscoped_log

                wep = self._wep(unscoped_log(self, run_id))
            except Exception:  # noqa: BLE001 - timeout-ok: writer thread
                continue
            if not wep.closed_for_good:
                self._write_final_marker(wep, message)

    def _write_final_marker(self, wep: WriterEpisode, message: str) -> None:
        """Append the ``disk_low`` marker as the last line of ``wep``'s log and close it for good."""
        sink = wep.sink
        seq = sink.next_seq_for_writer()
        wep.expected_seq = seq + 1
        wep.add_gap("disk_low", 1, seq)
        event = {
            "schema_version": SCHEMA_VERSION, "run_id": sink.run_id,
            "episode_id": sink.episode_id, "event_id": new_event_id(),
            "parent_event_id": None, "seq": seq, "wall_time": time.time(),
            "monotonic_time": time.monotonic(), "pid": os.getpid(),
            "thread": threading.current_thread().name, "sim_step": None, "sim_time": None,
            "sim_epoch": sink.sim_epoch, "event_type": "trace_warning", "status": "warning",
            "late": False, "refs": {}, "artifacts": [],
            "data": {"code": "disk_low", "writer_generated": True, "message": message},
        }
        line = (json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        if not wep.opened:  # a log that starts with the marker: its run is listed like any other
            wep.opened = True
            self._ensure_run_json(sink.run_dir, sink.run_id)
        if self._append_line(wep, line, event):
            wep.n_events += 1
            wep.last_seq = seq
        if not wep.is_unscoped:
            self._write_episode_json(wep)
        self._close_fd(wep)
        wep.closed_for_good = True

    # -- warnings -------------------------------------------------------------
    def warn_once(self, wep: WriterEpisode, code: str, message: str,
                  related_event_id: str | None = None, key: str | None = None) -> None:
        """A writer-detected problem: one ``trace_warning`` line per ``key`` (default: code) per log."""
        dedup = key or code
        if dedup in wep.warned_codes:
            return
        wep.warned_codes.add(dedup)
        try:
            wep.sink.emit("trace_warning", status="warning",
                          data={"code": code, "message": message, "related_event_id": related_event_id,
                                "writer_generated": True}, writer_generated=True)
        except Exception:  # noqa: BLE001 - timeout-ok: writer thread
            pass

    # -- heartbeat ------------------------------------------------------------
    def _heartbeat(self, force: bool = False) -> None:
        for wep in list(self._open_weps.values()):
            self._heartbeat_one(wep, force=force)

    def _heartbeat_one(self, wep: WriterEpisode, force: bool = False) -> None:
        if (wep.is_unscoped or not wep.start_written or wep.end_written or wep.end_processed
                or wep.closed_for_good):
            return
        now = time.monotonic()
        if not force and now - wep.last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        wep.last_heartbeat = now
        path = wep.base_dir / "heartbeat"
        try:
            if path.exists():
                os.utime(path, None)
            else:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o644)
                os.close(fd)
        except OSError:  # timeout-ok: writer thread
            pass

    # -- frame references -----------------------------------------------------
    def _resolve_frame_refs(self, wep: WriterEpisode, event: dict[str, Any]) -> None:
        """``written`` and ``reason`` of every key-frame reference (``frame_ref``, ``frame_refs``)."""
        data = event.get("data")
        if not isinstance(data, dict):
            return
        for field in ("frame_ref", "frame_refs"):
            refs = data.get(field)
            if not refs:
                continue
            items = refs if isinstance(refs, list) else [refs]
            for ref in items:
                if not isinstance(ref, dict) or not ref.get("path"):
                    continue
                key = str(wep.base_dir / ref["path"])
                written = key in wep.written_files
                if not written and wep.end_processed:
                    # an ended episode no longer keeps its file set (late events)
                    written = os.path.exists(key)
                ref["written"] = written
                if not written:
                    ref.setdefault("reason", wep.failed_files.get(key, "frame_not_written"))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_STORE: TraceStore | None = None
_STORE_LOCK = threading.Lock()


def init_store(config: TraceConfig) -> TraceStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = TraceStore(config)
        return _STORE


def get_store() -> TraceStore | None:
    return _STORE


def _reset_after_fork() -> None:
    """A forked child gets a fresh store (the writer thread does not survive fork)."""
    global _STORE, _STORE_LOCK, PID
    _STORE_LOCK = threading.Lock()
    PID = os.getpid()
    if _STORE is not None:
        _STORE = TraceStore(_STORE.config)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


__all__ = [
    "TraceStore",
    "EpisodeRecorder",
    "UnscopedLog",
    "EventSink",
    "BuildContext",
    "WriterEpisode",
    "Item",
    "init_store",
    "get_store",
    "process_info",
    "encode_jpeg",
    "encode_png",
]
