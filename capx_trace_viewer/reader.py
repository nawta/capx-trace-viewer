"""Tolerant reader for experiment traces (docs/trace-format.md sections 4.5 and 9.2).

Reads what the trace recorder writes under a trace root:

- ``events.jsonl``: split on ``b"\\n"``; lines that do not parse, anywhere in the file,
  are skipped and counted as ``corrupt_lines``; a last line without a newline is a torn
  tail and is not consumed. Paging is by byte offset, and pages are cached per file,
  keyed by (device, inode, size, mtime, ctime). Every page carries the file's
  ``generation`` (device, inode and a hash of the whole first line), so a client can tell
  that the file was replaced or rewritten in place; a cached page whose generation is
  not the file's current one is read again.
- ``episode.json`` plus the ``heartbeat`` file: the episode status of section 9.2,
  without reading ``events.jsonl``. Only for the summary fields, its first line (the
  ``episode_start`` event) is read when ``episode.json`` does not name one of the fields
  that event also holds (a field present with a null value counts as named: the recorder
  copies every start field, a null one too). A missing heartbeat file means the liveness
  is unknown, and an episode without ``episode_end`` is then ``incomplete``.
- ``run.json`` and ``runner_records.jsonl`` of a run.
- ``unscoped.<pid>.jsonl`` of a run (events with no episode, section 4.1): listed by
  ``run_detail`` and summarised by ``unscoped_summary`` under the id ``unscoped.<pid>``,
  so the viewer can open it like an episode (its files are under ``unscoped.<pid>.files/``).
  The listing also carries the recorder's ``unscoped.<pid>.status.json`` (lines of the log
  that could not be written) as ``write_status``, even when the log itself is missing.
- Files that exist but cannot be read are reported with a reason ("unreadable (EACCES)",
  "symlink_not_followed", ...), never as missing (section 2: gaps are never hidden).

Opening files below a root (``Dir``): the root is opened once, then every path component
is opened relative to its parent's file descriptor with ``O_NOFOLLOW``; a regular file is
checked with ``fstat`` on the descriptor that is then read. No symlink below the root is
followed, and a name swapped for a symlink between two requests (or between the check
and the read) is refused or has no effect on what is read. capx_trace_viewer/trace_api.py serves
every file through this class. The path-based helpers (``safe_join``, ``own_file``,
``read_events_page(path)``, ``episode_summary(path)``, ...) remain for local tools such
as local validation tools; they follow symlinks in the path they are given.

This module is shared with capx_trace_viewer/trace_api.py, which can run in the same process as a
live trial. It therefore follows the rules of section 2: it never prints, never emits a
Python warning, never configures logging and uses only the standard library (no numpy).
It never imports capx.web.

Fields read from ``episode.json`` as the trace recorder writes it (flat keys; nested
``start``/``end``/``env`` dicts and the ``episode_start`` event are fallbacks):
``status`` ("ended" once episode_end is written; any other value is ignored and the
heartbeat decides), ``trace_complete``, ``gaps``, ``dropped_events``, ``late_events``,
``n_warnings``, ``end_reason``, ``source``, ``task``, ``condition``, ``condition_label``,
``seed``, ``attempt_index``, ``runner_invocation``, ``started_wall_time`` (or
``start_wall_time``), ``ended_wall_time`` (or ``end_wall_time``), ``env.viser_port``
(or ``viser_port``), ``pid``, ``host``. The summary reports them under the names of
SUMMARY_FIELDS.

Related modules: capx_trace_viewer/trace_api.py (serves these results over HTTP),
scripts/make_demo_trace.py (writes a synthetic trace in this format),
tests/test_trace_reader.py.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "capx-trace/1"

# Section 4.2 and 9.2. Always use fullmatch: "$" in re.match also accepts a trailing "\n".
ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}"
_ID_RE = re.compile(ID_PATTERN)

EVENTS_FILE = "events.jsonl"
EPISODE_FILE = "episode.json"
HEARTBEAT_FILE = "heartbeat"
RUN_FILE = "run.json"
RUNNER_RECORDS_FILE = "runner_records.jsonl"

HEARTBEAT_STALE_S = 30.0
STATUSES = ("ended", "ended_with_gaps", "running", "incomplete")

# A run's log of events that belong to no episode (section 4.1): ``unscoped.<pid>.jsonl``,
# with its payloads and assets under ``unscoped.<pid>.files/`` (paths in its events are
# relative to the run dir). The viewer opens it like an episode under the id
# ``unscoped.<pid>``; its summary has status "unscoped".
UNSCOPED_STATUS = "unscoped"
_UNSCOPED_ID_RE = re.compile(r"unscoped\.[0-9]{1,12}")
# ``unscoped.<pid>.status.json``: written by the recorder next to a log whose lines could
# not be written (see _unscoped_logs); read up to this size.
UNSCOPED_STATUS_SUFFIX = ".status.json"
MAX_UNSCOPED_STATUS_BYTES = 64 * 1024

DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_EVENTS = 5000
MAX_PAGE_BYTES = 16 * 1024 * 1024
MAX_PAGE_EVENTS = 50_000
# A line longer than this (the recorder caps lines at 256 KB) is skipped as corrupt.
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_JSON_FILE_BYTES = 16 * 1024 * 1024
MAX_RUNNER_RECORDS_BYTES = 32 * 1024 * 1024
MAX_CORRUPT_OFFSETS = 100
MAX_FIRST_LINE_BYTES = 1024 * 1024
_FIRST_LINE_CHUNK = 16 * 1024
_SCAN_CHUNK = 1024 * 1024

# events "generation": the file's device and inode plus a hash of its whole first line.
# A file that is replaced (new inode) or rewritten in place (any byte of the first line
# changed) gets a new generation. The first line is hashed up to this many bytes, the
# longest line the reader serves (the recorder writes at most 256 KB per line); see
# events_generation for a longer one.
GENERATION_LINE_BYTES = MAX_LINE_BYTES
_GENERATION_CHUNK = 64 * 1024
# The generation reported while events.jsonl does not exist.
NO_FILE_GENERATION = "none"

# Episode fields copied into the summary (see the module docstring).
SUMMARY_FIELDS = (
    "source",
    "task",
    "condition",
    "condition_label",
    "seed",
    "attempt_index",
    "runner_invocation",
    "start_wall_time",
    "end_wall_time",
    "end_reason",
    "trace_complete",
    "gaps",
    "dropped_events",
    "late_events",
    "n_warnings",
    "viser_port",
    "pid",
    "host",
)
_START_FIELDS = (
    "source",
    "task",
    "condition",
    "condition_label",
    "seed",
    "attempt_index",
    "runner_invocation",
)
# Summary fields that the episode_start event also holds: when episode.json does not name
# one of them (see _has_field), the first line of events.jsonl is read.
_FIRST_EVENT_FIELDS = (*_START_FIELDS, "start_wall_time", "viser_port", "pid", "host")
_NESTED_KEYS = ("start", "episode_start", "end", "episode_end", "summary", "env")
# summary field -> other names the recorder may use for it
_ALIASES = {
    "start_wall_time": ("started_wall_time",),
    "end_wall_time": ("ended_wall_time",),
}
_NON_FINITE = {"NaN": "nan", "Infinity": "inf", "-Infinity": "-inf"}


# ---------------------------------------------------------------------------
# Ids and paths
# ---------------------------------------------------------------------------


def is_valid_id(value: object) -> bool:
    """True if ``value`` is an id of section 4.2 (also rejects "." and "..")."""
    return isinstance(value, str) and _ID_RE.fullmatch(value) is not None


def is_within(path: Path, root: Path) -> bool:
    """True if the resolved ``path`` lies under the resolved ``root`` (or is it)."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def safe_join(root: Path, *parts: str) -> Path | None:
    """Join id-checked parts below ``root``; None if a part is invalid or the result escapes.

    Every part must match the id pattern, so ``.``, ``..``, empty segments, slashes and
    control characters never reach the filesystem. The joined path is resolved (following
    symlinks) and must stay under ``root.resolve()``. Existence is not checked. For
    serving files use ``Dir``, which never follows a symlink and has no gap between the
    check and the open.
    """
    if not parts or not all(is_valid_id(p) for p in parts):
        return None
    try:
        base = root.resolve()
        target = base.joinpath(*parts).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not target.is_relative_to(base):
        return None
    return target


def split_relpath(relpath: str, *, max_len: int = 1024) -> list[str] | None:
    """Split a relative file path into id-checked segments, or None if any is invalid."""
    if not isinstance(relpath, str) or not relpath or len(relpath) > max_len:
        return None
    parts = relpath.split("/")
    if not all(is_valid_id(p) for p in parts):
        return None
    return parts


def regular_file(path: Path) -> os.stat_result | None:
    """stat() of ``path`` if it is a regular file (symlinks followed), else None."""
    try:
        st = path.stat()
    except (OSError, ValueError):
        return None
    return st if stat.S_ISREG(st.st_mode) else None


def is_dir(path: Path) -> bool:
    """True if ``path`` is a directory (symlinks followed); never raises."""
    try:
        return stat.S_ISDIR(path.stat().st_mode)
    except (OSError, ValueError):
        return False


def own_file(directory: Path, name: str) -> Path | None:
    """``directory/name`` resolved, if it is a regular file directly inside ``directory``.

    Used for the fixed file names (episode.json, events.jsonl, heartbeat, run.json,
    runner_records.jsonl): a symlink that points anywhere else is treated as missing.
    """
    try:
        base = directory.resolve()
        target = (directory / name).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if target.parent != base or regular_file(target) is None:
        return None
    return target


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Opening files below a root without following symlinks (section 9.2)
# ---------------------------------------------------------------------------

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a FIFO must not block the handler
_O_NOCTTY = getattr(os, "O_NOCTTY", 0)
_ROOT_FLAGS = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC
_SUBDIR_FLAGS = _ROOT_FLAGS | _O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC | _O_NONBLOCK | _O_NOCTTY
# Paths given by a local caller (validate_trace.py, tests): symlinks in them are followed.
_PATH_FILE_FLAGS = os.O_RDONLY | _O_CLOEXEC | _O_NONBLOCK | _O_NOCTTY


class NotRegularFileError(OSError):
    """The name exists but is not a regular file (a directory, FIFO, socket or device)."""


def _open_at(name: str, flags: int, dir_fd: int) -> int:
    """``os.open`` relative to ``dir_fd``. Every open below a root goes through here
    (tests replace it to swap a file for a symlink at an exact moment)."""
    return os.open(name, flags, dir_fd=dir_fd)


def _check_name(name: str) -> None:
    if not is_valid_id(name):
        raise FileNotFoundError(errno.ENOENT, "not a valid path segment", name)


def close_quietly(fd: int) -> None:
    """Close a file descriptor, ignoring errors."""
    try:
        os.close(fd)
    except OSError:
        pass


def pread_upto(fd: int, size: int, offset: int) -> bytes:
    """Up to ``size`` bytes at ``offset`` (fewer only at the end of the file)."""
    chunks: list[bytes] = []
    while size > 0:
        chunk = os.pread(fd, min(size, _SCAN_CHUNK), offset)
        if not chunk:
            break
        chunks.append(chunk)
        size -= len(chunk)
        offset += len(chunk)
    return b"".join(chunks)


class Dir:
    """An open directory below a trace root (or runner root), held as a file descriptor.

    ``Dir.open_root(root)`` opens the root itself (symlinks in the root's own path are
    followed: the root is configuration). ``subdir``, ``open_file`` and ``lstat`` then take
    one path segment at a time, relative to this descriptor, without following symlinks:
    a symlink named by a segment is refused (``ELOOP``/``ENOTDIR``), whatever it points
    to. A regular file is opened, then checked with ``fstat`` on the same descriptor,
    which the caller reads from. Use as a context manager, or call ``close``.
    """

    __slots__ = ("fd",)

    def __init__(self, fd: int) -> None:
        self.fd = fd

    @classmethod
    def open_root(cls, root: Path | str) -> Dir:
        return cls(os.open(os.fspath(root), _ROOT_FLAGS))

    @classmethod
    def open_below(cls, root: Path | str, parts: Sequence[str]) -> Dir:
        """``root/parts[0]/.../parts[-1]``, each segment opened with ``O_NOFOLLOW``."""
        current = cls.open_root(root)
        try:
            for part in parts:
                nxt = current.subdir(part)
                current.close()
                current = nxt
        except BaseException:
            current.close()
            raise
        return current

    def __enter__(self) -> Dir:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        fd, self.fd = self.fd, -1
        if fd >= 0:
            close_quietly(fd)

    def subdir(self, name: str) -> Dir:
        """The directory ``name`` directly inside this one (never through a symlink)."""
        _check_name(name)
        return Dir(_open_at(name, _SUBDIR_FLAGS, self.fd))

    def subdirs(self, parts: Sequence[str]) -> Dir:
        """``parts[0]/.../parts[-1]`` below this directory (this Dir stays open)."""
        current: Dir | None = None
        try:
            for part in parts:
                nxt = (current or self).subdir(part)
                if current is not None:
                    current.close()
                current = nxt
        except BaseException:
            if current is not None:
                current.close()
            raise
        if current is None:
            return Dir(os.dup(self.fd))
        return current

    def open_file(self, name: str) -> tuple[int, os.stat_result]:
        """(fd, fstat) of the regular file ``name`` directly inside this directory.

        Raises OSError: ``ELOOP`` for a symlink, ``ENOENT`` when missing,
        NotRegularFileError for a directory, FIFO, socket or device.
        """
        _check_name(name)
        fd = _open_at(name, _FILE_FLAGS, self.fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise NotRegularFileError(errno.EINVAL, "not a regular file", name)
        except BaseException:
            close_quietly(fd)
            raise
        return fd, st

    def open_file_below(self, parts: Sequence[str]) -> tuple[int, os.stat_result]:
        """(fd, fstat) of the regular file ``parts[0]/.../parts[-1]`` below this directory."""
        if not parts:
            raise FileNotFoundError(errno.ENOENT, "empty path")
        if len(parts) == 1:
            return self.open_file(parts[0])
        with self.subdirs(parts[:-1]) as parent:
            return parent.open_file(parts[-1])

    def lstat(self, name: str) -> os.stat_result | None:
        """lstat of ``name`` (a symlink is reported as a symlink), or None."""
        if not is_valid_id(name):
            return None
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except (OSError, ValueError):
            return None

    def file_stat(self, name: str) -> os.stat_result | None:
        """lstat of ``name`` if it is a regular file (not a symlink), else None."""
        st = self.lstat(name)
        return st if st is not None and stat.S_ISREG(st.st_mode) else None

    def has_subdir(self, name: str) -> bool:
        st = self.lstat(name)
        return st is not None and stat.S_ISDIR(st.st_mode)

    def is_symlink(self, name: str) -> bool:
        st = self.lstat(name)
        return st is not None and stat.S_ISLNK(st.st_mode)

    def names(self) -> list[str]:
        """Sorted entry names (an empty list when the directory cannot be listed)."""
        try:
            return sorted(os.listdir(self.fd))
        except OSError:
            return []


def is_symlink_refusal(exc: OSError) -> bool:
    """True when an open below a root failed because a segment is a symlink."""
    return exc.errno == errno.ELOOP


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


class EventsOffsetError(ValueError):
    """``after_offset`` is past the end of the file or not at the start of a line."""


@dataclass(frozen=True)
class EventsPage:
    """One page of ``events.jsonl`` starting at a line boundary.

    ``event_lines`` holds the JSON text of each valid event as UTF-8 bytes, in file order.
    Lines that parse are passed through unchanged, except that NaN and Infinity (not
    valid JSON) are rewritten as the strings "nan", "inf" and "-inf". ``generation``
    identifies the file (see ``events_generation``).
    """

    event_lines: tuple[bytes, ...]
    next_offset: int
    eof: bool
    torn_tail: bool
    corrupt_lines: int
    corrupt_offsets: tuple[int, ...]
    file_size: int
    stat_key: tuple[int, ...]
    generation: str = ""

    def events(self) -> list[dict[str, Any]]:
        """The events of this page as dicts."""
        return [json.loads(line) for line in self.event_lines]

    def to_json_bytes(self) -> bytes:
        """The ``/events`` response body: ``{events, next_offset, eof, torn_tail, corrupt_lines, ...}``."""
        meta = json.dumps(
            {
                "next_offset": self.next_offset,
                "eof": self.eof,
                "torn_tail": self.torn_tail,
                "corrupt_lines": self.corrupt_lines,
                "corrupt_offsets": list(self.corrupt_offsets),
                "file_size": self.file_size,
                "generation": self.generation,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return b'{"events":[' + b",".join(self.event_lines) + b"]," + meta[1:]


def _replace_constant(name: str) -> str:
    return _NON_FINITE.get(name, name)


def parse_event_line(line: bytes) -> bytes | None:
    """JSON text to serve for one ``events.jsonl`` line, or None if the line is corrupt.

    A line is valid when it is UTF-8 and parses to a JSON object.
    """
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError:
        return None
    constants: list[str] = []

    def on_constant(name: str) -> str:
        constants.append(name)
        return _replace_constant(name)

    try:
        obj = json.loads(text, parse_constant=on_constant)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict):
        return None
    if constants:
        return json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return line


def _stat_key(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def events_generation(fd: int, st: os.stat_result) -> str:
    """``<device>-<inode>-<hash>`` of an open events file.

    The hash covers the whole first line, its newline included, so a file rewritten in
    place is detected whichever byte of the first line changed. It is "pending" while
    the file holds no complete first line, so the generation changes once when that line
    is written. A replaced file (new inode) gets a new generation too.

    A first line longer than GENERATION_LINE_BYTES (the recorder never writes one; the
    reader skips it as corrupt) is not hashed whole. Its generation then also names the
    file's size and modification time (``<dev>-<inode>-<hash>-<size>-<mtime_ns>``), so
    it changes whenever the file changes: a client re-reads from the start instead of
    trusting an old offset. OSError propagates when the file cannot be read.
    """
    prefix = f"{st.st_dev:x}-{st.st_ino:x}"
    digest = hashlib.sha256()
    pos = 0
    while pos < GENERATION_LINE_BYTES:
        chunk = os.pread(fd, min(_GENERATION_CHUNK, GENERATION_LINE_BYTES - pos), pos)
        if not chunk:
            return f"{prefix}-pending"
        nl = chunk.find(b"\n")
        if nl != -1:
            digest.update(chunk[: nl + 1])
            return f"{prefix}-{digest.hexdigest()[:16]}"
        digest.update(chunk)
        pos += len(chunk)
    return f"{prefix}-{digest.hexdigest()[:16]}-{st.st_size:x}-{st.st_mtime_ns:x}"


def _open_path(path: Path | str) -> int:
    fd = os.open(os.fspath(path), _PATH_FILE_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NotRegularFileError(errno.EINVAL, "not a regular file", os.fspath(path))
    except BaseException:
        close_quietly(fd)
        raise
    return fd


def read_events_page(
    path: Path | int,
    after_offset: int = 0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> EventsPage:
    """Read complete event lines starting at byte ``after_offset``.

    ``path`` is a path (opened here; symlinks in it are followed) or an open file
    descriptor (read with ``pread``; its position is not used or changed).

    At most ``max_bytes`` bytes and ``max_events`` events are returned, except that a
    single line longer than ``max_bytes`` is still read whole (up to MAX_LINE_BYTES) so
    that paging always advances. ``next_offset`` is the offset after the last consumed
    line (valid or corrupt). A last line without a newline is left unconsumed and
    reported as ``torn_tail``; ``eof`` is true when no complete line remains after
    ``next_offset``.

    Raises EventsOffsetError when ``after_offset`` is negative, past the end of the
    file, or not preceded by a newline. OSError propagates when the file cannot be read.
    """
    if isinstance(path, int):
        return _read_events_page_fd(path, after_offset, max_bytes, max_events)
    fd = _open_path(path)
    try:
        return _read_events_page_fd(fd, after_offset, max_bytes, max_events)
    finally:
        close_quietly(fd)


def _read_events_page_fd(fd: int, after_offset: int, max_bytes: int, max_events: int) -> EventsPage:
    max_bytes = max(1, min(int(max_bytes), MAX_PAGE_BYTES))
    max_events = max(1, min(int(max_events), MAX_PAGE_EVENTS))
    after_offset = int(after_offset)
    if after_offset < 0:
        raise EventsOffsetError("after_offset must be >= 0")
    st = os.fstat(fd)
    size = st.st_size
    if after_offset > size:
        raise EventsOffsetError(f"after_offset {after_offset} is past the end of the file ({size} bytes)")
    if after_offset > 0 and os.pread(fd, 1, after_offset - 1) != b"\n":
        raise EventsOffsetError(f"after_offset {after_offset} is not at the start of a line")
    generation = events_generation(fd, st)
    buf = pread_upto(fd, min(max_bytes, size - after_offset), after_offset)
    lines: list[bytes] = []
    corrupt_offsets: list[int] = []
    n_corrupt = 0
    pos = 0
    torn = False

    def consume(line: bytes, offset: int) -> None:
        nonlocal n_corrupt
        parsed = parse_event_line(line)
        if parsed is None:
            n_corrupt += 1
            if len(corrupt_offsets) < MAX_CORRUPT_OFFSETS:
                corrupt_offsets.append(offset)
        else:
            lines.append(parsed)

    while len(lines) < max_events:
        nl = buf.find(b"\n", pos)
        if nl == -1:
            break
        consume(buf[pos:nl], after_offset + pos)
        pos = nl + 1
    if len(lines) < max_events and pos < len(buf):
        # The buffer ends inside a line (the loop found no further newline).
        if after_offset + len(buf) >= size:
            torn = True
        elif pos == 0:
            # One line longer than max_bytes: read it whole, or skip it if it is huge.
            pos, torn = _read_long_line(fd, after_offset, size, consume)
    next_offset = after_offset + pos
    # A torn tail runs to the end of the file, so nothing complete follows it.
    eof = torn or next_offset >= size
    return EventsPage(
        event_lines=tuple(lines),
        next_offset=next_offset,
        eof=eof,
        torn_tail=torn,
        corrupt_lines=n_corrupt,
        corrupt_offsets=tuple(corrupt_offsets),
        file_size=size,
        stat_key=_stat_key(st),
        generation=generation,
    )


def _read_long_line(fd: int, start: int, size: int, consume: Any) -> tuple[int, bool]:
    """Handle a line that starts at ``start`` and is longer than the page budget.

    Returns (consumed bytes relative to ``start``, torn_tail).
    """
    chunks: list[bytes] = []
    length = 0
    while True:
        want = min(_SCAN_CHUNK, max(0, size - start - length))
        chunk = os.pread(fd, want, start + length) if want else b""
        if not chunk:
            return 0, True  # reached the end of the file without a newline
        nl = chunk.find(b"\n")
        if nl != -1:
            length += nl
            if length <= MAX_LINE_BYTES:
                chunks.append(chunk[:nl])
                consume(b"".join(chunks), start)
            else:
                consume(b"", start)  # too long to serve: counted as corrupt
            return length + 1, False
        length += len(chunk)
        if length <= MAX_LINE_BYTES:
            chunks.append(chunk)
        else:
            chunks = []


class EventsPageCache:
    """LRU cache of event pages keyed by the file's (device, inode, size, mtime, ctime)
    and the query.

    An append changes the size and a replacement changes the inode, so a cached page is
    never served for a file that changed. A caller that has just computed the file's
    generation passes it: a cached page of another generation (the first line was
    rewritten in place within one timestamp tick) is read again. Thread-safe (FastAPI
    runs handlers in a thread pool).
    """

    def __init__(self, max_entries: int = 256, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._entries: OrderedDict[tuple[Any, ...], tuple[EventsPage, int]] = OrderedDict()
        self._bytes = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def get(
        self,
        path: Path | int,
        after_offset: int = 0,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_events: int = DEFAULT_MAX_EVENTS,
        *,
        generation: str | None = None,
    ) -> EventsPage:
        """A page of the file at ``path`` (or of the open file descriptor ``path``).

        ``generation``: the file's current generation, if the caller computed it; a cached
        page of another generation is not served.
        """
        if isinstance(path, int):
            return self._get_fd(path, after_offset, max_bytes, max_events, generation)
        fd = _open_path(path)
        try:
            return self._get_fd(fd, after_offset, max_bytes, max_events, generation)
        finally:
            close_quietly(fd)

    def _get_fd(
        self, fd: int, after_offset: int, max_bytes: int, max_events: int, generation: str | None
    ) -> EventsPage:
        st = os.fstat(fd)
        query = (int(after_offset), int(max_bytes), int(max_events))
        key = (*_stat_key(st), *query)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and (generation is None or hit[0].generation == generation):
                self._entries.move_to_end(key)
                return hit[0]
        page = _read_events_page_fd(fd, after_offset, max_bytes, max_events)
        cost = sum(len(line) for line in page.event_lines) + 256
        store_key = (*page.stat_key, *query)
        with self._lock:
            old = self._entries.pop(store_key, None)  # a page of an older generation
            if old is not None:
                self._bytes -= old[1]
            if cost <= self._max_bytes:
                self._entries[store_key] = (page, cost)
                self._bytes += cost
                while self._entries and (
                    len(self._entries) > self._max_entries or self._bytes > self._max_bytes
                ):
                    _, (_, old_cost) = self._entries.popitem(last=False)
                    self._bytes -= old_cost
        return page

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


def iter_events(path: Path | int, *, page_bytes: int = DEFAULT_MAX_BYTES) -> Iterator[dict[str, Any]]:
    """All valid events of a file in order (corrupt lines skipped, torn tail ignored)."""
    offset = 0
    while True:
        page = read_events_page(path, offset, page_bytes, MAX_PAGE_EVENTS)
        yield from page.events()
        if page.eof or page.next_offset == offset:
            return
        offset = page.next_offset


def _first_event_fd(fd: int) -> dict[str, Any] | None:
    buf = b""
    while len(buf) < MAX_FIRST_LINE_BYTES:
        try:
            chunk = os.pread(fd, min(_FIRST_LINE_CHUNK, MAX_FIRST_LINE_BYTES - len(buf)), len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        nl = chunk.find(b"\n")
        if nl != -1:
            parsed = parse_event_line(buf + chunk[:nl])
            return json.loads(parsed) if parsed is not None else None
        buf += chunk
    return None


def read_first_event(path: Path | int) -> dict[str, Any] | None:
    """The first line of ``events.jsonl`` as a dict, if it is complete, valid and <= 1 MB.

    Reads only up to the first newline (in 16 KB steps).
    """
    if isinstance(path, int):
        return _first_event_fd(path)
    try:
        fd = _open_path(path)
    except OSError:
        return None
    try:
        return _first_event_fd(fd)
    finally:
        close_quietly(fd)


def read_jsonl(path: Path | int, *, max_bytes: int = MAX_RUNNER_RECORDS_BYTES) -> dict[str, Any]:
    """Tolerant read of a small JSONL file (e.g. runner_records.jsonl).

    Returns ``{records, corrupt_lines, torn_tail, truncated}``; records beyond
    ``max_bytes`` are left out and ``truncated`` is set.
    """
    if not isinstance(path, int):
        fd = _open_path(path)
        try:
            return read_jsonl(fd, max_bytes=max_bytes)
        finally:
            close_quietly(fd)
    records: list[dict[str, Any]] = []
    corrupt = 0
    offset = 0
    torn = False
    truncated = False
    while True:
        page = read_events_page(path, offset, DEFAULT_MAX_BYTES, MAX_PAGE_EVENTS)
        records.extend(page.events())
        corrupt += page.corrupt_lines
        torn = page.torn_tail
        if page.eof or page.next_offset == offset:
            break
        offset = page.next_offset
        if offset >= max_bytes:
            truncated = True
            break
    return {"records": records, "corrupt_lines": corrupt, "torn_tail": torn, "truncated": truncated}


# ---------------------------------------------------------------------------
# JSON files (episode.json, run.json)
# ---------------------------------------------------------------------------


class _JsonFileCache:
    """Parsed small JSON files keyed by the file's (device, inode, size, mtime, ctime).

    Bounded by entry count and by the total size of the cached files.
    """

    def __init__(self, max_entries: int = 4096, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._entries: OrderedDict[tuple[Any, ...], tuple[tuple[Any, str | None], int]] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0
        self._lock = threading.Lock()

    def read(self, path: Path, max_bytes: int = MAX_JSON_FILE_BYTES) -> tuple[Any, str | None]:
        try:
            fd = _open_path(path)
        except (OSError, ValueError):
            return None, "missing"
        try:
            return self.read_fd(fd, os.fstat(fd), max_bytes)
        except OSError as exc:
            return None, f"unreadable ({errno_name(exc)})"
        finally:
            close_quietly(fd)

    def read_fd(self, fd: int, st: os.stat_result, max_bytes: int = MAX_JSON_FILE_BYTES) -> tuple[Any, str | None]:
        key = (*_stat_key(st), max_bytes)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit[0]
        result = _read_json_fd(fd, st, max_bytes)
        cost = st.st_size + 64
        if result[1] is not None and result[1].startswith("unreadable"):
            return result  # an I/O error may pass: it is not remembered for this file state
        with self._lock:
            if key not in self._entries and cost <= self._max_bytes:
                self._entries[key] = (result, cost)
                self._bytes += cost
                while self._entries and (
                    len(self._entries) > self._max_entries or self._bytes > self._max_bytes
                ):
                    _, (_, old_cost) = self._entries.popitem(last=False)
                    self._bytes -= old_cost
        return result


def _read_json_fd(fd: int, st: os.stat_result, max_bytes: int) -> tuple[Any, str | None]:
    if st.st_size > max_bytes:
        return None, f"too_large ({st.st_size} bytes)"
    try:
        data = pread_upto(fd, max_bytes + 1, 0)
    except OSError as exc:
        return None, f"unreadable ({errno_name(exc)})"
    if len(data) > max_bytes:
        return None, f"too_large ({len(data)} bytes)"
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_replace_constant), None
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, "invalid_json"


_json_cache = _JsonFileCache()


def read_json_file(path: Path, max_bytes: int = MAX_JSON_FILE_BYTES) -> tuple[Any, str | None]:
    """(parsed value, None) or (None, reason) for a small JSON file; cached by stat."""
    return _json_cache.read(path, max_bytes)


def errno_name(exc: OSError) -> str:
    """``EIO``, ``EACCES``, ... of an OSError (its class name when it has no errno)."""
    return errno.errorcode.get(exc.errno, type(exc).__name__) if exc.errno else type(exc).__name__


def open_failure_reason(exc: OSError) -> str:
    """Why ``Dir.open_file`` failed: "missing" only for a file that does not exist
    (ENOENT); "symlink_not_followed", "not_a_regular_file" or "unreadable (<errno>)"
    otherwise, so that a file that exists but cannot be read is never reported as absent."""
    if exc.errno == errno.ENOENT:
        return "missing"
    if is_symlink_refusal(exc):
        return "symlink_not_followed"
    if isinstance(exc, NotRegularFileError):
        return "not_a_regular_file"
    return f"unreadable ({errno_name(exc)})"


def read_json_at(
    directory: Dir, name: str, max_bytes: int = MAX_JSON_FILE_BYTES
) -> tuple[Any, str | None, os.stat_result | None]:
    """(value, None, stat) or (None, reason, stat or None) for ``name`` in ``directory``.

    Reasons for a file that cannot be opened come from ``open_failure_reason`` ("missing"
    only when there is no such file; a symlink is never followed).
    """
    try:
        fd, st = directory.open_file(name)
    except OSError as exc:
        return None, open_failure_reason(exc), None
    try:
        value, error = _json_cache.read_fd(fd, st, max_bytes)
    finally:
        close_quietly(fd)
    return value, error, st


def read_text_at(directory: Dir, name: str, limit: int) -> tuple[str, bool] | None:
    """(UTF-8 text with replacement characters, truncated) of a regular file, or None."""
    try:
        fd, _st = directory.open_file(name)
    except OSError:
        return None
    try:
        data = pread_upto(fd, limit + 1, 0)
    except OSError:
        return None
    finally:
        close_quietly(fd)
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


# ---------------------------------------------------------------------------
# Episodes and runs
# ---------------------------------------------------------------------------


def liveness_age(episode_dir: Path, now: float) -> tuple[float | None, str]:
    """Seconds since the heartbeat file was refreshed, and where that time came from.

    Only the ``heartbeat`` file counts ("heartbeat"); without it the liveness of the
    episode is unknown (None, "none"). File mtimes of events.jsonl or episode.json are not
    used: a copied or unpacked episode has fresh mtimes although nothing writes to it.
    """
    mtime = _mtime(own_file(episode_dir, HEARTBEAT_FILE))
    if mtime is not None:
        return now - mtime, "heartbeat"
    return None, "none"


def derive_status(episode_json: Any, liveness_age_s: float | None) -> tuple[str, str]:
    """Episode status of section 9.2 and a short reason.

    ``ended``: episode_end written and trace_complete true; ``ended_with_gaps``:
    episode_end written and trace_complete not true; ``running``: no episode_end and the
    heartbeat is younger than 30 s; ``incomplete``: no episode_end and a stale heartbeat,
    or no heartbeat file at all (liveness unknown).
    """
    if isinstance(episode_json, dict) and episode_json.get("status") in ("ended", "ended_with_gaps"):
        complete = episode_json.get("trace_complete")
        if complete is True:
            return "ended", "episode_end written; trace complete"
        if complete is None:
            return "ended_with_gaps", "episode_end written; trace_complete missing"
        return "ended_with_gaps", "episode_end written; trace has gaps"
    if liveness_age_s is None:
        return "incomplete", "no episode_end and no heartbeat file (liveness unknown)"
    if liveness_age_s < HEARTBEAT_STALE_S:
        return "running", "no episode_end; heartbeat file younger than 30 s"
    return "incomplete", "no episode_end; heartbeat file older than 30 s"


def _lookup(episode_json: Any, key: str) -> Any:
    if not isinstance(episode_json, dict):
        return None
    names = (key, *_ALIASES.get(key, ()))
    for name in names:
        if episode_json.get(name) is not None:
            return episode_json[name]
    for nested in _NESTED_KEYS:
        sub = episode_json.get(nested)
        if isinstance(sub, dict):
            for name in names:
                if sub.get(name) is not None:
                    return sub[name]
    return None


def _has_field(episode_json: Any, key: str) -> bool:
    """True if episode.json names ``key`` (or an alias of it), even with a null value.

    Looked up where ``_lookup`` looks: the top level and the nested dicts of _NESTED_KEYS.
    The recorder writes every start field, a null one too (a web UI episode has no seed,
    an env without a Viser server has ``env.viser_port: null``), so reading the
    ``episode_start`` event would give the same null.
    """
    if not isinstance(episode_json, dict):
        return False
    names = (key, *_ALIASES.get(key, ()))
    if any(name in episode_json for name in names):
        return True
    for nested in _NESTED_KEYS:
        sub = episode_json.get(nested)
        if isinstance(sub, dict) and any(name in sub for name in names):
            return True
    return False


def _from_start_event(event: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(event, dict) or event.get("event_type") != "episode_start":
        return None
    if key == "start_wall_time":
        return event.get("wall_time")
    if key == "pid":
        return event.get("pid")
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    if key == "viser_port":
        env = data.get("env")
        return env.get("viser_port") if isinstance(env, dict) else None
    return data.get(key)


def is_episode_dir(path: Path) -> bool:
    """A directory holding ``episode.json`` or ``events.jsonl`` as its own regular file."""
    return is_dir(path) and (
        own_file(path, EPISODE_FILE) is not None or own_file(path, EVENTS_FILE) is not None
    )


def is_episode_dir_at(directory: Dir) -> bool:
    """An open directory holding ``episode.json`` or ``events.jsonl`` as a regular file."""
    return directory.file_stat(EPISODE_FILE) is not None or directory.file_stat(EVENTS_FILE) is not None


def episode_summary(
    run_id: str, episode_dir: Path, *, episode_id: str | None = None, now: float | None = None
) -> dict[str, Any]:
    """Summary of one episode: status from episode.json and the heartbeat, plus its fields.

    ``episode_dir`` must already be validated by the caller; ``episode_id`` defaults to
    its name. See ``episode_summary_at``.
    """
    episode_id = episode_dir.name if episode_id is None else episode_id
    try:
        directory: Dir | None = Dir.open_root(episode_dir)
    except OSError:
        directory = None
    try:
        return episode_summary_at(run_id, episode_id, directory, now=now)
    finally:
        if directory is not None:
            directory.close()


def episode_summary_at(
    run_id: str,
    episode_id: str,
    directory: Dir | None,
    *,
    now: float | None = None,
    start_event: bool = True,
) -> dict[str, Any]:
    """Summary of the episode in the open directory ``directory`` (None: unreadable).

    ``events.jsonl`` is only stat'ed, except for its first line when ``start_event`` is
    true and episode.json does not name one of the fields the episode_start event also
    holds (``_has_field``). The status never depends on that line, so callers that need
    only the status pass ``start_event=False``. The fixed-name files are read only as
    regular files directly inside the directory: a symlink is never followed (for the
    status it counts as missing; ``episode_json_error`` says ``symlink_not_followed``).
    """
    now = time.time() if now is None else now
    episode_json: Any = None
    error: str | None = "missing"
    episode_st = heartbeat_st = events_st = None
    if directory is not None:
        episode_json, error, episode_st = read_json_at(directory, EPISODE_FILE)
        heartbeat_st = directory.file_stat(HEARTBEAT_FILE)
        events_st = directory.file_stat(EVENTS_FILE)
    if episode_json is not None and not isinstance(episode_json, dict):
        episode_json, error = None, "not_an_object"
    age = now - heartbeat_st.st_mtime if heartbeat_st is not None else None
    status, detail = derive_status(episode_json, age)
    first_event = None
    if (
        start_event
        and directory is not None
        and events_st is not None
        and not all(_has_field(episode_json, key) for key in _FIRST_EVENT_FIELDS)
    ):
        try:
            fd, _st = directory.open_file(EVENTS_FILE)
        except OSError:
            fd = None
        if fd is not None:
            try:
                first_event = _first_event_fd(fd)
            finally:
                close_quietly(fd)
    summary: dict[str, Any] = {"run_id": run_id, "episode_id": episode_id}
    for key in SUMMARY_FIELDS:
        value = _lookup(episode_json, key)
        if value is None:
            value = _from_start_event(first_event, key)
        summary[key] = value
    mtimes = [
        st.st_mtime for st in (events_st, episode_st, heartbeat_st) if st is not None
    ]
    summary.update(
        {
            "status": status,
            "status_detail": detail,
            "recorded_status": episode_json.get("status") if isinstance(episode_json, dict) else None,
            "heartbeat_age_s": round(age, 3) if age is not None else None,
            "liveness_source": "heartbeat" if heartbeat_st is not None else "none",
            "last_modified": max(mtimes) if mtimes else None,
            "events_bytes": events_st.st_size if events_st else None,
            "episode": episode_json,
            "episode_json_error": error,
        }
    )
    return summary


def _iter_episodes(run: Dir) -> Iterator[tuple[str, Dir]]:
    """(name, open directory) of each episode dir of a run; each Dir is closed after use."""
    for name in run.names():
        if not is_valid_id(name):
            continue
        try:
            directory = run.subdir(name)
        except OSError:
            continue  # a file, a symlink or gone
        try:
            if is_episode_dir_at(directory):
                yield name, directory
        finally:
            directory.close()


def _unscoped_logs(run: Dir) -> list[dict[str, Any]]:
    """The run's unscoped logs: ``{file, bytes, log_id, write_status}`` per log, sorted by
    file name.

    ``write_status`` is the recorder's ``unscoped.<pid>.status.json`` (lines of that log
    that could not be written; the trace recorder keeps it outside the log because
    the log itself failed), or None. A log that exists only as such a status file (its
    ``.jsonl`` is missing or not a regular file) is listed with ``bytes: null`` and
    ``log_id: null``, so a write failure is never hidden by the missing log. A status
    file that cannot be read gives ``write_status_error`` (reasons of
    ``read_json_at``).
    """
    names = run.names()
    logs: dict[str, dict[str, Any]] = {}
    for name in names:
        if name.startswith("unscoped.") and name.endswith(".jsonl") and is_valid_id(name):
            st = run.file_stat(name)
            if st is not None:
                log_id = name[: -len(".jsonl")]
                logs[log_id] = {
                    "file": name,
                    "bytes": st.st_size,
                    "log_id": log_id if is_unscoped_log_id(log_id) else None,
                    "write_status": None,
                }
    for name in names:
        if not (name.endswith(UNSCOPED_STATUS_SUFFIX) and is_valid_id(name)):
            continue
        log_id = name[: -len(UNSCOPED_STATUS_SUFFIX)]
        if not is_unscoped_log_id(log_id):
            continue
        value, error, _st = read_json_at(run, name, MAX_UNSCOPED_STATUS_BYTES)
        if error == "missing":
            continue  # removed since the listing
        entry = logs.get(log_id)
        if entry is None:
            entry = {"file": unscoped_log_file(log_id), "bytes": None, "log_id": None, "write_status": None}
            logs[log_id] = entry
        if error is None and not isinstance(value, dict):
            error = "not_an_object"
        if error is None:
            entry["write_status"] = value
        else:
            entry["write_status_error"] = error
    return [logs[key] for key in sorted(logs, key=lambda k: logs[k]["file"])]


def list_episode_dirs(trace_root: Path, run_id: str) -> list[Path]:
    """Episode directories of a run (ids checked, no symlinks followed below the root)."""
    if not is_valid_id(run_id):
        return []
    try:
        run = Dir.open_below(trace_root, [run_id])
    except OSError:
        return []
    with run:
        return [Path(trace_root) / run_id / name for name, _d in _iter_episodes(run)]


def list_run_ids(trace_root: Path) -> list[str]:
    """Run directories directly under the trace root (``_unscoped``, odd names and
    symlinks skipped)."""
    try:
        root = Dir.open_root(trace_root)
    except OSError:
        return []
    with root:
        return [name for name in root.names() if is_valid_id(name) and root.has_subdir(name)]


def run_overview(trace_root: Path, run_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """``{run_id, n_episodes, last_modified, status_counts}`` or None if it is not a run."""
    if not is_valid_id(run_id):
        return None
    try:
        run = Dir.open_below(trace_root, [run_id])
    except OSError:
        return None
    with run:
        return run_overview_at(run_id, run, now=now)


def run_overview_at(run_id: str, run: Dir, *, now: float | None = None) -> dict[str, Any] | None:
    """Overview of the run in the open directory ``run``, or None if it is not a run.

    A directory is a run when it holds run.json (readable or not), runner_records.jsonl,
    an episode directory or an unscoped log, the same rule as ``run_detail_at``.
    """
    now = time.time() if now is None else now
    counts = {s: 0 for s in STATUSES}
    run_st = run.file_stat(RUN_FILE)
    records_st = run.file_stat(RUNNER_RECORDS_FILE)
    mtimes = [st.st_mtime for st in (run_st, records_st) if st is not None]
    n_episodes = 0
    for name, directory in _iter_episodes(run):
        summary = episode_summary_at(run_id, name, directory, now=now, start_event=False)
        n_episodes += 1
        counts[summary["status"]] = counts.get(summary["status"], 0) + 1
        if summary["last_modified"] is not None:
            mtimes.append(summary["last_modified"])
    unscoped = _unscoped_logs(run)
    if not n_episodes and run_st is None and records_st is None and not unscoped:
        return None
    return {
        "run_id": run_id,
        "n_episodes": n_episodes,
        "last_modified": max(mtimes) if mtimes else None,
        "status_counts": counts,
    }


def list_runs(trace_root: Path, *, now: float | None = None) -> list[dict[str, Any]]:
    """Run overviews, newest first."""
    now = time.time() if now is None else now
    try:
        root = Dir.open_root(trace_root)
    except OSError:
        return []
    runs = []
    with root:
        for name in root.names():
            if not is_valid_id(name):
                continue
            try:
                run = root.subdir(name)
            except OSError:
                continue
            with run:
                overview = run_overview_at(name, run, now=now)
            if overview is not None:
                runs.append(overview)
    runs.sort(key=lambda r: (r["last_modified"] or 0.0, r["run_id"]), reverse=True)
    return runs


def run_detail(trace_root: Path, run_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """``{run_id, run, runner_records, episodes, ...}`` for one run, or None."""
    if not is_valid_id(run_id):
        return None
    try:
        run = Dir.open_below(trace_root, [run_id])
    except OSError:
        return None
    with run:
        return run_detail_at(run_id, run, now=now)


def _sort_time(summary: dict[str, Any]) -> int | float:
    """When an episode started, for ordering a run's episodes: ``start_wall_time`` if it is
    a finite number, else the newest file time, else 0. A damaged or foreign episode.json
    can hold a string, a list or NaN (read as "nan") there; one such value must not make
    the whole run fail to sort. The summary still reports the value as it was read."""
    for key in ("start_wall_time", "last_modified"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        return value
    return 0.0


def run_detail_at(run_id: str, run: Dir, *, now: float | None = None) -> dict[str, Any] | None:
    """Detail of the run in the open directory ``run``, or None if it is not a run.

    A run whose run.json is empty, damaged or unreadable is still returned, with ``run:
    null`` and ``run_json_error`` saying why (the run list shows it too). A
    runner_records.jsonl that exists but cannot be opened or read gives
    ``runner_records_info.error`` (``open_failure_reason``, e.g. ``unreadable (EACCES)`` or
    ``symlink_not_followed``) instead of looking like a run without runner records. What
    counts as a run is the rule of ``run_overview_at`` (regular files, readable or not).
    """
    now = time.time() if now is None else now
    run_json, run_error, _run_st = read_json_at(run, RUN_FILE)
    records: dict[str, Any] = {"records": [], "corrupt_lines": 0, "torn_tail": False, "truncated": False}
    try:
        fd, _records_st = run.open_file(RUNNER_RECORDS_FILE)
    except OSError as exc:
        fd = None
        if exc.errno != errno.ENOENT:
            records["error"] = open_failure_reason(exc)
    if fd is not None:
        try:
            records = read_jsonl(fd)
        except OSError as exc:
            records = {"records": [], "corrupt_lines": 0, "torn_tail": False, "truncated": False,
                       "error": open_failure_reason(exc)}
        finally:
            close_quietly(fd)
    episodes = [episode_summary_at(run_id, name, directory, now=now) for name, directory in _iter_episodes(run)]
    unscoped = _unscoped_logs(run)
    run_st, records_st = run.file_stat(RUN_FILE), run.file_stat(RUNNER_RECORDS_FILE)
    if not episodes and run_st is None and records_st is None and not unscoped:
        return None
    episodes.sort(key=lambda e: (_sort_time(e), e["episode_id"]))
    return {
        "run_id": run_id,
        "run": run_json,
        "run_json_error": None if run_error in (None, "missing") else run_error,
        "runner_records": records["records"],
        "runner_records_info": {k: v for k, v in records.items() if k != "records"},
        "episodes": episodes,
        "unscoped_files": unscoped,
    }


# ---------------------------------------------------------------------------
# Unscoped logs (events with no episode)
# ---------------------------------------------------------------------------


def is_unscoped_log_id(value: object) -> bool:
    """True for ``unscoped.<pid>``, the viewer's id of a run's unscoped log."""
    return isinstance(value, str) and _UNSCOPED_ID_RE.fullmatch(value) is not None


def unscoped_log_file(log_id: str) -> str:
    """File name of the unscoped log ``unscoped.<pid>`` inside its run dir."""
    return f"{log_id}.jsonl"


def unscoped_events_file(run_dir: Path, log_id: str) -> Path | None:
    """``<run>/unscoped.<pid>.jsonl`` as its own regular file, or None."""
    if not is_unscoped_log_id(log_id):
        return None
    return own_file(run_dir, unscoped_log_file(log_id))


def unscoped_files_dir(log_id: str) -> str:
    """Name of the directory beside the log that holds its payloads and assets."""
    return f"{log_id}.files"


def unscoped_summary(
    run_id: str, run_dir: Path, log_id: str, *, now: float | None = None
) -> dict[str, Any]:
    """Summary of a run's unscoped log, shaped like an episode summary (path version)."""
    try:
        directory: Dir | None = Dir.open_root(run_dir)
    except OSError:
        directory = None
    try:
        return unscoped_summary_at(run_id, directory, log_id, now=now)
    finally:
        if directory is not None:
            directory.close()


def unscoped_summary_at(
    run_id: str, run: Dir | None, log_id: str, *, now: float | None = None
) -> dict[str, Any]:
    """Summary of a run's unscoped log, shaped like an episode summary.

    ``status`` is "unscoped": the log belongs to no episode, so it has no
    ``episode_end`` and no heartbeat, and the episode statuses do not apply.
    """
    now = time.time() if now is None else now
    events_st = None
    if run is not None and is_unscoped_log_id(log_id):
        events_st = run.file_stat(unscoped_log_file(log_id))
    summary: dict[str, Any] = {"run_id": run_id, "episode_id": log_id}
    for key in SUMMARY_FIELDS:
        summary[key] = None
    mtime = events_st.st_mtime if events_st else None
    summary.update(
        {
            "kind": "unscoped",
            "status": UNSCOPED_STATUS,
            "status_detail": (
                "events that belong to no episode (runner preflight requests, events "
                "that could not be attributed to one episode)"
            ),
            "recorded_status": None,
            "heartbeat_age_s": round(now - mtime, 3) if mtime is not None else None,
            "liveness_source": "file_mtime" if mtime is not None else "none",
            "last_modified": mtime,
            "events_bytes": events_st.st_size if events_st else None,
            "pid": int(log_id.split(".", 1)[1]),
            "episode": None,
            "episode_json_error": None,
            "files_dir": unscoped_files_dir(log_id),
        }
    )
    return summary
