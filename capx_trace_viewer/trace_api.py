"""HTTP API of the experiment trace viewer (docs/trace-format.md sections 9.1, 9.2, 9.4).

``create_trace_router()`` returns the ``/api/traces`` router. An embedding server can
include it before its SPA catch-all; the standalone server includes it in viewer-only mode.

Rules this module keeps (sections 2 and 9.2):

- Handlers are plain ``def`` functions (FastAPI runs them in its thread pool). They never
  print, warn or log: in the full web UI they share the process with a running trial,
  whose ``exec`` redirects ``sys.stdout``/``sys.stderr`` for the whole process into the
  model's next input. Array work (capx_trace_viewer/trace_arrays.py) avoids every numpy warning
  path, and at most ARRAY_WORKERS array requests run at once, so a live trial in the same
  process is not starved.
- Every id matches ``^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$`` and so does every segment of a
  file path, so ``.``, ``..``, empty segments and a decoded ``%2F`` never reach the
  filesystem. Files are opened through ``capx_trace_viewer.reader.Dir``: the trace root (or
  runner root) is opened once and every path segment below it is opened relative to its
  parent's descriptor with ``O_NOFOLLOW``, so no symlink below the root is followed (one
  that stays inside is refused too). A file is checked with ``fstat`` on the descriptor
  it is then served from, so swapping the name for a symlink after the check changes
  nothing that is sent.
- Files are served with ``X-Content-Type-Options: nosniff``, ``Content-Security-Policy:
  sandbox`` and a MIME type from an allowlist; anything else downloads as
  ``application/octet-stream``. ``*.txt.gz``/``*.json.gz`` are sent with
  ``Content-Encoding: gzip`` so the browser decodes them. A single byte range
  (``Range: bytes=a-b``) is honoured, so videos can seek; a malformed ``Range`` header
  (non-ASCII digits, any number of digits) is ignored. A response part of at most
  FILE_SNAPSHOT_BYTES is read completely before its headers are sent, so it always has
  the length it announces, even when the file changes meanwhile. A larger part is sent
  without ``Content-Length``, so if the file becomes shorter while it is sent, the body
  ends early and the response is still complete (see OpenFileResponse).
- Unless the router is created with ``expose=True`` (``--expose-traces``), a request is
  answered only when (1) the TCP peer is a loopback address, (2) the ``Host`` header names
  this machine (``localhost``, ``127.0.0.0/8``, ``[::1]``, or the unspecified address
  ``0.0.0.0``/``[::]``, which a browser reaches only on this machine; any port, so an SSH
  port forward to another local port still works), and (3) an ``Origin`` header, if
  present, names such a host. (2) stops DNS-rebinding pages, whose requests reach the
  server from 127.0.0.1 with the attacker's host name. A reverse proxy on the same host
  looks like a loopback client: it must send ``X-Forwarded-For`` from 127.0.0.1 (uvicorn
  trusts it from there and then sees the real client) or the server must not be
  reachable through it. An embedding server's ``/viser-proxy`` should refuse ``api/`` paths and
  send ``X-Forwarded-For``; web-ui/vite.config.ts sets ``xfwd``.
- JSON answers are ASCII (``ensure_ascii=True``): an escaped lone surrogate in a recorded
  string is passed on as ``\\udcXX`` instead of failing the whole answer.

Responses not spelled out in section 9.2:

- ``/events?after_offset=&max_bytes=&max_events=&generation=`` returns ``{events,
  next_offset, eof, torn_tail, corrupt_lines, corrupt_offsets, file_size, generation}``;
  ``corrupt_lines`` counts the lines skipped in this page. ``generation`` identifies the
  file (device, inode and a hash of its whole first line, see
  ``reader.events_generation``; ``"none"`` while events.jsonl does not exist). When the
  request's ``generation`` differs from the file's, the answer is HTTP 409 ``{"error":
  "generation_changed", "generation": "<new>"}`` and the client starts again from offset
  0. An ``after_offset`` past the end of the file or not at a line start gives HTTP 409
  with ``detail``. Only a missing file (``ENOENT``) is "not written yet"; an
  events.jsonl that exists but cannot be opened or read is an error the viewer shows:
  403 for a symlink or a permission error, 500 for anything else (not a regular file,
  an I/O error), with the reason in ``detail``.
- ``/array/...`` returns ``{arrays: {name: {shape, dtype, stats, preview, preview_shape,
  truncated, is_mask, packed, image}}}`` (see capx_trace_viewer/trace_arrays.py). ``stats`` is
  always null: statistics were recorded in the event at capture time. ``image`` is
  ``{kind, count, height, width}`` or null and says how ``/array-image`` draws the array
  (``count`` > 1: ``index`` picks one image of a stack). Without ``?name=`` every array of
  the file is listed from its header and previews stop after a byte budget
  (``preview_omitted: "listing_budget"``); with ``?name=`` only that array is read. A
  bit-packed mask (npz members ``<name>__packbits`` and ``<name>__shape``) is one bool
  array ``<name>`` with ``packed: true``; ``?name=<name>__packbits`` (the summary's
  ``npz_key``) selects it too. When all array slots are busy for ARRAY_WAIT_S seconds
  the answer is HTTP 503.
- A run's unscoped log (``<run>/unscoped.<pid>.jsonl``, events with no episode, section
  4.1) is served through the episode endpoints under the episode id ``unscoped.<pid>``:
  the summary has ``kind: "unscoped"`` and ``status: "unscoped"``, ``/events`` pages the
  log, and ``/file``, ``/array`` and ``/array-image`` serve only paths below
  ``unscoped.<pid>.files/`` (the paths in its events are relative to the run dir).
  ``/runs/{run_id}`` lists these logs in ``unscoped_files`` (``{file, bytes, log_id,
  write_status}``; ``write_status`` is the recorder's ``unscoped.<pid>.status.json``, the
  lines of that log that could not be written, and a log that exists only as that file
  is listed with ``bytes`` and ``log_id`` null; see ``reader._unscoped_logs``).
  ``/array-image/...?name=&index=`` renders a mask, a depth map, an RGB image or one
  image of a stack as PNG.
- ``/runs/{run_id}`` also answers for a run whose run.json is empty, damaged or
  unreadable: ``run`` is null and ``run_json_error`` says why. A runner_records.jsonl
  that exists but cannot be read gives ``runner_records_info.error``.
- A run, episode or file that exists but cannot be opened is never answered as missing:
  404 means it is not there (or is not a regular file), 403 a symlink or a permission
  error, 500 anything else (an I/O error, too many open files) with the errno name in
  ``detail``.
- ``/viser`` returns ``{available, port, reason}``; with ``--viewer-only`` it always
  answers ``available: false`` because no Viser proxy routes exist. The probe is given
  this server's own port, so a web UI on a Viser port does not report itself.
- ``/legacy/runs`` returns ``{roots: [{index, path, exists}], runs: [{run_id,
  root_index, created_at, description, manifest_path, n_trials, last_modified}]}``.
- ``/legacy/runs/{run_id}?root=`` returns ``{run_id, root_index, run, trials: [{task,
  condition, trial_dir, trial, sandbox_rc, reward, task_completed, record}],
  records_info}``; ``record`` is a compact copy of the matching trials.jsonl record.
- ``/legacy/trials/{run_id}/{task}/{condition}/{trial_dir}?root=`` returns ``{run_id,
  task, condition, trial_dir, root_index, model_input_recorded, code,
  summary, trial_record, videos, harness_images, harness_prompts, other_files}``. File
  entries are ``{path, bytes, mime}`` (relative to the trial dir) and can be fetched from
  ``/legacy/file/{run_id}/{task}/{condition}/{trial_dir}/{path}?root=``.

Related modules: capx_trace_viewer/reader.py (``Dir``, paging, status),
capx_trace_viewer/trace_arrays.py (array previews and images), capx_trace_viewer/server.py
(includes this router), web-ui/src/trace/api.ts (client), scripts/make_demo_trace.py
(synthetic data), tests/test_trace_api.py (endpoints, gating, generations) and
tests/test_trace_api_files.py (symlink swap races, byte ranges, the array limit).
"""

from __future__ import annotations

import errno
import hashlib
import ipaddress
import json
import os
import re
import stat
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from email.utils import formatdate
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send

from capx_trace_viewer import reader
from capx_trace_viewer import trace_arrays
from capx_trace_viewer.trace_arrays import ArrayError, NameRequired, finite_json

# ---------------------------------------------------------------------------
# Limits and constants
# ---------------------------------------------------------------------------

MAX_TEXT_BYTES = 1024 * 1024
MAX_LEGACY_FILES = 500
# A file response part up to this size is read completely before its headers are sent
# (OpenFileResponse); larger parts are streamed without Content-Length. Every trace asset
# the viewer loads (payloads, text, images, state snapshots, VDM videos) is far smaller.
FILE_SNAPSHOT_BYTES = 16 * 1024 * 1024
# Reads of a part that keeps changing before the answer is 503.
FILE_SNAPSHOT_ATTEMPTS = 3
# Array requests (/array, /array-image) running at once per router, and how long a
# request waits for a slot before it is answered with 503.
ARRAY_WORKERS = 2
ARRAY_WAIT_S = 20.0

_JSON_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
# no-cache: browsers must revalidate instead of guessing freshness from Last-Modified, so a
# live episode.json or events.jsonl fetched through /file is never served stale.
_FILE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox",
    "Cache-Control": "no-cache",
}

# Suffix -> MIME type of files the viewer displays. Everything else is a download.
MIME_TYPES = {
    ".json": "application/json",
    ".jsonl": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".sh": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".csv": "text/plain; charset=utf-8",
    ".yaml": "text/plain; charset=utf-8",
    ".yml": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
# Compressed text sent with Content-Encoding: gzip (the browser decodes it).
GZIP_TEXT_TYPES = {
    ".txt.gz": "text/plain; charset=utf-8",
    ".json.gz": "application/json",
    ".jsonl.gz": "text/plain; charset=utf-8",
}

_TRIAL_DIR_RE = re.compile(
    r"trial_(?P<trial>\d+)_sandboxrc_(?P<rc>-?\d+)_reward_(?P<reward>[^_]+)_taskcompleted_(?P<tc>[01])"
)
_RECORD_KEYS = (
    "trial",
    "seed",
    "model",
    "vdm_model",
    "multi_turn",
    "outcome",
    "reward",
    "sandbox_rc",
    "num_code_blocks",
    "num_regenerations",
    "num_finishes",
    "duration_s",
    "num_attempts",
)

LEGACY_IMAGE_LABEL = {
    "label": "harness_saved_image",
    "label_ja": "ハーネス保存画像（送信を検証していない）",
}
LEGACY_PROMPT_LABEL = {
    "label": "harness_saved_prompt_text",
    "label_ja": "ハーネスが保存したプロンプト文字列（送信payloadではない）",
}

_TRACE_ROOT = "trace root"
_RUNNER_ROOT = "runner root"


# ---------------------------------------------------------------------------
# Small helpers (no printing, no warnings)
# ---------------------------------------------------------------------------


def _ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None


def _is_loopback_ip(text: str) -> bool:
    """A loopback address: 127.0.0.0/8, ::1 or ::ffff:127.x.y.z (used for the TCP peer)."""
    ip = _ip(text)
    if ip is None:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def client_is_loopback(request: Request) -> bool:
    """True if the TCP peer is a loopback address (127.0.0.0/8, ::1, ::ffff:127.x)."""
    client = request.client
    if client is None or not client.host:
        return False
    return _is_loopback_ip(client.host)


def _is_local_ip(text: str) -> bool:
    """An address in a Host or Origin header that can only mean this machine: a loopback
    address, or the unspecified address 0.0.0.0 / :: (a browser that opens
    http://0.0.0.0:<port>, the address the full web UI prints, reaches this machine)."""
    if _is_loopback_ip(text):
        return True
    ip = _ip(text)
    if ip is None:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_unspecified or bool(mapped is not None and mapped.is_unspecified)


def _is_local_name(name: str) -> bool:
    name = name.strip().lower()
    if name.endswith("."):
        name = name[:-1]
    return name == "localhost" or _is_local_ip(name)


def host_is_loopback(host: str | None) -> bool:
    """True if a ``Host`` header names this machine: ``localhost``, an address in
    127.0.0.0/8, ``[::1]``, ``[::ffff:127.x.y.z]`` or the unspecified address
    ``0.0.0.0``/``[::]``, with or without a port. A DNS-rebinding page sends its own
    host name, which is none of these."""
    if not host:
        return False
    host = host.strip()
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return False
        name, rest = host[1:end], host[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return False
        return _is_local_ip(name)
    if host.count(":") == 1:
        name, _, port = host.partition(":")
        if not port.isdigit():
            return False
    else:
        name = host  # no port (or a bare IPv6 address, which Host should bracket)
    return _is_local_name(name)


def origin_is_loopback(origin: str) -> bool:
    """True if an ``Origin`` header names this machine as ``host_is_loopback`` does (or is
    the opaque ``null``, whose page can read nothing: the CORS policy allows no such
    origin)."""
    origin = origin.strip()
    if origin == "null":
        return True
    try:
        hostname = urlsplit(origin).hostname
    except ValueError:
        return False
    return bool(hostname) and _is_local_name(hostname)


def json_response(content: Any, status_code: int = 200) -> Response:
    """JSON response without FastAPI's encoder (faster for large lists); never NaN.

    ASCII only: a lone surrogate in a recorded string (valid as a ``\\udcXX`` escape in
    the recorder's ASCII JSON) cannot make the encoding fail.
    """
    try:
        body = json.dumps(content, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except ValueError:
        body = json.dumps(finite_json(content), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return Response(
        body.encode("ascii"),
        status_code=status_code,
        media_type="application/json",
        headers=_JSON_HEADERS,
    )


def mime_for(name: str) -> tuple[str, bool, bool]:
    """(media type, gzip content-encoding, download) for a file name."""
    lower = name.lower()
    for suffix, mime in GZIP_TEXT_TYPES.items():
        if lower.endswith(suffix):
            return mime, True, False
    suffix = os.path.splitext(lower)[1]
    mime = MIME_TYPES.get(suffix)
    if mime is None:
        return "application/octet-stream", False, True
    return mime, False, False


_RANGE_NUMBER_RE = re.compile(r"[0-9]+")
# A range position with more significant digits than this is larger than any file; it is
# never converted (int() refuses more than 4300 digits and would be slow before that).
_MAX_RANGE_DIGITS = 18
_HUGE_POSITION = 10**_MAX_RANGE_DIGITS


def _range_number(text: str) -> int | None:
    """One side of a byte range: ASCII digits only, else None. ``str.isdigit`` would also
    accept superscripts such as ``²`` (Starlette decodes header values as Latin-1), which
    int() refuses. More than _MAX_RANGE_DIGITS significant digits give _HUGE_POSITION."""
    if _RANGE_NUMBER_RE.fullmatch(text) is None:
        return None
    digits = text.lstrip("0") or "0"
    return _HUGE_POSITION if len(digits) > _MAX_RANGE_DIGITS else int(digits)


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | str | None:
    """``(start, end)`` (end exclusive) of a single ``bytes=`` range, ``"unsatisfiable"``,
    or None when the header is absent, malformed or not a single byte range (the whole
    file is sent, as RFC 9110 lets a server ignore an invalid Range)."""
    if not value:
        return None
    unit, sep, spec = value.partition("=")
    if not sep or unit.strip().lower() != "bytes" or "," in spec:
        return None
    first, dash, last = spec.strip().partition("-")
    if not dash:
        return None
    first, last = first.strip(), last.strip()
    if first == "":
        suffix = _range_number(last)
        if suffix is None:
            return None
        if suffix == 0:
            return "unsatisfiable"
        start, end = max(0, size - suffix), size
    else:
        start = _range_number(first)
        if start is None:
            return None
        if last:
            last_pos = _range_number(last)
            if last_pos is None or last_pos < start:
                return None
            end = min(size, last_pos + 1)
        else:
            end = size
    if start >= size or start >= end:
        return "unsatisfiable"
    return start, end


def _validators(st: os.stat_result) -> tuple[str, str]:
    """(ETag, Last-Modified) of a file state."""
    etag_base = f"{st.st_mtime}-{st.st_size}".encode()
    etag = f'"{hashlib.md5(etag_base, usedforsecurity=False).hexdigest()}"'
    return etag, formatdate(st.st_mtime, usegmt=True)


class OpenFileResponse(Response):
    """Sends a regular file that is already open: the descriptor that was checked.

    Nothing is opened by name after the check, so a name swapped for a symlink after it
    changes nothing that is sent. One byte range is honoured (``bytes=a-b``, ``bytes=a-``,
    ``bytes=-n``, with ``If-Range``); other or malformed ``Range`` headers get the whole
    file. A ``HEAD`` request (the routes here accept only GET) would get the headers
    alone. The descriptor is closed once the response is sent, or when the response is
    discarded unsent.

    Length. A part of at most FILE_SNAPSHOT_BYTES (every trace asset the viewer loads) is
    read completely before its headers are sent, so the response always carries exactly
    the bytes it announces: the server never logs an incomplete response and the client
    never sees a short body. If the read comes up short (the file was truncated after
    the check), the file is checked again and the answer is built for its current size;
    after FILE_SNAPSHOT_ATTEMPTS short reads the answer is 503 with ``Retry-After``.

    Larger parts (200 or 206) are streamed up to the ``fstat`` size without a
    ``Content-Length`` header. The server then marks the end of the body itself: with
    chunked transfer coding for HTTP/1.1, by closing the connection for HTTP/1.0. If the
    file becomes shorter while it is sent, the body ends at the file's new end and the
    response is still completed, so the server logs nothing. Nothing is padded; a 206
    answer then carries fewer bytes than its ``Content-Range`` names. Trace files are
    replaced by rename, not truncated, so this needs a file over 16 MB that another
    program truncates during the transfer. A ``HEAD`` answer keeps the ``Content-Length``
    of the part, since it has no body that could end early.
    """

    chunk_size = 256 * 1024

    def __init__(self, fd: int, st: os.stat_result, filename: str) -> None:
        mime, gz, download = mime_for(filename)
        headers = dict(_FILE_HEADERS)
        if gz:
            headers["Content-Encoding"] = "gzip"
        if download:
            headers["Content-Disposition"] = "attachment"
        headers["Accept-Ranges"] = "bytes"
        self._fd = fd
        self._closer = weakref.finalize(self, reader.close_quietly, fd)
        self._st = st
        self.status_code = 200
        self.background = None
        self.media_type = mime
        self.init_headers(headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._send(scope, send)
        finally:
            self._closer()
        if self.background is not None:
            await self.background()

    async def _send(self, scope: Scope, send: Send) -> None:
        request_headers = Headers(scope=scope)
        range_header = request_headers.get("range")
        if_range = request_headers.get("if-range")
        head = str(scope.get("method", "GET")).upper() == "HEAD"
        st = self._st
        for _attempt in range(FILE_SNAPSHOT_ATTEMPTS):
            size = st.st_size
            etag, last_modified = _validators(st)
            byte_range = parse_byte_range(range_header, size)
            if if_range is not None and if_range not in (etag, last_modified):
                byte_range = None
            headers = [
                *self.raw_headers,
                (b"last-modified", last_modified.encode("latin-1")),
                (b"etag", etag.encode("latin-1")),
            ]
            if byte_range == "unsatisfiable":
                headers += [(b"content-range", f"bytes */{size}".encode("latin-1")), (b"content-length", b"0")]
                await send({"type": "http.response.start", "status": 416, "headers": headers})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            start, end, status = 0, size, 200
            if isinstance(byte_range, tuple):
                start, end = byte_range
                status = 206
                headers.append((b"content-range", f"bytes {start}-{end - 1}/{size}".encode("latin-1")))
            if end - start > FILE_SNAPSHOT_BYTES and not head:
                # No Content-Length, so the body can end early (see the class docstring).
                await send({"type": "http.response.start", "status": status, "headers": headers})
                await self._stream(send, start, end)
                return
            headers.append((b"content-length", str(end - start).encode("latin-1")))
            if head or start >= end:
                await send({"type": "http.response.start", "status": status, "headers": headers})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            data = await anyio.to_thread.run_sync(reader.pread_upto, self._fd, end - start, start)
            if len(data) == end - start:
                await send({"type": "http.response.start", "status": status, "headers": headers})
                await send({"type": "http.response.body", "body": data, "more_body": False})
                return
            # Shorter than announced: the file was truncated after the check. Nothing was
            # sent yet, so the answer is built again for the file as it is now.
            st = await anyio.to_thread.run_sync(os.fstat, self._fd)
        body = b"the file changed while it was read; try again\n"
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                    (b"retry-after", b"1"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _stream(self, send: Send, start: int, end: int) -> None:
        """Sends bytes ``start`` to ``end`` in ``chunk_size`` pieces, as the body of a
        response started without ``Content-Length``. The last message has ``more_body:
        False`` also when the file became shorter, so the response is completed. Only an
        error raised by a read (an I/O error) leaves it open: uvicorn then logs the error
        and closes the connection, and the client sees a broken transfer."""
        pos = start
        while pos < end:
            chunk = await anyio.to_thread.run_sync(os.pread, self._fd, min(self.chunk_size, end - pos), pos)
            if not chunk:
                # The file became shorter while it was sent: the body ends at its new end.
                break
            pos += len(chunk)
            await send({"type": "http.response.body", "body": chunk, "more_body": pos < end})
        if pos < end:
            await send({"type": "http.response.body", "body": b"", "more_body": False})


def _symlink_refused(root_name: str) -> HTTPException:
    return HTTPException(status_code=403, detail=f"symlinks below the {root_name} are not followed")


# Open failures that mean "there is no such file or directory here": the name is missing
# (ENOENT), a path segment is not a directory (ENOTDIR), or the name is a socket or device
# (ENXIO, ENODEV) or another non-regular file (reader.NotRegularFileError).
_NOT_THERE = (errno.ENOENT, errno.ENOTDIR, errno.ENXIO, errno.ENODEV)


def _open_error(exc: OSError, what: str, root_name: str) -> HTTPException:
    """HTTP answer for a file or directory below a root that could not be opened: 403 for
    a symlink or a permission error, 404 when it is not there (``_NOT_THERE``), and 500
    with the errno name for anything else (an I/O error, too many open files), so that an
    unreadable file is never reported as a missing one (section 2: gaps are never hidden)."""
    if reader.is_symlink_refusal(exc):
        return _symlink_refused(root_name)
    if exc.errno in (errno.EACCES, errno.EPERM):
        return HTTPException(status_code=403, detail="permission denied")
    if exc.errno in _NOT_THERE or isinstance(exc, reader.NotRegularFileError):
        return HTTPException(status_code=404, detail=f"{what} not found")
    return HTTPException(status_code=500, detail=f"{what} cannot be read ({reader.errno_name(exc)})")


def _events_read_error(exc: OSError, name: str) -> HTTPException:
    """An events file that exists but cannot be opened or read. Only ENOENT means "not
    written yet"; everything else is shown to the viewer as an error, never as an empty
    page (section 2: gaps are never hidden)."""
    if reader.is_symlink_refusal(exc):
        return HTTPException(status_code=403, detail=f"{name} is a symlink; symlinks below the trace root are not followed")
    if exc.errno in (errno.EACCES, errno.EPERM):
        return HTTPException(status_code=403, detail=f"{name} cannot be read: permission denied")
    if isinstance(exc, reader.NotRegularFileError):
        return HTTPException(status_code=500, detail=f"{name} is not a regular file")
    return HTTPException(status_code=500, detail=f"{name} cannot be read ({reader.errno_name(exc)})")


def server_port(scope: Scope) -> int | None:
    """The port this server accepted the request on (ASGI ``server``), if known."""
    server = scope.get("server")
    if isinstance(server, (tuple, list)) and len(server) == 2 and isinstance(server[1], int):
        return server[1]
    return None


def _subdir(parent: reader.Dir, name: str, what: str, root_name: str) -> reader.Dir:
    try:
        return parent.subdir(name)
    except OSError as exc:
        # O_DIRECTORY|O_NOFOLLOW on a symlink to a directory gives ENOTDIR on Linux.
        if parent.is_symlink(name):
            raise _symlink_refused(root_name) from None
        raise _open_error(exc, what, root_name) from None


def _open_file_below(base: reader.Dir, parts: Sequence[str], root_name: str) -> tuple[int, os.stat_result]:
    """(fd, fstat) of the regular file ``parts`` below ``base`` (HTTPException otherwise)."""
    opened: list[reader.Dir] = []
    try:
        current = base
        for part in parts[:-1]:
            current = _subdir(current, part, "file", root_name)
            opened.append(current)
        try:
            return current.open_file(parts[-1])
        except OSError as exc:
            raise _open_error(exc, "file", root_name) from None
    finally:
        for directory in opened:
            directory.close()


class _EpisodeTarget:
    """An open run dir plus either an open episode dir or the id of the run's unscoped log."""

    def __init__(self, run: reader.Dir, episode: reader.Dir | None, log_id: str | None) -> None:
        self.run = run
        self.episode = episode
        self.log_id = log_id

    def __enter__(self) -> _EpisodeTarget:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.episode is not None:
            self.episode.close()
        self.run.close()


# ---------------------------------------------------------------------------
# Legacy runner outputs (section 9.4)
# ---------------------------------------------------------------------------


def parse_trial_dir_name(name: str) -> dict[str, Any]:
    """trial number, sandbox_rc, reward and task_completed encoded in a trial dir name."""
    match = _TRIAL_DIR_RE.fullmatch(name)
    if match is None:
        return {"trial": None, "sandbox_rc": None, "reward": None, "task_completed": None}
    try:
        reward: Any = finite_json(float(match.group("reward")))
    except ValueError:
        reward = None
    return {
        "trial": int(match.group("trial")),
        "sandbox_rc": int(match.group("rc")),
        "reward": reward,
        "task_completed": match.group("tc") == "1",
    }


def parse_summary_txt(text: str) -> dict[str, Any]:
    """Fields of the summary.txt written by the runner that wrote the trace."""
    out: dict[str, Any] = {"parsed": False, "source": "summary.txt"}
    marker = text.find("Environment response:")
    program_at = text.find("Generated program:\n")
    if marker == -1:
        return out
    if program_at != -1 and program_at < marker:
        out["program"] = text[program_at + len("Generated program:\n") : marker].strip("\n")
    rest = text[marker:]
    so = rest.find("\n  Stdout: ")
    reward_at = rest.rfind("\n  Reward: ")
    if so == -1 or reward_at == -1 or reward_at < so:
        return out
    body = rest[so + len("\n  Stdout: ") : reward_at]
    split = body.find("\n  Stderr: ")
    if split == -1:
        return out
    out["stdout"] = body[:split]
    out["stderr"] = body[split + len("\n  Stderr: ") :]
    out["ambiguous_split"] = body.count("\n  Stderr: ") > 1
    tail = rest[reward_at:]
    patterns = {
        "sandbox_rc": (r"\n  Sandbox failed: (-?\d+)", int, rest[:so + 1]),
        "reward": (r"\n  Reward: ([^\n]*)", float, tail),
        "task_completed": (r"\n  Task Completed: ([^\n]*)", lambda v: v.strip() == "True", tail),
        "terminated": (r"\n  Terminated: (\w+)", lambda v: v == "True", tail),
        "truncated": (r"Truncated: (\w+)", lambda v: v == "True", tail),
        "num_regenerations": (r"\n  Num Regenerations: (\d+)", int, tail),
        "num_finishes": (r"\n  Num Finishes: (\d+)", int, tail),
        "num_code_blocks": (r"\n  Num Code Blocks: (\d+)", int, tail),
    }
    for key, (pattern, convert, where) in patterns.items():
        match = re.search(pattern, where)
        if match is not None:
            try:
                out[key] = finite_json(convert(match.group(1)))
            except ValueError:
                out[key] = None
    out["parsed"] = True
    return out


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {k: record.get(k) for k in _RECORD_KEYS if k in record}
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        out["attempt_statuses"] = [a.get("status") for a in attempts if isinstance(a, dict)]
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def create_trace_router(
    trace_root: Path,
    *,
    viewer_only: bool,
    expose: bool,
    runner_roots: Sequence[Path] = (),
    viser_probe: Callable[[int | None], int | None] | None = None,
) -> APIRouter:
    """Router for ``/api/traces`` (section 9.2) and the legacy views (section 9.4).

    Args:
        trace_root: Trace root to serve (need not exist yet).
        viewer_only: Set by ``--viewer-only``; reported by ``/config`` and ``/viser``.
        expose: Answer non-loopback clients too (``--expose-traces``).
        runner_roots: Experiment-runner output roots for the legacy views.
        viser_probe: Called with this server's own port (or None); returns the port the
            server's Viser proxy uses, or None. Full UI only.
    """
    base_dir = Path(trace_root)
    roots = [Path(p) for p in runner_roots]
    cache = reader.EventsPageCache()
    array_slots = threading.BoundedSemaphore(ARRAY_WORKERS)

    def gate(request: Request) -> None:
        if expose:
            return
        if not client_is_loopback(request):
            raise HTTPException(
                status_code=403,
                detail=(
                    "the trace API answers loopback clients only "
                    "(start the server with --expose-traces to change this)"
                ),
            )
        if not host_is_loopback(request.headers.get("host")):
            raise HTTPException(
                status_code=403,
                detail=(
                    "the trace API answers loopback requests only: the Host header must be "
                    "localhost, 127.0.0.1, [::1] or 0.0.0.0 (start the server with "
                    "--expose-traces to change this)"
                ),
            )
        origin = request.headers.get("origin")
        if origin is not None and not origin_is_loopback(origin):
            raise HTTPException(
                status_code=403,
                detail="the trace API answers loopback pages only: cross-origin requests are refused",
            )

    router = APIRouter(prefix="/api/traces", dependencies=[Depends(gate)])

    @contextmanager
    def array_slot() -> Iterator[None]:
        if not array_slots.acquire(timeout=ARRAY_WAIT_S):
            raise HTTPException(
                status_code=503, detail="array previews are busy; try again", headers={"Retry-After": "1"}
            )
        try:
            yield
        finally:
            array_slots.release()

    # -- opening run, episode and file (fd based, no symlink followed) -----------
    def open_run(run_id: str) -> reader.Dir:
        if not reader.is_valid_id(run_id):
            raise HTTPException(status_code=400, detail="invalid run id")
        try:
            root = reader.Dir.open_root(base_dir)
        except OSError:
            raise HTTPException(status_code=404, detail="run not found") from None
        with root:
            return _subdir(root, run_id, "run", _TRACE_ROOT)

    def open_episode_or_log(run_id: str, episode_id: str) -> _EpisodeTarget:
        """The episode dir, or the run's unscoped log for ``unscoped.<pid>``.

        ``unscoped.<pid>`` stands for ``<run>/unscoped.<pid>.jsonl``, the events that
        belong to no episode. A real episode directory of that name would win.
        """
        run = open_run(run_id)
        try:
            if not reader.is_valid_id(episode_id):
                raise HTTPException(status_code=400, detail="invalid episode id")
            episode: reader.Dir | None = None
            open_failure: OSError | None = None
            try:
                episode = run.subdir(episode_id)
            except OSError as exc:
                open_failure = exc
            if episode is not None:
                if reader.is_episode_dir_at(episode):
                    return _EpisodeTarget(run, episode, None)
                episode.close()
            if reader.is_unscoped_log_id(episode_id):
                if run.file_stat(reader.unscoped_log_file(episode_id)) is not None:
                    return _EpisodeTarget(run, None, episode_id)
                raise HTTPException(status_code=404, detail="unscoped log not found")
            if run.is_symlink(episode_id):
                raise _symlink_refused(_TRACE_ROOT)
            if open_failure is not None and open_failure.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                # A directory that exists but cannot be opened (permission, I/O error) is
                # reported as such, not as a missing episode.
                raise _open_error(open_failure, "episode", _TRACE_ROOT)
            raise HTTPException(status_code=404, detail="episode not found")
        except BaseException:
            run.close()
            raise

    def open_episode_file(run_id: str, episode_id: str, relpath: str) -> tuple[int, os.stat_result, str]:
        """(fd, fstat, file name) of a file below the episode dir (or the unscoped files dir)."""
        parts = reader.split_relpath(relpath)
        with open_episode_or_log(run_id, episode_id) as target:
            if target.episode is None:
                # Paths in an unscoped log are relative to the run dir and lie under
                # unscoped.<pid>.files/; nothing else in the run dir is served this way.
                if parts is None or len(parts) < 2 or parts[0] != reader.unscoped_files_dir(episode_id):
                    raise HTTPException(status_code=400, detail="invalid file path")
                base = target.run
            else:
                if parts is None:
                    raise HTTPException(status_code=400, detail="invalid file path")
                base = target.episode
            fd, st = _open_file_below(base, parts, _TRACE_ROOT)
        return fd, st, parts[-1]

    def open_legacy_run(run_id: str, root_index: int | None) -> tuple[int, reader.Dir]:
        if not reader.is_valid_id(run_id):
            raise HTTPException(status_code=400, detail="invalid run id")
        if root_index is not None and not 0 <= root_index < len(roots):
            raise HTTPException(status_code=400, detail="invalid runner root index")
        candidates = range(len(roots)) if root_index is None else [root_index]
        refused = False
        for i in candidates:
            try:
                root = reader.Dir.open_root(roots[i])
            except OSError:
                continue
            with root:
                try:
                    run = root.subdir(run_id)
                except OSError:
                    refused = refused or root.is_symlink(run_id)
                    continue
            if _is_runner_run(run):
                return i, run
            run.close()
        if refused:
            raise _symlink_refused(_RUNNER_ROOT)
        raise HTTPException(status_code=404, detail="runner run not found")

    def open_legacy_trial(
        run_id: str, task: str, condition: str, trial_dir: str, root_index: int | None
    ) -> tuple[int, reader.Dir]:
        i, run = open_legacy_run(run_id, root_index)
        opened: list[reader.Dir] = []
        try:
            for part in (task, condition, trial_dir):
                if not reader.is_valid_id(part):
                    raise HTTPException(status_code=400, detail="invalid path segment")
            current = run
            for part in (task, condition, trial_dir):
                current = _subdir(current, part, "trial", _RUNNER_ROOT)
                opened.append(current)
        except BaseException:
            for directory in opened:
                directory.close()
            raise
        finally:
            run.close()
        for directory in opened[:-1]:
            directory.close()
        return i, opened[-1]

    # -- 9.2 ----------------------------------------------------------------
    @router.get("/config")
    def get_config() -> Response:
        try:
            resolved = str(base_dir.resolve())
        except (OSError, RuntimeError):
            resolved = str(base_dir)
        return json_response(
            {
                "trace_root": resolved,
                "exists": reader.is_dir(base_dir),
                "viewer_only": viewer_only,
                "runner_roots": [str(p) for p in roots],
                "expose": expose,
                "schema_version": reader.SCHEMA_VERSION,
                "heartbeat_stale_s": reader.HEARTBEAT_STALE_S,
            }
        )

    @router.get("/runs")
    def get_runs() -> Response:
        return json_response({"runs": reader.list_runs(base_dir)})

    @router.get("/runs/{run_id}")
    def get_run(run_id: str) -> Response:
        with open_run(run_id) as run:
            detail = reader.run_detail_at(run_id, run)
        if detail is None:
            raise HTTPException(status_code=404, detail="run not found")
        return json_response(detail)

    @router.get("/runs/{run_id}/episodes/{episode_id}")
    def get_episode(run_id: str, episode_id: str) -> Response:
        with open_episode_or_log(run_id, episode_id) as target:
            if target.episode is None:
                summary = reader.unscoped_summary_at(run_id, target.run, episode_id)
            else:
                summary = reader.episode_summary_at(run_id, episode_id, target.episode)
        return json_response(summary)

    @router.get("/runs/{run_id}/episodes/{episode_id}/events")
    def get_events(
        run_id: str,
        episode_id: str,
        after_offset: int = Query(0, ge=0),
        max_bytes: int = Query(reader.DEFAULT_MAX_BYTES, ge=1),
        max_events: int = Query(reader.DEFAULT_MAX_EVENTS, ge=1),
        generation: str | None = Query(None, max_length=200),
    ) -> Response:
        with open_episode_or_log(run_id, episode_id) as target:
            opened: tuple[int, os.stat_result] | None
            if target.episode is not None:
                name, directory = reader.EVENTS_FILE, target.episode
            else:
                name, directory = reader.unscoped_log_file(episode_id), target.run
            try:
                opened = directory.open_file(name)
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise _events_read_error(exc, name) from None
                if target.episode is None:
                    raise HTTPException(status_code=404, detail="unscoped log not found") from None
                opened = None  # the episode has not written its first event yet
        if opened is None:
            current = reader.NO_FILE_GENERATION
            if generation is not None and generation != current:
                return json_response({"error": "generation_changed", "generation": current}, status_code=409)
            if after_offset:
                raise HTTPException(status_code=409, detail="events.jsonl does not exist yet")
            empty = {"events": [], "next_offset": 0, "eof": True, "torn_tail": False,
                     "corrupt_lines": 0, "corrupt_offsets": [], "file_size": 0, "generation": current}
            return json_response(empty)
        fd, st = opened
        try:
            try:
                current = reader.events_generation(fd, st)
                if generation is not None and generation != current:
                    return json_response({"error": "generation_changed", "generation": current}, status_code=409)
                page = cache.get(fd, after_offset, max_bytes, max_events, generation=current)
            except reader.EventsOffsetError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from None
            except OSError as exc:
                raise _events_read_error(exc, name) from None
        finally:
            reader.close_quietly(fd)
        return Response(page.to_json_bytes(), media_type="application/json", headers=_JSON_HEADERS)

    @router.get("/runs/{run_id}/episodes/{episode_id}/file/{relpath:path}")
    def get_file(run_id: str, episode_id: str, relpath: str) -> Response:
        fd, st, name = open_episode_file(run_id, episode_id, relpath)
        return OpenFileResponse(fd, st, name)

    @router.get("/runs/{run_id}/episodes/{episode_id}/array/{relpath:path}")
    def get_array(run_id: str, episode_id: str, relpath: str, name: str | None = None) -> Response:
        fd, st, filename = open_episode_file(run_id, episode_id, relpath)
        try:
            with array_slot():
                arrays = trace_arrays.describe_file(fd, st, filename, name)
        except ArrayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except (OSError, ValueError, MemoryError):
            raise HTTPException(status_code=400, detail="cannot read the array file") from None
        finally:
            reader.close_quietly(fd)
        if arrays is None:
            raise HTTPException(status_code=404, detail="no array with that name")
        return json_response({"arrays": arrays})

    @router.get("/runs/{run_id}/episodes/{episode_id}/array-image/{relpath:path}")
    def get_array_image(
        run_id: str,
        episode_id: str,
        relpath: str,
        name: str | None = None,
        index: int = Query(0, ge=0),
    ) -> Response:
        fd, st, filename = open_episode_file(run_id, episode_id, relpath)
        try:
            with array_slot():
                data = trace_arrays.render_file_png(fd, st, filename, name, index)
        except (NameRequired, ArrayError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except (OSError, ValueError, MemoryError):
            raise HTTPException(status_code=400, detail="cannot read the array file") from None
        finally:
            reader.close_quietly(fd)
        if data is None:
            raise HTTPException(status_code=404, detail="no array with that name")
        return Response(data, media_type="image/png", headers=dict(_FILE_HEADERS))

    @router.get("/viser")
    def get_viser(request: Request) -> Response:
        if viewer_only or viser_probe is None:
            return json_response({"available": False, "port": None, "reason": "viewer_only"})
        own_port = server_port(request.scope)
        try:
            port = viser_probe(own_port)
        except Exception:  # noqa: BLE001 - a probe failure means "not available"
            port = None
        if port is not None and port == own_port:
            port = None  # this server itself, not a Viser server
        return json_response(
            {"available": port is not None, "port": port, "reason": None if port else "not_running"}
        )

    # -- 9.4 legacy runner outputs --------------------------------------------
    @router.get("/legacy/runs")
    def get_legacy_runs() -> Response:
        runs = []
        root_info = []
        for i, base in enumerate(roots):
            try:
                root = reader.Dir.open_root(base)
            except OSError:
                root_info.append({"index": i, "path": str(base), "exists": False})
                continue
            root_info.append({"index": i, "path": str(base), "exists": True})
            with root:
                for name in root.names():
                    if not reader.is_valid_id(name):
                        continue
                    try:
                        run = root.subdir(name)
                    except OSError:
                        continue
                    with run:
                        if _is_runner_run(run):
                            runs.append(_legacy_run_overview(i, name, run))
        runs.sort(key=lambda r: (r.get("last_modified") or 0.0, r["run_id"]), reverse=True)
        return json_response({"roots": root_info, "runs": runs})

    @router.get("/legacy/runs/{run_id}")
    def get_legacy_run(run_id: str, root: int | None = Query(None, ge=0)) -> Response:
        i, run = open_legacy_run(run_id, root)
        with run:
            run_json = reader.read_json_at(run, "run.json")[0]
            records, info = _legacy_records(run)
            trial_dirs = _legacy_trial_dirs(run)
        by_trial: dict[tuple[str, str, str], dict[str, Any]] = {}
        for rec in records:
            artifacts = rec.get("artifacts")
            trial_path = artifacts.get("trial_dir") if isinstance(artifacts, dict) else None
            if isinstance(trial_path, str):
                key = (str(rec.get("task")), str(rec.get("condition")), trial_path.rstrip("/").rsplit("/", 1)[-1])
                by_trial[key] = rec
        trials = []
        for task, condition, trial_dir in trial_dirs:
            rec = by_trial.get((task, condition, trial_dir))
            trials.append(
                {
                    "task": task,
                    "condition": condition,
                    "trial_dir": trial_dir,
                    **parse_trial_dir_name(trial_dir),
                    "record": compact_record(rec) if rec else None,
                }
            )
        return json_response(
            {"run_id": run_id, "root_index": i, "run": run_json, "trials": trials, "records_info": info}
        )

    @router.get("/legacy/trials/{run_id}/{task}/{condition}/{trial_dir}")
    def get_legacy_trial(
        run_id: str, task: str, condition: str, trial_dir: str, root: int | None = Query(None, ge=0)
    ) -> Response:
        i, trial = open_legacy_trial(run_id, task, condition, trial_dir, root)
        code = None
        summary: dict[str, Any] | None = None
        trial_record = None
        with trial:
            files = _list_trial_files(trial)
            by_name = {f["path"]: f for f in files}
            if "code.py" in by_name:
                text = reader.read_text_at(trial, "code.py", MAX_TEXT_BYTES)
                if text is not None:
                    code = {"path": "code.py", "text": text[0], "truncated": text[1]}
            if "summary.txt" in by_name:
                text = reader.read_text_at(trial, "summary.txt", 4 * MAX_TEXT_BYTES)
                if text is not None:
                    summary = parse_summary_txt(text[0])
                    summary["truncated"] = text[1]
                    if not summary["parsed"]:
                        summary["raw"] = text[0][:20000]
            if "trial_record.json" in by_name:
                trial_record = reader.read_json_at(trial, "trial_record.json")[0]
        videos, images, prompts, other = [], [], [], []
        for f in files:
            name = f["path"].rsplit("/", 1)[-1]
            if f["mime"].startswith("video/"):
                videos.append(f)
            elif name.startswith("visual_feedback_") and name.endswith(".png"):
                images.append({**f, **LEGACY_IMAGE_LABEL})
            elif f["path"].startswith("prompts_and_responses/"):
                prompts.append({**f, **LEGACY_PROMPT_LABEL})
            elif f["path"] not in ("code.py", "summary.txt", "trial_record.json"):
                other.append(f)
        return json_response(
            {
                "run_id": run_id,
                "task": task,
                "condition": condition,
                "trial_dir": trial_dir,
                "root_index": i,
                **parse_trial_dir_name(trial_dir),
                "model_input_recorded": any(f["path"].startswith("codegen_turn_") and f["path"].endswith("_messages.json") for f in files),
                "code": code,
                "summary": summary,
                "trial_record": trial_record,
                "videos": videos,
                "harness_images": images,
                "harness_prompts": prompts,
                "other_files": other,
            }
        )

    @router.get("/legacy/file/{run_id}/{task}/{condition}/{trial_dir}/{relpath:path}")
    def get_legacy_file(
        run_id: str,
        task: str,
        condition: str,
        trial_dir: str,
        relpath: str,
        root: int | None = Query(None, ge=0),
    ) -> Response:
        _i, trial = open_legacy_trial(run_id, task, condition, trial_dir, root)
        parts = reader.split_relpath(relpath)
        with trial:
            if parts is None:
                raise HTTPException(status_code=400, detail="invalid file path")
            fd, st = _open_file_below(trial, parts, _RUNNER_ROOT)
        return OpenFileResponse(fd, st, parts[-1])

    # -- anything else under /api/traces is a JSON 404, never the SPA page ----
    all_methods = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]

    @router.api_route("", methods=all_methods, include_in_schema=False)
    @router.api_route("/{rest:path}", methods=all_methods, include_in_schema=False)
    def not_found(rest: str = "") -> Response:
        raise HTTPException(status_code=404, detail="unknown trace API path")

    return router


def _is_runner_run(run: reader.Dir) -> bool:
    return run.file_stat("run.json") is not None or run.file_stat("trials.jsonl") is not None


def _legacy_records(run: reader.Dir) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        fd, _st = run.open_file("trials.jsonl")
    except OSError:
        return [], {"corrupt_lines": 0, "torn_tail": False, "truncated": False, "exists": False}
    try:
        result = reader.read_jsonl(fd)
    except OSError:
        info = {"corrupt_lines": 0, "torn_tail": False, "truncated": False, "exists": True}
        return [], {**info, "error": "unreadable"}
    finally:
        reader.close_quietly(fd)
    info = {k: v for k, v in result.items() if k != "records"}
    info["exists"] = True
    return result["records"], info


def _legacy_trial_dirs(run: reader.Dir) -> list[tuple[str, str, str]]:
    """(task, condition, trial dir) of every trial below an open runner run dir."""
    out = []
    for task in run.names():
        if not reader.is_valid_id(task):
            continue
        try:
            task_dir = run.subdir(task)
        except OSError:
            continue
        with task_dir:
            for condition in task_dir.names():
                if not reader.is_valid_id(condition):
                    continue
                try:
                    cond_dir = task_dir.subdir(condition)
                except OSError:
                    continue
                with cond_dir:
                    for trial in cond_dir.names():
                        if trial.startswith("trial_") and reader.is_valid_id(trial) and cond_dir.has_subdir(trial):
                            out.append((task, condition, trial))
    return out


def _legacy_run_overview(index: int, run_id: str, run: reader.Dir) -> dict[str, Any]:
    run_json = reader.read_json_at(run, "run.json")[0]
    manifest = run_json.get("manifest") if isinstance(run_json, dict) else None
    mtimes = [st.st_mtime for st in (run.file_stat("run.json"), run.file_stat("trials.jsonl")) if st is not None]
    return {
        "run_id": run_id,
        "root_index": index,
        "created_at": run_json.get("created_at") if isinstance(run_json, dict) else None,
        "description": manifest.get("description") if isinstance(manifest, dict) else None,
        "manifest_path": run_json.get("manifest_path") if isinstance(run_json, dict) else None,
        "n_trials": len(_legacy_trial_dirs(run)),
        "last_modified": max(mtimes) if mtimes else None,
    }


def _list_trial_files(trial: reader.Dir) -> list[dict[str, Any]]:
    """Regular files of a trial dir and its direct subdirectories (symlinks skipped)."""
    out: list[dict[str, Any]] = []

    def visit(directory: reader.Dir, prefix: str, depth: int) -> None:
        for name in directory.names():
            if len(out) >= MAX_LEGACY_FILES or not reader.is_valid_id(name):
                continue
            st = directory.lstat(name)
            if st is None:
                continue
            rel = f"{prefix}{name}"
            if stat.S_ISREG(st.st_mode):
                out.append({"path": rel, "bytes": st.st_size, "mime": mime_for(name)[0]})
            elif stat.S_ISDIR(st.st_mode) and depth < 1:
                try:
                    sub = directory.subdir(name)
                except OSError:
                    continue
                with sub:
                    visit(sub, rel + "/", depth + 1)

    visit(trial, "", 0)
    return out
