"""How capx_trace_viewer/trace_api.py opens and serves files (docs/trace-format.md section 9.2).

A file below the trace root (or a runner root) is opened one path segment at a time from
the root's descriptor with ``O_NOFOLLOW`` and served from the descriptor that ``fstat``
checked (capx_trace_viewer.reader.Dir, trace_api.OpenFileResponse). These tests swap a name
for a symlink to a file outside the root at exact moments: right before a segment is
opened (``reader._open_at`` is the hook every such open goes through) and right after the
file was opened. Before: the open is refused (403). After: the response carries the bytes
of the file that was checked. A threaded test swaps continuously while requests run.

Also here: byte ranges of the streamed file responses, the limit on concurrent array
requests (``ARRAY_WORKERS``, HTTP 503 after ``ARRAY_WAIT_S``), and a check that no
request leaves a file descriptor open, on success and error paths alike.

A malformed ``Range`` header (Latin-1 superscript digits, thousands of digits)
is ignored or answered 416, never 500; a file truncated after the check is
answered completely for its current size, and a real uvicorn server logs nothing for it; a symlinked events.jsonl is an explicit 403, not an empty page.

For streaming tests, a part over FILE_SNAPSHOT_BYTES is sent without
``Content-Length``, so a file truncated after the check or while it is sent ends the body
early and the response still completes. ``serve_with_uvicorn_cycle`` runs such a request
in uvicorn's own h11 ``RequestResponseCycle`` with a fake transport (no server, no port)
and checks what uvicorn logged and what a client parses from the bytes written.
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h11
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image
from trace_viewer_testlib import LOOPBACK_BASE_URL, LOOPBACK_HOST, raw_get
from uvicorn.protocols.http.flow_control import FlowControl
from uvicorn.protocols.http.h11_impl import RequestResponseCycle

from capx_trace_viewer import reader
from capx_trace_viewer import trace_api, trace_arrays
from capx_trace_viewer.server import create_app

SECRET = "TOP-SECRET-OUTSIDE"
SECRET_NUMBER = 424242
LOCAL = ("127.0.0.1", 50000)
TRIAL = "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
INSIDE_ARRAY = np.arange(12, dtype=np.int64).reshape(3, 4)
EP_URL = "/api/traces/runs/r1/episodes/ep1"
LEGACY_URL = f"/api/traces/legacy/file/run1/task/cond/{TRIAL}"


def test_episode_video_directory_serves_mp4_without_escape(swap_env: dict[str, Any]) -> None:
    videos = swap_env["ep"] / "videos"
    videos.mkdir()
    (videos / "video_turn_00.mp4").write_bytes(b"portable mp4")
    response = swap_env["client"].get(f"{EP_URL}/file/videos/video_turn_00.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"portable mp4"
    (videos / "video_turn_01.mp4").symlink_to(swap_env["outside"] / "x.txt")
    escaped = swap_env["client"].get(f"{EP_URL}/file/videos/video_turn_01.mp4")
    assert escaped.status_code == 403
    assert SECRET.encode() not in escaped.content


@pytest.fixture
def swap_env(tmp_path: Path) -> dict[str, Any]:
    """A trace root and a runner root, each with a twin outside the root holding secrets."""
    root = tmp_path / "traces"
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True, "task": "inside_task"}))
    (ep / "events.jsonl").write_text(json.dumps({"seq": 0, "event_id": "ev_inside", "event_type": "episode_start", "data": {"task": "inside_task"}}) + "\n")
    (ep / "assets" / "x.txt").write_text("inside text")
    (ep / "assets" / "r.txt").write_bytes(b"0123456789abcdef")
    np.save(ep / "assets" / "x.npy", INSIDE_ARRAY)

    outside = tmp_path / "outside"  # the same names with secret contents
    (outside / "run" / "ep1" / "assets").mkdir(parents=True)
    for directory in (outside, outside / "run" / "ep1" / "assets"):
        (directory / "x.txt").write_text(SECRET)
        np.save(directory / "x.npy", np.full((3, 4), SECRET_NUMBER, dtype=np.int64))
    (outside / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True, "task": SECRET}))
    (outside / "events.jsonl").write_text(json.dumps({"seq": 0, "event_id": "ev_secret", "event_type": "episode_start", "data": {"task": SECRET}}) + "\n")
    (outside / "code.py").write_text(SECRET)
    for name in ("episode.json", "events.jsonl"):
        (outside / "run" / "ep1" / name).write_text((outside / name).read_text())

    legacy = tmp_path / "runner"
    trial = legacy / "run1" / "task" / "cond" / TRIAL
    trial.mkdir(parents=True)
    (legacy / "run1" / "run.json").write_text(json.dumps({"run_id": "run1"}))
    (trial / "code.py").write_text("print('inside')\n")

    app = create_app(viewer_only=True, trace_root=root, runner_roots=[legacy], frontend_dist=tmp_path / "no_dist")
    return {
        "root": root,
        "ep": ep,
        "trial": trial,
        "outside": outside,
        "client": TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL),
    }


def swap_for_symlink(path: Path, target: Path) -> None:
    """Replace ``path`` (a file or a directory) by a symlink to ``target``."""
    link = path.with_name(path.name + ".swap")
    link.symlink_to(target)
    if path.is_dir() and not path.is_symlink():
        path.rename(path.with_name(path.name + ".orig"))
    os.replace(link, path)


def hook_open(monkeypatch: pytest.MonkeyPatch, name: str, *, before: Callable[[], None] | None = None,
              after: Callable[[], None] | None = None) -> list[str]:
    """Run ``before``/``after`` around the first open of the path segment ``name``."""
    real = reader._open_at
    fired: list[str] = []

    def hooked(segment: str, flags: int, dir_fd: int) -> int:
        if segment != name or fired:
            return real(segment, flags, dir_fd)
        fired.append(segment)
        if before is not None:
            before()
        fd = real(segment, flags, dir_fd)
        if after is not None:
            after()
        return fd

    monkeypatch.setattr(reader, "_open_at", hooked)
    return fired


def assert_no_secret(resp: Any) -> None:
    assert SECRET.encode() not in resp.content and str(SECRET_NUMBER).encode() not in resp.content
    if resp.headers.get("content-type") == "image/png":
        assert np.asarray(Image.open(io.BytesIO(resp.content))).any()  # the secret array draws all black


# ---------------------------------------------------------------------------
# a name swapped for a symlink right before it is opened: refused
# ---------------------------------------------------------------------------

SWAP_BEFORE_CASES = [
    # (url, segment swapped, path relative to the fixture dir, symlink target relative to outside/)
    (f"{EP_URL}/file/assets/x.txt", "x.txt", "ep/assets/x.txt", "x.txt"),
    (f"{EP_URL}/array/assets/x.npy", "x.npy", "ep/assets/x.npy", "x.npy"),
    (f"{EP_URL}/array-image/assets/x.npy", "x.npy", "ep/assets/x.npy", "x.npy"),
    (f"{EP_URL}/file/assets/x.txt", "assets", "ep/assets", "run/ep1/assets"),
    (f"{EP_URL}/array/assets/x.npy", "assets", "ep/assets", "run/ep1/assets"),
    (f"{EP_URL}/file/assets/x.txt", "ep1", "ep", "run/ep1"),
    (f"{EP_URL}/file/assets/x.txt", "r1", "root/r1", "run"),
    (f"{LEGACY_URL}/code.py", "code.py", "trial/code.py", "code.py"),
    (f"{LEGACY_URL}/code.py", TRIAL, "trial", "."),  # outside/ holds a code.py too
]


def _fixture_path(env: dict[str, Any], rel: str) -> Path:
    head, _, tail = rel.partition("/")
    return env[head] / tail if tail else env[head]


@pytest.mark.parametrize(("url", "segment", "swapped", "target"), SWAP_BEFORE_CASES)
def test_a_name_swapped_before_its_open_is_refused(
    swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, url: str, segment: str, swapped: str, target: str
) -> None:
    fired = hook_open(
        monkeypatch, segment,
        before=lambda: swap_for_symlink(_fixture_path(swap_env, swapped), swap_env["outside"] / target),
    )
    resp = swap_env["client"].get(url)
    assert fired == [segment], "the swap never happened"
    assert resp.status_code in (403, 404), (url, resp.status_code)
    assert_no_secret(resp)
    if resp.status_code == 403:
        assert "symlink" in resp.json()["detail"]


def test_fixed_name_files_swapped_before_their_open_are_never_read(
    swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """episode.json: the summary has no episode and says why. events.jsonl: an explicit
    403, not the empty page of a file that does not exist yet."""
    ep, outside = swap_env["ep"], swap_env["outside"]
    fired = hook_open(monkeypatch, "episode.json", before=lambda: swap_for_symlink(ep / "episode.json", outside / "episode.json"))
    summary = swap_env["client"].get(EP_URL)
    assert fired and summary.status_code == 200 and summary.json()["episode"] is None
    assert summary.json()["episode_json_error"] == "symlink_not_followed"
    assert_no_secret(summary)
    monkeypatch.undo()
    fired = hook_open(monkeypatch, "events.jsonl", before=lambda: swap_for_symlink(ep / "events.jsonl", outside / "events.jsonl"))
    page = swap_env["client"].get(f"{EP_URL}/events")
    assert fired and page.status_code == 403 and "symlink" in page.json()["detail"]
    assert_no_secret(page)


# ---------------------------------------------------------------------------
# a name swapped right after the file was opened: the checked file is served
# ---------------------------------------------------------------------------


def test_file_swapped_after_its_open_serves_the_checked_file(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ep, outside = swap_env["ep"], swap_env["outside"]
    hook_open(monkeypatch, "x.txt", after=lambda: swap_for_symlink(ep / "assets" / "x.txt", outside / "x.txt"))
    resp = swap_env["client"].get(f"{EP_URL}/file/assets/x.txt")
    assert resp.status_code == 200 and resp.text == "inside text"
    assert (ep / "assets" / "x.txt").is_symlink()  # the swap did happen before the body was sent


def test_array_swapped_after_its_open_describes_the_checked_file(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ep, outside = swap_env["ep"], swap_env["outside"]
    hook_open(monkeypatch, "x.npy", after=lambda: swap_for_symlink(ep / "assets" / "x.npy", outside / "x.npy"))
    resp = swap_env["client"].get(f"{EP_URL}/array/assets/x.npy")
    assert resp.status_code == 200 and resp.json()["arrays"]["array"]["preview"] == INSIDE_ARRAY.tolist()
    assert_no_secret(resp)


def test_array_image_swapped_after_its_open_draws_the_checked_file(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ep, outside = swap_env["ep"], swap_env["outside"]
    hook_open(monkeypatch, "x.npy", after=lambda: swap_for_symlink(ep / "assets" / "x.npy", outside / "x.npy"))
    resp = swap_env["client"].get(f"{EP_URL}/array-image/assets/x.npy")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    expected = np.clip(INSIDE_ARRAY * (255.0 / 11.0), 0, 255).astype(np.uint8)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(resp.content))), expected)


def test_legacy_file_swapped_after_its_open_serves_the_checked_file(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    trial, outside = swap_env["trial"], swap_env["outside"]
    hook_open(monkeypatch, "code.py", after=lambda: swap_for_symlink(trial / "code.py", outside / "code.py"))
    resp = swap_env["client"].get(f"{LEGACY_URL}/code.py")
    assert resp.status_code == 200 and resp.text == "print('inside')\n"


def test_events_swapped_after_their_open_are_read_from_the_checked_file(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    ep, outside = swap_env["ep"], swap_env["outside"]
    hook_open(monkeypatch, "events.jsonl", after=lambda: swap_for_symlink(ep / "events.jsonl", outside / "events.jsonl"))
    page = swap_env["client"].get(f"{EP_URL}/events")
    assert page.status_code == 200 and [e["event_id"] for e in page.json()["events"]] == ["ev_inside"]


def test_continuous_swaps_never_serve_bytes_from_outside(swap_env: dict[str, Any]) -> None:
    """A thread flips assets/x.txt between the regular file and a symlink to outside while
    requests run. Every answer is the inside file or a refusal."""
    target = swap_env["ep"] / "assets" / "x.txt"
    regular = target.with_name("x.regular")
    link = target.with_name("x.link")
    stop = threading.Event()
    flips = [0]

    def flip() -> None:
        while not stop.is_set():
            link.symlink_to(swap_env["outside"] / "x.txt")
            os.replace(link, target)
            regular.write_text("inside text")
            os.replace(regular, target)
            flips[0] += 1

    thread = threading.Thread(target=flip, daemon=True)
    thread.start()
    statuses: Counter[int] = Counter()
    try:
        for _ in range(300):
            resp = swap_env["client"].get(f"{EP_URL}/file/assets/x.txt")
            assert_no_secret(resp)
            assert resp.status_code in (200, 403), resp.status_code
            if resp.status_code == 200:
                assert resp.text == "inside text"
            statuses[resp.status_code] += 1
    finally:
        stop.set()
        thread.join(10)
    assert flips[0] > 0 and statuses[200] > 0


# ---------------------------------------------------------------------------
# byte ranges of streamed files
# ---------------------------------------------------------------------------

R_URL = f"{EP_URL}/file/assets/r.txt"
R_DATA = b"0123456789abcdef"


@pytest.mark.parametrize(
    ("header", "status", "body", "content_range"),
    [
        ("bytes=2-5", 206, R_DATA[2:6], "bytes 2-5/16"),
        ("bytes=10-", 206, R_DATA[10:], "bytes 10-15/16"),
        ("bytes=-3", 206, R_DATA[-3:], "bytes 13-15/16"),
        ("bytes=5-100", 206, R_DATA[5:], "bytes 5-15/16"),
        ("bytes=-100", 206, R_DATA, "bytes 0-15/16"),
        ("bytes=16-", 416, b"", "bytes */16"),
        ("bytes=-0", 416, b"", "bytes */16"),
        ("bytes=0-1,4-5", 200, R_DATA, None),  # several ranges: the whole file
        ("bytes=5-2", 200, R_DATA, None),
        ("items=0-1", 200, R_DATA, None),
        ("bytes=x-y", 200, R_DATA, None),
    ],
)
def test_byte_ranges(swap_env: dict[str, Any], header: str, status: int, body: bytes, content_range: str | None) -> None:
    resp = swap_env["client"].get(R_URL, headers={"Range": header})
    assert resp.status_code == status and resp.content == body
    assert resp.headers.get("content-range") == content_range
    assert resp.headers["content-length"] == str(len(body))
    assert resp.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("url", [R_URL, f"{LEGACY_URL}/code.py"])
@pytest.mark.parametrize(
    ("header", "status", "expect"),
    [
        # Latin-1 superscripts: str.isdigit() accepted them and int() raised (HTTP 500)
        (b"bytes=\xb2-", 200, "all"),
        (b"bytes=0-\xb3", 200, "all"),
        (b"bytes=-\xb9", 200, "all"),
        (b"bytes=\xb2-\xb3", 200, "all"),
        # more digits than int() converts (4300) or than any file has bytes
        (b"bytes=" + b"9" * 5000 + b"-", 416, None),
        (b"bytes=0-" + b"9" * 5000, 206, "all"),
        (b"bytes=-" + b"9" * 5000, 206, "all"),
        (b"bytes=" + b"0" * 5000 + b"2-" + b"0" * 5000 + b"5", 206, (2, 6)),
        (b"bytes=1" + b"0" * 18 + b"-", 416, None),
        (b"bytes=-", 200, "all"),
        (b"bytes=--1", 200, "all"),
        (b"bytes=+1-2", 200, "all"),
    ],
)
def test_malformed_ranges_are_ignored_or_unsatisfiable_never_500(
    swap_env: dict[str, Any], url: str, header: bytes, status: int, expect: Any
) -> None:
    """ Sent as raw bytes: httpx refuses non-ASCII header values."""
    full = (swap_env["ep"] / "assets" / "r.txt").read_bytes() if url == R_URL else (swap_env["trial"] / "code.py").read_bytes()
    got, body, headers = raw_get(swap_env["client"].app, url, headers=[(b"host", LOOPBACK_HOST.encode()), (b"range", header)])
    assert got == status, (header[:40], got, body[:200])
    if expect == "all":
        assert body == full and headers["content-length"] == str(len(full))
    elif expect is not None:
        assert body == full[expect[0] : expect[1]]
        assert headers["content-range"] == f"bytes {expect[0]}-{expect[1] - 1}/{len(full)}"
    else:
        assert body == b"" and headers["content-range"] == f"bytes */{len(full)}"


def test_parse_byte_range_bounds_its_work() -> None:
    assert trace_api.parse_byte_range("bytes=" + "7" * 100_000 + "-", 16) == "unsatisfiable"
    assert trace_api.parse_byte_range("bytes=0-" + "7" * 100_000, 16) == (0, 16)
    assert trace_api.parse_byte_range("bytes=٣-", 16) is None  # Arabic-Indic digit three
    assert trace_api.parse_byte_range("bytes=2-5", 16) == (2, 6)


def test_if_range(swap_env: dict[str, Any]) -> None:
    full = swap_env["client"].get(R_URL)
    assert full.status_code == 200 and full.content == R_DATA and full.headers["accept-ranges"] == "bytes"
    etag, modified = full.headers["etag"], full.headers["last-modified"]
    for validator in (etag, modified):
        part = swap_env["client"].get(R_URL, headers={"Range": "bytes=0-3", "If-Range": validator})
        assert part.status_code == 206 and part.content == R_DATA[:4]
    stale = swap_env["client"].get(R_URL, headers={"Range": "bytes=0-3", "If-Range": '"another-version"'})
    assert stale.status_code == 200 and stale.content == R_DATA


def truncate_after_the_check(monkeypatch: pytest.MonkeyPatch, path: Path, size: int) -> list[int]:
    """Truncate ``path`` to ``size`` bytes right after the endpoint's fstat check of it."""
    inode = path.stat().st_ino
    real_fstat = os.fstat
    checked: list[int] = []

    def fstat_then_truncate(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        if st.st_ino == inode and not checked:
            checked.append(st.st_size)
            os.truncate(path, size)
        return st

    monkeypatch.setattr(os, "fstat", fstat_then_truncate)
    return checked


@pytest.mark.parametrize(
    ("size", "range_header", "status", "body", "content_range"),
    [
        (4, None, 200, R_DATA[:4], None),
        (4, "bytes=2-9", 206, R_DATA[2:4], "bytes 2-3/4"),
        (0, None, 200, b"", None),
        (0, "bytes=2-9", 416, b"", "bytes */0"),
    ],
)
def test_a_file_that_shrinks_after_the_check_is_answered_for_its_current_size(
    swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, size: int, range_header: str | None,
    status: int, body: bytes, content_range: str | None,
) -> None:
    """The part is read before its headers are sent, so a truncation after
    the check gives a complete answer for the file as it is now, never a response shorter
    than its Content-Length (which the server logs) and never padding."""
    checked = truncate_after_the_check(monkeypatch, swap_env["ep"] / "assets" / "r.txt", size)
    headers = {"Range": range_header} if range_header else {}
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(swap_env["client"].get(R_URL, headers=headers)), daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive(), "the response never ended"
    (resp,) = result
    assert checked == [16] and resp.status_code == status and resp.content == body
    assert resp.headers["content-length"] == str(len(body)) and resp.headers.get("content-range") == content_range


def test_a_file_that_keeps_changing_is_answered_503(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Every read comes up short: after FILE_SNAPSHOT_ATTEMPTS reads the answer is 503."""
    reads: list[int] = []

    def short_read(fd: int, size: int, offset: int) -> bytes:
        reads.append(size)
        return b"x" * (size // 2)

    monkeypatch.setattr(trace_api.reader, "pread_upto", short_read)
    resp = swap_env["client"].get(R_URL)
    assert resp.status_code == 503 and resp.headers["retry-after"] == "1" and "changed" in resp.text
    assert len(reads) == trace_api.FILE_SNAPSHOT_ATTEMPTS


def test_parts_over_the_snapshot_size_are_streamed(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Parts over FILE_SNAPSHOT_BYTES are streamed without Content-Length; a part
    up to that size of the same file is still read first and carries its length."""
    monkeypatch.setattr(trace_api, "FILE_SNAPSHOT_BYTES", 5)
    monkeypatch.setattr(trace_api.OpenFileResponse, "chunk_size", 3)
    full = swap_env["client"].get(R_URL)
    assert full.status_code == 200 and full.content == R_DATA and "content-length" not in full.headers
    assert full.headers["accept-ranges"] == "bytes" and full.headers["x-content-type-options"] == "nosniff"
    part = swap_env["client"].get(R_URL, headers={"Range": "bytes=1-9"})
    assert part.status_code == 206 and part.content == R_DATA[1:10]
    assert part.headers["content-range"] == "bytes 1-9/16" and "content-length" not in part.headers
    small = swap_env["client"].get(R_URL, headers={"Range": "bytes=1-5"})
    assert small.status_code == 206 and small.content == R_DATA[1:6] and small.headers["content-length"] == "5"


class _ServerThread:
    """A real uvicorn server on an ephemeral 127.0.0.1 port, stopped by ``stop()``."""

    def __init__(self, app: Any) -> None:
        config = uvicorn.Config(
            app, host="127.0.0.1", port=0, log_config=None, access_log=False, lifespan="off", http="h11", ws="none"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> int:
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started:
            assert time.monotonic() < deadline and self.thread.is_alive(), "uvicorn did not start"
            time.sleep(0.01)
        return self.server.servers[0].sockets[0].getsockname()[1]

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(20)


@pytest.mark.parametrize("snapshot_bytes", [None, 5], ids=["read_before_headers", "streamed"])
def test_truncation_after_the_check_logs_nothing_on_a_real_server(
    swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    snapshot_bytes: int | None,
) -> None:
    """ Before the fix uvicorn logged "ASGI callable returned without
    completing response." at ERROR and the client read fewer bytes than announced.
    ``streamed`` lowers FILE_SNAPSHOT_BYTES so that the 16-byte file is a part
    over it: that answer has no Content-Length and its chunked body ends after 4 bytes."""
    if snapshot_bytes is not None:
        monkeypatch.setattr(trace_api, "FILE_SNAPSHOT_BYTES", snapshot_bytes)
    server = _ServerThread(swap_env["client"].app)
    port = server.start()
    try:
        checked = truncate_after_the_check(monkeypatch, swap_env["ep"] / "assets" / "r.txt", 4)
        with caplog.at_level(logging.INFO):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            try:
                conn.request("GET", R_URL, headers={"Host": LOOPBACK_HOST})
                resp = conn.getresponse()
                body = resp.read()  # raises IncompleteRead if fewer bytes than announced arrive
                status, length = resp.status, resp.getheader("content-length")
                transfer = resp.getheader("transfer-encoding")
            finally:
                conn.close()
            monkeypatch.undo()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)  # the server still answers
            try:
                conn.request("GET", f"{EP_URL}/file/assets/x.txt", headers={"Host": LOOPBACK_HOST})
                again = conn.getresponse().read()
            finally:
                conn.close()
    finally:
        server.stop()
    assert checked == [16] and status == 200 and body == R_DATA[:4]
    if snapshot_bytes is None:
        assert length == "4" and transfer is None
    else:
        assert length is None and transfer == "chunked"
    assert again == b"inside text"
    problems = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert problems == [], [r.getMessage() for r in problems]


# ---------------------------------------------------------------------------
# parts over FILE_SNAPSHOT_BYTES in uvicorn's own request cycle (streaming tests)
# ---------------------------------------------------------------------------

# One byte over the size that is read before the headers are sent: the real boundary.
LARGE = trace_api.FILE_SNAPSHOT_BYTES + 1
# The first bytes of the large test files; the rest is a hole, which reads as zeros.
BIG_HEAD = b"0123456789"


class _FakeTransport:
    """Stands in for the socket of uvicorn's RequestResponseCycle: keeps what is written
    and runs ``on_first_body`` right after the first body write (the second write; the
    first one holds the status line and headers)."""

    def __init__(self, on_first_body: Callable[[], None] | None = None) -> None:
        self.written: list[bytes] = []
        self.closed = False
        self._on_first_body = on_first_body

    def write(self, data: bytes) -> None:
        self.written.append(bytes(data))
        if len(self.written) == 2 and self._on_first_body is not None:
            self._on_first_body()

    def close(self) -> None:
        self.closed = True

    def is_closing(self) -> bool:
        return self.closed


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def serve_with_uvicorn_cycle(
    app: Any, *, method: str = "GET", http_version: str = "1.1", headers: dict[str, str] | None = None,
    on_first_body: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Runs one request (``headers`` besides ``Host``) through uvicorn's h11
    ``RequestResponseCycle`` with a fake transport, then parses the written bytes with an
    h11 client.

    Returns ``complete`` (uvicorn saw the response completed), ``logged`` (the messages
    uvicorn logged, any level), ``closed`` (uvicorn closed the connection), and what the
    client read: ``status``, ``headers``, ``body``, ``ended`` (the end of the body was
    framed properly) and ``error`` (the client's protocol error, if any)."""
    request_headers = [(b"host", b"localhost")]
    for name, value in (headers or {}).items():
        request_headers.append((name.lower().encode("latin-1"), value.encode("latin-1")))

    async def run() -> dict[str, Any]:
        transport = _FakeTransport(on_first_body)
        conn = h11.Connection(h11.SERVER)
        conn.receive_data(
            f"{method} / HTTP/{http_version}\r\n".encode()
            + b"".join(name + b": " + value + b"\r\n" for name, value in request_headers)
            + b"\r\n"
        )
        assert isinstance(conn.next_event(), h11.Request) and isinstance(conn.next_event(), h11.EndOfMessage)
        records = _Records()
        logger = logging.Logger("uvicorn-cycle-test")  # not in the logging tree: nothing propagates
        logger.addHandler(records)
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": http_version,
            "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 50000), "scheme": "http",
            "method": method, "root_path": "", "path": "/", "raw_path": b"/", "query_string": b"",
            "headers": request_headers,
        }
        cycle = RequestResponseCycle(
            scope=scope, conn=conn, transport=transport, flow=FlowControl(transport),  # type: ignore[arg-type]
            logger=logger, access_logger=logger, access_log=False, default_headers=[],
            message_event=asyncio.Event(), on_response=lambda: None,
        )
        await cycle.run_asgi(app)
        return {
            "complete": cycle.response_complete,
            "logged": [record.getMessage() for record in records.records],
            "closed": transport.closed,
            "written": b"".join(transport.written),
        }

    result = asyncio.run(run())
    client = h11.Connection(h11.CLIENT)
    client.send(h11.Request(method=method, target="/", headers=[("Host", "localhost")]))
    client.send(h11.EndOfMessage())
    client.receive_data(result.pop("written"))
    if result["closed"]:
        client.receive_data(b"")
    status, headers, body, ended, error = None, {}, [], False, None
    while not ended:
        try:
            event = client.next_event()
        except h11.RemoteProtocolError as exc:
            error = str(exc)
            break
        if event is h11.NEED_DATA:
            break
        if isinstance(event, h11.Response):
            status, headers = event.status_code, {k.decode(): v.decode() for k, v in event.headers}
        elif isinstance(event, h11.Data):
            body.append(bytes(event.data))
        elif isinstance(event, h11.EndOfMessage):
            ended = True
    result.update(status=status, headers=headers, body=b"".join(body), ended=ended, error=error)
    return result


def _sparse_file(path: Path, size: int) -> Path:
    with path.open("wb") as f:
        f.write(BIG_HEAD)
        f.truncate(size)
    return path


def _open_file_response(path: Path) -> trace_api.OpenFileResponse:
    """The response the file endpoints build: a descriptor opened and checked with fstat."""
    fd = os.open(path, os.O_RDONLY)
    return trace_api.OpenFileResponse(fd, os.fstat(fd), path.name)


@pytest.mark.parametrize(
    ("headers", "when", "http_version", "status", "first", "ranged"),
    [
        pytest.param({}, "before_send", "1.1", 200, 0, False, id="whole_before_send"),
        pytest.param({}, "while_sending", "1.1", 200, 0, False, id="whole_while_sending"),
        pytest.param({"Range": "bytes=0-"}, "before_send", "1.1", 206, 0, True, id="range_before_send"),
        pytest.param({"Range": "bytes=2-"}, "while_sending", "1.1", 206, 2, True, id="range_while_sending"),
        pytest.param({"Range": f"bytes=-{LARGE}"}, "before_send", "1.1", 206, 3, True, id="suffix_range_before_send"),
        # a small part, but If-Range names another version: the whole file
        pytest.param({"Range": "bytes=0-3", "If-Range": '"another-version"'}, "before_send", "1.1", 200, 0, False,
                     id="if_range_mismatch_before_send"),
        pytest.param({}, "before_send", "1.0", 200, 0, False, id="http10_before_send"),
        pytest.param({}, "while_sending", "1.0", 200, 0, False, id="http10_while_sending"),
    ],
)
def test_a_part_over_the_snapshot_size_ends_early_when_the_file_shrinks(
    tmp_path: Path, headers: dict[str, str], when: str, http_version: str, status: int, first: int,
    ranged: bool,
) -> None:
    """Streaming test. The file has ``LARGE + first`` bytes, so the part from
    byte ``first`` is one byte over FILE_SNAPSHOT_BYTES and is streamed. The file shrinks
    to 4 bytes after the check (``before_send``) or right after the first chunk was
    written (``while_sending``). Before the fix the answer announced the ``fstat``
    length, so uvicorn logged "ASGI callable returned without completing response." and
    closed the connection, and the client got a protocol error. Now the part has no
    Content-Length: the body ends at the file's new end, the response is completed and
    nothing is logged."""
    size = LARGE + first
    content_range = f"bytes {first}-{size - 1}/{size}" if ranged else None
    path = _sparse_file(tmp_path / "big.txt", size)
    response = _open_file_response(path)

    def shrink() -> None:
        os.truncate(path, 4)

    if when == "before_send":
        shrink()
    got = serve_with_uvicorn_cycle(
        response, http_version=http_version, headers=headers,
        on_first_body=shrink if when == "while_sending" else None,
    )
    assert got["complete"] and got["logged"] == [], got["logged"]
    assert got["error"] is None and got["ended"]
    chunk = trace_api.OpenFileResponse.chunk_size
    if when == "before_send":
        expected = BIG_HEAD[first:4]
    else:  # exactly the first chunk, read before the file shrank
        expected = (BIG_HEAD + bytes(chunk))[first : first + chunk]
    assert got["status"] == status and got["body"] == expected
    answer = got["headers"]
    assert "content-length" not in answer and answer.get("content-range") == content_range
    if http_version == "1.1":
        assert answer["transfer-encoding"] == "chunked" and not got["closed"]
    else:  # no chunked coding in HTTP/1.0: the end of the body is the end of the connection
        assert "transfer-encoding" not in answer and answer["connection"] == "close" and got["closed"]
    assert answer["x-content-type-options"] == "nosniff" and answer["content-security-policy"] == "sandbox"
    assert answer["content-type"] == "text/plain; charset=utf-8" and answer["accept-ranges"] == "bytes"


@pytest.mark.parametrize(
    ("size", "method", "range_header", "shrink", "status", "length", "body", "content_range"),
    [
        # at the limit: read before the headers, answered for the size after the truncation
        pytest.param(LARGE - 1, "GET", None, True, 200, "4", (0, 4), None, id="at_limit_shrunk"),
        # a small range of a large file is read first as well
        pytest.param(LARGE, "GET", "bytes=0-9", True, 206, "4", (0, 4), "bytes 0-3/4", id="small_range_shrunk"),
        pytest.param(LARGE, "GET", "bytes=1-9", False, 206, "9", (1, 10), f"bytes 1-9/{LARGE}", id="small_range"),
        # a large part of a file that does not change arrives whole, chunked (the new framing)
        pytest.param(LARGE, "GET", None, False, 200, None, (0, LARGE), None, id="large_unchanged"),
        # HEAD has no body that could end early: it keeps the length of the part
        pytest.param(LARGE, "HEAD", None, False, 200, str(LARGE), (0, 0), None, id="head"),
    ],
)
def test_other_file_answers_near_the_snapshot_size_complete(
    tmp_path: Path, size: int, method: str, range_header: str | None, shrink: bool, status: int,
    length: str | None, body: tuple[int, int], content_range: str | None,
) -> None:
    """Answers near the boundary that do not end early, in the same request cycle: all are
    complete with nothing logged. A part read before its headers and a HEAD answer carry
    Content-Length, as before the fix; a large part of a file that does not change is
    chunked and arrives whole. ``body`` is the slice of the file's content (after the
    truncation to 4 bytes, when ``shrink``)."""
    path = _sparse_file(tmp_path / "big.txt", size)
    response = _open_file_response(path)
    content = BIG_HEAD + bytes(size - len(BIG_HEAD))
    if shrink:
        os.truncate(path, 4)
        content = BIG_HEAD[:4]
    got = serve_with_uvicorn_cycle(response, method=method, headers={"Range": range_header} if range_header else {})
    summary = {key: got[key] for key in ("complete", "logged", "error", "ended", "status", "headers")}
    assert got["complete"] and got["logged"] == [] and got["error"] is None and got["ended"], summary
    assert got["status"] == status and len(got["body"]) == body[1] - body[0], summary
    assert got["body"] == content[body[0] : body[1]]
    assert got["headers"].get("content-length") == length and got["headers"].get("content-range") == content_range
    assert got["headers"].get("transfer-encoding") == (None if length is not None else "chunked")


# ---------------------------------------------------------------------------
# Concurrent array requests must not starve a live trial in the same process.
# ---------------------------------------------------------------------------


def test_array_requests_beyond_the_worker_limit_get_503(swap_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(trace_api, "ARRAY_WORKERS", 1)
    monkeypatch.setattr(trace_api, "ARRAY_WAIT_S", 0.2)
    app = create_app(viewer_only=True, trace_root=swap_env["root"], frontend_dist=tmp_path / "no_dist")
    started, release = threading.Event(), threading.Event()
    real = trace_arrays.describe_file

    def slow(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(20)
        return real(*args, **kwargs)

    monkeypatch.setattr(trace_arrays, "describe_file", slow)
    url = f"{EP_URL}/array/assets/x.npy"
    client = TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL)
    first: list[Any] = []
    thread = threading.Thread(target=lambda: first.append(TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL).get(url)))
    thread.start()
    try:
        assert started.wait(20)
        for busy_url in (url, f"{EP_URL}/array-image/assets/x.npy"):
            busy = client.get(busy_url)
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1", busy.text
        assert client.get(f"{EP_URL}/file/assets/x.txt").status_code == 200  # other endpoints are not limited
    finally:
        release.set()
        thread.join(20)
    assert first and first[0].status_code == 200
    assert client.get(url).json()["arrays"]["array"]["preview"] == INSIDE_ARRAY.tolist()  # the slot was released


# ---------------------------------------------------------------------------
# every descriptor opened for a request is closed again
# ---------------------------------------------------------------------------


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd")
def test_no_descriptor_leaks_on_success_and_error_paths(swap_env: dict[str, Any]) -> None:
    ep = swap_env["ep"]
    (ep / "assets" / "link.txt").symlink_to(swap_env["outside"] / "x.txt")
    (ep / "assets" / "bad.npz").write_bytes(b"PK\x03\x04 not a zip")
    np.savez(ep / "assets" / "two.npz", a=np.zeros(3), b=np.ones((2, 2)))
    urls = [
        "/api/traces/config", "/api/traces/runs", "/api/traces/runs/r1", EP_URL, f"{EP_URL}/events",
        f"{EP_URL}/events?after_offset=3", f"{EP_URL}/events?generation=x", f"{EP_URL}/file/assets/x.txt",
        f"{EP_URL}/file/assets/link.txt", f"{EP_URL}/file/assets/missing.txt", f"{EP_URL}/file/assets",
        f"{EP_URL}/array/assets/x.npy", f"{EP_URL}/array/assets/bad.npz", f"{EP_URL}/array/assets/two.npz",
        f"{EP_URL}/array-image/assets/two.npz", f"{EP_URL}/array-image/assets/x.npy", f"{EP_URL}/array/assets/x.txt",
        "/api/traces/runs/r1/episodes/nope", "/api/traces/runs/nope", "/api/traces/legacy/runs",
        "/api/traces/legacy/runs/run1", f"/api/traces/legacy/trials/run1/task/cond/{TRIAL}", f"{LEGACY_URL}/code.py",
        f"{LEGACY_URL}/missing.py",
    ]
    client = swap_env["client"]
    for url in urls:  # warm up (imports, caches, thread pool)
        client.get(url)
    before = _open_fds()
    statuses = Counter(client.get(url).status_code for _ in range(5) for url in urls)
    assert _open_fds() <= before, (before, _open_fds())
    assert {200, 400, 403, 404, 409} <= set(statuses)
    for _ in range(5):  # byte ranges and unsatisfiable ranges
        client.get(f"{EP_URL}/file/assets/r.txt", headers={"Range": "bytes=2-3"})
        client.get(f"{EP_URL}/file/assets/r.txt", headers={"Range": "bytes=99-"})
    assert _open_fds() <= before
