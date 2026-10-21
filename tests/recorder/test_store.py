"""Writer robustness (docs/trace-format.md, sections 4.3-4.5).

Each test uses its own ``TraceStore`` (not the process singleton), so nothing
here depends on ``CAPX_TRACE_DIR`` or on the bootstrap.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from tracing_testlib import REPO, clean_env
from trace_reader import by_type, read_events


def make_store(work_dir, **kwargs) -> S.TraceStore:
    return S.TraceStore(TraceConfig(trace_dir=work_dir / "traces", **kwargs))


def open_ep(store, name="ep1"):
    ep = C.open_episode(store, run_id="run-test", episode_id=name, source="test")
    ep.emit("episode_start", status="started", data={"source": "test", "task": "t", "tracer": {}})
    return ep


def finish(store, ep):
    with ep._lock:
        ep.emit("episode_end", data={"end_reason": "completed"})
        ep.ended = True
    C.close_episode(ep)
    assert store.flush(20)


def frame_builder(arr):
    def builder(ctx):
        ctx.write_file("frames/cam/e0_0.jpg", lambda: S.encode_jpeg(arr, 85), label="debug_frame",
                       kind="image", mime="image/jpeg", binary=True)
    return builder


def test_no_thread_and_no_file_before_first_event(work_dir):
    store = make_store(work_dir)
    assert store._writer is None
    assert not (work_dir / "traces").exists()
    ep = open_ep(store)
    assert store.writer_alive()
    finish(store, ep)
    store.shutdown(5)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "a//b", "a/./b", "link/escape"])
def test_write_file_rejects_unsafe_path(work_dir, name):
    store = make_store(work_dir)
    ep = open_ep(store)
    assert store.flush(5)
    outside = work_dir / "outside"
    outside.mkdir()
    (ep.base_dir / "link").symlink_to(outside, target_is_directory=True)
    ep.emit("frame", builder=lambda ctx: ctx.write_file(
        name, b"private", label="test", kind="text", mime="text/plain", binary=False))
    finish(store, ep)
    events, _, _ = read_events(ep.events_path)
    artifact = by_type(events, "frame")[0]["artifacts"][0]
    assert artifact["written"] is False and artifact["reason"] == "unsafe_path"
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert "unsafe_path" in {gap["code"] for gap in summary["gaps"]}
    assert not (outside / "escape").exists()
    store.shutdown(5)


@pytest.mark.parametrize("error", [RuntimeError, ValueError, OSError])
def test_write_file_resolve_failure_is_unsafe_path(work_dir, monkeypatch, error):
    store = make_store(work_dir)
    ep = open_ep(store)
    assert store.flush(5)
    original_resolve = S.Path.resolve

    def broken_resolve(path, *args, **kwargs):
        if path.name == "bad.txt":
            raise error("cannot resolve")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(S.Path, "resolve", broken_resolve)
    ep.emit("frame", builder=lambda ctx: ctx.write_file(
        "bad.txt", b"secret", label="test", kind="text", mime="text/plain", binary=False))
    finish(store, ep)
    events, _, _ = read_events(ep.events_path)
    artifact = by_type(events, "frame")[0]["artifacts"][0]
    assert artifact["written"] is False and artifact["reason"] == "unsafe_path"
    store.shutdown(5)


def test_ordering_under_concurrent_emits(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)

    def worker(k):
        for i in range(200):
            ep.emit("trace_warning", data={"code": "test", "counts_as_gap": False, "k": k, "i": i})

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finish(store, ep)
    events, corrupt, torn = read_events(ep.events_path)
    assert corrupt == 0 and not torn
    assert [e["seq"] for e in events] == list(range(1, 8 * 200 + 3))
    per_worker = {}
    for e in by_type(events, "trace_warning"):
        per_worker.setdefault(e["data"]["k"], []).append(e["data"]["i"])
    assert all(v == list(range(200)) for v in per_worker.values())
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["status"] == "ended" and summary["trace_complete"] is True
    store.shutdown(5)


def test_short_write_is_truncated_and_recorded(work_dir, monkeypatch):
    store = make_store(work_dir)
    ep = open_ep(store)
    assert store.flush(10)
    real_write = S._os_write
    calls = {"n": 0}

    def short_write(fd, data):
        if isinstance(data, (bytes, bytearray)) and data.endswith(b"\n") and b'"victim"' in data:
            calls["n"] += 1
            return real_write(fd, bytes(data[:10]))
        return real_write(fd, data)

    monkeypatch.setattr(S, "_os_write", short_write)
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "victim": True})
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "after": True})
    assert store.flush(10)
    monkeypatch.setattr(S, "_os_write", real_write)
    finish(store, ep)
    assert calls["n"] == 1
    events, corrupt, torn = read_events(ep.events_path)
    assert corrupt == 0 and not torn  # the partial line was truncated away
    assert not any(e["data"].get("victim") for e in events)
    assert any(e["data"].get("after") for e in events)
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["trace_complete"] is False
    assert {g["code"] for g in summary["gaps"]} == {"write_failed"}
    warnings = [e for e in by_type(events, "trace_warning") if e["data"]["code"] == "write_failed"]
    assert warnings and warnings[0]["data"]["writer_generated"] is True
    store.shutdown(5)


def test_queue_budget_drops_binary_items_but_keeps_lines(work_dir):
    store = make_store(work_dir, max_queue_mb=0.001)  # ~1 KB
    ep = open_ep(store)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    ep.emit("frame", data={"camera": "cam"}, builder=frame_builder(arr), nbytes=arr.nbytes)
    finish(store, ep)
    events, _, _ = read_events(ep.events_path)
    frame = by_type(events, "frame")[0]
    assert frame["artifacts"][0]["written"] is False
    assert frame["artifacts"][0]["reason"] == "queue_full"
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["trace_complete"] is False
    assert "queue_full" in {g["code"] for g in summary["gaps"]}
    assert not (ep.base_dir / "frames/cam/e0_0.jpg").exists()
    store.shutdown(5)


def test_disk_low_skips_binaries_then_stops(work_dir, monkeypatch):
    store = make_store(work_dir, min_free_gb=1.0)
    free = {"bytes": 10 * 1024 ** 3}
    monkeypatch.setattr(S, "_disk_free", lambda path: free["bytes"])
    monkeypatch.setattr(S, "DISK_CHECK_INTERVAL_S", 0.0)
    ep = open_ep(store)
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    ep.emit("frame", data={"n": 1}, builder=frame_builder(arr), nbytes=arr.nbytes)
    assert store.flush(10)
    free["bytes"] = int(0.7 * 1024 ** 3)  # below the floor: binaries skipped
    ep.emit("frame", data={"n": 2}, builder=frame_builder(arr), nbytes=arr.nbytes)
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "text_line": True})
    assert store.flush(10)
    free["bytes"] = int(0.2 * 1024 ** 3)  # below half: one marker, then nothing
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "lost": True})
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "lost": True})
    assert store.flush(10)
    events, corrupt, torn = read_events(ep.events_path)
    assert corrupt == 0 and not torn
    frames = by_type(events, "frame")
    assert frames[0]["artifacts"][0]["written"] is True
    assert frames[1]["artifacts"][0]["written"] is False
    assert frames[1]["artifacts"][0]["reason"] == "disk_low"
    assert any(e["data"].get("text_line") for e in events)
    assert not any(e["data"].get("lost") for e in events)
    last = events[-1]
    assert last["event_type"] == "trace_warning" and last["data"]["code"] == "disk_low"
    assert "recording stopped" in last["data"]["message"]
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["status"] == "running"  # never ended: the viewer shows it incomplete
    C.close_episode(ep)
    store.shutdown(5)


def test_late_events_mark_the_episode_incomplete(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)
    with ep._lock:
        ep.emit("episode_end", data={"end_reason": "completed"})
        ep.ended = True
    assert store.flush(10)
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["trace_complete"] is True
    ep.emit("model_response", data={"content": "arrived late"})
    assert store.flush(10)
    events, _, _ = read_events(ep.events_path)
    assert events[-1]["late"] is True and events[-2]["event_type"] == "episode_end"
    summary = json.loads((ep.base_dir / "episode.json").read_text())
    assert summary["status"] == "ended" and summary["trace_complete"] is False
    assert summary["late_events"] == 1
    assert {g["code"] for g in summary["gaps"]} == {"late_event"}
    C.close_episode(ep)
    store.shutdown(5)


def test_seq_gap_is_counted_as_dropped(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)
    with ep._lock:
        ep._seq += 3  # three events lost before reaching the queue
    finish(store, ep)
    events, _, _ = read_events(ep.events_path)
    end = by_type(events, "episode_end")[0]["data"]
    assert end["dropped_events"] == 3 and end["trace_complete"] is False
    assert end["gaps"][0] == {"code": "dropped_events", "count": 3, "first_seq": 2}
    store.shutdown(5)


def test_builder_failure_is_recorded_and_line_still_written(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)

    def bad(ctx):
        raise RuntimeError("builder bug")

    ep.emit("code_block_start", data={"code": "x"}, builder=bad)
    finish(store, ep)
    events, _, _ = read_events(ep.events_path)
    block = by_type(events, "code_block_start")[0]
    assert "builder bug" in block["data"]["recorder_error"]
    codes = {e["data"]["code"] for e in by_type(events, "trace_warning")}
    assert "internal_error" in codes
    store.shutdown(5)


def test_oversize_event_data_moves_to_an_asset(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)
    ep.emit("api_call_end", data={"name": "big", "blob": ["x" * 3000] * 200})
    finish(store, ep)
    raw = ep.events_path.read_bytes()
    assert max(len(line) for line in raw.split(b"\n")) <= 256 * 1024
    events, _, _ = read_events(ep.events_path)
    big = by_type(events, "api_call_end")[0]["data"]
    assert big["oversize"] is True and big["name"] == "big"
    assert (ep.base_dir / big["data_asset"]).is_file()
    store.shutdown(5)


def test_files_are_written_atomically(work_dir):
    store = make_store(work_dir)
    ep = open_ep(store)
    arr = np.full((8, 8, 3), 7, dtype=np.uint8)
    ep.emit("frame", data={}, builder=frame_builder(arr), nbytes=arr.nbytes)
    finish(store, ep)
    names = [p.name for p in ep.base_dir.rglob("*")]
    assert not [n for n in names if ".tmp." in n]
    assert (ep.base_dir / "frames/cam/e0_0.jpg").read_bytes()[:3] == b"\xff\xd8\xff"
    assert (ep.base_dir / "heartbeat").exists()
    run_json = json.loads((ep.run_dir / "run.json").read_text())
    assert run_json["run_id"] == "run-test"
    store.shutdown(5)


def test_heartbeat_is_refreshed_while_running(work_dir, monkeypatch):
    monkeypatch.setattr(S, "HEARTBEAT_INTERVAL_S", 0.2)
    store = make_store(work_dir)
    ep = open_ep(store)
    assert store.flush(10)
    hb = ep.base_dir / "heartbeat"
    first = hb.stat().st_mtime_ns
    deadline = time.time() + 5
    while time.time() < deadline and hb.stat().st_mtime_ns == first:
        time.sleep(0.1)
    assert hb.stat().st_mtime_ns > first
    finish(store, ep)
    store.shutdown(5)


def test_dead_writer_is_noticed(work_dir, monkeypatch):
    store = make_store(work_dir)
    ep = open_ep(store)
    assert store.flush(10)

    class Boom(BaseException):
        pass

    real_queue = store._queue

    class ExplodingQueue:
        def put(self, item):
            real_queue.put(item)

        def get(self, *args, **kwargs):
            raise Boom()

    monkeypatch.setattr(store, "_queue", ExplodingQueue())
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False})  # wakes the writer
    deadline = time.time() + 5
    while time.time() < deadline and store.writer_alive():
        time.sleep(0.05)
    assert not store.writer_alive()
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False})
    assert ep.dead is True
    C.close_episode(ep)


_ATEXIT_SCRIPT = r"""
import sys, time
from pathlib import Path
from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
root = Path(sys.argv[1])
store = S.init_store(TraceConfig(trace_dir=root))
ep = C.open_episode(store, run_id="r", episode_id="e", source="test")
ep.emit("episode_start", data={"tracer": {}})
def slow(ctx):
    time.sleep(float(sys.argv[2]))
for i in range(50):
    ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "i": i})
ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "slow": True}, builder=slow)
ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False, "last": True})
print("exiting", flush=True)
"""


@pytest.mark.parametrize("sleep_s,expect_last", [(0.2, True), (30.0, False)])
def test_atexit_drains_the_queue_within_ten_seconds(work_dir, sleep_s, expect_last):
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", _ATEXIT_SCRIPT, str(work_dir / "t"), str(sleep_s)],
                          cwd=str(REPO), env=clean_env(), capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    events, corrupt, torn = read_events(work_dir / "t/r/e/events.jsonl")
    assert corrupt == 0 and not torn
    assert len([e for e in events if "i" in e["data"]]) == 50
    assert any(e["data"].get("last") for e in events) is expect_last
    if not expect_last:
        assert elapsed < 25  # the atexit wait is bounded at 10 s
