"""Tests of capx_trace_viewer/reader.py (docs/trace-format.md sections 4.5 and 9.2).

Covers the tolerant events.jsonl reader (torn and corrupt lines, byte-offset paging, the
(inode, size) page cache), episode status from episode.json and the heartbeat, run
listing, path checks, and the section 2 rule that the reader imports nothing heavy and
emits nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from capx_trace_viewer import reader
from trace_viewer_testlib import EPISODE_ENDED, EPISODE_INCOMPLETE, EPISODE_RUNNING, REPO_ROOT, RUN_ID


def _event(seq: int, **extra: object) -> dict:
    return {"schema_version": "capx-trace/1", "event_id": f"ev_{seq:020d}", "seq": seq, "event_type": "frame", "data": {}, **extra}


def _line(seq: int, **extra: object) -> bytes:
    return json.dumps(_event(seq, **extra)).encode() + b"\n"


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# ids and paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["run1", "a", "cube_stack__M4__s1__i0__a1", "A.b-c_d", "x" * 200])
def test_valid_ids(value: str) -> None:
    assert reader.is_valid_id(value)


@pytest.mark.parametrize(
    "value", ["", ".", "..", "...", ".hidden", "-x", "_unscoped", "a/b", "a\\b", "a\n", "a b", "%2E%2E", "x" * 201, "é", None, 3]
)
def test_invalid_ids(value: object) -> None:
    assert not reader.is_valid_id(value)


def test_safe_join_rejects_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "run").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link_out").symlink_to(outside)
    (root / "link_in").symlink_to(root / "run")
    assert reader.safe_join(root, "run") == (root / "run").resolve()
    assert reader.safe_join(root, "link_in") == (root / "run").resolve()
    assert reader.safe_join(root, "link_out") is None
    assert reader.safe_join(root, "..") is None
    assert reader.safe_join(root, "run", "..") is None
    assert reader.safe_join(root) is None


def test_split_relpath() -> None:
    assert reader.split_relpath("assets/ab/abc.png") == ["assets", "ab", "abc.png"]
    for bad in ["", "/etc/passwd", "a//b", "a/../b", "./a", "a/", "..", "a/.", "a\x00b", "x" * 2000]:
        assert reader.split_relpath(bad) is None, bad


def test_own_file_rejects_symlinks_that_leave_the_directory(tmp_path: Path) -> None:
    ep = tmp_path / "ep"
    ep.mkdir()
    secret = _write(tmp_path / "secret.json", b'{"secret": 1}')
    (ep / "episode.json").symlink_to(secret)
    _write(ep / "events.jsonl", b"")
    assert reader.own_file(ep, "episode.json") is None
    assert reader.own_file(ep, "events.jsonl") == (ep / "events.jsonl").resolve()
    assert reader.own_file(ep, "missing") is None


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------


def test_page_reads_complete_lines(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", b"".join(_line(i) for i in range(5)))
    page = reader.read_events_page(path)
    assert [e["seq"] for e in page.events()] == [0, 1, 2, 3, 4]
    assert page.next_offset == path.stat().st_size
    assert page.eof and not page.torn_tail and page.corrupt_lines == 0


def test_torn_tail_is_not_consumed_and_completes_later(tmp_path: Path) -> None:
    full = _line(2)
    path = _write(tmp_path / "events.jsonl", _line(0) + _line(1) + full[:20])
    page = reader.read_events_page(path)
    assert [e["seq"] for e in page.events()] == [0, 1]
    assert page.torn_tail and page.eof
    assert page.next_offset == len(_line(0) + _line(1))
    with open(path, "ab") as fh:
        fh.write(full[20:])
    page2 = reader.read_events_page(path, page.next_offset)
    assert [e["seq"] for e in page2.events()] == [2]
    assert not page2.torn_tail and page2.eof


def test_corrupt_lines_anywhere_are_skipped_and_counted(tmp_path: Path) -> None:
    data = (
        _line(0)
        + b'{"seq": 1, "event_id": "ev_\x00\x00\n'  # damaged line
        + b"not json\n"
        + b"[1, 2, 3]\n"  # valid JSON but not an object
        + b'"just a string"\n'
        + b"\xff\xfe{}\n"  # not UTF-8
        + b"\n"  # empty line
        + _line(7)
    )
    path = _write(tmp_path / "events.jsonl", data)
    page = reader.read_events_page(path)
    assert [e["seq"] for e in page.events()] == [0, 7]
    assert page.corrupt_lines == 6
    assert page.corrupt_offsets[0] == len(_line(0))
    assert page.eof and not page.torn_tail


def test_deeply_nested_line_is_corrupt_not_a_crash(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", b"[" * 100000 + b"]" * 100000 + b"\n" + _line(1))
    page = reader.read_events_page(path, max_bytes=10_000_000)
    assert [e["seq"] for e in page.events()] == [1]
    assert page.corrupt_lines == 1


def test_non_finite_numbers_become_strings(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", b'{"seq": 0, "data": {"x": NaN, "y": Infinity, "z": -Infinity}}\n')
    page = reader.read_events_page(path)
    body = page.to_json_bytes()
    parsed = json.loads(body, parse_constant=lambda c: pytest.fail(f"non-standard constant {c}"))
    assert parsed["events"][0]["data"] == {"x": "nan", "y": "inf", "z": "-inf"}


def test_paging_by_byte_offset_covers_every_event_once(tmp_path: Path) -> None:
    data = b"".join(_line(i, pad="x" * (i % 7) * 50) for i in range(200))
    path = _write(tmp_path / "events.jsonl", data)
    for max_bytes, max_events in [(1, 5000), (700, 5000), (5000, 3), (10**9, 1)]:
        offset, seqs, pages = 0, [], 0
        while True:
            page = reader.read_events_page(path, offset, max_bytes, max_events)
            seqs += [e["seq"] for e in page.events()]
            pages += 1
            assert page.next_offset >= offset
            if page.eof:
                break
            assert page.next_offset > offset, "paging must advance"
            offset = page.next_offset
        assert seqs == list(range(200)), (max_bytes, max_events)
        assert pages > 1


def test_line_longer_than_max_bytes_is_returned_whole(tmp_path: Path) -> None:
    big = _line(0, pad="y" * 5000)
    path = _write(tmp_path / "events.jsonl", big + _line(1))
    page = reader.read_events_page(path, 0, max_bytes=100)
    assert [e["seq"] for e in page.events()] == [0]
    assert page.next_offset == len(big) and not page.eof


def test_line_longer_than_the_hard_limit_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader, "MAX_LINE_BYTES", 1000)
    monkeypatch.setattr(reader, "_SCAN_CHUNK", 256)
    big = _line(0, pad="z" * 5000)
    path = _write(tmp_path / "events.jsonl", big + _line(1))
    page = reader.read_events_page(path, 0, max_bytes=100)
    assert page.events() == [] and page.corrupt_lines == 1
    assert page.next_offset == len(big)
    assert [e["seq"] for e in reader.read_events_page(path, page.next_offset).events()] == [1]


def test_huge_torn_tail_does_not_advance(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", _line(0) + b'{"seq": 1, "pad": "' + b"q" * 50000)
    page = reader.read_events_page(path, 0, max_bytes=100)
    assert [e["seq"] for e in page.events()] == [0]
    page2 = reader.read_events_page(path, page.next_offset, max_bytes=100)
    assert page2.events() == [] and page2.torn_tail and page2.eof
    assert page2.next_offset == page.next_offset


@pytest.mark.parametrize("offset", [-1, 1, 5, 10**9])
def test_bad_offsets_are_rejected(tmp_path: Path, offset: int) -> None:
    path = _write(tmp_path / "events.jsonl", _line(0) + _line(1))
    with pytest.raises(reader.EventsOffsetError):
        reader.read_events_page(path, offset)


def test_offset_at_line_start_and_at_eof_are_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", _line(0) + _line(1))
    assert [e["seq"] for e in reader.read_events_page(path, len(_line(0))).events()] == [1]
    end = reader.read_events_page(path, path.stat().st_size)
    assert end.events() == [] and end.eof


def test_page_cache_is_keyed_by_inode_and_size(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", _line(0))
    cache = reader.EventsPageCache()
    first = cache.get(path)
    assert cache.get(path) is first
    with open(path, "ab") as fh:
        fh.write(_line(1))
    second = cache.get(path)
    assert second is not first and [e["seq"] for e in second.events()] == [0, 1]
    replacement = tmp_path / "new.jsonl"
    replacement.write_bytes(_line(5) + _line(6))
    os.replace(replacement, path)  # new inode
    third = cache.get(path)
    assert [e["seq"] for e in third.events()] == [5, 6]


def test_page_cache_evicts_by_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", b"".join(_line(i, pad="p" * 500) for i in range(20)))
    cache = reader.EventsPageCache(max_entries=1000, max_bytes=3000)
    for offset_events in range(1, 15):
        cache.get(path, 0, max_bytes=10**6, max_events=offset_events)
    assert cache._bytes <= 3000


def test_iter_events_and_first_event(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", b"".join(_line(i) for i in range(10)) + b"garbage\n" + b'{"torn"')
    assert [e["seq"] for e in reader.iter_events(path, page_bytes=64)] == list(range(10))
    assert reader.read_first_event(path)["seq"] == 0


# ---------------------------------------------------------------------------
# status and summaries
# ---------------------------------------------------------------------------


def _episode(tmp_path: Path, name: str, *, episode_json: dict | None, heartbeat_age: float | None) -> Path:
    ep = tmp_path / "run" / name
    ep.mkdir(parents=True)
    _write(ep / "events.jsonl", json.dumps({"event_type": "episode_start", "seq": 0, "wall_time": 123.0, "pid": 7, "data": {"task": "t_from_event", "seed": 9, "env": {"viser_port": 8081}}}).encode() + b"\n")
    if episode_json is not None:
        _write(ep / "episode.json", json.dumps(episode_json).encode())
    if heartbeat_age is not None:
        hb = ep / "heartbeat"
        hb.touch()
        t = time.time() - heartbeat_age
        os.utime(hb, (t, t))
    return ep


@pytest.mark.parametrize(
    ("episode_json", "age", "status"),
    [
        ({"status": "ended", "trace_complete": True}, 999, "ended"),
        ({"status": "ended", "trace_complete": False, "gaps": [{"code": "disk_low", "count": 1, "first_seq": 3}]}, 999, "ended_with_gaps"),
        ({"status": "ended"}, 1, "ended_with_gaps"),  # trace_complete missing is not complete
        ({"status": "open"}, 1, "running"),
        ({"status": "open"}, 29, "running"),
        ({"status": "open"}, 31, "incomplete"),
        ({"status": "running"}, 3600, "incomplete"),  # the recorded word is never trusted
        (None, 3600, "incomplete"),
    ],
)
def test_status_from_episode_json_and_heartbeat(tmp_path: Path, episode_json: dict | None, age: float, status: str) -> None:
    ep = _episode(tmp_path, "ep", episode_json=episode_json, heartbeat_age=age)
    # events.jsonl mtime must not keep a stale episode alive when the heartbeat exists
    old = time.time() - 3600
    os.utime(ep / "events.jsonl", (old, old))
    summary = reader.episode_summary("run", ep)
    assert summary["status"] == status
    assert summary["liveness_source"] == "heartbeat"


def test_missing_heartbeat_is_incomplete_even_with_fresh_files(tmp_path: Path) -> None:
    """A copied or unpacked crashed episode has fresh mtimes but no heartbeat: not running."""
    ep = _episode(tmp_path, "ep", episode_json={"status": "open"}, heartbeat_age=None)
    now = time.time()
    for name in ("events.jsonl", "episode.json"):
        os.utime(ep / name, (now, now))
    summary = reader.episode_summary("run", ep)
    assert summary["status"] == "incomplete"
    assert summary["liveness_source"] == "none" and summary["heartbeat_age_s"] is None
    assert "no heartbeat file" in summary["status_detail"]
    assert reader.liveness_age(ep, now) == (None, "none")
    (ep / "heartbeat").touch()
    running = reader.episode_summary("run", ep)
    assert running["status"] == "running" and running["status_detail"] == "no episode_end; heartbeat file younger than 30 s"
    assert running["liveness_source"] == "heartbeat"


def test_symlinked_heartbeat_does_not_count(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep", episode_json={"status": "open"}, heartbeat_age=None)
    fresh = _write(tmp_path / "fresh", b"")
    (ep / "heartbeat").symlink_to(fresh)
    assert reader.episode_summary("run", ep)["status"] == "incomplete"


def test_summary_fields_from_episode_json_or_first_event(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep1", episode_json={"status": "open", "task": "t_json", "seed": 1, "gaps": []}, heartbeat_age=1)
    summary = reader.episode_summary("run", ep)
    assert summary["task"] == "t_json" and summary["seed"] == 1 and summary["episode_id"] == "ep1"
    ep2 = _episode(tmp_path, "ep2", episode_json=None, heartbeat_age=1)
    summary2 = reader.episode_summary("run", ep2)
    assert summary2["task"] == "t_from_event" and summary2["seed"] == 9 and summary2["viser_port"] == 8081
    assert summary2["start_wall_time"] == 123.0 and summary2["episode_json_error"] == "missing"


def test_nested_episode_json_fields_are_found(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep", episode_json={"status": "ended", "trace_complete": True, "start": {"task": "nested"}, "end": {"end_reason": "completed"}}, heartbeat_age=999)
    summary = reader.episode_summary("run", ep)
    assert summary["task"] == "nested" and summary["end_reason"] == "completed"


def test_invalid_episode_json_is_reported(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep", episode_json=None, heartbeat_age=999)
    _write(ep / "episode.json", b'{"status": "ended", "trace_complete": tru')
    summary = reader.episode_summary("run", ep)
    assert summary["episode_json_error"] == "invalid_json" and summary["status"] == "incomplete"


def test_symlinked_episode_json_outside_is_ignored(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep", episode_json=None, heartbeat_age=999)
    secret = _write(tmp_path / "secret.json", b'{"status": "ended", "trace_complete": true, "task": "TOP-SECRET"}')
    (ep / "episode.json").symlink_to(secret)
    summary = reader.episode_summary("run", ep)
    assert summary["episode"] is None and "TOP-SECRET" not in json.dumps(summary)


def test_list_runs_skips_unscoped_and_odd_names_and_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "traces"
    for name, age in (("old_run", 500), ("new_run", 10)):
        ep = root / name / "ep"
        ep.mkdir(parents=True)
        _write(ep / "episode.json", b'{"status": "ended", "trace_complete": true}')
        t = time.time() - age
        os.utime(ep / "episode.json", (t, t))
    (root / "_unscoped").mkdir()
    _write(root / "_unscoped" / "unscoped.1.jsonl", _line(0))
    (root / ".hidden").mkdir()
    (root / "empty_dir").mkdir()
    _write(root / "file.txt", b"x")
    runs = reader.list_runs(root)
    assert [r["run_id"] for r in runs] == ["new_run", "old_run"]
    assert runs[0]["status_counts"]["ended"] == 1 and runs[0]["n_episodes"] == 1


def test_list_runs_on_missing_root(tmp_path: Path) -> None:
    assert reader.list_runs(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# demo data
# ---------------------------------------------------------------------------


def test_demo_statuses(demo_root: Path) -> None:
    run_dir = demo_root / RUN_ID
    os.utime(run_dir / EPISODE_RUNNING / "heartbeat")  # the demo's running episode is fresh
    detail = reader.run_detail(demo_root, RUN_ID)
    assert detail is not None
    statuses = {e["episode_id"]: e["status"] for e in detail["episodes"]}
    assert statuses == {EPISODE_ENDED: "ended_with_gaps", EPISODE_RUNNING: "running", EPISODE_INCOMPLETE: "incomplete"}
    ended = next(e for e in detail["episodes"] if e["episode_id"] == EPISODE_ENDED)
    assert {g["code"] for g in ended["gaps"]} == {"disk_low", "late_event"}
    assert len(detail["runner_records"]) == 2
    assert detail["unscoped_files"] == [
        {"file": "unscoped.412300.jsonl", "bytes": (run_dir / "unscoped.412300.jsonl").stat().st_size,
         "log_id": "unscoped.412300", "write_status": None}
    ]
    runs = reader.list_runs(demo_root)
    assert [r["run_id"] for r in runs] == [RUN_ID]


def test_demo_incomplete_episode_has_corrupt_line_and_torn_tail(demo_root: Path) -> None:
    path = demo_root / RUN_ID / EPISODE_INCOMPLETE / "events.jsonl"
    page = reader.read_events_page(path)
    assert page.corrupt_lines == 1 and page.torn_tail and page.eof
    seqs = [e["seq"] for e in page.events()]
    assert seqs[0] == 0 and len(set(seqs)) == len(seqs)
    assert any(b - a == 2 for a, b in zip(seqs, seqs[1:])), "the damaged line leaves a seq gap"


def test_demo_covers_every_event_type(demo_root: Path) -> None:
    types = set()
    for ep in (EPISODE_ENDED, EPISODE_RUNNING, EPISODE_INCOMPLETE):
        types |= {e["event_type"] for e in reader.iter_events(demo_root / RUN_ID / ep / "events.jsonl")}
    expected = {
        "episode_start", "episode_end", "task_text", "model_request", "model_response", "model_error",
        "model_output_parsed", "feedback_built", "code_block_start", "code_block_end", "stdout", "stderr",
        "api_call_start", "api_call_end", "api_call_error", "service_call_start", "service_call_end",
        "service_call_error", "observation_snapshot", "video_frames_read", "frame", "env_reset",
        "evaluator_result", "trace_warning",
    }
    assert expected <= types


def test_demo_input_summary_offsets_point_at_the_segments(demo_root: Path) -> None:
    ep = demo_root / RUN_ID / EPISODE_ENDED
    events = list(reader.iter_events(ep / "events.jsonl"))
    stdout_texts = {e["event_id"]: e["data"]["text"] for e in events if e["event_type"] in ("stdout", "stderr")}
    decision = [e for e in events if e["event_type"] == "model_request" and e["data"]["call_kind"] == "multi_turn_decision"][0]
    payload = json.loads((ep / decision["data"]["payload_path"]).read_text())
    user_text = payload["messages"][1]["content"][0]["text"]
    hits = decision["data"]["input_summary"]["categories"]["stdout_stderr"]
    assert {h["kind"] for h in hits} == {"stdout", "stderr"}
    for hit in hits:
        assert hit["path"] == "messages[1].content[0].text"
        (source,) = hit["source_event_ids"]
        assert user_text[hit["start"]:hit["end"]] == stdout_texts[source]
    not_found = decision["data"]["input_summary"]["registered_not_found"]
    assert any(n["kind"] == "vdm_description" for n in not_found)


# ---------------------------------------------------------------------------
# section 2 rules
# ---------------------------------------------------------------------------


def test_reader_imports_nothing_heavy() -> None:
    code = (
        "import sys, capx_trace_viewer.reader as r\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'PIL', 'fastapi', 'torch', 'robosuite') or m.startswith('capx.web'))\n"
        "print(bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True, env={**os.environ, "CAPX_TRACE_DIR": ""})
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_reader_emits_nothing_under_warnings_as_errors(demo_root: Path, tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    broken = _write(tmp_path / "events.jsonl", b'{"x": NaN}\n\xff\n[' * 3 + b'{"torn')
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reader.list_runs(demo_root)
        detail = reader.run_detail(demo_root, RUN_ID)
        for ep in detail["episodes"]:
            list(reader.iter_events(demo_root / RUN_ID / ep["episode_id"] / "events.jsonl"))
        reader.read_events_page(broken).to_json_bytes()
        reader.read_jsonl(broken)
    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""


def test_recorder_field_names_are_read(tmp_path: Path) -> None:
    """episode.json as the trace recorder writes it (started_/ended_wall_time, env)."""
    ep = _episode(
        tmp_path,
        "ep",
        episode_json={
            "status": "ended",
            "episode_end_written": True,
            "trace_complete": True,
            "gaps": [],
            "started_wall_time": 100.0,
            "ended_wall_time": 150.0,
            "env": {"class": "X", "viser_port": 8080},
            "task": "t",
        },
        heartbeat_age=999,
    )
    summary = reader.episode_summary("run", ep)
    assert summary["status"] == "ended"
    assert (summary["start_wall_time"], summary["end_wall_time"], summary["viser_port"]) == (100.0, 150.0, 8080)


def test_running_word_in_episode_json_is_not_trusted(tmp_path: Path) -> None:
    ep = _episode(tmp_path, "ep", episode_json={"status": "running", "episode_end_written": False}, heartbeat_age=45)
    assert reader.episode_summary("run", ep)["status"] == "incomplete"


# ---------------------------------------------------------------------------
# unscoped logs (events with no episode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "ok"),
    [("unscoped.412300", True), ("unscoped.1", True), ("unscoped.", False), ("unscoped.x", False),
     ("unscoped.1.jsonl", False), ("unscoped.1\n", False), ("xunscoped.1", False), ("unscoped." + "1" * 13, False), (None, False)],
)
def test_is_unscoped_log_id(value: object, ok: bool) -> None:
    assert reader.is_unscoped_log_id(value) is ok


def test_unscoped_summary(demo_root: Path) -> None:
    run_dir = demo_root / RUN_ID
    assert reader.unscoped_events_file(run_dir, "unscoped.412300") == (run_dir / "unscoped.412300.jsonl").resolve()
    assert reader.unscoped_events_file(run_dir, "unscoped.1") is None
    assert reader.unscoped_events_file(run_dir, "../x") is None
    summary = reader.unscoped_summary(RUN_ID, run_dir, "unscoped.412300", now=10**10)
    assert summary["status"] == reader.UNSCOPED_STATUS and summary["kind"] == "unscoped"
    assert summary["episode_id"] == "unscoped.412300" and summary["pid"] == 412300
    assert summary["events_bytes"] == (run_dir / "unscoped.412300.jsonl").stat().st_size
    assert summary["task"] is None and summary["episode"] is None
    assert set(reader.SUMMARY_FIELDS) <= set(summary)


# ---------------------------------------------------------------------------
# start fields, damaged run.json, generation, fd-based opening, demo cases
# ---------------------------------------------------------------------------


def test_first_event_fills_every_missing_start_field(tmp_path: Path) -> None:
    """episode.json with a task but no seed/pid/viser_port: the rest comes from episode_start."""
    ep = _episode(tmp_path, "ep", episode_json={"status": "open", "task": "t_json"}, heartbeat_age=1)
    summary = reader.episode_summary("run", ep)
    assert summary["task"] == "t_json"  # episode.json wins where it has a value
    assert (summary["seed"], summary["viser_port"], summary["pid"], summary["start_wall_time"]) == (9, 8081, 7, 123.0)


def test_first_event_is_not_read_when_episode_json_has_every_start_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = {
        "status": "open", "source": "experiments.run", "task": "t", "condition": "c", "condition_label": "l",
        "seed": 1, "attempt_index": 1, "runner_invocation": 0, "started_wall_time": 1.0,
        "env": {"viser_port": 8080}, "pid": 3, "host": "h",
    }
    ep = _episode(tmp_path, "ep", episode_json=full, heartbeat_age=1)
    calls: list[int] = []
    real = reader._first_event_fd
    monkeypatch.setattr(reader, "_first_event_fd", lambda fd: calls.append(fd) or real(fd))
    assert reader.episode_summary("run", ep)["viser_port"] == 8080
    assert calls == []


def test_start_fields_recorded_as_null_are_not_looked_up_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorder writes every start field, a null one too (a web UI episode has no seed
    and no condition; an env without a Viser server has env.viser_port null)."""
    as_recorded = {
        "status": "running", "source": "web_ui", "task": "t", "condition": None, "condition_label": None,
        "seed": None, "attempt_index": None, "runner_invocation": None, "started_wall_time": 5.0,
        "env": {"class": "X", "viser_port": None}, "pid": 3, "host": "h",
    }
    ep = _episode(tmp_path, "ep", episode_json=as_recorded, heartbeat_age=1)
    calls: list[int] = []
    real = reader._first_event_fd
    monkeypatch.setattr(reader, "_first_event_fd", lambda fd: calls.append(fd) or real(fd))
    summary = reader.episode_summary("run", ep)
    assert calls == [] and summary["seed"] is None and summary["viser_port"] is None and summary["task"] == "t"
    for missing in ("host", "env", "started_wall_time"):  # a field that is absent is looked up
        ep2 = _episode(tmp_path, f"ep_{missing}", episode_json={k: v for k, v in as_recorded.items() if k != missing}, heartbeat_age=1)
        calls.clear()
        summary = reader.episode_summary("run", ep2)
        assert len(calls) == 1, missing
    assert summary["start_wall_time"] == 123.0  # from the episode_start event


def test_run_listing_needs_only_the_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run list reads no events.jsonl; the run detail reads first lines where needed."""
    root = tmp_path / "traces"
    ep = root / "r1" / "ep"
    ep.mkdir(parents=True)
    _write(ep / "episode.json", b'{"status": "ended", "trace_complete": true}')
    _write(ep / "events.jsonl", json.dumps({"event_type": "episode_start", "seq": 0, "data": {"task": "from_event"}}).encode() + b"\n")

    def must_not_run(fd: int) -> None:
        raise AssertionError("events.jsonl was read for the run list")

    monkeypatch.setattr(reader, "_first_event_fd", must_not_run)
    (run,) = reader.list_runs(root)
    assert run["status_counts"]["ended"] == 1 and reader.run_overview(root, "r1")["n_episodes"] == 1
    monkeypatch.undo()
    assert reader.run_detail(root, "r1")["episodes"][0]["task"] == "from_event"


@pytest.mark.parametrize("content", [b"", b'{"run_id": "r1", "created_at": "2026-09'])
def test_run_with_a_damaged_run_json_is_listed_and_detailed(tmp_path: Path, content: bytes) -> None:
    root = tmp_path / "traces"
    _write(root / "r1" / "run.json", content)
    assert [r["run_id"] for r in reader.list_runs(root)] == ["r1"]
    detail = reader.run_detail(root, "r1")
    assert detail is not None and detail["run"] is None and detail["run_json_error"] == "invalid_json"
    assert detail["episodes"] == []


def test_what_counts_as_a_run(tmp_path: Path) -> None:
    root = tmp_path / "traces"
    _write(root / "only_unscoped" / "unscoped.42.jsonl", _line(0))
    _write(root / "only_records" / "runner_records.jsonl", b"")
    (root / "junk" / "sub").mkdir(parents=True)
    _write(root / "junk" / "notes.txt", b"x")
    assert sorted(r["run_id"] for r in reader.list_runs(root)) == ["only_records", "only_unscoped"]
    assert reader.run_detail(root, "only_unscoped")["unscoped_files"][0]["log_id"] == "unscoped.42"
    assert reader.run_detail(root, "junk") is None


def test_generation_identifies_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "events.jsonl", _line(0) + _line(1))
    st = path.stat()
    page = reader.read_events_page(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert reader.events_generation(fd, os.fstat(fd)) == page.generation
    finally:
        os.close(fd)
    gen = page.generation
    assert gen.startswith(f"{st.st_dev:x}-{st.st_ino:x}-") and not gen.endswith("-pending")
    assert json.loads(page.to_json_bytes())["generation"] == gen
    with open(path, "ab") as fh:  # an append keeps the generation
        fh.write(_line(2))
    assert reader.read_events_page(path).generation == gen
    assert len(_line(7)) == len(_line(0))
    with open(path, "r+b") as fh:  # the first line rewritten in place (same inode)
        fh.write(_line(7))
    assert path.stat().st_ino == st.st_ino and reader.read_events_page(path).generation != gen
    replacement = _write(tmp_path / "new.jsonl", _line(0) + _line(1) + _line(2))
    os.replace(replacement, path)  # a new inode, even with the original first line
    assert reader.read_events_page(path).generation.split("-")[1] != f"{st.st_ino:x}"


def test_generation_of_a_file_without_a_complete_first_line(tmp_path: Path) -> None:
    torn = _write(tmp_path / "torn.jsonl", b'{"seq": 0, "event_type": "episode_st')
    assert reader.read_events_page(torn).generation.endswith("-pending")
    empty = _write(tmp_path / "empty.jsonl", b"")
    assert reader.read_events_page(empty).generation.endswith("-pending")
    # a long first line without its newline is still incomplete ("pending"); it
    # used to be hashed by its first 4096 bytes
    long_line = _write(tmp_path / "long.jsonl", b'{"pad": "' + b"x" * 5000)
    assert reader.read_events_page(long_line).generation.endswith("-pending")
    with open(long_line, "ab") as fh:
        fh.write(b'"}\n')
    assert not reader.read_events_page(long_line).generation.endswith("-pending")


def _generation(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY)
    try:
        return reader.events_generation(fd, os.fstat(fd))
    finally:
        os.close(fd)


def test_generation_hashes_the_whole_first_line(tmp_path: Path) -> None:
    """A change anywhere in the first line changes the generation, also
    past its first 4096 bytes and in a line of 256 KB, the recorder's largest."""
    for length in (5000, 256 * 1024):
        head = b'{"pad": "' + b"x" * (length - 14) + b'"}'
        path = _write(tmp_path / f"e{length}.jsonl", head + b"A\n" + _line(1))
        before = _generation(path)
        with open(path, "r+b") as fh:
            fh.seek(len(head))
            fh.write(b"B")  # the byte right before the newline
        after = _generation(path)
        assert before != after and before.rsplit("-", 1)[0] == after.rsplit("-", 1)[0]
        with open(path, "ab") as fh:  # appends keep it
            fh.write(_line(2))
        assert _generation(path) == after


def test_generation_of_a_first_line_longer_than_the_limit_follows_every_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader, "GENERATION_LINE_BYTES", 256)
    path = _write(tmp_path / "long.jsonl", b'{"pad": "' + b"x" * 1000 + b'"}\n' + _line(1))
    first = _generation(path)
    st = path.stat()
    assert first.endswith(f"-{st.st_size:x}-{st.st_mtime_ns:x}") and _generation(path) == first
    with open(path, "ab") as fh:
        fh.write(_line(2))
    assert _generation(path) != first


def test_generation_read_errors_are_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An I/O error is no longer reported as a pending first line."""
    path = _write(tmp_path / "events.jsonl", _line(0))

    def failing_pread(fd: int, size: int, offset: int) -> bytes:
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(reader.os, "pread", failing_pread)
    with pytest.raises(OSError):
        _generation(path)


def test_open_failure_reasons(tmp_path: Path) -> None:
    d = tmp_path / "d"
    _write(d / "f.json", b"{}")
    (d / "link.json").symlink_to(d / "f.json")
    (d / "sub.json").mkdir()
    with reader.Dir.open_root(d) as directory:
        assert reader.read_json_at(directory, "f.json")[:2] == ({}, None)
        assert reader.read_json_at(directory, "nope.json")[:2] == (None, "missing")
        assert reader.read_json_at(directory, "link.json")[:2] == (None, "symlink_not_followed")
        assert reader.read_json_at(directory, "sub.json")[:2] == (None, "not_a_regular_file")
    assert reader.open_failure_reason(PermissionError(13, "denied")) == "unreadable (EACCES)"
    assert reader.open_failure_reason(OSError(5, "io")) == "unreadable (EIO)"
    assert reader.open_failure_reason(OSError("no errno")) == "unreadable (OSError)"


def test_a_json_read_error_is_named_and_not_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A read error of episode.json or run.json said "unreadable (OSError)" and was cached
    for that file state, so the file stayed "unreadable" after the error passed."""
    d = tmp_path / "d"
    _write(d / "run.json", b'{"a": 1}')
    real = reader.pread_upto

    def failing(fd: int, size: int, offset: int) -> bytes:
        raise OSError(5, "Input/output error")

    with reader.Dir.open_root(d) as directory:
        monkeypatch.setattr(reader, "pread_upto", failing)
        assert reader.read_json_at(directory, "run.json")[:2] == (None, "unreadable (EIO)")
        monkeypatch.setattr(reader, "pread_upto", real)
        assert reader.read_json_at(directory, "run.json")[:2] == ({"a": 1}, None)


def test_unscoped_status_files_are_listed(tmp_path: Path) -> None:
    """The recorder keeps the lines of a run's unscoped log that could not be written in
    unscoped.<pid>.status.json. The run detail lists it with the log,
    and also when the log itself could not be created."""
    run = tmp_path / "traces" / "r1"
    status = {"schema_version": "capx-trace/1", "run_id": "r1", "file": "unscoped.100.jsonl", "pid": 100,
              "lines_not_written": 2, "last_reason": "OSError: [Errno 28] No space left on device"}
    _write(run / "run.json", json.dumps({"run_id": "r1"}).encode())
    _write(run / "unscoped.100.jsonl", b"")
    _write(run / "unscoped.100.status.json", json.dumps(status).encode())
    (run / "unscoped.200.jsonl").mkdir()  # the log could not be created at all
    _write(run / "unscoped.200.status.json", json.dumps({**status, "pid": 200, "lines_not_written": 5}).encode())
    _write(run / "unscoped.300.jsonl", b"")
    _write(run / "unscoped.400.status.json", b"{broken")
    _write(run / "unscoped.x.status.json", b"{}")  # not an unscoped log id
    files = reader.run_detail(tmp_path / "traces", "r1")["unscoped_files"]
    assert [f["file"] for f in files] == [f"unscoped.{pid}.jsonl" for pid in (100, 200, 300, 400)]
    by_file = {f["file"]: f for f in files}
    assert by_file["unscoped.100.jsonl"] == {"file": "unscoped.100.jsonl", "bytes": 0, "log_id": "unscoped.100", "write_status": status}
    assert by_file["unscoped.200.jsonl"]["bytes"] is None and by_file["unscoped.200.jsonl"]["log_id"] is None
    assert by_file["unscoped.200.jsonl"]["write_status"]["lines_not_written"] == 5
    assert by_file["unscoped.300.jsonl"] == {"file": "unscoped.300.jsonl", "bytes": 0, "log_id": "unscoped.300", "write_status": None}
    assert by_file["unscoped.400.jsonl"]["write_status"] is None and by_file["unscoped.400.jsonl"]["write_status_error"] == "invalid_json"
    # a run whose only file is such a status file is still a run, in the list and the detail
    _write(tmp_path / "traces" / "r2" / "unscoped.7.status.json", json.dumps(status).encode())
    assert "r2" in {r["run_id"] for r in reader.list_runs(tmp_path / "traces")}
    assert reader.run_detail(tmp_path / "traces", "r2")["unscoped_files"][0]["write_status"] == status


def test_dir_opens_one_segment_at_a_time_without_following_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root / "real" / "f.txt", b"x")
    (root / "real" / "sub").mkdir()
    (root / "inside_link").symlink_to(root / "real")  # stays inside the root: refused too
    (root / "real" / "file_link").symlink_to(root / "real" / "f.txt")
    with reader.Dir.open_root(root) as d:
        with pytest.raises(OSError):
            d.subdir("inside_link")
        assert d.is_symlink("inside_link") and not d.has_subdir("inside_link") and d.has_subdir("real")
        for bad in ("..", ".", "a/b", "", "a\x00b"):
            with pytest.raises(FileNotFoundError):
                d.subdir(bad)
        with d.subdir("real") as real:
            fd, st = real.open_file("f.txt")
            os.close(fd)
            assert st.st_size == 1 and real.file_stat("f.txt") is not None
            with pytest.raises(OSError) as refused:
                real.open_file("file_link")
            assert reader.is_symlink_refusal(refused.value) and real.file_stat("file_link") is None
            with pytest.raises(reader.NotRegularFileError):
                real.open_file("sub")
    with pytest.raises(OSError):
        reader.Dir.open_below(root, ["inside_link", "f.txt"])
    fd, _st = reader.Dir.open_below(root, ["real"]).open_file("f.txt")
    os.close(fd)


@pytest.mark.skipif(sys.platform != "linux", reason="FIFO path test requires Linux")
def test_dir_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    os.mkfifo(root / "fifo")
    with reader.Dir.open_root(root) as d:
        with pytest.raises(reader.NotRegularFileError):
            d.open_file("fifo")  # O_NONBLOCK: opening a FIFO does not wait for a writer


def test_demo_has_an_oversize_event_with_its_data_in_an_asset(demo_root: Path) -> None:
    """Use the trace recorder's oversized event format."""
    ep = demo_root / RUN_ID / EPISODE_ENDED
    assert max(len(line) for line in (ep / "events.jsonl").read_bytes().split(b"\n")) <= 256 * 1024
    events = list(reader.iter_events(ep / "events.jsonl"))
    (over,) = [e for e in events if e["data"].get("oversize")]
    data = over["data"]
    assert over["event_type"] == "api_call_end" and data["name"] == "summarize_workspace"
    assert data["data_asset_written"] is True
    art = next(a for a in over["artifacts"] if a["path"] == data["data_asset"])
    assert (art["label"], art["kind"], art["mime"], art["written"]) == ("payload", "json", "application/gzip", True)
    blob = (ep / data["data_asset"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == art["sha256"] and len(blob) == art["bytes"]
    full = json.loads(gzip.decompress(blob))
    assert full["return"]["type"] == "dict" and full["return"]["items"]["n_objects"] == 36
    inline = {k: v for k, v in data.items() if k not in ("oversize", "data_asset", "data_asset_written")}
    assert inline and all(full[k] == v for k, v in inline.items())
    assert all(v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200) for v in inline.values())
    assert {"return", "stdout_slice", "stderr_slice", "frame_ref", "state_ref"} <= set(full) - set(inline)
    assert len(json.dumps({**over, "data": full}, separators=(",", ":"))) > 256 * 1024


def test_demo_has_an_exception_message_over_4000_characters(demo_root: Path) -> None:
    """As the recorder stores it: text_value() nodes whose txt.gz is not in artifacts."""
    ep = demo_root / RUN_ID / EPISODE_ENDED
    events = list(reader.iter_events(ep / "events.jsonl"))
    (err,) = [e for e in events if e["event_type"] == "api_call_error" and isinstance(e["data"]["message"], dict)]
    texts = {}
    for key in ("message", "traceback"):
        node = err["data"][key]
        text = gzip.decompress((ep / node["asset"]).read_bytes()).decode("utf-8")
        assert node["truncated"] is True and node["written"] is True and node["length"] == len(text) > 4000
        assert node["preview"] == text[:4000] and node["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert not any(a["path"] == node["asset"] for e in events for a in e["artifacts"])
        texts[key] = text
    assert err["data"]["error_type"] == "ValueError"
    assert texts["traceback"].endswith("ValueError: " + texts["message"])


def test_demo_video_frames_read_has_the_recorder_fields(demo_root: Path) -> None:
    """The demo's get_video_frames_range event carries the fields the trace recorder
    writes (start, end, n_frames, camera, frame_pixel_sha256), so the viewer is tested on
    the recorder's shape."""
    events = list(reader.iter_events(demo_root / RUN_ID / EPISODE_RUNNING / "events.jsonl"))
    reads = [e for e in events if e["event_type"] == "video_frames_read"]
    assert reads
    for ev in reads:
        data = ev["data"]
        assert set(data) == {"start", "end", "n_frames", "camera", "frame_pixel_sha256"}
        assert len(data["frame_pixel_sha256"]) == data["n_frames"]
