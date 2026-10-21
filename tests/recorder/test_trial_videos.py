"""Completed upstream videos are portable episode artifacts."""

from __future__ import annotations

import hashlib
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.hooks.trial import end_episode
from capx_trace_recorder.hooks import trial as TRIAL
from capx_trace_viewer.reader import read_events_page
from trace_reader import by_type, read_events


def test_trial_import_does_not_require_fcntl():
    proc = subprocess.run([sys.executable, "-c",
                           "import sys; sys.modules['fcntl'] = None; import capx_trace_recorder.hooks.trial"],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr


def record(tmp_path, trial_dir, *, limit=256, disk_low=False, free_after=None, queue_mb=512, monkeypatch=None,
           code_path=None):
    store = S.TraceStore(TraceConfig(trace_dir=tmp_path / "traces", video_file_mb=limit,
                                      max_queue_mb=queue_mb))
    if disk_low:
        original = store.disk_allows
        monkeypatch.setattr(store, "disk_allows", lambda binary, trigger=None: (False, "disk_low") if binary else original(binary, trigger))
    if free_after is not None:
        monkeypatch.setattr(TRIAL.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=store.config.min_free_bytes + free_after))
    ep = C.open_episode(store, run_id="run", episode_id="ep", source="envs.runner")
    ep.emit("episode_start", status="started", data={"source": "envs.runner", "task": "t", "tracer": {}})
    ep.n_code_blocks = 1
    ep.started_wall -= 1
    end_episode(ep, end_reason="completed", video_dir=trial_dir, code_path=code_path)
    C.close_episode(ep)
    assert store.flush(20)
    events, corrupt, torn = read_events(ep.events_path)
    assert not corrupt and not torn
    store.shutdown(5)
    return ep.base_dir, by_type(events, "episode_end")[0], by_type(events, "trial_videos")[0], by_type(events, "trace_warning")


def test_copies_turn_and_combined_videos_with_hashes(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    for name, data in (("video_turn_00.mp4", b"turn"), ("video_combined.mp4", b"combined")):
        (trial / name).write_bytes(data)
    episode, end, video_event, warnings = record(tmp_path, trial)
    assert not warnings
    assert end["seq"] < video_event["seq"] and video_event["late"] is False
    page = read_events_page(episode / "events.jsonl")
    assert page.eof and [json.loads(e)["event_type"] for e in page.event_lines][-2:] == ["episode_end", "trial_videos"]
    assert json.loads((episode / "episode.json").read_text())["trace_complete"] is True
    arts = {art["path"]: art for art in video_event["artifacts"]}
    assert set(arts) == {"videos/video_turn_00.mp4", "videos/video_combined.mp4"}
    for path, art in arts.items():
        data = (episode / path).read_bytes()
        assert data == (trial / path.split("/")[-1]).read_bytes()
        assert art["bytes"] == len(data)
        assert art["sha256"] == hashlib.sha256(data).hexdigest()
        assert art["written"] is True
    assert arts["videos/video_turn_00.mp4"]["turn_index"] == 0
    assert arts["videos/video_combined.mp4"]["turn_index"] is None


def test_copy_refreshes_heartbeat(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_combined.mp4").write_bytes(b"video")
    touched = []
    original = os.utime

    def track(path, *args, **kwargs):
        # Count only refreshes made while a copy's temporary file exists.
        if str(path).endswith("heartbeat") and any(Path(path).parent.glob("videos/.video_*.tmp.*")):
            touched.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(TRIAL.os, "utime", track)
    episode, _, videos, _ = record(tmp_path, trial)
    assert videos["artifacts"][0]["written"] is True
    assert episode / "heartbeat" in touched


def test_video_copy_unsupported_platform_warns_once(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_combined.mp4").write_bytes(b"video")
    monkeypatch.setattr(TRIAL, "_video_copy_supported", lambda: False)
    _, _, videos, warnings = record(tmp_path, trial)
    assert videos["artifacts"] == []
    assert [w["data"]["reason"] for w in warnings] == ["unsupported_platform"]


@pytest.mark.skipif(sys.platform != "linux", reason="FIFO video test requires Linux")
def test_fifo_is_skipped_and_later_events_are_written(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    os.mkfifo(trial / "video_turn_00.mp4")
    store = S.TraceStore(TraceConfig(trace_dir=tmp_path / "traces"))
    ep = C.open_episode(store, run_id="run", episode_id="ep", source="envs.runner")
    ep.emit("episode_start", status="started", data={"source": "envs.runner", "tracer": {}})
    ep.n_code_blocks = 1
    end_episode(ep, end_reason="completed", video_dir=trial)
    ep.emit("trace_warning", status="warning", writer_generated=True,
            data={"code": "after_video", "counts_as_gap": False})
    C.close_episode(ep)
    assert store.flush(5), "opening the FIFO must not block the writer"
    events, _, _ = read_events(ep.events_path)
    assert by_type(events, "trial_videos")[0]["artifacts"][0]["reason"] == "not_regular"
    assert any(e["data"].get("code") == "after_video" for e in events)
    store.shutdown(5)


def test_video_mtime_uses_code_file_clock_and_excludes_old_run(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    code = trial / "code.py"
    code.write_text("pass")
    (trial / "video_turn_00.mp4").write_bytes(b"current")
    (trial / "video_combined.mp4").write_bytes(b"old")
    os.utime(code, (1000, 1000))
    os.utime(trial / "video_turn_00.mp4", (999, 999))
    os.utime(trial / "video_combined.mp4", (700, 700))
    episode, _, videos, _ = record(tmp_path, trial, code_path=code)
    arts = {a["path"]: a for a in videos["artifacts"]}
    assert arts["videos/video_turn_00.mp4"]["written"] is True
    assert arts["videos/video_combined.mp4"]["reason"] == "stale"
    assert json.loads((episode / "episode.json").read_text())["trace_complete"] is True


def test_copy_pending_is_persisted_before_copy_and_cleared_after(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_combined.mp4").write_bytes(b"video")
    entered, release = threading.Event(), threading.Event()
    original = os.read
    def blocked(fd, count):
        entered.set()
        assert release.wait(5)
        return original(fd, count)
    monkeypatch.setattr(os, "read", blocked)
    store = S.TraceStore(TraceConfig(trace_dir=tmp_path / "traces"))
    ep = C.open_episode(store, run_id="run", episode_id="ep", source="envs.runner")
    ep.emit("episode_start", status="started", data={"source": "envs.runner", "tracer": {}})
    ep.n_code_blocks = 1
    end_episode(ep, end_reason="completed", video_dir=trial)
    C.close_episode(ep)
    try:
        assert entered.wait(5)
        doc = json.loads((ep.base_dir / "episode.json").read_text())
        assert doc["status"] == "ended" and doc["video_copy_pending"] is True
        assert not by_type(read_events(ep.events_path)[0], "trial_videos")
    finally:
        release.set()
    assert store.flush(5)
    assert json.loads((ep.base_dir / "episode.json").read_text())["video_copy_pending"] is False
    store.shutdown(5)


def test_large_video_is_copied_in_bounded_chunks(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    content = b"a" * (2 * 1024 * 1024 + 17)
    (trial / "video_combined.mp4").write_bytes(content)
    seen = []
    original = os.read

    def measured(fd, count):
        seen.append(count)
        return original(fd, count)

    monkeypatch.setattr(os, "read", measured)
    episode, _, videos, _ = record(tmp_path, trial)
    assert (episode / "videos/video_combined.mp4").read_bytes() == content
    assert videos["artifacts"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert seen and max(seen) <= 1024 * 1024


def test_video_checks_free_space_after_copy_and_ignores_queue_budget(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_combined.mp4").write_bytes(b"video")
    with monkeypatch.context() as patch:
        episode, _, videos, warnings = record(tmp_path / "low", trial, free_after=4, monkeypatch=patch)
    assert videos["artifacts"][0]["written"] is False
    assert len(json.loads((episode / "episode.json").read_text())["gaps"]) == 1
    assert [w["data"]["code"] for w in warnings] == ["disk_low", "video_skipped"]
    episode, _, videos, warnings = record(tmp_path / "queue", trial, queue_mb=0)
    assert videos["artifacts"][0]["written"] is True
    assert not warnings
    assert json.loads((episode / "episode.json").read_text())["trace_complete"] is True


def test_zero_limit_warns_once_and_disk_low_warns_once_per_episode(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    for name in ("video_turn_00.mp4", "video_combined.mp4"):
        (trial / name).write_bytes(b"video")
    _, _, videos, warnings = record(tmp_path / "zero", trial, limit=0)
    assert videos["artifacts"] == []
    assert [(w["data"]["code"], w["data"].get("reason")) for w in warnings] == [("video_size_limit", "disabled")]
    with monkeypatch.context() as patch:
        _, _, videos, warnings = record(tmp_path / "low", trial, free_after=0, monkeypatch=patch)
    assert len(videos["artifacts"]) == 2
    assert [w["data"]["code"] for w in warnings].count("disk_low") == 1


def test_non_ascii_video_name_is_not_selected(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_turn_０１.mp4").write_bytes(b"video")
    _, _, videos, warnings = record(tmp_path, trial)
    assert videos["artifacts"] == []
    assert [w["data"]["code"] for w in warnings] == ["video_missing"]


@pytest.mark.parametrize(("mode", "code"), [("missing", "video_missing"), ("oversize", "video_size_limit"), ("disk_low", "disk_low")])
def test_video_skip_warns_and_episode_ends(tmp_path, monkeypatch, mode, code):
    trial = tmp_path / "trial"
    if mode != "missing":
        trial.mkdir()
        (trial / "video_turn_00.mp4").write_bytes(b"1234")
    episode, end, video_event, warnings = record(tmp_path, trial, limit=0.000001 if mode == "oversize" else 256,
                                    disk_low=mode == "disk_low", monkeypatch=monkeypatch)
    assert not list((episode / "videos").glob("*.mp4")) if (episode / "videos").exists() else True
    assert end["data"]["end_reason"] == "completed"
    assert code in [event["data"]["code"] for event in warnings]
    if mode in ("missing", "oversize"):
        assert end["data"]["trace_complete"] is True
        assert not json.loads((episode / "episode.json").read_text())["gaps"]
    else:
        assert len(json.loads((episode / "episode.json").read_text())["gaps"]) == 1


@dataclasses.dataclass
class Summary:
    code_path: str | None
    task_completed: bool = True


@pytest.mark.parametrize("source,record_video,code_path,raises,expect_video", [
    ("upstream", True, True, False, True),
    ("upstream", False, True, False, False),
    ("upstream", True, False, False, False),
    ("upstream", True, True, "timeout", False),
    ("upstream", True, True, "exception", False),
    ("fork", True, True, False, False),
    ("symlink", True, True, False, True),
])
def test_h4_video_trigger(tmp_path, monkeypatch, source, record_video, code_path, raises, expect_video):
    store = S.TraceStore(TraceConfig(trace_dir=tmp_path / "traces"))
    monkeypatch.setattr(TRIAL, "get_store", lambda: store)
    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "videos_h4")
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    monkeypatch.setattr(C.REGISTRY, "runner", C.RunnerContext())
    output = tmp_path / "output"
    if source == "fork":
        runner = tmp_path / "runner"
        runner.mkdir()
        (runner / "run.json").write_text('{"invocations":[],"conditions":[]}')
        output = runner / "task" / "condition"
    output.mkdir(parents=True)
    trial = output / "trial_1"
    trial.mkdir()
    video = trial / "video_combined.mp4"
    if source == "symlink":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"private")
        video.symlink_to(outside)
    else:
        video.write_bytes(b"video")
    if source == "upstream":
        monkeypatch.chdir(tmp_path)
    summary = Summary(str((trial / "code.py").relative_to(tmp_path)) if code_path and source == "upstream"
                      else str(trial / "code.py") if code_path else None)

    def fake(env, trial_number, args, config):
        ep, _ = C.resolve_episode()
        ep.n_code_blocks = 1
        if source != "symlink":
            os.utime(video, (ep.started_wall + 1, ep.started_wall + 1))
        if raises == "timeout":
            raise TimeoutError("trial timeout")
        if raises == "exception":
            raise RuntimeError("trial error")
        return summary

    wrapped = TRIAL.make_run_single_trial_wrapper(fake)
    config = {"output_dir": str(output), "record_video": record_video}
    args = types.SimpleNamespace(config_path="config.yaml")
    if raises:
        with pytest.raises(TimeoutError if raises == "timeout" else RuntimeError):
            wrapped(types.SimpleNamespace(low_level_env=None), 1, args, config)
    else:
        assert wrapped(types.SimpleNamespace(low_level_env=None), 1, args, config) is summary
    assert store.flush(20)
    episodes = list((tmp_path / "traces").glob("*/**/events.jsonl"))
    assert len(episodes) == 1
    events, _, _ = read_events(episodes[0])
    assert bool(by_type(events, "trial_videos")) is expect_video
    if expect_video:
        assert by_type(events, "trial_videos")[0]["artifacts"][0]["written"] is (source != "symlink")
    store.shutdown(5)


def test_stale_and_symlink_videos_are_skipped_without_gap(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    for name in ("video_turn_00.mp4", "video_turn_01.mp4", "video_combined.mp4"):
        (trial / name).write_bytes(b"old")
    os.utime(trial / "video_turn_00.mp4", (1, 1))
    os.utime(trial / "video_combined.mp4", (1, 1))
    (trial / "video_turn_02.mp4").symlink_to(trial / "video_turn_01.mp4")
    episode, end, videos, warnings = record(tmp_path, trial)
    assert end["data"]["trace_complete"] is True
    assert not list((episode / "videos").glob("*.mp4"))
    assert len([w for w in warnings if w["data"]["reason"] == "stale"]) == 2
    assert len([w for w in warnings if w["data"]["reason"] == "turn_index_out_of_range"]) == 1
    assert any(w["data"]["reason"] == "not_regular" for w in warnings)


def test_reward_named_video_without_frames_is_not_a_gap(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "video_1.000.mp4").write_bytes(b"upstream fallback")
    episode, end, videos, warnings = record(tmp_path, trial)
    assert videos["artifacts"] == []
    assert warnings[0]["data"]["code"] == "video_missing"
    assert json.loads((episode / "episode.json").read_text())["trace_complete"] is True


@pytest.mark.parametrize("failure", ["open", "read", "write"])
def test_h4_returns_original_summary_after_video_io_failure(tmp_path, monkeypatch, failure):
    store = S.TraceStore(TraceConfig(trace_dir=tmp_path / "traces"))
    monkeypatch.setattr(TRIAL, "get_store", lambda: store)
    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "io_failure")
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    output = tmp_path / "output"
    output.mkdir()
    trial = output / "trial"
    trial.mkdir()
    (trial / "video_combined.mp4").write_bytes(b"video")
    summary = Summary(str(trial / "code.py"))

    def fake(env, trial_number, args, config):
        ep, _ = C.resolve_episode()
        ep.n_code_blocks = 1
        os.utime(trial / "video_combined.mp4", (ep.started_wall + 1, ep.started_wall + 1))
        return summary

    with monkeypatch.context() as patch:
        if failure == "open":
            real = os.open
            def fail_video_open(path, flags, mode=0o777, *, dir_fd=None):
                if path == "video_combined.mp4" and dir_fd is not None:
                    raise PermissionError("open failed")
                return real(path, flags, mode, dir_fd=dir_fd)
            patch.setattr(os, "open", fail_video_open)
        elif failure == "read":
            patch.setattr(os, "read", lambda fd, n: (_ for _ in ()).throw(OSError("read failed")))
        else:
            real = os.replace
            def fail_video_replace(src, dst):
                if str(dst).endswith("video_combined.mp4"):
                    raise OSError("write failed")
                return real(src, dst)
            patch.setattr(os, "replace", fail_video_replace)
        wrapped = TRIAL.make_run_single_trial_wrapper(fake)
        assert wrapped(types.SimpleNamespace(low_level_env=None), 1,
                       types.SimpleNamespace(config_path="config.yaml"),
                       {"output_dir": str(output), "record_video": True}) is summary
        assert store.flush(20)
    events_path = next((tmp_path / "traces").glob("*/**/events.jsonl"))
    events, _, _ = read_events(events_path)
    assert by_type(events, "episode_end")
    assert by_type(events, "trial_videos")
    doc = json.loads((events_path.parent / "episode.json").read_text())
    assert doc["trace_complete"] is False
    assert doc["video_copy_pending"] is False
    assert [g["code"] for g in doc["gaps"]] == ["read_failed" if failure != "write" else "write_failed"]
    store.shutdown(5)
