"""Queue budget, disk floor and write failures (docs/trace-format.md, sections 3, 4.3 and 4.4).

* Over ``CAPX_TRACE_MAX_QUEUE_MB`` an item's binary copies are released when it
  is queued (``TraceStore.submit`` swaps its builder for a light one), so a
  writer that falls behind cannot make the process grow without bound. Every
  dropped item is one ``queue_full`` gap, whichever hook made it (frames,
  harness renders, VDM video frames, H12, API and service arrays, the
  decision-prompt image). Model request bodies are text: they are always
  written, only the images in them are not.
* The disk-low stop writes its final marker to the logs that are still open,
  to the log of the item that noticed it (the next episode, when none was open)
  and to the run's unscoped log; ended episodes keep their complete trace.
  After the stop, ``episode_end`` still releases the episode's references (env,
  frame buffer). A stop that happens inside a builder does not append that
  item's line after the marker.
* A failed ``episode.json`` write, a failed or short ``run.json`` write, a
  failed run-level line (``runner_records.jsonl``) and a failed line of the
  run's unscoped log are recorded (``write_failed``; the unscoped log's own
  failures in ``unscoped.<pid>.status.json``). A gap counted after
  ``episode_end`` reaches ``episode.json``.
* Ground-truth state snapshots count in the queue budget like frames.
* Key-frame references of every camera (``frame_refs``) are resolved.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import sys
import threading
import tracemalloc
import types
import weakref

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import frames as FR
from capx_trace_recorder import store as S
from capx_trace_recorder.hooks import env as ENV
from capx_trace_recorder.hooks import http as HTTP
from capx_trace_recorder.hooks import prompts as PROMPTS
from capx_trace_recorder.hooks import services as SVC
from capx_trace_recorder.hooks import video as VID
from capx_trace_recorder.hooks.api import install_api, uninstall_api
from inprocess import Recorder, by_type, warning_codes

MIB = 1024 * 1024
BUDGET_MB = 2.0
FRAME_SHAPE = (512, 683, 3)  # just over 1 MiB of uint8


def _frame() -> np.ndarray:
    return np.ones(FRAME_SHAPE, dtype=np.uint8)


def _block_writer(ep):
    """Hold the writer inside an item until the returned event is set."""
    gate = threading.Event()
    started = threading.Event()

    def builder(ctx):
        started.set()
        gate.wait(60)

    ep.internal(builder)
    assert started.wait(10)
    return gate


class Low:
    save_camera_name = "cam"

    def __init__(self) -> None:
        self._frame_buffer = []
        self._sim_step_count = 0

    def _record_frame(self) -> None:
        self._frame_buffer.append(_frame())


class RenderEnv:
    """render / get_video_frames_range with the real H7 wrappers."""

    low_level_env = None

    def __init__(self) -> None:
        self.image = _frame()

    def render(self):
        return self.image  # the same object: only the recorder's copy is new memory

    def get_video_frames_range(self, start, end):
        return [_frame()]  # fresh copies, as the real method makes


RenderEnv.render = ENV.make_render_wrapper(RenderEnv.__dict__["render"], "main")
RenderEnv.get_video_frames_range = ENV.make_frames_range_wrapper(
    RenderEnv.__dict__["get_video_frames_range"], "main")


class ArrayApi:
    def __init__(self) -> None:
        self._env = None

    def functions(self):
        return {"take": self.take}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def take(self):
        """Return a fresh 1 MiB array."""
        return np.ones((512, 512), dtype=np.float32)


def _build_prompt(complete_multi_turn_prompt, visual_feedback=None, visual_differencing_feedback=None,
                  is_video_feedback=False):
    return [{"role": "user", "content": complete_multi_turn_prompt}]


def _producers(ep):
    """One callable per hook; each call queues one item holding about 1 MiB (or 2)."""
    low = Low()
    ep.low_env = low
    tap, reason = FR.install_tap(low)
    assert tap is not None, reason
    tap.attach(ep)
    ep.frame_tap = tap
    env = RenderEnv()
    api = ArrayApi()
    assert install_api(api)[0]
    seg = SVC.make_service_wrapper(lambda image, text: np.ones((512, 512), dtype=np.float32),
                                   "sam3", "sam3_seg_fn")
    small = np.zeros((16, 16, 3), dtype=np.uint8)
    encode = VID.make_encode_wrapper(lambda frames, fps=5: "data:video/mp4;base64," + "A" * MIB)
    build = PROMPTS.make_build_wrapper(_build_prompt, "_build_multi_turn_decision_prompt")
    frame = _frame()

    def keyframe(i):
        low._frame_buffer.append(frame)  # one frame object, a new buffer index each time
        tap.keyframe(f"ev_{i:020x}")

    return {
        "frame": keyframe,
        "render": lambda i: env.render(),
        "video_frames_read": lambda i: env.get_video_frames_range(0, 1),
        "video_encode": lambda i: encode([_frame()]),
        "api_return": lambda i: api.take(),
        "service_output": lambda i: seg(small, "cube"),
        "feedback_image": lambda i: build("prompt text of the next decision",
                                          "data:image/png;base64," + "B" * MIB),
    }, api


@pytest.fixture()
def rec(tmp_path, monkeypatch):
    r = Recorder(tmp_path / "traces", monkeypatch, max_queue_mb=BUDGET_MB)
    yield r
    r.close()


@pytest.mark.parametrize("producer", ["frame", "render", "video_frames_read", "video_encode",
                                      "api_return", "service_output", "feedback_image"])
def test_dropped_items_release_their_copies(rec, producer):
    ep = rec.open()
    producers, api = _producers(ep)
    produce = producers[producer]
    C.push_episode(ep)
    n = 24
    try:
        gate = _block_writer(ep)  # the writer falls behind: nothing leaves the queue
        dropped_before = rec.store.stats["queue_full"]
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for i in range(n):
                produce(i)
            gc.collect()
            held = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()
            gate.set()
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    dropped = rec.store.stats["queue_full"] - dropped_before
    # without the release every item would keep its copy (n MiB or more)
    assert held < (BUDGET_MB + 3) * MIB, (producer, held / MIB, dropped)
    assert dropped >= n - 3, dropped
    rec.end(ep)
    doc = rec.summary(ep)
    gaps = {g["code"]: g["count"] for g in doc["gaps"]}
    assert doc["trace_complete"] is False
    assert gaps.get("queue_full") == dropped, gaps  # one gap per dropped item
    events = rec.events(ep)
    assert "queue_full" in warning_codes(events)
    if producer == "video_frames_read":
        reads = by_type(events, "video_frames_read")
        assert len(reads) == n and sum(1 for r in reads if r["data"].get("reason") == "queue_full") == dropped
    if producer == "frame":
        frames = by_type(events, "frame")
        assert sum(1 for f in frames if f["artifacts"][0].get("reason") == "queue_full") == dropped


def test_dropped_copies_are_freed_at_once(rec, monkeypatch):
    """The copy a producer made is unreachable as soon as the item is queued."""
    ep = rec.open()
    low = Low()
    tap, _ = FR.install_tap(low)
    tap.attach(ep)
    copies = []
    real_copy = FR._copy_frame

    def tracked_copy(frame):
        out = real_copy(frame)
        copies.append(weakref.ref(out))
        return out

    monkeypatch.setattr(FR, "_copy_frame", tracked_copy)
    gate = _block_writer(ep)
    try:
        frame = _frame()
        for i in range(6):
            low._frame_buffer.append(frame)
            tap.keyframe(None)
        gc.collect()
        alive = [r() is not None for r in copies]
    finally:
        gate.set()
    # the first copy fits into the 2 MiB budget and waits for the writer; the rest were dropped
    assert alive == [True] + [False] * 5
    rec.end(ep)


class StateModel:
    def __init__(self, nbody: int) -> None:
        self.nbody = nbody

    def body_id2name(self, i: int) -> str:
        return f"body{i}"


class StateData:
    def __init__(self, nbody: int) -> None:
        self.qpos = np.zeros(64)
        self.xpos = np.zeros((nbody, 3))
        self.xquat = np.zeros((nbody, 4))
        self.time = 0.0


class StateLow(Low):
    """A robosuite-shaped low-level env whose ground-truth state is about 230 KB."""

    def __init__(self, nbody: int = 4096) -> None:
        super().__init__()
        self.robosuite_env = types.SimpleNamespace(sim=types.SimpleNamespace(
            data=StateData(nbody), model=StateModel(nbody)))


class PingApi(ArrayApi):
    def functions(self):
        return {"ping": self.ping}

    def ping(self):
        """Return a small value (only the state snapshots are large)."""
        return 1


class StepEnv:
    def __init__(self, low) -> None:
        self.low_level_env = low
        self._apis = {}
        self._exec_globals = {}

    def step(self, code):
        return {}, 0.0, False, False, {"stdout": "", "stderr": "", "sandbox_rc": 0}


StepEnv.step = ENV.make_step_wrapper(StepEnv.__dict__["step"])


@pytest.mark.parametrize("where", ["api_call", "code_block"])
def test_state_snapshots_count_in_the_queue_budget(rec, where):
    """State copies are held by the item until the writer takes it: they count in
    the budget, a dropped item releases them and its state_ref says why (earlier regression)."""
    ep = rec.open()
    low = StateLow()
    ep.low_env = low
    api = PingApi()
    env = StepEnv(low)
    assert install_api(api)[0]
    C.push_episode(ep)
    n = 20
    try:
        gate = _block_writer(ep)
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            for i in range(n):
                api.ping() if where == "api_call" else env.step(f"x = {i}")
            gc.collect()
            held = tracemalloc.get_traced_memory()[0] - base
        finally:
            tracemalloc.stop()
            gate.set()
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    per_item = 64 * 8 + 4096 * 7 * 8
    assert held < (BUDGET_MB + 1.5) * MIB, (held / MIB, 2 * n * per_item / MIB)
    rec.end(ep)
    events = rec.events(ep)
    kinds = ("api_call_start", "api_call_end") if where == "api_call" else ("code_block_start", "code_block_end")
    refs = [e["data"]["state_ref"] for e in events if e["event_type"] in kinds]
    dropped = [r for r in refs if r.get("reason") == "queue_full"]
    assert len(refs) == 2 * n and len(dropped) >= 2 * n - int(BUDGET_MB * MIB // per_item) - 1
    assert all(r == {"path": None, "written": False, "reason": "queue_full"} for r in dropped)
    assert all(r["written"] is True for r in refs if r not in dropped)
    gaps = {g["code"]: g["count"] for g in rec.summary(ep)["gaps"]}
    assert gaps["queue_full"] == len(dropped)


# ---------------------------------------------------------------------------
# Model bodies under queue pressure
# ---------------------------------------------------------------------------


def _png_data_url() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 7, dtype=np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_request_text_is_kept_when_only_its_images_are_dropped(tmp_path, monkeypatch):
    rec = Recorder(tmp_path / "traces", monkeypatch, max_queue_mb=0.0001)  # ~100 bytes
    try:
        ep = rec.open()
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Goal: lift the red cube.\nAPIs:\nget_object_pose(name)"},
            {"type": "image_url", "image_url": {"url": _png_data_url()}}]}]})
        ep.emit("model_request", data={"role": "code_generator"}, keep_payload=True, nbytes=len(body),
                builder=functools.partial(HTTP._build_request, body=body, segments=(), event_id="ev_body"))
        response = json.dumps({"choices": [{"message": {"content": "```python\nprint(1)\n```"}}]}).encode()
        ep.emit("model_response", data={"role": "code_generator"}, keep_payload=True,
                nbytes=len(response),
                builder=functools.partial(HTTP._build_response, content=response, event_id="ev_resp"))
        rec.end(ep)
        events = rec.events(ep)
        request = by_type(events, "model_request")[0]["data"]
        stored = json.loads((ep.base_dir / request["payload_path"]).read_text())
        text = stored["messages"][0]["content"][0]["text"]
        assert text.startswith("Goal: lift the red cube.")
        placeholder = stored["messages"][0]["content"][1]["image_url"]["url"]
        assert placeholder["written"] is False and placeholder["reason"] == "queue_full"
        assert request["images"][0]["written"] is False
        assert request["reconstruction"]["verified"] is True  # checked in memory
        assert request["input_summary"]["n_images"] == 1
        response_data = by_type(events, "model_response")[0]["data"]
        assert response_data["content"] == "```python\nprint(1)\n```"
        doc = rec.summary(ep)
        # one gap: the request's image; the response lost nothing
        assert doc["gaps"] == [{"code": "queue_full", "count": 1,
                                "first_seq": by_type(events, "model_request")[0]["seq"]}]
    finally:
        rec.close()


# ---------------------------------------------------------------------------
# Disk floor
# ---------------------------------------------------------------------------


@pytest.fixture()
def disk(tmp_path, monkeypatch):
    rec = Recorder(tmp_path / "traces", monkeypatch, min_free_gb=1.0)
    free = {"bytes": 10 * 1024 ** 3}
    monkeypatch.setattr(S, "_disk_free", lambda path: free["bytes"])
    monkeypatch.setattr(S, "DISK_CHECK_INTERVAL_S", 0.0)
    yield rec, free
    rec.close()


class EnvStandIn:
    """Something heavy the episode references (an env and its frame buffer)."""

    def __init__(self) -> None:
        self.buffer = [np.zeros((64, 64, 3), np.uint8) for _ in range(4)]
        self.low_level_env = None


def test_disk_stop_leaves_ended_episodes_alone(disk):
    rec, free = disk
    ended = rec.open("ended")
    rec.end(ended)
    rec.flush()
    before = {name: (ended.base_dir / name).read_bytes() for name in ("episode.json", "events.jsonl")}
    assert json.loads(before["episode.json"])["trace_complete"] is True
    running = rec.open("running")
    rec.flush()
    free["bytes"] = 0  # below half of the 1 GB floor: all writing stops
    running.emit("trace_warning", data={"code": "x", "counts_as_gap": False})
    rec.flush()
    after = {name: (ended.base_dir / name).read_bytes() for name in ("episode.json", "events.jsonl")}
    assert after == before  # no marker line, no rewritten episode.json
    events = rec.events(running)
    assert events[-1]["event_type"] == "trace_warning" and events[-1]["data"]["code"] == "disk_low"
    doc = json.loads((running.base_dir / "episode.json").read_text())
    assert [g["code"] for g in doc["gaps"]] == ["disk_low"]
    rec.end(running)


def test_episode_end_after_the_stop_releases_the_env(disk):
    rec, free = disk
    during = rec.open("during")
    rec.flush()
    free["bytes"] = 0
    envs = []
    later = []
    for ep in [during] + [rec.open(f"after{i}") for i in range(2)]:
        env = EnvStandIn()
        ep.env = env
        envs.append(weakref.ref(env))
        later.append(ep)
        del env
    for ep in later:
        rec.end(ep)
    rec.flush()
    gc.collect()
    assert all(ep.released and ep.env is None for ep in later)
    assert [r() for r in envs] == [None, None, None]
    assert rec.store.stats["lost_after_stop"] >= 4
    # the first item after the drop (after0's start) noticed the stop: the final marker
    # is the only line of that episode; nothing is written for the episodes after it
    assert [e["data"]["code"] for e in rec.events(later[1])] == ["disk_low"]
    assert not later[2].base_dir.exists()


def test_a_stop_inside_a_builder_keeps_the_marker_last(disk):
    rec, free = disk
    ep = rec.open("inside")
    rec.flush()

    def builder(ctx):
        free["bytes"] = 0  # the disk check inside write_file stops all writing
        ctx.write_file("frames/cam/e0_0.jpg", b"\xff\xd8\xff", label="debug_frame", kind="image",
                       mime="image/jpeg", binary=True)

    ep.emit("frame", data={"camera": "cam"}, builder=builder)
    ep.emit("trace_warning", data={"code": "after", "counts_as_gap": False})
    rec.flush()
    events = rec.events(ep)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)  # file order == seq order
    assert [e["event_type"] for e in events] == ["episode_start", "trace_warning"]
    assert events[-1]["data"]["code"] == "disk_low"
    wep = ep._writer_state
    assert wep.closed_for_good and wep.fd is None  # no reopened descriptor
    rec.end(ep)
    rec.flush()
    assert rec.events(ep)[-1]["data"]["code"] == "disk_low"


def test_a_stop_between_episodes_is_marked_in_the_next_episode_and_the_run_log(disk):
    """regression: no episode was open when the stop was noticed (earlier regression)."""
    rec, free = disk
    first = rec.open("first")
    rec.end(first)
    rec.flush()
    complete = {name: (first.base_dir / name).read_bytes() for name in ("episode.json", "events.jsonl")}
    free["bytes"] = 0  # noticed at the next item: the start of the next episode
    second = rec.open("second")
    third = rec.open("third")
    for ep in (second, third):
        rec.end(ep)
    rec.flush()
    assert {n: (first.base_dir / n).read_bytes() for n in complete} == complete  # untouched
    events = rec.events(second)
    assert [(e["event_type"], e["data"]["code"]) for e in events] == [("trace_warning", "disk_low")]
    assert "later events and episodes of this process are not recorded" in events[0]["data"]["message"]
    doc = rec.summary(second)
    assert [g["code"] for g in doc["gaps"]] == ["disk_low"] and doc["task"] == "t"
    assert not third.base_dir.exists()
    markers = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "disk_low"]
    assert len(markers) == 1


def test_a_stop_at_the_first_episode_of_a_run_still_lists_the_run(disk):
    rec, free = disk
    free["bytes"] = 0
    ep = rec.open("only", run_id="run2")
    rec.end(ep)
    rec.flush()
    assert [e["data"]["code"] for e in rec.events(ep)] == ["disk_low"]
    assert json.loads((rec.root / "run2" / "run.json").read_text())["run_id"] == "run2"
    assert [e["data"]["code"] for e in rec.unscoped("run2")] == ["disk_low"]


def test_a_stop_noticed_by_a_runner_line_is_marked_in_the_run_log(disk):
    """The runner's parent process writes only runner_records.jsonl: it notices the stop too."""
    rec, free = disk
    run_dir = rec.root / "run"
    assert rec.store.append_run_line(run_dir, "runner_records.jsonl", b'{"trial":1}\n')
    rec.flush()
    free["bytes"] = 0
    assert rec.store.append_run_line(run_dir, "runner_records.jsonl", b'{"trial":2}\n')
    rec.flush()
    assert (run_dir / "runner_records.jsonl").read_text().splitlines() == ['{"trial":1}']
    assert [e["data"]["code"] for e in rec.unscoped()] == ["disk_low"]


# ---------------------------------------------------------------------------
# Write failures
# ---------------------------------------------------------------------------


def test_a_failed_episode_json_write_is_a_gap(rec, monkeypatch):
    ep = rec.open("json-fails")
    rec.flush()
    real = S.TraceStore.atomic_write
    state = {"fail": 1}

    def atomic_write(self, target, data, wep=None, count_disk=True):
        if target.name == "episode.json" and state["fail"] > 0:
            state["fail"] -= 1
            return False, "OSError: injected"
        return real(self, target, data, wep, count_disk)

    monkeypatch.setattr(S.TraceStore, "atomic_write", atomic_write)
    rec.end(ep)
    rec.flush()
    assert state["fail"] == 0
    doc = rec.summary(ep)
    assert doc["status"] == "ended" and doc["trace_complete"] is False
    assert [g["code"] for g in doc["gaps"]] == ["write_failed"]
    events = rec.events(ep)
    assert events[-1]["event_type"] == "trace_warning" and events[-1]["data"]["code"] == "write_failed"
    assert "episode.json" in events[-1]["data"]["message"]


def test_a_failed_run_level_line_is_recorded(rec):
    run_dir = rec.root / "run"
    (run_dir / "runner_records.jsonl").mkdir(parents=True)  # cannot be opened for appending
    assert rec.store.append_run_line(run_dir, "runner_records.jsonl", b'{"trial":1}\n')
    rec.flush()
    assert rec.store.stats["run_write_failed"] == 1
    warnings = [e["data"] for e in rec.unscoped() if e["event_type"] == "trace_warning"]
    assert [(w["code"], w["file"]) for w in warnings] == [("write_failed", "runner_records.jsonl")]


def _fd_path(fd: int) -> str:
    return os.readlink(f"/proc/self/fd/{fd}")


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc/self/fd")
def test_run_json_short_writes_are_completed(rec, monkeypatch):
    """A short write of run.json is continued (earlier regression: it was left cut)."""
    real = S._os_write

    def twelve_bytes(fd, data):
        if _fd_path(fd).endswith("/run.json"):
            return real(fd, bytes(data[:12]))
        return real(fd, data)

    monkeypatch.setattr(S, "_os_write", twelve_bytes)
    rec.end(rec.open("rj"))
    rec.flush()
    doc = json.loads((rec.root / "run" / "run.json").read_text())
    assert doc["run_id"] == "run" and rec.store.stats["run_write_failed"] == 0


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc/self/fd")
def test_a_failed_run_json_write_is_recorded_and_retried(rec, monkeypatch):
    real = S._os_write
    state = {"fail": 1}

    def no_progress(fd, data):
        if state["fail"] and _fd_path(fd).endswith("/run.json"):
            state["fail"] -= 1
            return 0
        return real(fd, data)

    monkeypatch.setattr(S, "_os_write", no_progress)
    rec.end(rec.open("rj"))
    rec.flush()
    assert state["fail"] == 0 and rec.store.stats["run_write_failed"] == 1
    warnings = [e["data"] for e in rec.unscoped() if e["event_type"] == "trace_warning"]
    assert [(w["code"], w["file"]) for w in warnings] == [("write_failed", "run.json")]
    assert json.loads((rec.root / "run" / "run.json").read_text())["run_id"] == "run"  # the next try


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc/self/fd")
def test_a_failed_unscoped_log_is_recorded_outside_it(rec):
    """An unscoped log open failure reaches a status file beside that log."""
    run_dir = rec.root / "run"
    name = f"unscoped.{os.getpid()}"
    (run_dir / f"{name}.jsonl").mkdir(parents=True)
    log = C.unscoped_log(rec.store, "run")
    log.emit("model_request", data={"role": "preflight"})
    log.emit("trace_warning", data={"code": "hook_install_failed", "run_level": True})
    rec.flush()
    status = json.loads((run_dir / f"{name}.status.json").read_text())
    assert status["file"] == f"{name}.jsonl" and status["lines_not_written"] == 2
    assert "cannot open" in status["last_reason"]
    assert rec.store.stats["run_write_failed"] == 2


def test_an_unscoped_log_that_recovers_says_how_many_lines_it_lost(rec, monkeypatch):
    real = S._os_write
    state = {"fail": 2}

    def no_space(fd, data):
        if state["fail"] and "/unscoped." in _fd_path(fd):
            state["fail"] -= 1
            raise OSError(28, "No space left on device")
        return real(fd, data)

    monkeypatch.setattr(S, "_os_write", no_space)
    log = C.unscoped_log(rec.store, "run")
    for i in range(3):
        log.emit("model_request", data={"role": "preflight", "i": i})
    rec.flush()
    events = rec.unscoped()
    assert [e["data"].get("i") for e in events if e["event_type"] == "model_request"] == [2]
    reports = [e["data"] for e in events if e["data"].get("code") == "write_failed"]
    assert [r["count"] for r in reports] == [2] and reports[0]["file"].startswith("unscoped.")
    status = json.loads((rec.root / "run" / f"unscoped.{os.getpid()}.status.json").read_text())
    assert status["lines_not_written"] == 2


def test_a_drop_after_episode_end_reaches_episode_json(tmp_path, monkeypatch):
    """a second queue_full (its warning is not repeated) after the end."""
    rec = Recorder(tmp_path / "traces", monkeypatch, max_queue_mb=0.0001)  # about 100 bytes
    try:
        ep = rec.open("late-drop")
        ep.emit("frame", data={"camera": "c"}, builder=lambda ctx: None, drop_builder=lambda ctx: None,
                nbytes=10_000)  # dropped before the end: one gap and its warning
        rec.end(ep)
        rec.flush()
        assert {g["code"]: g["count"] for g in rec.summary(ep)["gaps"]} == {"queue_full": 1}
        # an internal item dropped after the end (as H12 does for a VDM video)
        ep.internal(lambda ctx: None, nbytes=10_000, drop_builder=lambda ctx: None)
        rec.flush()
        assert {g["code"]: g["count"] for g in rec.summary(ep)["gaps"]} == {"queue_full": 2}
    finally:
        rec.close()


def test_a_gap_noted_after_episode_end_reaches_episode_json(rec):
    ep = rec.open("late-gap")
    rec.end(ep)
    rec.flush()
    assert rec.summary(ep)["trace_complete"] is True
    ep.note_gap("ambiguous_episode")
    rec.flush()
    doc = rec.summary(ep)
    assert doc["trace_complete"] is False and [g["code"] for g in doc["gaps"]] == ["ambiguous_episode"]


# ---------------------------------------------------------------------------
# Key-frame references of every camera
# ---------------------------------------------------------------------------


def test_frame_refs_of_every_camera_are_resolved(rec):
    ep = rec.open("refs")

    def write_main(ctx):
        ctx.write_file("frames/main/e0_1.jpg", b"\xff\xd8\xff", label="debug_frame", kind="image",
                       mime="image/jpeg", binary=True)

    ep.emit("frame", data={"camera": "main"}, builder=write_main)
    main = {"epoch": 0, "buffer_index": 1, "camera": "main", "path": "frames/main/e0_1.jpg"}
    wrist = {"epoch": 0, "buffer_index": 1, "camera": "wrist", "path": "frames/wrist/e0_1.jpg"}
    ep.emit("code_block_start", data={"frame_ref": dict(main), "frame_refs": [dict(main), dict(wrist)]})
    rec.end(ep)
    block = by_type(rec.events(ep), "code_block_start")[0]["data"]
    assert block["frame_ref"]["written"] is True
    assert [r["written"] for r in block["frame_refs"]] == [True, False]
    assert block["frame_refs"][1]["reason"] == "frame_not_written"
