"""The trial alarm is never swallowed by recorder code (docs/trace-format.md, section 2).

The alarm is a ``TimeoutError`` raised by a SIGALRM handler wherever the trial
thread is. These tests make the call the alarm would interrupt raise
``TimeoutError`` (``np.array`` of a frame, ``hashlib`` of an array over the
budget, the simulator clock read inside ``emit``, a state copy) and check that
it leaves the H7, H8, H9 and H10 wrappers; two tests use a real SIGALRM that
lands inside a recorder window. They also check that an alarm between a
context push and the wrapper's ``try`` cannot leave a stale context on the
thread (section 7.3), and H4's cleanup of contexts left by earlier episodes.
"""

from __future__ import annotations

import asyncio
import errno
import json
import signal
import sys
import threading
import time
import types

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import frames as FR
from capx_trace_recorder import serialize as SER
from capx_trace_recorder import state_probe as SP
from capx_trace_recorder import tbfilter
from capx_trace_recorder.hooks import api as API
from capx_trace_recorder.hooks import common as COMMON
from capx_trace_recorder.hooks import env as ENV
from capx_trace_recorder.hooks import llm as LLM
from capx_trace_recorder.hooks import services as SVC
from capx_trace_recorder.hooks import trial as TRIAL
from capx_trace_recorder.hooks import web as WEB
from capx_trace_recorder.hooks.api import install_api, uninstall_api
from inprocess import Recorder, by_type, thread_stacks, warning_codes

pytestmark = pytest.mark.skipif(sys.platform != "linux" or threading.current_thread() is not threading.main_thread(),
                                reason="SIGALRM tests need Linux and the main thread")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class AlarmFrame:
    """A buffer entry whose conversion to an array is where the alarm lands."""

    def __array__(self, *args, **kwargs):
        raise TimeoutError("alarm inside np.array")


class SlowFrame:
    """A buffer entry whose conversion takes long enough for a real alarm."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __array__(self, *args, **kwargs):
        time.sleep(self.seconds)
        return np.zeros((8, 8, 3), dtype=np.uint8)


class Low:
    """A low-level env with a frame buffer and a step counter (no simulator)."""

    def __init__(self, entry=None) -> None:
        self._frame_buffer = []
        self._sim_step_count = 0
        self.entry = entry

    def _record_frame(self) -> None:
        self._frame_buffer.append(self.entry if self.entry is not None else np.zeros((8, 8, 3), np.uint8))


class AlarmQpos:
    @property
    def qpos(self):
        raise TimeoutError("alarm inside the state copy")

    xpos = np.zeros((1, 3))
    xquat = np.zeros((1, 4))
    time = 0.0


class Model:
    nbody = 1

    def body_id2name(self, i):
        return "world"


class Env:
    """Stands in for CodeExecutionEnvBase (the H7 wrappers go on this class)."""

    def __init__(self, low) -> None:
        self.low_level_env = low
        self._apis = {}
        self._exec_globals = {}

    def step(self, code):
        return {}, 0.0, False, False, {"stdout": "", "stderr": "", "sandbox_rc": 0}


Env.step = ENV.make_step_wrapper(Env.__dict__["step"])


class ArrayApi:
    """An API instance for H8 (``functions()`` / ``combined_doc()`` contract)."""

    def __init__(self, low) -> None:
        self._env = low

    def functions(self):
        return {"take": self.take}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def take(self, arr):
        """Take an array."""
        return 1


@pytest.fixture()
def rec(tmp_path, monkeypatch):
    r = Recorder(tmp_path / "traces", monkeypatch, array_call_mb=0.001)
    yield r
    r.close()


def _episode(rec, low, env=None, tap=True):
    ep = rec.open()
    ep.low_env = low
    ep.env = env
    ep.trial_thread = ep.sim_thread = threading.get_ident()
    if tap:
        t, reason = FR.install_tap(low)
        assert t is not None, reason
        t.attach(ep)
        ep.frame_tap = t
    return ep


def _raise_timeout(*_args, **_kwargs):
    raise TimeoutError("alarm")


# ---------------------------------------------------------------------------
# Injected alarms leave every wrapper
# ---------------------------------------------------------------------------


def test_emit_lets_the_alarm_out_of_the_clock_read(rec, monkeypatch):
    ep = _episode(rec, Low(), tap=False)
    monkeypatch.setattr(SP, "sim_clock", _raise_timeout)
    with pytest.raises(TimeoutError):
        ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False})


def test_h10_frame_copy(rec):
    low = Low(AlarmFrame())
    ep = _episode(rec, low)
    with pytest.raises(TimeoutError):
        low._record_frame()
    assert ep.frame_tap is not None


@pytest.mark.parametrize("where", ["keyframe_np_array", "state_copy", "sim_clock_in_emit"])
def test_h7_step(rec, monkeypatch, where):
    if where == "keyframe_np_array":
        low = Low()
        low._frame_buffer.append(AlarmFrame())
        tap = True
    elif where == "state_copy":
        low = Low()
        low.robosuite_env = types.SimpleNamespace(sim=types.SimpleNamespace(data=AlarmQpos(), model=Model()))
        tap = False
    else:
        low = Low()
        tap = False
    env = Env(low)
    ep = _episode(rec, low, env, tap=tap)
    C.push_episode(ep)
    try:
        if where == "sim_clock_in_emit":
            calls = {"n": 0}
            real = SP.sim_clock

            def clock_then_alarm(low_env):
                calls["n"] += 1
                if calls["n"] >= 2:  # the first read is block.sim_step0; the next is inside emit
                    raise TimeoutError("alarm")
                return real(low_env)

            monkeypatch.setattr(SP, "sim_clock", clock_then_alarm)
            monkeypatch.setattr(rec.store.config.__class__, "state", property(lambda self: False))
        with pytest.raises(TimeoutError):
            env.step("x = 1")
        assert C._stack("frames") == []  # the block's context is not left behind
    finally:
        C.pop_episode(ep)


@pytest.mark.parametrize("where", ["hashlib", "sim_clock_in_emit"])
def test_h8_api_call(rec, monkeypatch, where):
    low = Low()
    ep = _episode(rec, low, tap=False)
    api = ArrayApi(low)
    ok, reason, _ = install_api(api)
    assert ok, reason
    C.push_episode(ep)
    try:
        if where == "hashlib":
            # an argument over the array budget is hashed on the calling thread
            monkeypatch.setattr(SER, "hashlib", types.SimpleNamespace(sha256=_raise_timeout))
        else:
            monkeypatch.setattr(SP, "sim_clock", _raise_timeout)
        with pytest.raises(TimeoutError):
            api.take(np.ones((64, 64)))
        assert C._stack("frames") == []
    finally:
        C.pop_episode(ep)
        uninstall_api(api)


@pytest.mark.parametrize("where", ["hashlib", "sim_clock_in_emit"])
def test_h9_service_call(rec, monkeypatch, where):
    low = Low()
    ep = _episode(rec, low, tap=False)
    C.push_episode(ep)
    try:
        if where == "hashlib":
            monkeypatch.setattr(SER, "hashlib", types.SimpleNamespace(sha256=_raise_timeout))
        else:
            monkeypatch.setattr(SP, "sim_clock", _raise_timeout)
        seg = SVC.make_service_wrapper(lambda image, text: [], "sam3", "sam3_seg_fn")
        with pytest.raises(TimeoutError):
            seg(np.zeros((64, 64, 3), dtype=np.uint8), "cube")
        assert C._stack("frames") == []
    finally:
        C.pop_episode(ep)


def test_h1_and_h3_let_the_alarm_out(rec, monkeypatch):
    """The model-call wrappers: an alarm inside their recorder code leaves the call."""
    import requests

    from capx_trace_recorder.hooks import http as HTTP
    from capx_trace_recorder.segments import SegmentRegistry

    ep = _episode(rec, Low(), tap=False)
    calls = []
    query = LLM.make_query_model_wrapper(lambda args, prompt: calls.append("query") or {"content": "x"})
    C.push_episode(ep)
    try:
        with monkeypatch.context() as mp:  # the segment snapshot taken before each request
            mp.setattr(SegmentRegistry, "snapshot", _raise_timeout)
            with pytest.raises(TimeoutError):
                query(types.SimpleNamespace(model="m"), [])
        assert calls == [] and C._stack("frames") == []
        lc = C.LogicalCall("lc_" + "0" * 20, ep, "code_generator", "initial_code", 0, "m", None, "test")
        C.push_ctx(lc)
        try:
            send = HTTP.make_send_wrapper(lambda self, request, **kw: calls.append("send"))
            request = requests.Request("POST", "http://127.0.0.1:9/v1/chat/completions",
                                       data=b"{}").prepare()
            with monkeypatch.context() as mp:  # the clock read inside emit (model_request)
                mp.setattr(SP, "sim_clock", _raise_timeout)
                with pytest.raises(TimeoutError):
                    send(None, request)
        finally:
            C.pop_ctx(lc)
        assert calls == []
    finally:
        C.pop_episode(ep)


def test_traceback_filter_reraises_instead_of_printing_recorder_frames(monkeypatch):
    def walk_raising(tb):
        raise TimeoutError("alarm inside the walk")
        yield  # noqa: B901 - a generator

    monkeypatch.setitem(tbfilter._ORIGINALS, "walk_tb", walk_raising)
    with pytest.raises(TimeoutError):
        list(tbfilter._filtered_walk_tb(None))

    def is_frame_raising(frame):
        raise TimeoutError("alarm inside the frame check")

    monkeypatch.setitem(tbfilter._ORIGINALS, "walk_tb", lambda tb: iter([(sys._getframe(), 1)]))
    monkeypatch.setattr(tbfilter, "is_recorder_frame", is_frame_raising)
    with pytest.raises(TimeoutError):
        list(tbfilter._filtered_walk_tb(None))


# ---------------------------------------------------------------------------
# A real SIGALRM inside a recorder window
# ---------------------------------------------------------------------------


@pytest.fixture()
def alarm():
    def handler(signum, frame):
        raise TimeoutError("Trial exceeded the wall-clock limit")

    previous = signal.signal(signal.SIGALRM, handler)
    try:
        yield lambda seconds: signal.setitimer(signal.ITIMER_REAL, seconds)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_real_alarm_inside_the_frame_copy_leaves_h10(rec, alarm):
    low = Low(SlowFrame(2.0))
    _episode(rec, low)
    alarm(0.2)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock limit"):
        low._record_frame()
    assert time.monotonic() - t0 < 1.5  # it did not wait for the copy to finish


def test_real_alarm_inside_the_array_hash_leaves_h8(rec, alarm, monkeypatch):
    low = Low()
    ep = _episode(rec, low, tap=False)
    api = ArrayApi(low)
    assert install_api(api)[0]

    def slow_sha256(data=b""):
        time.sleep(2.0)
        raise AssertionError("the alarm should have interrupted the hash")

    monkeypatch.setattr(SER, "hashlib", types.SimpleNamespace(sha256=slow_sha256))
    C.push_episode(ep)
    try:
        alarm(0.2)
        with pytest.raises(TimeoutError, match="wall-clock limit"):
            api.take(np.ones((64, 64)))
        assert C._stack("frames") == []
    finally:
        C.pop_episode(ep)
        uninstall_api(api)


class FailingApi(ArrayApi):
    def functions(self):
        return {"fail": self.fail}

    def fail(self):
        """Raise inside a public function."""
        raise ValueError("the value is out of range for this call")


def test_api_error_traceback_is_looked_up_off_the_trial_thread(rec, monkeypatch):
    """regression: linecache.checkcache, whose ``except OSError`` swallows a TimeoutError,
    never runs on the trial thread; the writer formats the traceback."""
    import linecache

    seen = []
    real = linecache.checkcache

    def spy(*args, **kwargs):
        seen.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(linecache, "checkcache", spy)
    low = Low()
    ep = _episode(rec, low, tap=False)
    api = FailingApi(low)
    assert install_api(api)[0]
    C.push_episode(ep)
    try:
        with pytest.raises(ValueError):
            api.fail()
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    assert threading.get_ident() not in seen
    rec.end(ep)
    error = by_type(rec.events(ep), "api_call_error")[0]["data"]
    assert error["traceback"].startswith("Traceback (most recent call last):\n")
    assert 'raise ValueError("the value is out of range for this call")' in error["traceback"]
    assert error["traceback"].endswith("ValueError: the value is out of range for this call")
    assert "capx_trace_recorder" not in error["traceback"]


def test_the_recorder_keeps_no_weak_reference_with_a_callback(rec):
    """A weakref callback runs on whatever thread drops the object; an alarm inside
    it would be printed as "Exception ignored" (regression)."""
    import weakref

    low = Low()
    tap, reason = FR.install_tap(low)
    assert tap is not None, reason
    api = ArrayApi(low)
    assert install_api(api)[0]
    ep = rec.open()
    C.REGISTRY.bind_env(low, ep)
    try:
        for obj in (api, tap, low):
            refs = weakref.getweakrefs(obj)
            assert refs and all(r.__callback__ is None for r in refs), obj
    finally:
        uninstall_api(api)


# ---------------------------------------------------------------------------
# Bounded hashing of arrays over the budget
# ---------------------------------------------------------------------------


def test_over_budget_arrays_are_hashed_within_a_bound():
    budget = SER.ArrayBudget(0, None)  # nothing is copied: every array is over the budget
    small = np.arange(1000, dtype=np.float64)  # 8 KB: hashed whole
    big = np.ones((1024, 1024), dtype=np.float32)  # 4 MiB: only a prefix
    node_small = SER.summarize(small, [], budget)
    assert node_small["not_saved"] == "array_budget" and len(node_small["sha256"]) == 64
    node_big = SER.summarize(big, [], budget)
    assert node_big["sha256"] is None
    assert node_big["sha256_prefix_bytes"] == SER.HASH_FULL_BYTES
    import hashlib

    assert node_big["sha256_prefix"] == hashlib.sha256(big.reshape(-1)[: SER.HASH_FULL_BYTES // 4].tobytes()).hexdigest()
    # one summary hashes at most HASH_BUDGET_BYTES in total
    many = [np.full((512, 512), i, dtype=np.float32) for i in range(8)]  # 1 MiB each
    node = SER.summarize(many, [], budget)
    items = node["items"]
    hashed = [i for i in items if i.get("sha256")]
    skipped = [i for i in items if i.get("sha256_skipped") == "hash_budget"]
    assert len(hashed) == SER.HASH_BUDGET_BYTES // SER.HASH_FULL_BYTES and len(skipped) == 8 - len(hashed)
    # a non-contiguous view is hashed over the same bytes as its contiguous copy
    view = np.arange(20000, dtype=np.int64).reshape(100, 200)[:, ::2]
    assert SER.summarize(view, [], budget)["sha256"] == hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Contexts: pushed inside the wrapper's try (an alarm cannot leave one behind)
# ---------------------------------------------------------------------------


def _alarm_on_first_call(module, name, monkeypatch):
    real = getattr(module, name)
    state = {"armed": True}

    def wrapped(*args, **kwargs):
        if state["armed"]:
            state["armed"] = False
            raise TimeoutError("alarm")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)


def _alarm_after_push(monkeypatch, name="push_ctx"):
    """The alarm lands right after the context was pushed."""
    real = getattr(C, name)
    state = {"armed": True}

    def push_then_alarm(entry):
        real(entry)
        if state["armed"]:
            state["armed"] = False
            raise TimeoutError("alarm")

    monkeypatch.setattr(C, name, push_then_alarm)


@pytest.mark.parametrize("gap", ["before_push", "after_push"])
def test_api_context_not_left_behind(rec, monkeypatch, gap):
    low = Low()
    ep = _episode(rec, low, tap=False)
    api = ArrayApi(low)
    assert install_api(api)[0]
    C.push_episode(ep)
    try:
        if gap == "before_push":
            _alarm_on_first_call(API, "add_overhead", monkeypatch)
        else:
            _alarm_after_push(monkeypatch)
        with pytest.raises(TimeoutError):
            api.take(1)
        assert C._stack("frames") == []
        assert api.take(2) == 1  # the next call is at depth 0 again
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    starts = by_type(rec.events(ep), "api_call_start")
    assert [s["data"]["depth"] for s in starts] == [0, 0]


@pytest.mark.parametrize("gap", ["before_push", "after_push"])
def test_block_service_and_model_call_contexts_not_left_behind(rec, monkeypatch, gap):
    low = Low()
    env = Env(low)
    ep = _episode(rec, low, env, tap=False)
    seg = SVC.make_service_wrapper(lambda image, text: [], "sam3", "sam3_seg_fn")
    query = LLM.make_query_model_wrapper(lambda args, prompt: {"content": "ok"})
    C.push_episode(ep)
    try:
        for module, call in ((ENV, lambda: env.step("x = 1")),
                             (SVC, lambda: seg(np.zeros((4, 4, 3), np.uint8), "cube")),
                             (LLM, lambda: query(types.SimpleNamespace(model="m"), []))):
            with monkeypatch.context() as mp:
                if gap == "before_push":
                    _alarm_on_first_call(module, "add_overhead", mp)
                else:
                    _alarm_after_push(mp)
                with pytest.raises(TimeoutError):
                    call()
            assert C._stack("frames") == [], module.__name__
    finally:
        C.pop_episode(ep)


def test_role_context_not_left_behind(rec, monkeypatch):
    ep = _episode(rec, Low(), tap=False)
    wrapped = TRIAL.make_role_wrapper(lambda: "done", "vdm", "image_diff", "current")
    C.push_episode(ep)
    try:
        _alarm_after_push(monkeypatch, "push_role")
        with pytest.raises(TimeoutError):
            wrapped()
        assert C._stack("roles") == [] and ep.current_role is None
        assert wrapped() == "done" and C._stack("roles") == []
    finally:
        C.pop_episode(ep)


# ---------------------------------------------------------------------------
# H4: episodes, and contexts left on the trial thread
# ---------------------------------------------------------------------------


class Args:
    config_path = "env_configs/fake/alarm.yaml"
    model = "fake/model"


def _h4(fake):
    def _run_single_trial(env, trial, args, config, multi_turn_prompt=None, partial_artifacts=None):
        return fake(env)

    return TRIAL.make_run_single_trial_wrapper(_run_single_trial)


@pytest.fixture()
def h4_env(rec, monkeypatch, tmp_path):
    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "run")
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    return Env(Low()), {"output_dir": str(tmp_path / "harness")}


def _only_episode_dirs(rec) -> list:
    run = rec.root / "run"
    return sorted(p for p in run.iterdir() if p.is_dir()) if run.exists() else []


def _opened_episodes(monkeypatch) -> list:
    """Every episode H4 opens from now on (also one whose start is undone)."""
    opened = []
    real = C.open_episode

    def spy(*args, **kwargs):
        ep = real(*args, **kwargs)
        opened.append(ep)
        return ep

    monkeypatch.setattr(C, "open_episode", spy)
    return opened


def test_h4_episode_not_left_open_by_an_alarm_after_the_start(rec, monkeypatch, h4_env):
    """The alarm lands right after the episode was opened and pushed (earlier regression)."""
    env, config = h4_env
    opened = _opened_episodes(monkeypatch)
    wrapped = _h4(lambda e: "summary")
    _alarm_on_first_call(TRIAL, "add_overhead", monkeypatch)
    with pytest.raises(TimeoutError):
        wrapped(env, 1, Args(), config)
    assert thread_stacks() == {"frames": [], "roles": [], "episodes": []}
    assert C.REGISTRY.open_episodes() == []
    assert wrapped(env, 2, Args(), config) == "summary"
    assert C.REGISTRY.open_episodes() == []
    rec.flush()
    first, second = opened
    assert first.released and first.env is None and first.low_env is None  # references released
    dirs = _only_episode_dirs(rec)
    assert len(dirs) == 2
    docs = {d.name: json.loads((d / "episode.json").read_text()) for d in dirs}
    ends = sorted((doc["end_reason"], doc["status"]) for doc in docs.values())
    assert ends == [("completed", "ended"), ("timeout_alarm", "ended")]
    assert second.released and second.env is None
    interrupted = [d for d in dirs if docs[d.name]["end_reason"] == "timeout_alarm"][0]
    events = [json.loads(line) for line in (interrupted / "events.jsonl").read_text().splitlines()]
    assert [e["event_type"] for e in events][-1] == "episode_end"


def test_h4_file_io_timeout_at_unique_episode_id_runs_trial(rec, monkeypatch, h4_env):
    env, config = h4_env
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    real_exists = TRIAL.Path.exists
    hit = []
    trial_thread = threading.get_ident()

    def timed_out(path):
        if threading.get_ident() == trial_thread and path.parent == rec.root / "run":
            hit.append(path)
            raise OSError(errno.ETIMEDOUT, "timed out")
        return real_exists(path)

    monkeypatch.setattr(TRIAL.Path, "exists", timed_out)
    calls = []
    assert _h4(lambda e: calls.append(e) or "summary")(env, 1, Args(), config) == "summary"
    assert hit and calls == [env]
    monkeypatch.setattr(TRIAL.Path, "exists", real_exists)
    notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
    assert len(notes) == 1 and notes[0]["outside_episode"] is True


@pytest.mark.parametrize("where", ["attach_frames", "push_episode"])
def test_h4_file_io_timeout_inside_start_releases_registration(rec, monkeypatch, h4_env, where):
    env, config = h4_env
    opened = []
    slots = []
    real_open = C.open_episode

    def track_open(*args, **kwargs):
        ep = real_open(*args, **kwargs)
        opened.append(ep)
        slots.append(kwargs["slot"])
        return ep

    monkeypatch.setattr(C, "open_episode", track_open)
    _raise_once(TRIAL if where == "attach_frames" else C, where,
                OSError(errno.ETIMEDOUT, "timed out"), monkeypatch)
    calls = []
    assert _h4(lambda e: calls.append(e) or "summary")(env, 1, Args(), config) == "summary"
    assert calls == [env]
    assert len(opened) == len(slots) == 1 and slots[0].abandoned
    ep = opened[0]
    rec.flush()  # a queued start releases its references when the writer handles its end
    assert ep.released and ep.env is None and ep.low_env is None
    assert ep.frame_tap is None
    assert C.REGISTRY.open_episodes() == [] and thread_stacks()["episodes"] == []
    assert C.REGISTRY.binding(env) is None and C.REGISTRY.binding(env.low_level_env) is None
    notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
    assert len(notes) == 1 and notes[0]["outside_episode"] is True


@pytest.mark.parametrize("where", ["attach_frames", "push_episode"])
def test_h4_alarm_inside_the_start_undoes_the_registration(rec, monkeypatch, h4_env, where):
    """Between open_episode() and the push: the episode is taken out of the open set
    and the bindings; if its start was queued it is ended, else the run log says so."""
    env, config = h4_env
    opened = _opened_episodes(monkeypatch)
    calls = []
    if where == "attach_frames":
        _alarm_on_first_call(TRIAL, "attach_frames", monkeypatch)  # before episode_start is queued
    else:
        _alarm_on_first_call(C, "push_episode", monkeypatch)  # after it
    with pytest.raises(TimeoutError):
        _h4(lambda e: calls.append(1))(env, 1, Args(), config)
    assert calls == []  # the trial did not start
    assert C.REGISTRY.open_episodes() == [] and thread_stacks()["episodes"] == []
    assert C.REGISTRY.binding(env) is None and C.REGISTRY.binding(env.low_level_env) is None
    rec.flush()
    assert len(opened) == 1 and opened[0].released and opened[0].env is None
    dirs = _only_episode_dirs(rec)
    if where == "attach_frames":
        assert dirs == []  # nothing of the episode was queued
        notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
        assert len(notes) == 1 and notes[0]["outside_episode"] is True and notes[0]["alarm"] is True
        assert "did not start" in notes[0]["message"]
    else:
        doc = json.loads((dirs[0] / "episode.json").read_text())
        assert doc["status"] == "ended" and doc["end_reason"] == "timeout_alarm"
        events = [json.loads(line) for line in (dirs[0] / "events.jsonl").read_text().splitlines()]
        end = [e for e in events if e["event_type"] == "episode_end"][0]["data"]
        assert "starting the episode" in end["end_note"]


class StartStop(BaseException):
    """A non-``Exception`` exception (as ``KeyboardInterrupt``) raised inside a start."""


def _raise_once(module, name, exc, monkeypatch):
    """``module.name`` raises ``exc`` on its first call, then behaves as before."""
    real = getattr(module, name)
    state = {"armed": True}

    def wrapped(*args, **kwargs):
        if state["armed"]:
            state["armed"] = False
            raise exc
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)


def _failing_cleanup(*_args, **_kwargs):
    raise RuntimeError("cleanup failed (a recorder bug)")


# where the start is interrupted: before its episode_start was queued, and after
_START_POINTS = {"H4": ((TRIAL, "attach_frames"), (C, "push_episode")),
                 "H13": ((TRIAL, "tracer_info"), (WEB, "_WebState"))}


@pytest.mark.parametrize("raised", [TimeoutError, StartStop])
@pytest.mark.parametrize("queued", [False, True], ids=["before-queue", "after-queue"])
@pytest.mark.parametrize("hook", ["H4", "H13"])
def test_a_failing_cleanup_never_replaces_the_start_exception(rec, monkeypatch, h4_env, hook, queued, raised):
    """the alarm (or another non-Exception exception) lands inside the start and
    undoing the start fails as well. The original exception leaves the wrapper, the
    wrapped function does not run, and the cleanup failure is recorded."""
    env, config = h4_env
    monkeypatch.setattr(COMMON, "_NOTED_OUTSIDE_EPISODES", set())
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    monkeypatch.setattr(TRIAL, "_record_abandoned", _failing_cleanup)
    original = raised("alarm at start")
    module, name = _START_POINTS[hook][queued]
    _raise_once(module, name, original, monkeypatch)
    ran = []
    if hook == "H4":
        with pytest.raises(raised) as info:
            _h4(lambda e: ran.append(e))(env, 1, Args(), config)
        assert thread_stacks()["episodes"] == []
        assert C.REGISTRY.binding(env) is None
    else:
        async def run_trial_async(session, args):
            ran.append(session)
            return "done"

        session = types.SimpleNamespace(session_id="session-A", config={}, env_factory={"cfg": {}},
                                        config_path="env_configs/fake/A.yaml", await_user_input_each_turn=False,
                                        state=None, env=None)
        args = types.SimpleNamespace(model="m")
        with pytest.raises(raised) as info:
            asyncio.run(WEB.make_run_trial_async_wrapper(run_trial_async)(session=session, args=args))
        assert "session-A" not in C.REGISTRY.web_sessions and C.REGISTRY.binding(args) is None
    assert info.value is original
    assert ran == []
    assert C.REGISTRY.open_episodes() == []
    notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
    assert [n["message"] for n in notes] == ["_abandon_start: RuntimeError: cleanup failed (a recorder bug)"]


class RaisingCameraLow(Low):
    """A low-level env whose camera name cannot be read yet (a recorder error in H10's install)."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    @property
    def save_camera_name(self):
        self.reads += 1
        if self.reads == 1:
            raise RuntimeError("camera config not loaded yet")
        return "cam"


def test_h4_recorder_error_after_open_episode_leaves_no_zombie(rec, monkeypatch, h4_env):
    """regression: the trial runs untraced, nothing stays open, and the run log says why."""
    from capx_trace_recorder import bootstrap

    monkeypatch.setattr(bootstrap, "frame_hooks_allowed", lambda: True)  # as with H11 installed
    _env, config = h4_env
    env = Env(RaisingCameraLow())
    opened = _opened_episodes(monkeypatch)
    ran = []
    assert _h4(lambda e: ran.append(C._stack("episodes")) or "summary")(env, 1, Args(), config) == "summary"
    assert ran == [[]]  # untraced: no episode on the stack
    assert C.REGISTRY.open_episodes() == [] and C.REGISTRY.binding(env) is None
    assert opened[0].released and opened[0].env is None and opened[0].frame_tap is None
    resolved = {}

    def trial(e):
        worker = threading.Thread(target=lambda: resolved.update(r=C.resolve_episode()))
        worker.start()  # an ensemble member thread: the single open episode, not ambiguous
        worker.join(10)
        resolved["own"] = C._stack("episodes")[-1]
        return "summary"

    assert _h4(trial)(env, 2, Args(), config) == "summary"
    ep, candidates = resolved["r"]
    assert ep is resolved["own"] and candidates == [ep]
    assert C.REGISTRY.open_episodes() == []
    rec.flush()
    for d in _only_episode_dirs(rec):  # no events.jsonl without episode.json
        assert (d / "episode.json").exists(), d
    notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
    assert len(notes) == 1 and notes[0]["outside_episode"] is True
    assert notes[0]["message"] == "_episode_start: RuntimeError: camera config not loaded yet"
    assert notes[0]["note"].endswith("the trial ran untraced")


def test_h4_alarm_after_the_trial_returned_still_ends_the_episode(rec, monkeypatch, h4_env):
    """The alarm lands after the original returned, before H4 recorded the end."""
    env, config = h4_env
    seen = {}

    def trial(e):
        seen["ep"] = C._stack("episodes")[-1]
        return "summary"

    real_finish = TRIAL._episode_finish
    state = {"armed": True}

    def finish_interrupted(ep, summary, exc):
        if state["armed"]:
            state["armed"] = False
            raise TimeoutError("Trial exceeded the wall-clock limit")
        return real_finish(ep, summary, exc)

    monkeypatch.setattr(TRIAL, "_episode_finish", finish_interrupted)
    with pytest.raises(TimeoutError):
        _h4(trial)(env, 1, Args(), config)
    ep = seen["ep"]
    assert thread_stacks() == {"frames": [], "roles": [], "episodes": []}
    assert ep.ended and ep.episode_id not in C.REGISTRY.open
    end = by_type(rec.events(ep), "episode_end")
    assert len(end) == 1 and end[0]["data"]["end_reason"] == "timeout_alarm"
    assert end[0]["data"]["alarm_seen"] is True


def test_h4_ends_an_episode_left_open_on_the_thread(rec, h4_env):
    env, config = h4_env
    left = rec.open("left-open")
    C.push_episode(left)  # an interrupted exit path left it pushed and open
    seen = {}

    def trial(e):
        seen["ep"] = C._stack("episodes")[-1]
        return "summary"

    assert _h4(trial)(env, 1, Args(), config) == "summary"
    assert seen["ep"] is not left
    assert left.ended and left.episode_id not in C.REGISTRY.open
    end = by_type(rec.events(left), "episode_end")[0]["data"]
    assert end["end_reason"] == "error" and "next episode started" in end["end_note"]
    assert rec.summary(left)["status"] == "ended"
    new = rec.events(seen["ep"])
    stale = [e["data"] for e in by_type(new, "trace_warning") if e["data"]["code"] == "stale_context"]
    assert stale and stale[0]["removed"] == {"episode": 1}
    assert thread_stacks() == {"frames": [], "roles": [], "episodes": []}


def test_h4_clears_contexts_left_by_earlier_episodes(rec, h4_env):
    env, config = h4_env
    old = rec.open("old-episode")
    rec.end(old)
    C.push_ctx(C.ApiCallCtx("ev_" + "0" * 20, old, "stale_call", 0))  # left by an interrupted wrapper
    seen = {}

    def trial(e):
        seen["frames"] = list(C._stack("frames"))
        seen["ep"] = C._stack("episodes")[-1]
        return "summary"

    assert _h4(trial)(env, 1, Args(), config) == "summary"
    assert seen["frames"] == []
    ep = seen["ep"]
    events = rec.events(ep)
    stale = [e for e in by_type(events, "trace_warning") if e["data"]["code"] == "stale_context"]
    assert len(stale) == 1 and stale[0]["data"]["counts_as_gap"] is False
    assert stale[0]["data"]["removed"] == {"api_call": 1}
    assert rec.summary(ep)["trace_complete"] is True
    assert thread_stacks() == {"frames": [], "roles": [], "episodes": []}


def test_h4_records_contexts_this_episode_left_behind(rec, h4_env):
    env, config = h4_env
    seen = {}

    def trial(e):
        ep = C._stack("episodes")[-1]
        seen["ep"] = ep
        C.push_ctx(C.CodeBlockCtx("ev_" + "1" * 20, ep, "x", e))  # never popped
        return "summary"

    _h4(trial)(env, 1, Args(), config)
    ep = seen["ep"]
    events = rec.events(ep)
    assert "stale_context" in warning_codes(events)
    doc = rec.summary(ep)
    assert doc["trace_complete"] is False and [g["code"] for g in doc["gaps"]] == ["stale_context"]
    assert thread_stacks() == {"frames": [], "roles": [], "episodes": []}
