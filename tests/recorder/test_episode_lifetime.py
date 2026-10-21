"""Episode lifetime, late events and attribution (docs/trace-format.md, sections 4.2, 4.4 and 7.3).

* After ``episode_end`` the writer keeps only a small record of the episode and
  the recorder drops its references to the env (a long batch or web server run
  must not grow with every episode).
* An ended episode stays reachable through its env binding (a tombstone):
  events from its env are appended to it as late events, never dropped and
  never given to another open episode.
* A binding has a generation; a thread keeps the one it used first, so a
  thread started for one job and still working for the old episode stays with
  it after the env is reused by the next episode, while new work goes to the
  new one. A pool worker's pin does not decide (it may be running a new job):
  its event is ambiguous unless a per-job signal (the web context variable, a
  pushed episode or context) says where it belongs.
* Two web episodes open at once: each env is bound through its session, the
  streaming thread through the launch arguments.
* Events that cannot be attributed add an ``ambiguous_episode`` gap to every
  candidate, with one warning per (hook, run, candidates).
* Long ids get their ``__dup<n>`` suffix inside 200 characters.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks import env as ENV
from capx_trace_recorder.hooks import web as WEB
from capx_trace_recorder.ids import join_id
from inprocess import Recorder, by_type, warning_codes


class Low:
    def __init__(self) -> None:
        self._frame_buffer = []
        self._sim_step_count = 0


class Env:
    """Stands in for CodeExecutionEnvBase, with the real H7 wrappers."""

    def __init__(self, low=None) -> None:
        self.low_level_env = low if low is not None else Low()
        self._apis = {}
        self._exec_globals = {}

    def reset(self, *args, **kwargs):
        return {}, {}

    def step(self, code):
        return {}, 0.0, False, False, {"stdout": f"ran {code}", "stderr": "", "sandbox_rc": 0}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


Env.reset = ENV.make_reset_wrapper(Env.__dict__["reset"])
Env.step = ENV.make_step_wrapper(Env.__dict__["step"])
Env.render = ENV.make_render_wrapper(Env.__dict__["render"], "main")


@pytest.fixture()
def rec(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "run")
    r = Recorder(tmp_path / "traces", monkeypatch)
    yield r
    r.close()


def _bind(env, ep):
    C.REGISTRY.bind_env(env, ep)
    C.REGISTRY.bind_env(env.low_level_env, ep)


def _in_thread(fn):
    """Run ``fn`` on a new thread (no episode stack, no pins) and return its result."""
    out = {}

    def run():
        out["value"] = fn()

    t = threading.Thread(target=run)
    t.start()
    t.join(20)
    assert not t.is_alive()
    return out.get("value")


def _snapshots(rec, ep):
    return [(e["late"]) for e in by_type(rec.events(ep), "observation_snapshot")]


# ---------------------------------------------------------------------------
# (a) writer state and references after episode_end
# ---------------------------------------------------------------------------


def test_ended_episode_keeps_only_a_small_record(rec):
    a = rec.open("a")
    b = rec.open("b")
    env = Env()
    env_ref = weakref.ref(env)
    a.env, a.low_env = env, env.low_level_env
    _bind(env, a)
    a.emit("frame", data={"camera": "c"}, builder=lambda ctx: ctx.write_file(
        "frames/c/e0_0.jpg", b"\xff\xd8\xff", label="debug_frame", kind="image", mime="image/jpeg",
        binary=True))
    rec.flush()
    assert a._writer_state.written_files
    assert set(rec.store._open_weps) == {id(a), id(b)}
    rec.end(a)
    rec.flush()
    wep = a._writer_state
    assert wep.end_processed and wep.written_files == set() and wep.made_dirs == set()
    assert set(rec.store._open_weps) == {id(b)}  # the heartbeat visits only open logs
    assert a.released and a.env is None and a.low_env is None
    del env
    import gc

    gc.collect()
    assert env_ref() is None  # neither the store nor the binding tombstone keeps the env alive
    rec.end(b)


# ---------------------------------------------------------------------------
# (b), (c) late events through the env binding
# ---------------------------------------------------------------------------


def test_events_from_an_ended_episodes_env_are_late_events_of_it(rec):
    ep = rec.open("a")
    env = Env()
    _bind(env, ep)
    rec.end(ep)  # queues episode_end and closes the episode
    _in_thread(env.render)  # a thread still working for the ended episode
    assert _snapshots(rec, ep) == [True]
    doc = rec.summary(ep)
    assert doc["status"] == "ended" and doc["trace_complete"] is False
    assert doc["late_events"] == 1 and {g["code"] for g in doc["gaps"]} == {"late_event"}


def test_an_old_env_never_goes_to_the_currently_open_episode(rec):
    old = rec.open("old")
    old_env = Env()
    _bind(old_env, old)
    rec.end(old)
    new = rec.open("new")  # the only open episode
    _in_thread(old_env.render)
    unknown_env = Env()
    _in_thread(unknown_env.render)  # never bound: the single open episode
    assert _snapshots(rec, old) == [True]
    assert _snapshots(rec, new) == [False]
    rec.end(new)


# ---------------------------------------------------------------------------
# Env reuse across episodes in one worker (generations and pins)
# ---------------------------------------------------------------------------


def test_env_reuse_across_episodes(rec):
    env = Env()
    ep1 = rec.open("e1")
    _bind(env, ep1)
    used = threading.Event()
    go = threading.Event()

    def straggler():
        env.render()  # during ep1, through the binding: this thread now works for ep1
        used.set()
        go.wait(10)
        env.render()  # after the env was bound to ep2: still ep1's work

    t = threading.Thread(target=straggler)
    t.start()
    assert used.wait(10)
    env.render()  # the test thread uses the binding too (pinned to ep1)
    rec.end(ep1)
    ep2 = rec.open("e2")
    _bind(env, ep2)  # H4 binds the reused env to the next episode ...
    C.push_episode(ep2)  # ... and pushes the episode on the trial thread, which drops its pins
    env.render()
    C.pop_episode(ep2)
    go.set()
    t.join(10)
    _in_thread(env.render)  # new work on another thread: ep2
    rec.end(ep2)
    env.render()  # after ep2 ended, on the test thread: ep2, late (not ep1)
    assert _snapshots(rec, ep1) == [False, False, True]
    assert _snapshots(rec, ep2) == [False, False, True]


def test_a_reused_pool_thread_does_not_pin_new_work_to_the_old_episode(rec):
    """A one-worker ThreadPoolExecutor renders the reused env for episode 1, then
    for episode 2 (earlier regression). A pool worker may be running a new job, so its
    pin to episode 1 does not decide: without a per-job signal the event is
    ambiguous (a gap in both episodes), never a late event of episode 1. A per-job
    signal wins over the pin: the context that ``asyncio.to_thread`` copies into
    each job carries the web episode."""
    env = Env()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        ep1 = rec.open("e1")
        _bind(env, ep1)
        pool.submit(env.render).result(10)  # episode 1's work pins the worker to episode 1
        rec.end(ep1)
        ep2 = rec.open("e2")
        _bind(env, ep2)
        pool.submit(env.render).result(10)  # a new job on the same worker, no per-job signal
        job = contextvars.copy_context()
        job.run(C.WEB_EPISODE.set, ep2)
        pool.submit(job.run, env.render).result(10)  # a job that carries its episode
        rec.end(ep2)
    finally:
        pool.shutdown(wait=True)
    assert _snapshots(rec, ep1) == [False]
    assert _snapshots(rec, ep2) == [False]
    assert rec.summary(ep2)["gaps"] == [{"code": "ambiguous_episode", "count": 1, "first_seq": None}]
    ended = {g["code"]: g["count"] for g in rec.summary(ep1)["gaps"]}
    assert ended["ambiguous_episode"] == 1  # the ended episode may have lost a late event


# ---------------------------------------------------------------------------
# (d) two web episodes open at once
# ---------------------------------------------------------------------------


def test_two_open_web_episodes_are_attributed(rec):
    streams = {}

    def gen(args, prompt):
        streams[args.tag] = C.resolve_episode()[0]
        yield {"type": "content_delta", "content": "x"}

    stream = WEB.make_web_stream_wrapper(gen)

    async def fake_run_trial_async(session, args):
        """The flow of capx/web/async_trial_runner.py: env and streaming on executor threads."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="env")
        stream_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream")
        loop = asyncio.get_running_loop()
        async def on_executor(pool, fn, *call_args):
            future = loop.run_in_executor(pool, fn, *call_args)
            # Polling also keeps this isolated test reliable in environments
            # where a cross-thread loop wakeup is lost after an episode opens.
            while not future.done():
                await asyncio.sleep(0.01)
            return future.result()

        env = await on_executor(executor, Env)
        session.env = env
        await asyncio.sleep(0.1)  # both episodes are open from here on
        await on_executor(executor, env.reset)
        await on_executor(stream_executor, lambda: list(stream(args, [])))
        for i in range(2):
            await on_executor(executor, env.step, f"block_{i}()")
        await on_executor(executor, env.render)
        executor.shutdown(wait=True)
        stream_executor.shutdown(wait=True)
        return "done"

    wrapped = WEB.make_run_trial_async_wrapper(fake_run_trial_async)
    sessions = []
    for tag in ("A", "B"):
        sessions.append(types.SimpleNamespace(session_id=f"session-{tag}", config={}, env_factory={"cfg": {}},
                                              config_path=f"env_configs/fake/{tag}.yaml",
                                              await_user_input_each_turn=False, state=None, env=None))

    async def main():
        await asyncio.wait_for(asyncio.gather(
            *(wrapped(session=s, args=types.SimpleNamespace(tag=s.session_id[-1], model="m"))
              for s in sessions)), timeout=10)

    asyncio.run(main())
    rec.flush()
    episodes = sorted(C.REGISTRY.used_ids)
    run_dir = rec.root / "run"
    by_session = {}
    for ep_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and "__web__" in p.name):
        from trace_reader import load_episode

        doc, events = load_episode(ep_dir)
        by_session[ep_dir.name] = (doc, events)
    assert len(by_session) == 2, episodes
    for doc, events in by_session.values():
        assert len(by_type(events, "code_block_start")) == 2
        assert len(by_type(events, "observation_snapshot")) == 1
        assert doc["trace_complete"] is True, doc["gaps"]
        assert "ambiguous_episode" not in warning_codes(events)
    assert {streams["A"].episode_id, streams["B"].episode_id} == set(by_session)
    assert rec.unscoped() == []


def test_a_web_start_error_leaves_no_zombie(rec, monkeypatch):
    """regression for H13: the open set, the session table and the launch-args binding
    are undone, the trial runs untraced and the run log says why."""
    from capx_trace_recorder.hooks import trial as TRIAL

    def failing_tracer_info(store):
        raise RuntimeError("tracer info not ready")

    monkeypatch.setattr(TRIAL, "tracer_info", failing_tracer_info)
    seen = []

    async def fake_run_trial_async(session, args):
        seen.append(C.WEB_EPISODE.get())
        return "done"

    wrapped = WEB.make_run_trial_async_wrapper(fake_run_trial_async)
    session = types.SimpleNamespace(session_id="session-Z", config={}, env_factory={"cfg": {}},
                                    config_path="env_configs/fake/Z.yaml", await_user_input_each_turn=False,
                                    state=None, env=None)
    args = types.SimpleNamespace(model="m")
    assert asyncio.run(wrapped(session=session, args=args)) == "done"
    assert seen == [None]  # untraced
    assert C.REGISTRY.open_episodes() == []
    assert "session-Z" not in C.REGISTRY.web_sessions
    assert C.REGISTRY.binding(args) is None
    notes = [e["data"] for e in rec.unscoped() if e["data"].get("code") == "internal_error"]
    assert len(notes) == 1 and notes[0]["outside_episode"] is True
    assert notes[0]["message"] == "_web_start: RuntimeError: tracer info not ready"
    assert not any(p.is_dir() for p in (rec.root / "run").iterdir())


# ---------------------------------------------------------------------------
# (e) ambiguous events
# ---------------------------------------------------------------------------


def test_ambiguous_events_count_in_every_candidate(rec):
    a, b = rec.open("a"), rec.open("b")
    C.note_ambiguous("H7", [a, b])
    C.note_ambiguous("H7", [a, b])  # same key: no second warning, but a second gap
    c = rec.open("c")
    C.note_ambiguous("H7", [a, b, c])  # other candidates: a new warning
    C.note_ambiguous("H8", [a, b])  # other hook: a new warning
    for ep in (a, b, c):
        rec.end(ep)
    gaps = {ep.episode_id: rec.summary(ep)["gaps"] for ep in (a, b, c)}
    assert gaps == {"a": [{"code": "ambiguous_episode", "count": 4, "first_seq": None}],
                    "b": [{"code": "ambiguous_episode", "count": 4, "first_seq": None}],
                    "c": [{"code": "ambiguous_episode", "count": 1, "first_seq": None}]}
    unscoped = [(e["data"]["hook"], e["data"]["candidates"]) for e in rec.unscoped()]
    assert unscoped == [("H7", ["a", "b"]), ("H7", ["a", "b", "c"]), ("H8", ["a", "b"])]
    assert warning_codes(rec.events(a)).count("ambiguous_episode") == 3


def test_an_unbound_env_with_two_open_episodes_is_not_attributed(rec):
    a, b = rec.open("a"), rec.open("b")
    _in_thread(Env().render)
    for ep in (a, b):
        rec.end(ep)
        assert by_type(rec.events(ep), "observation_snapshot") == []
        assert rec.summary(ep)["trace_complete"] is False


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def test_duplicate_ids_of_200_characters_get_a_suffix(rec):
    result = {}

    def run():
        base = "a" * 200
        result["ids"] = [C.unique_episode_id(rec.store, "run", base) for _ in range(3)]
        (rec.root / "run" / ("b" * 199)).mkdir(parents=True)
        result["dir"] = C.unique_episode_id(rec.store, "run", "b" * 199)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "unique_episode_id did not return"
    ids = result["ids"]
    assert ids[0] == "a" * 200 and ids[1] == "a" * 194 + "__dup2" and ids[2] == "a" * 194 + "__dup3"
    assert all(len(i) <= 200 for i in ids + [result["dir"]])
    assert result["dir"] == "b" * 194 + "__dup2"


def test_long_episode_ids_keep_the_parts_that_tell_trials_apart():
    head = "task_" + "x" * 300
    ep_id = join_id(f"{head}__cond", "__s12__i3__a2")
    assert len(ep_id) == 200 and ep_id.endswith("__s12__i3__a2")
    assert join_id("t", "__s1") == "t__s1"


def test_a_duplicate_runner_id_keeps_the_trial_tail(rec):
    """regression: __dup<n> shortens the part before the tail, like join_id."""
    tail = "__s1__i0__a1"
    base = join_id("t" * 120 + "__" + "c" * 70, tail)
    assert len(base) == 200 and base.endswith(tail)
    assert C.unique_episode_id(rec.store, "run", base, tail) == base
    dup = C.unique_episode_id(rec.store, "run", base, tail)
    assert len(dup) == 200 and dup.endswith(tail + "__dup2")


def test_h4_duplicate_ids_keep_the_trial_tail(rec, monkeypatch, tmp_path):
    """The same through H4, for a runner episode whose id fills the 200 characters."""
    from capx_trace_recorder.hooks import trial as TRIAL

    runner = tmp_path / "runner" / "run1"
    runner.mkdir(parents=True)
    (runner / "run.json").write_text('{"invocations": [{}], "conditions": []}')
    output_dir = runner / ("t" * 120) / ("c" * 70)
    output_dir.mkdir(parents=True)
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    monkeypatch.setattr(C.REGISTRY, "runner", C.RunnerContext())
    first_id = join_id("t" * 120 + "__" + "c" * 70, "__s1__i1__a1")
    (rec.root / "run" / first_id).mkdir(parents=True)  # the id is taken
    seen = {}

    def _run_single_trial(env, trial, args, config, multi_turn_prompt=None, partial_artifacts=None):
        seen["id"] = C._stack("episodes")[-1].episode_id
        return "summary"

    TRIAL.make_run_single_trial_wrapper(_run_single_trial)(Env(), 1, None, {"output_dir": str(output_dir)})
    assert len(seen["id"]) == 200 and seen["id"].endswith("__s1__i1__a1__dup2")


def test_late_events_after_the_release_carry_no_clock(rec):
    """regression: the env's last clock is not stamped on events after the episode ended."""
    ep = rec.open("clock")
    low = Low()
    low._sim_step_count = 1234
    ep.low_env = low
    ep.emit("trace_warning", data={"code": "during", "counts_as_gap": False})
    rec.end(ep)
    rec.flush()
    assert ep.released and ep.low_env is None
    ep.emit("trace_warning", data={"code": "late", "counts_as_gap": False})
    events = {e["data"].get("code"): e for e in by_type(rec.events(ep), "trace_warning")}
    assert events["during"]["sim_step"] == 1234 and events["during"]["late"] is False
    assert events["late"]["sim_step"] is None and events["late"]["late"] is True


# ---------------------------------------------------------------------------
# Resets inside an episode, and the LIBERO goal
# ---------------------------------------------------------------------------


def test_a_reset_inside_the_episode_starts_a_new_sim_epoch(rec):
    """The web UI resets the env after a block timeout (section 5.2 env_reset)."""
    ep = rec.open("resets")
    env = Env()
    C.push_episode(ep)
    try:
        env.reset()
        env.render()
        env.reset()
        env.render()
    finally:
        C.pop_episode(ep)
    rec.end(ep)
    events = rec.events(ep)
    resets = by_type(events, "env_reset")
    assert [r["data"]["sim_epoch"] for r in resets] == [1]
    assert [s["sim_epoch"] for s in by_type(events, "observation_snapshot")] == [0, 1]


def test_the_libero_goal_is_registered_as_task_text(rec):
    from capx_trace_recorder.hooks import trial as TRIAL

    ep = rec.open("libero")
    goal = TRIAL.make_patch_libero_goal_wrapper(lambda env, obs: obs)
    text = "Put the bowl on the plate.\nAPIs:\nget_object_pose(name)"
    obs = {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}]}
    C.push_episode(ep)
    try:
        assert goal(Env(), obs) is obs
    finally:
        C.pop_episode(ep)
    segments = [(s.kind, s.text) for s in ep.segments.dynamic()]
    assert ("task_prompt", "Put the bowl on the plate.") in segments
    rec.end(ep)
