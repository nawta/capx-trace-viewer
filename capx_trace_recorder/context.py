"""Episode registry, per-thread stacks and role context (docs/trace-format.md, section 7.3).

Episode resolution for an event with no enclosing context, first match wins:

1. the web episode of the current ``contextvars`` context (set by H13; it
   reaches ``asyncio.to_thread`` workers but not ``run_in_executor`` threads);
2. the thread's own episode stack (pushed by H4 on the trial thread);
3. the innermost open context of this thread (code block, API call, logical
   model call, service call), each of which carries its episode;
4. the binding of a known object: the env or its low-level env (bound by H4 and
   H7), or the web runner's launch arguments (bound by H13). A binding outlives
   its episode as a tombstone (weakly referenced, so it never keeps an env
   alive): events from an env of an ended episode go to that episode as late
   events and never to another open episode. Each binding has a generation, and
   a thread remembers the binding it used for an object ("pin"). When the object
   has been bound to a newer episode since, the pin decides only where the work
   is known to be one job: a thread started for one target (not a pool worker)
   still works for the older episode, so its events stay there (late). A pool
   worker (``ThreadPoolExecutor``, ``multiprocessing.pool.ThreadPool``), the main
   thread or a thread of unknown kind may be running a new job, and nothing
   tells the two apart, so the event is ambiguous between the pinned and the
   current episode (recorded as below, never given to the older one). Per-job
   signals win over pins because they come first: the web contextvar (copied
   into each ``asyncio.to_thread`` job), the episode a wrapper pushes on the
   thread (H4, the H13 streaming thread; ``push_episode`` also drops the
   thread's pins) and the innermost open context;
5. an open web episode whose session holds the env (``session.env``, H13), so
   the env of each web trial is bound even while two web episodes are open;
6. the single open episode of the process.

With more than one candidate the event is not attributed: each candidate
episode gets an ``ambiguous_episode`` gap per event, and one warning per
(hook, run, candidates) goes to the run's ``unscoped`` log and to each
candidate. Model requests without an episode go to the ``unscoped`` log.

Role context (``role``, ``call_kind``, ``turn_index``) is per thread; ensemble
pool threads fall back to the episode's ``current_role`` set by the caller
before the pool starts.

Contexts are pushed by the wrappers as the first statement inside their own
``try`` and popped in its ``finally``, so an alarm cannot leave one behind
between the push and the ``try`` or after the original returned. H4 clears
whatever is still left on the trial thread when an episode starts or ends
(``clear_stale``), records it, and ends an episode that an interrupted exit
path left open.

Related: ``capx_trace_recorder/store.py`` (``EpisodeRecorder``), ``capx_trace_recorder/hooks/*``.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from capx_trace_recorder.ids import random_hex, sanitize_id, with_suffix
from capx_trace_recorder.store import EpisodeRecorder, TraceStore, UnscopedLog, get_store

_THREAD = threading.local()
WEB_EPISODE: contextvars.ContextVar[EpisodeRecorder | None] = contextvars.ContextVar(
    "capx_trace_web_episode", default=None
)


# ---------------------------------------------------------------------------
# Context objects (each carries its episode)
# ---------------------------------------------------------------------------


class RoleCtx:
    __slots__ = ("role", "call_kind", "turn_index", "episode")

    def __init__(self, role: str, call_kind: str | None, turn_index: int | None,
                 episode: EpisodeRecorder | None) -> None:
        self.role = role
        self.call_kind = call_kind
        self.turn_index = turn_index
        self.episode = episode


class Ctx:
    """Base of the typed context entries on a thread's stack."""

    kind = "ctx"
    __slots__ = ("event_id", "episode")

    def __init__(self, event_id: str, episode: EpisodeRecorder) -> None:
        self.event_id = event_id
        self.episode = episode


class CodeBlockCtx(Ctx):
    kind = "code_block"
    __slots__ = ("code", "exec_globals", "module_codes", "start_mono", "sim_step0", "buf_len0",
                 "block_index", "exec_result", "exec_duration", "env")

    def __init__(self, event_id: str, episode: EpisodeRecorder, code: Any, env: Any) -> None:
        super().__init__(event_id, episode)
        self.code = code if isinstance(code, str) else None
        self.env = env
        self.exec_globals: Any = None
        self.module_codes: dict[int, set[int]] = {}
        self.start_mono = time.monotonic()
        self.sim_step0: int | None = None
        self.buf_len0: int | None = None
        self.block_index = 0
        self.exec_result: Any = None
        self.exec_duration: float | None = None


class ApiCallCtx(Ctx):
    kind = "api_call"
    __slots__ = ("name", "depth", "start_mono", "sim_step0", "epoch0", "buf_len0",
                 "stdout_buf", "stderr_buf", "stdout0", "stderr0", "api")

    def __init__(self, event_id: str, episode: EpisodeRecorder, name: str, depth: int) -> None:
        super().__init__(event_id, episode)
        self.name = name
        self.depth = depth
        self.start_mono = time.monotonic()
        self.sim_step0: int | None = None
        self.epoch0: int | None = None
        self.buf_len0: int | None = None
        self.stdout_buf: Any = None
        self.stderr_buf: Any = None
        self.stdout0: int | None = None
        self.stderr0: int | None = None
        self.api: Any = None


class ServiceCallCtx(Ctx):
    kind = "service_call"
    __slots__ = ("service", "function", "attempts", "endpoint", "start_mono", "molmo_call",
                 "molmo_calls")

    def __init__(self, event_id: str, episode: EpisodeRecorder, service: str, function: str) -> None:
        super().__init__(event_id, episode)
        self.service = service
        self.function = function
        self.attempts: list[dict[str, Any]] = []
        self.endpoint: Any = None
        self.start_mono = time.monotonic()
        self.molmo_call: "LogicalCall | None" = None
        self.molmo_calls: list[str] = []


class LogicalCall(Ctx):
    """A logical model call (created by H3, or per Molmo object by H1/H14)."""

    kind = "logical_call"
    __slots__ = ("role", "call_kind", "turn_index", "model", "attempts", "last_request_id",
                 "last_response_id", "last_status", "segments", "parent", "stream_info",
                 "start_mono", "origin")

    def __init__(self, event_id: str, episode: EpisodeRecorder, role: str, call_kind: str | None,
                 turn_index: int | None, model: Any, parent: str | None, origin: str) -> None:
        super().__init__(event_id, episode)
        self.role = role
        self.call_kind = call_kind
        self.turn_index = turn_index
        self.model = model
        self.attempts = 0
        self.last_request_id: str | None = None
        self.last_response_id: str | None = None
        self.last_status: int | None = None
        self.segments: tuple = ()
        self.parent = parent
        self.stream_info: dict[str, Any] = {}
        self.start_mono = time.monotonic()
        self.origin = origin


# ---------------------------------------------------------------------------
# Per-thread stacks
# ---------------------------------------------------------------------------


def _stack(name: str) -> list:
    stack = getattr(_THREAD, name, None)
    if stack is None:
        stack = []
        setattr(_THREAD, name, stack)
    return stack


def push_ctx(ctx: Ctx) -> None:
    _stack("frames").append(ctx)


def pop_ctx(ctx: Ctx) -> None:
    stack = _stack("frames")
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is ctx:
            del stack[i:]
            return


def innermost(kind: str | None = None) -> Ctx | None:
    stack = _stack("frames")
    for i in range(len(stack) - 1, -1, -1):
        if kind is None or stack[i].kind == kind:
            return stack[i]
    return None


def api_depth() -> int:
    return sum(1 for c in _stack("frames") if c.kind == "api_call")


def push_role(role: RoleCtx) -> None:
    _stack("roles").append(role)


def pop_role(role: RoleCtx) -> None:
    stack = _stack("roles")
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is role:
            del stack[i:]
            return


def thread_role() -> RoleCtx | None:
    stack = _stack("roles")
    return stack[-1] if stack else None


def push_episode(ep: EpisodeRecorder) -> None:
    """Make ``ep`` this thread's episode (H4, the web streaming thread).

    The thread now works for ``ep`` explicitly, so its env pins (step 4 of the
    resolution order) no longer apply.
    """
    _stack("episodes").append(ep)
    pins = getattr(_THREAD, "env_pins", None)
    if pins:
        pins.clear()


def pop_episode(ep: EpisodeRecorder) -> None:
    stack = _stack("episodes")
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is ep:
            del stack[i:]
            return


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RunnerContext:
    """What H15 learned from the experiment runner."""

    def __init__(self) -> None:
        self.manifest: Any = None
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.exported = False


class Binding:
    """An object (env, low-level env, web launch args) bound to an episode."""

    __slots__ = ("ref", "strong", "ep", "generation")

    def __init__(self, obj: Any, ep: EpisodeRecorder, generation: int) -> None:
        try:
            self.ref: Any = weakref.ref(obj)
            self.strong: Any = None
        except TypeError:  # not weak-referenceable: held only while the episode is open
            self.ref = None
            self.strong = obj
        self.ep = ep
        self.generation = generation

    def target(self) -> Any:
        if self.strong is not None:
            return self.strong
        return self.ref() if self.ref is not None else None


MAX_BINDINGS = 256


class Registry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.open: dict[str, EpisodeRecorder] = {}
        self.env_bind: dict[int, Binding] = {}
        self.generation = 0
        self.unscoped: dict[str, UnscopedLog] = {}
        self.used_ids: set[str] = set()
        self.runner = RunnerContext()
        self.web_sessions: dict[str, EpisodeRecorder] = {}
        self.web_stop_requested: set[str] = set()
        self.attempts: dict[tuple[str, Any], int] = {}
        self.run_json_cache: dict[str, Any] = {}
        self.any_episode = False
        self.hooks_installed: list[str] = []
        self.hooks_failed: list[dict[str, Any]] = []
        self.hooks_skipped: list[dict[str, Any]] = []
        self.failures_logged = 0  # hooks_failed entries already written to the run's unscoped log

    def h4_failed(self) -> bool:
        """True when H4 (episodes) is not installed: no trial opens an episode."""
        with self.lock:
            return any(f.get("hook") == "H4" for f in self.hooks_failed)

    def open_episodes(self) -> list[EpisodeRecorder]:
        with self.lock:
            return list(self.open.values())

    def bind_env(self, env: Any, ep: EpisodeRecorder) -> None:
        """Bind ``env`` (or any object) to ``ep``; a new binding gets a new generation."""
        if env is None:
            return
        with self.lock:
            key = id(env)
            current = self.env_bind.get(key)
            if current is not None and current.ep is ep and current.target() is env:
                return
            self.generation += 1
            self.env_bind[key] = Binding(env, ep, self.generation)
            if len(self.env_bind) > MAX_BINDINGS:
                for k in [k for k, b in self.env_bind.items() if b.target() is None]:
                    del self.env_bind[k]

    def binding(self, obj: Any) -> Binding | None:
        """The current binding of ``obj`` (its episode may have ended), else None."""
        if obj is None:
            return None
        with self.lock:
            entry = self.env_bind.get(id(obj))
        if entry is None or entry.target() is not obj:
            return None
        return entry

    def episode_for_env(self, env: Any) -> EpisodeRecorder | None:
        """The episode ``env`` is bound to, open or ended (a tombstone), else None."""
        entry = self.binding(env)
        return entry.ep if entry is not None else None

    def forget_strong(self, ep: EpisodeRecorder) -> None:
        """At episode close: keep tombstones only for weakly referenced objects."""
        with self.lock:
            for key in [k for k, b in self.env_bind.items() if b.ep is ep and b.strong is not None]:
                del self.env_bind[key]

    def unbind_episode(self, ep: EpisodeRecorder) -> None:
        """Remove every binding to ``ep`` (an abandoned start, tests)."""
        with self.lock:
            for key in [k for k, b in self.env_bind.items() if b.ep is ep]:
                del self.env_bind[key]

    def web_episode_owning(self, env: Any) -> EpisodeRecorder | None:
        """The open web episode whose session holds ``env`` (or its low-level env).

        The web runner stores the env it created in ``session.env`` before the
        first reset; this binds each web trial's env to its own episode even
        when two web episodes are open (H13).
        """
        found = None
        for ep in self.open_episodes():
            session = getattr(ep, "web_session", None)
            if session is None:
                continue
            owned = _instance_attr(session, "env")
            if owned is None:
                continue
            if owned is env or _instance_attr(owned, "low_level_env") is env:
                if found is not None and found is not ep:
                    return None  # two sessions claim it: not attributable
                found = ep
        return found


REGISTRY = Registry()


def reset_registry_after_fork() -> None:
    global REGISTRY, _THREAD
    REGISTRY = Registry()
    _THREAD = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=reset_registry_after_fork)


def _instance_attr(obj: Any, name: str) -> Any:
    """``obj.__dict__[name]`` without descriptors or ``__getattr__``, else None."""
    try:
        return obj.__dict__.get(name)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001 - no instance dict
        return None


def _pins() -> dict[int, Binding]:
    pins = getattr(_THREAD, "env_pins", None)
    if pins is None:
        pins = {}
        _THREAD.env_pins = pins
    return pins


# thread targets that run many jobs, one after another, on the same thread
_POOL_WORKER_MODULES = frozenset({"concurrent.futures.thread", "multiprocessing.pool"})


def thread_is_one_job() -> bool:
    """True for a thread started for one target that is not a pool worker loop.

    Such a thread does one piece of work for its whole life, so a pin taken on it
    belongs to that work. Pool workers, the main thread and threads of unknown
    kind (no target, for example a ``Thread`` subclass or a thread not started by
    ``threading``) can run several jobs: a pin taken in an earlier job says
    nothing about the current one.
    """
    thread = threading.current_thread()
    try:
        target = thread.__dict__.get("_target")
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return False
    if target is None:
        return False
    module = getattr(target, "__module__", None)
    return isinstance(module, str) and module not in _POOL_WORKER_MODULES


class _Ambiguous:
    """``_bound_episode`` result: the pinned and the current episode both qualify."""

    __slots__ = ("candidates",)

    def __init__(self, candidates: list[EpisodeRecorder]) -> None:
        self.candidates = candidates


def _bound_episode(obj: Any) -> EpisodeRecorder | _Ambiguous | None:
    """Step 4 of the resolution order, with this thread's pin for ``obj``."""
    entry = REGISTRY.binding(obj)
    if entry is None:
        return None
    pins = _pins()
    pin = pins.get(id(obj))
    if pin is not None and pin is not entry and pin.target() is obj:
        # the object was bound to a newer episode after this thread used it
        if pin.ep is entry.ep:
            pins[id(obj)] = entry
            return entry.ep
        if thread_is_one_job():
            return pin.ep  # the same job, still working for the older episode
        return _Ambiguous([pin.ep, entry.ep])  # maybe a new job: cannot tell
    pins[id(obj)] = entry
    if len(pins) > 64:
        for key in [k for k, b in pins.items() if b.target() is None]:
            del pins[key]
    return entry.ep


def resolve_episode(env: Any = None, owners: tuple[Any, ...] = ()) -> tuple[EpisodeRecorder | None, list[EpisodeRecorder]]:
    """``(episode, candidates)``; ``episode`` is None when none or ambiguous.

    ``env``: the env (or low-level env) the event concerns; ``owners``: other
    bound objects (the web runner's launch arguments).
    """
    ep = WEB_EPISODE.get()
    if ep is not None:
        return ep, [ep]
    stack = _stack("episodes")
    if stack:
        return stack[-1], [stack[-1]]
    ctx = innermost()
    if ctx is not None:
        return ctx.episode, [ctx.episode]
    objects: list[Any] = []
    if env is not None:
        objects.append(env)
        low = _instance_attr(env, "low_level_env")
        if low is not None:
            objects.append(low)
    objects.extend(o for o in owners if o is not None)
    for obj in objects:
        bound = _bound_episode(obj)
        if isinstance(bound, _Ambiguous):
            return None, bound.candidates
        if bound is not None:
            return bound, [bound]
    if env is not None:
        web = REGISTRY.web_episode_owning(env)
        if web is not None:
            return web, [web]
    candidates = REGISTRY.open_episodes()
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def resolve_or_note(where: str, env: Any = None, owners: tuple[Any, ...] = ()) -> EpisodeRecorder | None:
    """``resolve_episode`` for a hook that records an event; an ambiguous event is noted."""
    ep, candidates = resolve_episode(env, owners)
    if ep is None and len(candidates) > 1:
        note_ambiguous(where, candidates)
    return ep


def unscoped_log(store: TraceStore, run_id: str) -> UnscopedLog:
    """The run's ``unscoped.<pid>.jsonl`` log of ``store`` (one per run and store)."""
    with REGISTRY.lock:
        log = REGISTRY.unscoped.get(run_id)
        if log is None or log.store is not store:
            log = UnscopedLog(store, run_id, store.config.trace_dir / run_id)
            REGISTRY.unscoped[run_id] = log
        return log


def unscoped_sink_for(candidates: list[EpisodeRecorder]) -> UnscopedLog | None:
    """Where an event with no (or an ambiguous) episode goes, if anywhere.

    Nothing is recorded in a process that never opened an episode and is not
    the experiment runner (for example a perception server that inherited
    ``CAPX_TRACE_DIR``): "tracing enabled but no episode running" creates no
    episode file or writer thread. The startup hook may write a process status file.
    """
    store = get_store()
    if store is None:
        return None
    if candidates:
        return unscoped_log(store, candidates[0].run_id)
    runner = REGISTRY.runner
    if runner.run_id:
        return unscoped_log(store, runner.run_id)
    return None


_AMBIGUOUS_NOTED: set[tuple[str, str, tuple[str, ...]]] = set()


def note_ambiguous(where: str, candidates: list[EpisodeRecorder]) -> None:
    """An event of hook ``where`` could not be attributed (section 7.3).

    Every candidate episode gets one ``ambiguous_episode`` gap per event, so none
    of them is shown as complete. The explaining warning is written once per
    (hook, run, candidates) to the run's unscoped log and to each candidate.
    """
    if len(candidates) < 2:
        return
    for ep in candidates:
        ep.note_gap("ambiguous_episode")
    ids = tuple(sorted(str(c.episode_id) for c in candidates))
    key = (where, str(candidates[0].run_id), ids)
    with REGISTRY.lock:
        if key in _AMBIGUOUS_NOTED:
            return
        _AMBIGUOUS_NOTED.add(key)
    message = (f"{where}: {len(candidates)} open episodes; events from this hook that could "
               "not be attributed were not recorded (each counts as an ambiguous_episode gap)")
    data = {"code": "ambiguous_episode", "message": message, "hook": where, "candidates": list(ids),
            "counts_as_gap": False}
    for ep in candidates:
        ep.emit("trace_warning", status="warning", data=dict(data))
    sink = unscoped_sink_for(candidates)
    if sink is not None:
        sink.emit("trace_warning", status="warning", data=dict(data))


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def current_role(ep: EpisodeRecorder | None) -> RoleCtx | None:
    role = thread_role()
    if role is not None:
        return role
    if ep is not None and ep.current_role is not None:
        return ep.current_role
    return None


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


MAX_DUP = 10000


def unique_episode_id(store: TraceStore, run_id: str, episode_id: str, tail: str = "") -> str:
    """``episode_id``, or ``<id>__dup<n>`` when the id or its directory is taken.

    The suffix always fits in 200 characters: the part before ``tail`` (the
    parts that tell trials apart, see ``ids.join_id``) is shortened, so the tail
    survives. After ``MAX_DUP`` numbered tries a random suffix is used, so the
    loop always ends.
    """
    base = sanitize_id(episode_id)
    run_dir = store.config.trace_dir / run_id
    candidate = base
    n = 1
    with REGISTRY.lock:
        while candidate in REGISTRY.used_ids or (run_dir / candidate).exists():
            n += 1
            if n > MAX_DUP + 16:
                break  # 16 random suffixes collided too: use the last one
            suffix = f"__dup{n}" if n <= MAX_DUP else f"__dup{random_hex(12)}"
            candidate = with_suffix(base, suffix, tail)
        REGISTRY.used_ids.add(candidate)
    return candidate


class StartSlot:
    """Filled in by ``open_episode`` the moment the episode is registered.

    H4 and H13 pass one, so that whatever interrupts the start (an alarm, a
    recorder error) finds the episode and undoes the registration.
    """

    __slots__ = ("ep", "start_queued", "started", "abandoned")

    def __init__(self) -> None:
        self.ep: EpisodeRecorder | None = None
        self.start_queued = False  # episode_start was (or was about to be) queued
        self.started = False  # the start completed (H4: pushed on the trial thread)
        self.abandoned = False


def open_episode(store: TraceStore, *, run_id: str, episode_id: str, source: str,
                 slot: StartSlot | None = None, tail: str = "") -> EpisodeRecorder:
    run_id = sanitize_id(run_id)
    episode_id = unique_episode_id(store, run_id, episode_id, tail)
    ep = EpisodeRecorder(store, run_id, episode_id, store.config.trace_dir / run_id, source)
    if slot is not None:
        slot.ep = ep  # before the registration: an interruption after it finds the episode
    with REGISTRY.lock:
        REGISTRY.open[episode_id] = ep
        REGISTRY.any_episode = True
    return ep


def abandon_episode(ep: EpisodeRecorder) -> None:
    """Undo the registration of an episode whose start did not complete.

    The episode leaves the thread's stack and the open set, its bindings (env,
    low-level env, web launch arguments) and its web session entry are removed,
    so later events are never attributed to it and no other episode becomes
    ambiguous because of it. Recording what happened is the caller's part.
    """
    try:
        pop_episode(ep)
    finally:
        close_episode(ep)
        REGISTRY.unbind_episode(ep)


def close_episode(ep: EpisodeRecorder) -> None:
    """The episode is no longer open. Its bindings stay as tombstones (late events)."""
    with REGISTRY.lock:
        if REGISTRY.open.get(ep.episode_id or "") is ep:
            REGISTRY.open.pop(ep.episode_id or "", None)
        for sid in [s for s, e in REGISTRY.web_sessions.items() if e is ep]:
            REGISTRY.web_sessions.pop(sid, None)
    REGISTRY.forget_strong(ep)


def clear_stale(stacks: tuple[str, ...] = ("frames", "roles", "episodes")) -> list[Any]:
    """Remove and return every entry of this thread's context stacks ``stacks``.

    H4 calls it on the trial thread when an episode starts (anything still there
    was left by an earlier episode) and, for ``frames`` and ``roles``, when the
    episode ends (anything still there was left by this episode).
    """
    removed: list[Any] = []
    for name in stacks:
        stack = _stack(name)
        if stack:
            removed.extend(stack)
            stack.clear()
    return removed


def entry_episode(entry: Any) -> Any:
    """The episode a stack entry belongs to."""
    return entry if isinstance(entry, EpisodeRecorder) else getattr(entry, "episode", None)


__all__ = [
    "REGISTRY",
    "WEB_EPISODE",
    "RoleCtx",
    "CodeBlockCtx",
    "ApiCallCtx",
    "ServiceCallCtx",
    "LogicalCall",
    "push_ctx",
    "pop_ctx",
    "innermost",
    "api_depth",
    "push_role",
    "pop_role",
    "thread_role",
    "push_episode",
    "pop_episode",
    "resolve_episode",
    "resolve_or_note",
    "unscoped_sink_for",
    "note_ambiguous",
    "current_role",
    "open_episode",
    "close_episode",
    "abandon_episode",
    "StartSlot",
    "clear_stale",
    "Binding",
]
