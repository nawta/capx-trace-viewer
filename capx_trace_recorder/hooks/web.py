"""H13: the web UI trial runner (docs/trace-format.md, sections 5.2, 7.1, 7.3).

``capx.web.async_trial_runner.run_trial_async`` is rebound in its defining
module by an ``async`` wrapper, so ``capx.web.server`` (also when it runs as
``__main__``) binds the wrapper at import. The wrapper never blocks the event
loop: episode start and end only queue events. The web episode is put into a
``contextvars`` variable (set inside the wrapper's ``try``), which reaches
``asyncio.to_thread`` workers (VDM calls). Threads started with
``run_in_executor`` do not see it: the env thread finds the episode through the
session that holds the env (``session.env``, then the env binding), and the
streaming thread through the launch arguments object the runner passes to
``_query_model_streaming`` (bound to the episode here). Both bindings stay as
tombstones after the episode ends, so work that is still running then is
recorded as late events of its own episode, even while two web episodes are
open. ``end_reason`` is read from the session state after the runner returns
(``COMPLETE`` -> completed, ``ERROR`` -> error, ``IDLE`` -> stopped_by_user
when ``SessionManager.stop_session`` was called for the session, else
cancelled_replaced).

The web runner's own aliases get role wrappers: ``_query_model`` is used only
for VDM calls (initial scene description before the first code generation,
image differencing after it) and ``_query_model_streaming`` only for code
generation (a generator wrapper using ``yield from``; the H3 wrapper inside it
records the streamed response). ``SessionManager.inject_prompt`` registers the
injected text as a ``user_feedback`` segment.

Related: ``capx_trace_recorder/hooks/llm.py`` (H3), ``capx_trace_recorder/hooks/env.py``
(binds the env created inside the episode), ``capx_trace_recorder/hooks/prompts.py``.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import guard, is_wrapper, mark, mark_only, rebind_aliases
from capx_trace_recorder.ids import sanitize_id, utc_stamp
from capx_trace_recorder.store import get_store
from capx_trace_recorder.timeout import is_trial_alarm


def _state_name(session: Any) -> str | None:
    state = getattr(session, "state", None)
    value = getattr(state, "value", state)
    return str(value) if value is not None else None


def _web_start(args: tuple, kwargs: dict, slot: Any) -> Any:
    """Open the web episode and queue ``episode_start``.

    As in H4 (``hooks/trial._episode_start``), everything after ``open_episode``
    is one unit with its cleanup: a recorder error there takes the episode out of
    the open set, the web session table and the launch-args binding, ends it if
    its start was queued, records an ``internal_error`` in the run's unscoped log,
    and the trial runs untraced (None is returned). The alarm and other
    non-``Exception`` exceptions are re-raised; a cleanup that fails after them runs
    in its own ``guard()``, so it never replaces them.
    """
    store = get_store()
    if store is None:
        return None
    from capx_trace_recorder.hooks.trial import (
        _abandon_start,
        config_flags,
        models_of,
        resolve_run_id,
        tracer_info,
    )

    session = args[0] if args else kwargs.get("session")
    launch_args = args[1] if len(args) > 1 else kwargs.get("args")
    session_id = str(getattr(session, "session_id", "") or "")
    run_id = resolve_run_id(None)
    tail = f"__web__{sanitize_id(session_id[:8])}"
    episode_id = f"{utc_stamp()}{tail}"
    ep = C.open_episode(store, run_id=run_id, episode_id=episode_id, source="web_ui", slot=slot, tail=tail)
    try:
        ep.web_session_id = session_id
        ep.web_session = session  # its ``env`` binds the env created inside the episode
        with C.REGISTRY.lock:
            C.REGISTRY.web_sessions[session_id] = ep
        # the runner passes this object to _query_model_streaming on a run_in_executor thread
        C.REGISTRY.bind_env(launch_args, ep)
        config = getattr(session, "config", None)
        config = config if isinstance(config, dict) else {}
        factory = getattr(session, "env_factory", None)
        cfg = factory.get("cfg") if isinstance(factory, dict) else None
        multi_turn_prompt = cfg.get("multi_turn_prompt") if isinstance(cfg, dict) else None
        ep.multi_turn_prompt = multi_turn_prompt
        flags = config_flags(config, launch_args, multi_turn_prompt)
        flags["await_user_input_each_turn"] = bool(getattr(session, "await_user_input_each_turn", False))
        ep.config_flags = flags
        config_path = getattr(session, "config_path", None)
        data = {
            "source": "web_ui",
            "task": Path(str(config_path)).stem if config_path else None,
            "condition": None,
            "seed": None,
            "attempt_index": 1,
            "runner_invocation": None,
            "condition_label": None,
            "runner_condition": None,
            "config_flags": flags,
            "models": models_of(launch_args, flags),
            "env": None,
            "env_pending": "the web UI creates the env inside the episode; see episode.json env",
            "tracer": tracer_info(store),
            "web_session_id": session_id,
            "output_dir": config.get("output_dir"),
        }

        def builder(ctx):
            from capx_trace_recorder.hooks.trial import _start_builder

            _start_builder(ctx)

        slot.start_queued = True  # from here on the start may be queued: undoing it ends the episode
        ep.emit("episode_start", status="started", data=data, builder=builder)
        state = _WebState(ep, session)
        slot.started = True
    except TimeoutError as exc:
        if is_trial_alarm(exc):
            # a failing cleanup cannot replace the alarm
            guard(_abandon_start, ep, slot, exc, "_web_start")
            raise
        _abandon_start(ep, slot, exc, "_web_start")
        return None
    except Exception as exc:  # noqa: BLE001 - the alarm is re-raised by the handler above
        _abandon_start(ep, slot, exc, "_web_start")
        return None
    except BaseException as exc:
        guard(_abandon_start, ep, slot, exc, "_web_start")
        raise
    return state


class _WebState:
    __slots__ = ("ep", "session", "token")

    def __init__(self, ep: Any, session: Any) -> None:
        self.ep = ep
        self.session = session
        self.token: Any = None


def _web_end(state: Any, result: Any, exc: BaseException | None) -> None:
    ep, token, session = state.ep, state.token, state.session
    from capx_trace_recorder.hooks.trial import end_episode

    session_id = ep.web_session_id or ""
    stop_requested = session_id in C.REGISTRY.web_stop_requested
    state_name = _state_name(session)
    if exc is not None:
        if type(exc).__name__ == "CancelledError":
            reason = "stopped_by_user" if stop_requested else "cancelled_replaced"
        elif isinstance(exc, KeyboardInterrupt):
            reason = "stopped_by_user"
        else:
            reason = "exception"
    elif result is not None:
        reason = "completed"
    elif state_name == "error":
        reason = "error"
    elif state_name == "idle":
        reason = "stopped_by_user" if stop_requested else "cancelled_replaced"
    else:
        reason = "error"
    try:
        end_episode(ep, end_reason=reason, summary=result, exc=exc,
                    extra={"session_state": state_name})
    finally:
        if token is not None:
            try:
                C.WEB_EPISODE.reset(token)
            except (ValueError, RuntimeError):
                C.WEB_EPISODE.set(None)
        elif C.WEB_EPISODE.get() is ep:
            C.WEB_EPISODE.set(None)
        tap = ep.frame_tap
        if tap is not None:
            tap.detach(ep)
        C.close_episode(ep)
        with C.REGISTRY.lock:
            C.REGISTRY.web_stop_requested.discard(session_id)


def _web_exit(slot: Any, state: Any, result: Any, exc: BaseException | None) -> None:
    """End the web episode (once), or undo a start that did not complete."""
    ep = slot.ep
    if ep is None or slot.abandoned or ep.ended:
        return
    if state is None:
        from capx_trace_recorder.hooks.trial import _abandon_start

        _abandon_start(ep, slot, exc, "_web_start")  # interrupted between open_episode and the end of the start
        return
    _web_end(state, result, exc)


def make_run_trial_async_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    async def run_trial_async(*args: Any, **kwargs: Any) -> Any:
        slot = C.StartSlot()
        state = None
        try:
            state = guard(_web_start, args, kwargs, slot)
            if state is not None:
                state.token = C.WEB_EPISODE.set(state.ep)
            result = await original(*args, **kwargs)  # untraced when the start did not complete
        except BaseException as exc:
            guard(_web_exit, slot, state, None, exc)
            raise
        guard(_web_exit, slot, state, result, None)
        return result

    return mark(run_trial_async, original)


# ---------------------------------------------------------------------------
# Role wrappers for the web runner's aliases
# ---------------------------------------------------------------------------


class _WebRole:
    __slots__ = ("ep", "role", "push_episode", "pushed")

    def __init__(self, ep: Any, role: Any, push_episode: bool) -> None:
        self.ep = ep
        self.role = role
        self.push_episode = push_episode
        self.pushed = False

    def push(self) -> None:
        """Called as the first statement inside the wrapper's ``try``.

        ``_pop_web_role`` pops both entries whether or not they were pushed
        (popping an absent entry does nothing), so an alarm here leaves nothing.
        """
        self.pushed = True
        if self.push_episode:
            C.push_episode(self.ep)
        C.push_role(self.role)


def _make_web_role(kind: str, launch_args: Any = None) -> _WebRole | None:
    """The role (and, on a thread without the context variable, the episode) to push.

    The streaming thread is started with ``run_in_executor``: it finds its
    episode through ``launch_args``, bound by ``_web_start``.
    """
    ep = C.resolve_or_note("H13", owners=(launch_args,) if launch_args is not None else ())
    if ep is None or not hasattr(ep, "codegen_requests"):
        return None
    if kind == "vdm":
        if ep.codegen_requests == 0:
            role = C.RoleCtx("vdm", "initial_scene_description", 0, ep)
        else:
            # the image diff feeds the next decision (as in the batch path)
            role = C.RoleCtx("vdm", "image_diff", ep.turn_index + 1, ep)
    else:
        if ep.codegen_requests == 0:
            role = C.RoleCtx("code_generator", "initial_code", 0, ep)
        else:
            ep.turn_index += 1
            role = C.RoleCtx("code_generator", "multi_turn_decision", ep.turn_index, ep)
    stack = getattr(C._THREAD, "episodes", None)
    push_episode = C.WEB_EPISODE.get() is None and not stack
    return _WebRole(ep, role, push_episode)


def _pop_web_role(token: _WebRole | None) -> None:
    if token is None:
        return
    C.pop_role(token.role)
    if token.push_episode:
        C.pop_episode(token.ep)


def _launch_args(args: tuple, kwargs: dict) -> Any:
    return args[0] if args else kwargs.get("args")


def make_web_vdm_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _query_model(*args: Any, **kwargs: Any) -> Any:
        token = guard(_make_web_role, "vdm", _launch_args(args, kwargs))
        try:
            if token is not None:
                token.push()
            result = original(*args, **kwargs)
        finally:
            guard(_pop_web_role, token)  # on every way out
        return result

    return mark_only(_query_model, original)


def make_web_stream_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _query_model_streaming(*args: Any, **kwargs: Any):
        token = guard(_make_web_role, "codegen", _launch_args(args, kwargs))
        try:
            if token is not None:
                token.push()
            result = yield from original(*args, **kwargs)
        finally:
            guard(_pop_web_role, token)  # on every way out, GeneratorExit included
        return result

    return mark_only(_query_model_streaming, original)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


def _after_inject(session_id: Any, text: Any, ok: Any) -> None:
    if ok is not True or not isinstance(text, str) or not text:
        return
    with C.REGISTRY.lock:
        ep = C.REGISTRY.web_sessions.get(str(session_id))
    if ep is None:
        return
    ep.pending_user_feedback.append(text)
    ep.segments.register("user_feedback", text, (), "web_ui inject_prompt")


def make_inject_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    async def inject_prompt(self: Any, session_id: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        ok = await original(self, session_id, text, *args, **kwargs)
        guard(_after_inject, session_id, text, ok)
        return ok

    return mark(inject_prompt, original)


def _before_stop(session_id: Any) -> None:
    with C.REGISTRY.lock:
        C.REGISTRY.web_stop_requested.add(str(session_id))


def make_stop_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    async def stop_session(self: Any, session_id: Any, *args: Any, **kwargs: Any) -> Any:
        guard(_before_stop, session_id)
        return await original(self, session_id, *args, **kwargs)

    return mark(stop_session, original)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install_runner(module: Any) -> list[str]:
    problems = []
    original = getattr(module, "run_trial_async", None)
    if original is None or not inspect.iscoroutinefunction(original):
        problems.append("run_trial_async: not found or not a coroutine function")
    elif not is_wrapper(original):
        module.run_trial_async = make_run_trial_async_wrapper(original)
    rebind_aliases(module, ("_query_model", "_query_model_streaming", "_build_multi_turn_decision_prompt",
                            "_extract_code", "_parse_multi_turn_decision"))
    vdm = getattr(module, "_query_model", None)
    if vdm is not None and not getattr(vdm, "__capx_trace_web__", False):
        wrapper = make_web_vdm_wrapper(vdm)
        wrapper.__capx_trace_web__ = True  # type: ignore[attr-defined]
        module._query_model = wrapper
    stream = getattr(module, "_query_model_streaming", None)
    if stream is not None and not getattr(stream, "__capx_trace_web__", False):
        if not inspect.isgeneratorfunction(stream):
            problems.append("_query_model_streaming is not a generator function")
        else:
            wrapper = make_web_stream_wrapper(stream)
            wrapper.__capx_trace_web__ = True  # type: ignore[attr-defined]
            module._query_model_streaming = wrapper
    return problems


def install_session_manager(module: Any) -> list[str]:
    cls = getattr(module, "SessionManager", None)
    if not isinstance(cls, type):
        return ["SessionManager not found"]
    problems = []
    for name, factory in (("inject_prompt", make_inject_wrapper), ("stop_session", make_stop_wrapper)):
        current = cls.__dict__.get(name)
        if current is None or not inspect.iscoroutinefunction(current):
            problems.append(f"SessionManager.{name}: not found or not async")
            continue
        if not is_wrapper(current):
            setattr(cls, name, factory(current))
    return problems


__all__ = ["install_runner", "install_session_manager"]
