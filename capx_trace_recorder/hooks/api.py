"""H8: public API methods (docs/trace-format.md, sections 5.2, 5.3, 7.2).

``combined_doc()`` loops over ``self.functions()`` (bound methods
``self.<name>``) and writes ``name + str(inspect.signature(fn))`` and
``inspect.getdoc(fn)``. Each method is replaced by an *instance* attribute
``types.MethodType(functools.wraps(fn.__func__)(traced), fn.__self__)``: the
signature, the docstring (including an inherited one found by
``inspect._finddoc``), ``__name__`` and ``__qualname__`` stay identical.
Per API instance ``combined_doc()`` is computed before and after installing;
on any difference the instance attributes are removed again and
``hook_install_failed`` is recorded.

Each thread keeps a stack of open API calls: ``depth`` is its size and the
parent is the open call, or the code block for depth 0. Depth-0 calls record
the call site in the generated code (the ``<module>`` frame whose globals are
the sandbox's ``_exec_globals``), key frames, a ground-truth state snapshot and
the offsets of the Tee buffers that capture stdout/stderr during ``exec``.
Arguments are summarised before the call and the return value right after it
(arrays copied at that moment). The call's context is pushed as the first
statement inside the wrapper's ``try`` (see ``capx_trace_recorder/hooks/common.py``).
Exception messages, causes and tracebacks are redacted on the writer
(``payload.redact``, with ``message_redacted``, ``traceback_redacted`` and
``cause.message_redacted`` when a rule replaced something in the stored text);
the exception re-raised to the caller is untouched. The traceback is kept on the
trial thread as raw ``(filename, lineno, name)`` tuples and formatted by the
writer, so ``linecache`` (whose ``except OSError`` could swallow the trial
alarm) is never touched on the trial thread. A ground-truth state snapshot is
counted in the queue budget like the array copies; a dropped item releases it
and its ``state_ref`` says ``reason: queue_full``. A public function that cannot
be wrapped makes the install partial, which the caller records as
``hook_install_failed``.

Related: ``capx_trace_recorder/serialize.py``, ``capx_trace_recorder/frames.py``,
``capx_trace_recorder/state_probe.py``, ``capx_trace_recorder/hooks/env.py`` (installs H8/H9
lazily at the first code block), ``capx_trace_recorder/hooks/services.py`` (H9).
"""

from __future__ import annotations

import functools
import inspect
import io
import sys
import threading
import time
import types
import weakref
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import add_overhead, guard, is_wrapper, mark
from capx_trace_recorder.ids import new_event_id
from capx_trace_recorder.serialize import (
    ArrayBudget,
    pending_builders,
    summarize,
    summarize_arguments,
    text_value,
    units_for,
)
from capx_trace_recorder.timeout import is_trial_alarm

MAX_SLICE_BUFFER_CHARS = 8 * 1024 * 1024
# id(api) -> (weak reference, names of public functions that could not be
# wrapped): a long-running web server creates new API instances per trial, and
# the recorder must not keep old ones (and their envs) alive. The weak reference
# has no callback (a callback runs on whatever thread drops the instance, where
# an alarm inside it would be printed as "Exception ignored"); dead entries are
# removed at the next install.
_INSTALLED: dict[int, tuple[Any, list[str]]] = {}
_INSTALL_LOCK = threading.Lock()
_SIGNATURES: dict[int, inspect.Signature | None] = {}
_WRAPPED_FUNCS: dict[tuple[int, str], Any] = {}


# ---------------------------------------------------------------------------
# Call-time helpers
# ---------------------------------------------------------------------------


def _signature(func: Any) -> inspect.Signature | None:
    key = id(func)
    if key not in _SIGNATURES:
        try:
            _SIGNATURES[key] = inspect.signature(func)
        except (TypeError, ValueError):
            _SIGNATURES[key] = None
    return _SIGNATURES[key]


def bind_arguments(func: Any, self_obj: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    sig = _signature(func)
    if sig is not None:
        try:
            bound = sig.bind_partial(self_obj, *args, **kwargs).arguments
            bound = dict(bound)
            if bound:
                bound.pop(next(iter(bound)))  # drop self
            return bound
        except TypeError:
            pass
    out = {f"arg{i}": v for i, v in enumerate(args)}
    out.update(kwargs)
    return out


def _tee_buffers() -> tuple[Any, Any]:
    base = sys.modules.get("capx.envs.tasks.base")
    tee = getattr(base, "Tee", None) if base is not None else None
    out = []
    for stream in (sys.stdout, sys.stderr):
        buf = None
        if tee is not None and type(stream) is tee:
            streams = getattr(stream, "streams", None)
            if isinstance(streams, tuple) and streams and isinstance(streams[-1], io.StringIO):
                buf = streams[-1]
        out.append(buf)
    return out[0], out[1]


def _offset(buf: Any) -> int | None:
    if buf is None:
        return None
    try:
        return int(buf.tell())
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None


def _slice(buf: Any, start: int | None, pending: list[Any]) -> dict[str, Any] | None:
    if buf is None or start is None:
        return None
    end = _offset(buf)
    if end is None:
        return None
    out: dict[str, Any] = {"start": start, "end": end, "text": ""}
    if end > start:
        if end > MAX_SLICE_BUFFER_CHARS:
            out["text"] = None
            out["text_omitted"] = "buffer_too_large"
        else:
            try:
                out["text"] = text_value(buf.getvalue()[start:end], pending, "text")
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                out["text"] = None
    return out


def _module_code_ids(block: Any, module_code: Any) -> set[int]:
    key = id(module_code)
    ids = block.module_codes.get(key)
    if ids is None:
        ids = set()
        stack = [module_code]
        while stack:
            code = stack.pop()
            ids.add(id(code))
            for const in getattr(code, "co_consts", ()):
                if hasattr(const, "co_consts"):
                    stack.append(const)
        block.module_codes[key] = ids
    return ids


def call_site(block: Any, frame: Any) -> dict[str, Any] | None:
    """Line in the generated code that led to this depth-0 call."""
    if block is None:
        return None
    site: dict[str, Any] = {"code_block_id": block.event_id, "line": None, "source_line": None}
    globs = block.exec_globals
    if globs is None:
        return site
    inner = None
    module_frame = None
    f = frame
    n = 0
    while f is not None and n < 300:
        if f.f_globals is globs:
            if inner is None:
                inner = f
            if f.f_code.co_name == "<module>":
                module_frame = f
                break
        f = f.f_back
        n += 1
    lines = block.code.splitlines() if isinstance(block.code, str) else []
    if module_frame is not None:
        line = module_frame.f_lineno
        site["line"] = line
        if 1 <= line <= len(lines):
            site["source_line"] = lines[line - 1]
    if inner is not None and inner is not module_frame:
        site["function"] = inner.f_code.co_name
        site["function_line"] = inner.f_lineno
        if module_frame is not None and id(inner.f_code) in _module_code_ids(block, module_frame.f_code):
            if 1 <= inner.f_lineno <= len(lines):
                site["function_source_line"] = lines[inner.f_lineno - 1]
        else:
            site["function_defined_in_earlier_block"] = True
    return site


def keyframe_and_state(ep: Any, for_event_id: str) -> tuple[Any, list[Any], Any, int]:
    """Key frame reference(s) and a state snapshot (with its byte size)."""
    frame_ref, frame_refs = (None, [])
    tap = ep.frame_tap
    if tap is not None:
        frame_ref, frame_refs = tap.keyframe(for_event_id)
    state = None
    nbytes = 0
    if ep.config.state and not ep.store.stopped:
        from capx_trace_recorder.state_probe import capture_state

        snapshot, reason = capture_state(ep.low_env)
        if snapshot is not None:
            state = snapshot
            nbytes = snapshot.nbytes
        else:
            state = reason
    return frame_ref, frame_refs, state, nbytes


def state_ref_builder(ctx: Any, state: Any, key: str = "state_ref") -> None:
    """Write a state snapshot; a snapshot that could not be taken is a gap.

    ``unsupported_simulator`` is by design (section 8.4) and is not a gap;
    any other reason (``unexpected_state_shape``, ``state_read_failed``) means
    state the recorder should have had is missing: ``state_unavailable``.
    """
    from capx_trace_recorder.state_probe import StateSnapshot

    if isinstance(state, StateSnapshot):
        art = ctx.json_asset(state.to_json(), label="state")
        ctx.data[key] = {"path": art["path"], "written": art["written"], "sim_time": state.sim_time}
        if not art["written"]:
            ctx.data[key]["reason"] = art.get("reason")
    elif isinstance(state, str):
        ctx.data[key] = {"path": None, "reason": state}
        if state != "unsupported_simulator":
            ctx.warn("state_unavailable", f"ground-truth state not captured: {state}",
                     ctx.event.get("event_id") if ctx.event else None)
    else:
        ctx.data[key] = None


def state_parts(state: Any, key: str = "state_ref") -> tuple[Any, Any, int]:
    """``(write, drop, nbytes)`` for a state snapshot taken on the calling thread.

    ``write(ctx)`` stores it (``state_ref_builder``); ``drop(ctx)`` holds no
    reference to the snapshot and records ``state_ref`` as not written because of
    the queue budget; ``nbytes`` (the copied arrays) is counted in the item's
    queue bytes, so a queue_full drop releases the copy at once.
    """
    from capx_trace_recorder.state_probe import StateSnapshot

    if isinstance(state, StateSnapshot):
        def write(ctx: Any, state: Any = state) -> None:
            state_ref_builder(ctx, state, key)

        def drop(ctx: Any) -> None:
            ctx.data[key] = {"path": None, "written": False, "reason": "queue_full"}

        return write, drop, int(state.nbytes)

    def small(ctx: Any, state: Any = state) -> None:  # a reason string or None: nothing to release
        state_ref_builder(ctx, state, key)

    return small, small, 0


def _chain(*steps: Any) -> Any:
    """One builder step that runs ``steps`` (None entries skipped) in order."""
    todo = [s for s in steps if s is not None]

    def run(ctx: Any) -> None:
        for step in todo:
            step(ctx)

    return run


def _frames_len(ep: Any) -> tuple[int | None, int | None]:
    tap = ep.frame_tap
    if tap is None:
        return None, None
    lengths = tap.buffer_lengths()
    if not lengths:
        return None, None
    epoch, length = next(iter(lengths.values()))
    return epoch, length


# ---------------------------------------------------------------------------
# Before / after
# ---------------------------------------------------------------------------


def _api_before(api: Any, func: Any, name: str, args: tuple, kwargs: dict, caller: Any) -> Any:
    """Emit ``api_call_start`` and return the call's context (the wrapper pushes it)."""
    ep = C.resolve_or_note("H8", getattr(api, "_env", None))
    if ep is None:
        return None
    depth = C.api_depth()
    parent_ctx = C.innermost("api_call") if depth > 0 else C.innermost("code_block")
    parent = parent_ctx.event_id if parent_ctx is not None else None
    block = C.innermost("code_block")
    event_id = new_event_id()
    pending: list[Any] = []
    budget = ArrayBudget(ep.config.array_call_bytes, ep)
    bound = bind_arguments(func, api, args, kwargs)
    arg_summary = summarize_arguments(bound, pending, budget, label="value")
    api_class = type(api).__name__
    rec = C.ApiCallCtx(event_id, ep, name, depth)
    rec.api = api
    rec.stdout_buf, rec.stderr_buf = _tee_buffers()
    rec.stdout0 = _offset(rec.stdout_buf)
    rec.stderr0 = _offset(rec.stderr_buf)
    rec.sim_step0 = ep.clock()[0]
    rec.epoch0, rec.buf_len0 = _frames_len(ep)
    data: dict[str, Any] = {
        "api_call_id": event_id,
        "name": name,
        "qualname": getattr(func, "__qualname__", name),
        "api_class": api_class,
        "depth": depth,
        "privileged": getattr(getattr(ep.env, "cfg", None), "privileged", None),
        "args": arg_summary,
        "units": units_for(api_class, name),
        "call_site": call_site(block, caller) if depth == 0 else None,
        "stdout_offset": rec.stdout0,
        "stderr_offset": rec.stderr0,
        "frame_ref": None,
        "state_ref": None,
    }
    if budget.exceeded:
        data["array_budget_exceeded"] = budget.exceeded
    write_state = drop_state = None
    state_bytes = 0
    if depth == 0:
        frame_ref, frame_refs, state, _nbytes = keyframe_and_state(ep, event_id)
        data["frame_ref"] = frame_ref
        if len(frame_refs) > 1:
            data["frame_refs"] = frame_refs
        write_state, drop_state, state_bytes = state_parts(state)
        del state

    builder, drop_builder, nbytes = pending_builders(pending, write_state, drop_after=drop_state,
                                                     extra_bytes=state_bytes)
    refs = {"api_call_id": event_id}
    if block is not None:
        refs["code_block_id"] = block.event_id
    ep.emit("api_call_start", status="started", data=data, parent=parent, refs=refs,
            builder=builder, drop_builder=drop_builder, event_id=event_id, nbytes=nbytes)
    if budget.exceeded:
        _array_budget_warning(ep, event_id, budget.exceeded)
    return rec


def _array_budget_warning(ep: Any, event_id: str, count: int) -> None:
    ep.emit("trace_warning", status="warning",
            data={"code": "array_budget", "related_event_id": event_id,
                  "message": f"{count} array(s) above CAPX_TRACE_ARRAY_CALL_MB / "
                             "CAPX_TRACE_ARRAY_EPISODE_MB; only shape, dtype and sha256 kept"})


def _finish_common(rec: Any, pending: list[Any]) -> tuple[dict[str, Any], Any, int]:
    ep = rec.episode
    data: dict[str, Any] = {
        "api_call_id": rec.event_id,
        "name": rec.name,
        "depth": rec.depth,
        "duration_s": round(time.monotonic() - rec.start_mono, 6),
        "stdout_slice": _slice(rec.stdout_buf, rec.stdout0, pending),
        "stderr_slice": _slice(rec.stderr_buf, rec.stderr0, pending),
        "frame_ref": None,
        "state_ref": None,
    }
    step1 = ep.clock()[0]
    data["sim_step_delta"] = (step1 - rec.sim_step0) if (
        isinstance(step1, int) and isinstance(rec.sim_step0, int)) else None
    epoch1, len1 = _frames_len(ep)
    if len1 is not None and rec.buf_len0 is not None and epoch1 == rec.epoch0 and len1 > rec.buf_len0:
        data["buffer_frames"] = [rec.buf_len0, len1 - 1]
    else:
        data["buffer_frames"] = None
    state = None
    nbytes = 0
    if rec.depth == 0:
        frame_ref, frame_refs, state, nbytes = keyframe_and_state(ep, rec.event_id)
        data["frame_ref"] = frame_ref
        if len(frame_refs) > 1:
            data["frame_refs"] = frame_refs
    return data, state, nbytes


def _api_after(rec: Any, result: Any) -> None:
    ep = rec.episode
    pending: list[Any] = []
    budget = ArrayBudget(ep.config.array_call_bytes, ep)
    summary = summarize(result, pending, budget, label="value")
    data, state, _nbytes = _finish_common(rec, pending)
    data["return"] = summary
    if budget.exceeded:
        data["array_budget_exceeded"] = budget.exceeded
    write_state, drop_state, state_bytes = state_parts(state) if rec.depth == 0 else (None, None, 0)
    del state
    builder, drop_builder, nbytes = pending_builders(pending, write_state, drop_after=drop_state,
                                                     extra_bytes=state_bytes)
    ep.emit("api_call_end", status="ok", data=data, parent=rec.event_id,
            refs={"api_call_id": rec.event_id}, builder=builder, drop_builder=drop_builder,
            nbytes=nbytes)
    if budget.exceeded:
        _array_budget_warning(ep, rec.event_id, budget.exceeded)


def exception_name(exc: BaseException) -> str:
    name = type(exc).__qualname__
    if type(exc).__module__ not in ("builtins", "__main__"):
        name = f"{type(exc).__module__}.{name}"
    return name


def safe_str(exc: BaseException) -> str:
    """``str(exc)`` (taken on the calling thread: the exception may change later)."""
    try:
        return str(exc)
    except Exception as err:  # noqa: BLE001
        if is_trial_alarm(err):
            raise
        return ""


def fill_error_text(ctx: Any, message: str, pending: list[Any], field: str = "message") -> Any:
    """Writer side: store ``message`` redacted (inline or as a text asset).

    Sets ``<field>_redacted`` only when a rule replaced something in the stored
    text, and ``<field>_cut`` when the message was longer than ``payload.REDACT_WINDOW``.
    Returns the ``payload.Redacted`` result.
    """
    from capx_trace_recorder.payload import redact

    result = redact(message, None)
    ctx.data[field] = text_value(result.text, pending, "text")
    if result.redacted:
        ctx.data[field + "_redacted"] = True
    if result.cut:
        ctx.data[field + "_cut"] = True
    return result


def raw_frames(exc: BaseException) -> list[tuple[str, int, str]] | None:
    """``(filename, lineno, name)`` of the traceback's frames, recorder frames removed.

    Runs on the trial thread and reads only frame attributes: the source lines
    are looked up when the writer formats the traceback (``format_frames``).
    """
    import traceback

    from capx_trace_recorder.tbfilter import is_recorder_frame

    return [(f.f_code.co_filename, lineno, f.f_code.co_name)
            for f, lineno in traceback.walk_tb(exc.__traceback__) if not is_recorder_frame(f)]


def format_frames(frames: list[tuple[str, int, str]]) -> str:
    """Writer side: the ``Traceback (most recent call last):`` block of ``frames``."""
    import traceback

    summary = traceback.StackSummary.from_list([(f, n, name, None) for f, n, name in frames])
    return "Traceback (most recent call last):\n" + "".join(summary.format())


def _api_error(rec: Any, exc: BaseException) -> None:
    ep = rec.episode
    pending: list[Any] = []
    data, state, _nbytes = _finish_common(rec, pending)
    alarm = is_trial_alarm(exc)
    if alarm:
        ep.alarm_seen = True
    message = safe_str(exc)
    exc_name = exception_name(exc)
    data["error_type"] = exc_name
    data["message"] = None  # filled in, redacted, by the writer
    data["alarm_timeout"] = alarm
    try:
        frames = raw_frames(exc)
    except Exception as err:  # noqa: BLE001
        if is_trial_alarm(err):
            raise
        frames = None
    cause = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    cause_type = type(cause).__qualname__ if cause is not None else None
    cause_message = safe_str(cause) if cause is not None else None

    def texts_step(ctx, message=message, frames=frames, exc_name=exc_name, cause_type=cause_type,
                   cause_message=cause_message):
        from capx_trace_recorder.payload import redact

        texts: list[Any] = []
        clean = fill_error_text(ctx, message, texts)
        last_line = f"{exc_name}: {clean.text}" if clean.text else exc_name
        if frames is not None:
            try:
                tb = redact(format_frames(frames), None)
                ctx.data["traceback"] = text_value(tb.text + last_line, texts, "text")
                if tb.redacted or clean.redacted:
                    ctx.data["traceback_redacted"] = True
                if tb.cut:
                    ctx.data["traceback_cut"] = True
            except Exception:  # noqa: BLE001 - timeout-ok: writer thread
                ctx.data["traceback"] = None
        if cause_type is not None:
            ctx.data["cause"] = {"type": cause_type}
            if cause_message is not None:
                from capx_trace_recorder.payload import put_redacted

                put_redacted(ctx.data["cause"], "message", cause_message, 2000)
        ctx.fill_pending(texts)

    write_state, drop_state, state_bytes = state_parts(state) if rec.depth == 0 else (None, None, 0)
    del state
    builder, drop_builder, nbytes = pending_builders(
        pending, _chain(texts_step, write_state), drop_after=_chain(texts_step, drop_state),
        extra_bytes=state_bytes)
    ep.emit("api_call_error", status="error", data=data, parent=rec.event_id,
            refs={"api_call_id": rec.event_id}, builder=builder, drop_builder=drop_builder,
            nbytes=nbytes)


# ---------------------------------------------------------------------------
# Wrapper and install
# ---------------------------------------------------------------------------


def make_api_function(func: Any, name: str) -> Any:
    key = (id(func), name)
    cached = _WRAPPED_FUNCS.get(key)
    if cached is not None and cached.__capx_trace_original__ is func:
        return cached

    @functools.wraps(func)
    def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        caller = sys._getframe(1)
        rec = guard(_api_before, self, func, name, args, kwargs, caller)
        del caller
        if rec is None:
            return func(self, *args, **kwargs)
        add_overhead(rec.episode, t0)
        try:
            C.push_ctx(rec)
            result = func(self, *args, **kwargs)
        except BaseException as exc:
            C.pop_ctx(rec)
            t1 = time.perf_counter()
            guard(_api_error, rec, exc)
            add_overhead(rec.episode, t1)
            raise
        finally:
            C.pop_ctx(rec)  # also when an alarm lands after the call returned
        t1 = time.perf_counter()
        guard(_api_after, rec, result)
        add_overhead(rec.episode, t1)
        return result

    mark(traced, func)
    _WRAPPED_FUNCS[key] = traced
    return traced


def _attribute_name(api: Any, key: str, func: Any) -> str | None:
    cls = type(api)
    for attr in (key, getattr(func, "__name__", None)):
        if not attr:
            continue
        try:
            candidate = inspect.getattr_static(cls, attr)
        except AttributeError:
            continue
        if isinstance(candidate, (staticmethod, classmethod)):
            return None
        if candidate is func:
            return attr
    return None


def install_api(api: Any) -> tuple[bool, str | None, list[str]]:
    """Install H8 on one API instance; ``(ok, reason, names wrapped now)``.

    ``ok`` with a ``reason`` means a partial install: the public functions named
    in ``reason`` (also ``skipped_functions(api)``) are not wrapped, so their
    calls are not recorded; the caller records ``hook_install_failed``. A repeated
    call on the same instance returns the same ``reason``.
    """
    with _INSTALL_LOCK:
        entry = _INSTALLED.get(id(api))
        if entry is not None and entry[0]() is api:
            return True, ("; ".join(entry[1]) if entry[1] else None), []
        for dead in [k for k, (ref, _skipped) in _INSTALLED.items() if ref() is None]:
            del _INSTALLED[dead]
        try:
            before = api.combined_doc()
            functions = api.functions()
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            return False, f"combined_doc()/functions() raised {type(exc).__name__}", []
        installed: list[str] = []
        skipped: list[str] = []
        for key, fn in functions.items():
            if not isinstance(fn, types.MethodType) or fn.__self__ is not api:
                skipped.append(f"{key}: not a bound method of the instance")
                continue
            func = fn.__func__
            if is_wrapper(func):
                continue
            attr = _attribute_name(api, key, func)
            if attr is None:
                skipped.append(f"{key}: attribute name not found")
                continue
            if attr in api.__dict__:
                skipped.append(f"{key}: instance attribute already set")
                continue
            try:
                setattr(api, attr, types.MethodType(make_api_function(func, key), api))
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                skipped.append(f"{key}: cannot set the attribute ({type(exc).__name__})")
                continue
            installed.append(attr)
        try:
            after = api.combined_doc()
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            after = f"<raised {type(exc).__name__}>"
        if after != before:
            for attr in installed:
                try:
                    delattr(api, attr)
                except AttributeError:
                    pass
            return False, "combined_doc() changed after wrapping; wrappers removed", []
        try:
            _INSTALLED[id(api)] = (weakref.ref(api), list(skipped))  # no callback (see _INSTALLED)
        except TypeError:  # not weak-referenceable: remember nothing (install stays idempotent)
            pass
        return True, ("; ".join(skipped) if skipped else None), installed


def skipped_functions(api: Any) -> list[str]:
    """Public functions of ``api`` that ``install_api`` could not wrap."""
    entry = _INSTALLED.get(id(api))
    if entry is None or entry[0]() is not api:
        return []
    return list(entry[1])


def wrapped_attributes(api: Any) -> list[str]:
    """Names of the H8 wrappers installed on ``api`` now, whenever they were installed.

    An env (and its API instances) is reused across the episodes of a worker, so
    the wrappers installed in the first episode are still in place later on.
    """
    try:
        items = list(api.__dict__.items())
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return []
    return [attr for attr, value in items
            if isinstance(value, types.MethodType) and value.__self__ is api and is_wrapper(value.__func__)]


def uninstall_api(api: Any) -> None:
    """Remove the instance attributes again (tests)."""
    with _INSTALL_LOCK:
        for attr, value in list(api.__dict__.items()):
            func = getattr(value, "__func__", None)
            if isinstance(value, types.MethodType) and is_wrapper(func):
                delattr(api, attr)
        _INSTALLED.pop(id(api), None)


__all__ = ["install_api", "uninstall_api", "wrapped_attributes", "skipped_functions",
           "make_api_function", "call_site", "keyframe_and_state", "state_ref_builder", "state_parts",
           "bind_arguments", "exception_name", "safe_str", "fill_error_text", "raw_frames",
           "format_frames"]
