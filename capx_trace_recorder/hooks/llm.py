"""H3: logical model calls in ``capx.llm.client`` (docs/trace-format.md, sections 6.4 and 7.3).

``query_model`` and ``query_model_streaming`` are rebound in their defining
module right after it executes, so every later ``from capx.llm.client import
query_model as _query_model`` (``launch_utils``, ``trial``, the web runner)
binds the wrapper. ``client.py`` also binds its own underscore aliases at the
end of the module; those are rebound too. The wrappers here are the only place
a ``logical_call_id`` is created for code-generation and VDM calls.

Role context comes from the thread's role stack (H2, H13), else from the
episode's ``current_role`` (ensemble pool threads), else ``preflight`` in the
experiment runner before any episode, else ``unknown``. Segments are
registered right before each request from the env's state at that moment.

``query_model_streaming`` is wrapped as a generator (PEP 380 delegation with a
per-chunk callback); the context is set in the iterating thread for the whole
iteration, ``model_response`` comes from the ``done`` chunk and ``model_error``
carries the partial text on an exception or ``GeneratorExit``.

Related: ``capx_trace_recorder/hooks/http.py`` (H1 attempts), ``capx_trace_recorder/hooks/trial.py``
(H2 roles), ``capx_trace_recorder/hooks/web.py`` (H13).
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import add_overhead, guard, mark, patch_attribute
from capx_trace_recorder.hooks.http import new_logical_call_id
from capx_trace_recorder.ids import new_event_id
from capx_trace_recorder.segments import find_task_text
from capx_trace_recorder.serialize import text_value


# ---------------------------------------------------------------------------
# Logical calls
# ---------------------------------------------------------------------------


def start_logical(args: Any, prompt: Any, origin: str) -> Any:
    """Create a logical call (the wrapper pushes it); None when nothing is recorded.

    Without an episode the call goes to the run's unscoped log; with several
    candidate episodes each of them also gets an ``ambiguous_episode`` gap.
    """
    ep, candidates = C.resolve_episode()
    sink: Any = ep
    if ep is None:
        sink = C.unscoped_sink_for(candidates)
        if sink is None:
            return None
        if len(candidates) > 1:
            C.note_ambiguous(f"H3 {origin}", candidates)
    role_ctx = C.current_role(ep)
    if role_ctx is not None:
        role, call_kind, turn_index = role_ctx.role, role_ctx.call_kind, role_ctx.turn_index
    elif ep is None and not candidates and C.REGISTRY.runner.run_id and not C.REGISTRY.h4_failed():
        # the experiment runner before any episode; without H4 trial requests
        # would look the same, so they are then "unknown", not "preflight"
        role, call_kind, turn_index = "preflight", None, None
    else:
        role, call_kind, turn_index = "unknown", None, None
    lc = C.LogicalCall(new_logical_call_id(), sink, role, call_kind, turn_index,
                       getattr(args, "model", None), None, origin)
    if ep is not None:
        ep.count("n_logical_calls")
        if role == "code_generator":
            ep.count("codegen_requests")
            _register_task_text(ep, lc, prompt)
        lc.segments = ep.segments.snapshot(ep.env, ep.multi_turn_prompt)
    return lc


def _register_task_text(ep: Any, lc: Any, prompt: Any) -> None:
    text, reason = find_task_text(prompt)
    if text is not None:
        ep.segments.register("task_prompt", text, (), "sent_task_text")
    with ep._lock:
        if ep.task_text_emitted:
            return
        ep.task_text_emitted = True
    pending: list[Any] = []
    data = {
        "text": text_value(text, pending, "text") if text is not None else None,
        "reason": reason,
        "logical_call_id": lc.event_id,
        "source": "text before \"\\nAPIs:\\n\" in the last user message of the first "
                  "code-generation request",
    }
    builder = None
    if pending:
        def builder(ctx, pending=pending):
            ctx.fill_pending(pending)
    ep.emit("task_text", data=data, refs={"logical_call_id": lc.event_id}, builder=builder)


def end_logical_ok(lc: Any, result: Any) -> None:
    ep = lc.episode
    if not hasattr(ep, "segments"):
        return
    content = result.get("content") if isinstance(result, dict) else result if isinstance(result, str) else None
    if lc.role == "vdm" and isinstance(content, str):
        ids = (lc.last_response_id,) if lc.last_response_id else ()
        ep.segments.register("vdm_description", content, ids, lc.call_kind or "vdm")
        ep.vdm_responses.append((lc.last_response_id, content))
    elif lc.role == "code_generator" and lc.call_kind != "ensemble_member":
        ep.last_codegen_response = (lc.last_response_id, content)


def end_logical_error(lc: Any, exc: BaseException) -> None:
    ep = lc.episode
    if isinstance(exc, TimeoutError) and hasattr(ep, "alarm_seen"):
        ep.alarm_seen = True
    try:
        message = str(exc)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        message = type(exc).__name__
    from capx_trace_recorder.hooks.http import fill_message

    data = {"scope": "logical", "role": lc.role, "call_kind": lc.call_kind,
            "logical_call_id": lc.event_id, "http_attempt": lc.attempts,
            "http_status": lc.last_status, "error_type": type(exc).__name__,
            "message": None}  # filled in, redacted, by the writer
    refs = {"logical_call_id": lc.event_id}
    if lc.turn_index is not None:
        refs["turn_index"] = lc.turn_index
    ep.emit("model_error", status="error", data=data, parent=lc.last_request_id, refs=refs,
            builder=functools.partial(fill_message, message=message))


def make_query_model_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def query_model(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        a = args[0] if args else kwargs.get("args")
        p = args[1] if len(args) > 1 else kwargs.get("prompt")
        lc = guard(start_logical, a, p, "query_model")
        if lc is None:
            return original(*args, **kwargs)
        add_overhead(lc.episode, t0)
        try:
            C.push_ctx(lc)
            result = original(*args, **kwargs)
        except BaseException as exc:
            C.pop_ctx(lc)
            t1 = time.perf_counter()
            guard(end_logical_error, lc, exc)
            add_overhead(lc.episode, t1)
            raise
        finally:
            C.pop_ctx(lc)  # also when an alarm lands after the call returned
        t1 = time.perf_counter()
        guard(end_logical_ok, lc, result)
        add_overhead(lc.episode, t1)
        return result

    return mark(query_model, original)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamState:
    def __init__(self, lc: Any) -> None:
        self.lc = lc
        self.n_chunks = 0
        self.first_chunk: float | None = None
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.done: dict[str, Any] | None = None
        self.t_start = time.monotonic()

    def on_chunk(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        ctype = chunk.get("type")
        if ctype in ("content_delta", "reasoning_delta"):
            self.n_chunks += 1
            if self.first_chunk is None:
                self.first_chunk = time.monotonic()
            text = chunk.get("content")
            if isinstance(text, str):
                (self.content_parts if ctype == "content_delta" else self.reasoning_parts).append(text)
        elif ctype == "done":
            self.done = chunk

    def _timing(self) -> tuple[float | None, float | None, Any]:
        info = self.lc.stream_info or {}
        t_send = info.get("t_send", self.t_start)
        latency = time.monotonic() - t_send
        ttft = (self.first_chunk - t_send) if self.first_chunk is not None else None
        return latency, ttft, info

    def finished(self) -> None:
        lc = self.lc
        ep = lc.episode
        latency, ttft, info = self._timing()
        done = self.done or {}
        content = done.get("content") if self.done is not None else "".join(self.content_parts)
        reasoning = done.get("reasoning") if self.done is not None else None
        content_type = (info.get("content_type") or "").lower()
        field = "choices[0].message.reasoning" if (
            "application/json" in content_type and "text/event-stream" not in content_type
        ) else "delta.reasoning"
        event_id = new_event_id()
        pending: list[Any] = []
        data = {
            "role": lc.role, "call_kind": lc.call_kind, "logical_call_id": lc.event_id,
            "http_attempt": lc.attempts, "http_status": info.get("http_status"),
            "latency_s": round(latency, 6) if latency is not None else None,
            "streamed": True, "n_chunks": self.n_chunks,
            "ttft_s": round(ttft, 6) if ttft is not None else None,
            "content": text_value(content, pending, "text") if isinstance(content, str) else content,
            "reasoning": ({"text": text_value(reasoning, pending, "text"), "field": field}
                          if isinstance(reasoning, str) and reasoning else None),
            "usage": None, "finish_reason": None, "response_model": None, "response_id": None,
            "done_chunk_seen": self.done is not None,
        }
        assembled = {"streamed": True, "content": content,
                     "reasoning": reasoning if isinstance(reasoning, str) else None,
                     "n_chunks": self.n_chunks}

        def builder(ctx, pending=pending, assembled=assembled, event_id=event_id):
            art = ctx.write_file(f"payloads/{event_id}.response.json",
                                 json.dumps(assembled, ensure_ascii=True).encode("ascii"),
                                 label="payload", kind="json", mime="application/json", binary=False,
                                 extra={"assembled_from_stream": True})
            ctx.data["payload_path"] = art["path"] if art["written"] else None
            ctx.fill_pending(pending)

        refs = {"logical_call_id": lc.event_id, "http_attempt": lc.attempts,
                "model_event_id": lc.last_request_id}
        if lc.turn_index is not None:
            refs["turn_index"] = lc.turn_index
        ep.emit("model_response", status="ok", data=data, parent=lc.last_request_id, refs=refs,
                builder=builder, event_id=event_id)
        lc.last_response_id = event_id
        end_logical_ok(lc, {"content": content})

    def failed(self, exc: BaseException, error_type: str | None = None) -> None:
        lc = self.lc
        ep = lc.episode
        latency, ttft, info = self._timing()
        partial = "".join(self.content_parts)
        pending: list[Any] = []
        try:
            message = str(exc)
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            message = type(exc).__name__
        from capx_trace_recorder.hooks.http import fill_message

        data = {"scope": "logical", "role": lc.role, "call_kind": lc.call_kind,
                "logical_call_id": lc.event_id, "http_attempt": lc.attempts,
                "http_status": info.get("http_status"),
                "error_type": error_type or type(exc).__name__,
                "message": None,  # filled in, redacted, by the writer
                "streamed": True,
                "n_chunks": self.n_chunks, "partial_content": text_value(partial, pending, "text"),
                "latency_s": round(latency, 6) if latency is not None else None}

        def builder(ctx, pending=pending, message=message):
            fill_message(ctx, message=message)
            ctx.fill_pending(pending)
        if isinstance(exc, TimeoutError) and hasattr(ep, "alarm_seen"):
            ep.alarm_seen = True
        ep.emit("model_error", status="error", data=data, parent=lc.last_request_id,
                refs={"logical_call_id": lc.event_id}, builder=builder)


def delegate(gen: Any, on_chunk: Any):
    """``yield from gen`` (PEP 380) with ``on_chunk(item)`` for every item."""
    try:
        item = next(gen)
    except StopIteration as stop:
        return stop.value
    while True:
        guard(on_chunk, item)
        try:
            sent = yield item
        except GeneratorExit:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
            raise
        except BaseException as exc:  # noqa: BLE001 - timeout-ok: forwarded into the generator (PEP 380), which decides
            throw = getattr(gen, "throw", None)
            if throw is None:
                raise
            try:
                item = throw(exc)
            except StopIteration as stop:
                return stop.value
        else:
            try:
                item = next(gen) if sent is None else gen.send(sent)
            except StopIteration as stop:
                return stop.value


def make_streaming_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def query_model_streaming(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        a = args[0] if args else kwargs.get("args")
        p = args[1] if len(args) > 1 else kwargs.get("prompt")
        lc = guard(start_logical, a, p, "query_model_streaming")
        if lc is None:
            return (yield from original(*args, **kwargs))
        state = StreamState(lc)
        add_overhead(lc.episode, t0)
        try:
            C.push_ctx(lc)
            result = yield from delegate(original(*args, **kwargs), state.on_chunk)
        except GeneratorExit as exc:
            C.pop_ctx(lc)
            guard(state.failed, exc, "GeneratorExit")
            raise
        except BaseException as exc:
            C.pop_ctx(lc)
            guard(state.failed, exc)
            raise
        finally:
            C.pop_ctx(lc)  # also when an alarm lands after the stream ended
        guard(state.finished)
        return result

    return mark(query_model_streaming, original)


# ---------------------------------------------------------------------------
# Ensembles (members run on a ThreadPoolExecutor)
# ---------------------------------------------------------------------------


class _EnsembleState:
    __slots__ = ("ep", "member", "synth", "previous_current", "pushed")

    def __init__(self, ep: Any, member: Any, synth: Any) -> None:
        self.ep = ep
        self.member = member
        self.synth = synth
        self.previous_current: Any = None
        self.pushed = False

    def push(self) -> None:
        """Pool threads read ``ep.current_role``; this thread gets the synthesis role.

        ``pushed`` is set first: ``_ensemble_end`` undoes any part that ran.
        """
        self.previous_current = self.ep.current_role
        self.pushed = True
        self.ep.current_role = self.member
        C.push_role(self.synth)


def _ensemble_start() -> _EnsembleState | None:
    ep, _ = C.resolve_episode()
    if ep is None:
        return None
    caller = C.current_role(ep)
    role = caller.role if caller is not None else "code_generator"
    turn = caller.turn_index if caller is not None else None
    return _EnsembleState(ep, C.RoleCtx(role, "ensemble_member", turn, ep),
                          C.RoleCtx(role, "ensemble_synthesis", turn, ep))


def _ensemble_end(token: _EnsembleState | None) -> None:
    if token is None:
        return
    C.pop_role(token.synth)
    if token.pushed and token.ep.current_role is token.member:
        token.ep.current_role = token.previous_current


def make_ensemble_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def ensemble(*args: Any, **kwargs: Any) -> Any:
        token = guard(_ensemble_start)
        try:
            if token is not None:
                token.push()
            result = original(*args, **kwargs)
        finally:
            guard(_ensemble_end, token)  # on every way out, an alarm included
        return result

    return mark(ensemble, original)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

_ALIASES = {
    "_query_model": "query_model",
    "_query_model_streaming": "query_model_streaming",
    "_query_model_ensemble": "query_model_ensemble",
    "_query_single_model_ensemble": "query_single_model_ensemble",
}


def install(client_module: Any) -> list[str]:
    """Patch ``capx.llm.client``; returns a list of problems (empty = ok)."""
    problems = []
    for name, factory in (
        ("query_model", make_query_model_wrapper),
        ("query_model_streaming", make_streaming_wrapper),
        ("query_model_ensemble", make_ensemble_wrapper),
        ("query_single_model_ensemble", make_ensemble_wrapper),
    ):
        err = patch_attribute(client_module, name, factory)
        if err:
            problems.append(err)
    import inspect

    stream = getattr(client_module, "query_model_streaming", None)
    original_stream = getattr(stream, "__capx_trace_original__", None)
    if original_stream is not None and not inspect.isgeneratorfunction(original_stream):
        problems.append("query_model_streaming is not a generator function")
    for alias, target in _ALIASES.items():
        if alias in vars(client_module):
            setattr(client_module, alias, getattr(client_module, target))
    return problems


__all__ = ["install", "make_query_model_wrapper", "make_streaming_wrapper", "make_ensemble_wrapper",
           "start_logical", "end_logical_ok", "end_logical_error", "delegate"]
