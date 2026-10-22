"""H9: perception and motion service closures on API instances.

Implements docs/trace-format.md sections 5.2 (``service_call_*``) and 7.1. The
closures (``sam3_seg_fn``, ``molmo_point_fn``, ``grasp_net_plan_fn``,
``ik_solve_fn``, ...) are instance attributes read through ``self.<name>`` at
call time, so replacing the instance attribute is enough. The image each
service received is stored as PNG (``perception_input``); array outputs (masks
bit-packed with ``np.packbits``, boxes, scores, grasp poses, IK solutions) are
copied when the closure returns and stored together as one ``npz``
(``perception_output``). HTTP attempts inside the call are collected by H1; a
call whose attempts failed but which still returned (SAM3 and Molmo swallow
transport errors) is ``degraded``. A Molmo call's HTTP requests are
``perception_vlm`` model requests (H1), one logical call per object (H14).
Arrays beyond ``CAPX_TRACE_ARRAY_CALL_MB`` keep only shape, dtype and a bounded
hash (``capx_trace_recorder/serialize.py``); the event then carries
``array_budget_exceeded`` and an ``array_budget`` warning follows (SAM3 returns
up to 200 masks of 512x512, 256 KiB each as bool). The error message of a
failed call and the error of each failed HTTP attempt are stored redacted by
the writer (``payload.redact``: URL userinfo and query, secret values; with
``message_redacted`` / ``http_attempts[].error_redacted`` when a rule replaced
something in the stored text); the exception re-raised to the API is untouched. A closure
that cannot be replaced is reported by ``install_services`` and recorded as
``hook_install_failed``.

Related: ``capx_trace_recorder/hooks/http.py``, ``capx_trace_recorder/hooks/molmo.py``,
``capx_trace_recorder/hooks/api.py`` (installs H9 with H8).
"""

from __future__ import annotations

import functools
import time
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.api import _array_budget_warning
from capx_trace_recorder.hooks.common import add_overhead, guard, is_wrapper, mark_only
from capx_trace_recorder.ids import new_event_id
from capx_trace_recorder.serialize import (
    ArrayBudget,
    PendingArray,
    pending_binary_bytes,
    pending_builders,
    split_pending,
    summarize,
    summarize_arguments,
)
from capx_trace_recorder.timeout import is_trial_alarm

SERVICE_ATTRS = {
    "sam3_seg_fn": "sam3",
    "sam3_point_prompt_fn": "sam3",
    "sam2_seg_fn": "sam2",
    "sam2_point_prompt_fn": "sam2",
    "owl_vit_det_fn": "owlvit",
    "molmo_point_fn": "molmo",
    "grasp_net_plan_fn": "graspnet",
    "grasp_net_plan_point_clouds_fn": "graspnet",
    "ik_solve_fn": "pyroki",
    "trajopt_plan_fn": "pyroki",
}


def _service_name(attr: str, fn: Any) -> str:
    service = SERVICE_ATTRS[attr]
    module = getattr(fn, "__module__", "") or ""
    if service == "pyroki" and "curobo" in module:
        return "curobo"
    return service


def _bind(fn: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    import inspect

    try:
        return dict(inspect.signature(fn).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        out = {f"arg{i}": v for i, v in enumerate(args)}
        out.update(kwargs)
        return out


def _service_before(service: str, function: str, fn: Any, args: tuple, kwargs: dict) -> Any:
    """Emit ``service_call_start`` and return the call's context (the wrapper pushes it)."""
    parent_ctx = C.innermost()
    if parent_ctx is not None:
        ep = parent_ctx.episode
    else:
        ep = C.resolve_or_note("H9")
    if ep is None or not hasattr(ep, "frame_tap"):
        return None
    api_ctx = C.innermost("api_call")
    parent = api_ctx.event_id if api_ctx is not None else None
    event_id = new_event_id()
    pending: list[Any] = []
    budget = ArrayBudget(ep.config.array_call_bytes, ep)
    request = summarize_arguments(_bind(fn, args, kwargs), pending, budget,
                                  label="perception_input", image_args=True)
    data = {"service": service, "function": function, "request_summary": request, "endpoint": None}
    if budget.exceeded:
        data["array_budget_exceeded"] = budget.exceeded

    builder, drop_builder, nbytes = pending_builders(pending)
    refs = {"api_call_id": parent} if parent else {}
    ep.emit("service_call_start", status="started", data=data, parent=parent, refs=refs,
            builder=builder, drop_builder=drop_builder, event_id=event_id, nbytes=nbytes)
    if budget.exceeded:
        _array_budget_warning(ep, event_id, budget.exceeded)
    return C.ServiceCallCtx(event_id, ep, service, function)


def _packed_outputs(pending: list[Any]) -> tuple[list[Any], list[tuple[str, Any, dict]]]:
    """Split arrays out of ``pending`` so they go into one npz."""
    others = []
    arrays = []
    for item in pending:
        if isinstance(item, PendingArray):
            key = f"a{len(arrays)}"
            arrays.append((key, item.array, item.node))
        else:
            others.append(item)
    return others, arrays


def _npz_builder(ctx: Any, arrays: list[tuple[str, Any, dict]]) -> None:
    if not arrays:
        return
    import numpy as np

    from capx_trace_recorder.serialize import array_stats

    payload = {}
    for key, arr, node in arrays:
        stats = array_stats(arr)
        if stats is not None:
            node["stats"] = stats
        if arr.dtype == np.bool_:
            payload[key + "__packbits"] = np.packbits(arr.reshape(-1))
            payload[key + "__shape"] = np.asarray(arr.shape, dtype=np.int64)
            node["npz_key"] = key + "__packbits"
            node["packbits"] = True
            node["is_mask"] = True
        else:
            payload[key] = arr
            node["npz_key"] = key
    art = ctx.npz_asset(payload, label="perception_output")
    for _key, _arr, node in arrays:
        node["npz"] = art["path"]
        node["written"] = art["written"]
        if not art["written"]:
            node["reason"] = art.get("reason")


def _service_end(rec: Any, result: Any, exc: BaseException | None) -> None:
    ep = rec.episode
    if rec.molmo_call is not None:
        rec.molmo_call = None
    from capx_trace_recorder.hooks.http import RAW_ERROR

    failed = [a for a in rec.attempts if a.get("error") or a.get(RAW_ERROR)]
    # raw error texts stay out of the event until the writer has redacted them
    attempts = [{k: v for k, v in a.items() if k != RAW_ERROR} for a in rec.attempts]
    raw_errors = [(i, a[RAW_ERROR]) for i, a in enumerate(rec.attempts) if a.get(RAW_ERROR) is not None]
    data: dict[str, Any] = {
        "service": rec.service,
        "function": rec.function,
        "endpoint": rec.endpoint,
        "latency_s": round(time.monotonic() - rec.start_mono, 6),
        "http_attempts": attempts,
    }
    if rec.molmo_calls:
        data["model_logical_call_ids"] = list(rec.molmo_calls)
    pending: list[Any] = []
    arrays: list[tuple[str, Any, dict]] = []
    budget_exceeded = 0
    message = None
    if exc is None:
        budget = ArrayBudget(ep.config.array_call_bytes, ep)
        data["response_summary"] = summarize(result, pending, budget, label="perception_output")
        budget_exceeded = budget.exceeded
        if budget_exceeded:
            # e.g. SAM3 returns up to 200 masks of 512x512 (256 KiB each as bool)
            data["array_budget_exceeded"] = budget_exceeded
        pending, arrays = _packed_outputs(pending)
        status = "degraded" if failed else "ok"
        if failed:
            data["degraded_reason"] = "an HTTP attempt failed but the closure returned"
        etype = "service_call_end"
    else:
        from capx_trace_recorder.hooks.api import safe_str

        message = safe_str(exc)
        data["error_type"] = type(exc).__name__
        data["message"] = None  # filled in, redacted, by the writer
        status = "error"
        etype = "service_call_error"
        if is_trial_alarm(exc) and hasattr(ep, "alarm_seen"):
            ep.alarm_seen = True

    def after(ctx, message=message, raw_errors=raw_errors):
        from capx_trace_recorder.payload import put_redacted

        for i, text in raw_errors:
            put_redacted(ctx.data["http_attempts"][i], "error", text, 300)
        if message is not None:
            from capx_trace_recorder.hooks.api import fill_error_text

            texts: list[Any] = []
            fill_error_text(ctx, message, texts)
            ctx.fill_pending(texts)

    kept, dropped = split_pending(pending)
    dropped += [node for _k, _a, node in arrays]

    def builder(ctx, pending=pending, arrays=arrays):
        ctx.fill_pending(pending)
        _npz_builder(ctx, arrays)
        after(ctx)

    def drop_builder(ctx, kept=kept, dropped=dropped):
        # holds no array: the queue budget drop releases the outputs (counted as a gap)
        ctx.fill_pending(kept)
        ctx.fill_dropped(dropped)
        after(ctx)

    nbytes = pending_binary_bytes(pending) + sum(int(a.nbytes) for _k, a, _n in arrays)
    end_id = ep.emit(etype, status=status, data=data, parent=rec.event_id, refs={}, builder=builder,
                     drop_builder=drop_builder, nbytes=nbytes)
    if budget_exceeded:
        _array_budget_warning(ep, end_id, budget_exceeded)


def make_service_wrapper(fn: Any, service: str, function: str) -> Any:
    @functools.wraps(fn)
    def service_call(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        rec = guard(_service_before, service, function, fn, args, kwargs)
        if rec is None:
            return fn(*args, **kwargs)
        add_overhead(rec.episode, t0)
        try:
            C.push_ctx(rec)
            result = fn(*args, **kwargs)
        except BaseException as exc:
            C.pop_ctx(rec)
            t1 = time.perf_counter()
            guard(_service_end, rec, None, exc)
            add_overhead(rec.episode, t1)
            raise
        finally:
            C.pop_ctx(rec)  # also when an alarm lands after the closure returned
        t1 = time.perf_counter()
        guard(_service_end, rec, result, None)
        add_overhead(rec.episode, t1)
        return result

    # flagged, but not kept in the process-wide wrapper table: the closures belong
    # to API instances that come and go with each env (a web server runs many)
    return mark_only(service_call, fn)


def install_services(api: Any) -> tuple[list[str], list[str]]:
    """Wrap the service closures that exist on ``api``.

    Returns ``(wrapped names, failures)``: a closure whose attribute cannot be
    replaced is a failure (its calls would not be recorded).
    """
    wrapped: list[str] = []
    failed: list[str] = []
    try:
        attrs = api.__dict__
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return wrapped, failed
    for attr in SERVICE_ATTRS:
        fn = attrs.get(attr)
        if fn is None or not callable(fn) or is_wrapper(fn):
            continue
        try:
            setattr(api, attr, make_service_wrapper(fn, _service_name(attr, fn), attr))
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            failed.append(f"{attr}: cannot set the attribute ({type(exc).__name__})")
            continue
        if not is_wrapper(attrs.get(attr)):
            failed.append(f"{attr}: the attribute did not keep the wrapper")
            continue
        wrapped.append(attr)
    return wrapped, failed


def wrapped_services(api: Any) -> list[str]:
    """Service closures of ``api`` that carry an H9 wrapper now (new or from an earlier episode)."""
    try:
        attrs = api.__dict__
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return []
    return [attr for attr in SERVICE_ATTRS if is_wrapper(attrs.get(attr))]


__all__ = ["install_services", "wrapped_services", "make_service_wrapper", "SERVICE_ATTRS"]
