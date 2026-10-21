"""H1: ``requests.Session.send`` (docs/trace-format.md, sections 6.1-6.3).

Every model request leaves through ``Session.send``; there ``PreparedRequest.body``
is exactly what is sent. The wrapper keeps a reference to that immutable body
(``str`` or ``bytes``) and never to the headers. Classification uses the
thread's innermost context:

* a logical model call (H3) -> ``model_request`` / ``model_response`` /
  ``model_error`` for this HTTP attempt;
* a Molmo service call (H9) -> a ``perception_vlm`` request; its logical call
  is one object's retry loop (created here, closed by H14);
* any other service call -> one HTTP attempt of that call;
* anything else is ignored.

For ``stream=True`` only the request, the status and the timing are recorded
here; the body is never touched (``r.content``, ``r.raw``, ``iter_*``). The
streamed text is recorded by the H3 generator wrapper. For non-streamed
responses ``Session.send`` has already read the body, so ``r.content`` is the
cached ``bytes``. Parsing, hashing and file writes happen on the writer thread.

Request and response bodies are text (the model input and output): they are
always parsed and written. Over the writer queue budget only the images and
videos in a request are not written (``written: false, reason: queue_full``).

Related: ``capx_trace_recorder/payload.py``, ``capx_trace_recorder/segments.py``,
``capx_trace_recorder/hooks/llm.py`` (H3), ``capx_trace_recorder/hooks/services.py`` (H9).
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import add_overhead, guard, mark
from capx_trace_recorder.ids import new_event_id, random_hex
from capx_trace_recorder.payload import (
    Media,
    body_to_bytes,
    ext_for_mime,
    fill_placeholder,
    has_nonfinite,
    image_dimensions,
    mask_secrets,
    normalize_request,
    parse_response_body,
    pixel_sha256,
    put_redacted,
    reconstruct,
    sanitize_url,
)
from capx_trace_recorder.segments import compute_input_summary, text_parts
from capx_trace_recorder.serialize import json_safe, text_value


class _SendInfo:
    __slots__ = ("kind", "ctx", "lc", "event_id", "t_send", "stream", "endpoint", "sink")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.ctx: Any = None
        self.lc: Any = None
        self.event_id: str | None = None
        self.t_send = time.monotonic()
        self.stream = False
        self.endpoint: dict[str, Any] = {}
        self.sink: Any = None


def new_logical_call_id() -> str:
    return "lc_" + random_hex(20)


# ---------------------------------------------------------------------------
# Before / after
# ---------------------------------------------------------------------------


def _before(request: Any, kwargs: dict[str, Any]) -> _SendInfo | None:
    ctx = C.innermost()
    if ctx is None:
        return None
    if ctx.kind == "logical_call":
        return _model_request(ctx, request, kwargs)
    if ctx.kind == "service_call":
        if ctx.service == "molmo":
            lc = ctx.molmo_call
            if lc is None or lc.episode is not ctx.episode:
                lc = C.LogicalCall(new_logical_call_id(), ctx.episode, "perception_vlm", "point",
                                   None, None, ctx.event_id, "molmo")
                ctx.molmo_call = lc
                ctx.molmo_calls.append(lc.event_id)
                ep = ctx.episode
                if hasattr(ep, "count"):
                    ep.count("n_logical_calls")
            info = _model_request(lc, request, kwargs)
            info.ctx = ctx
            if ctx.endpoint is None:
                ctx.endpoint = info.endpoint
            return info
        info = _SendInfo("service")
        info.ctx = ctx
        info.endpoint = sanitize_url(getattr(request, "url", None))
        if ctx.endpoint is None:
            ctx.endpoint = info.endpoint
        return info
    return None


def _model_request(lc: Any, request: Any, kwargs: dict[str, Any]) -> _SendInfo:
    ep = lc.episode
    lc.attempts += 1
    if hasattr(ep, "count"):
        ep.count("n_http_attempts")
    info = _SendInfo("model")
    info.lc = lc
    info.sink = ep
    info.stream = bool(kwargs.get("stream"))
    info.endpoint = sanitize_url(getattr(request, "url", None))
    body = getattr(request, "body", None)
    body_type = type(body).__name__
    if not isinstance(body, (str, bytes)):
        body = None
    data = {
        "role": lc.role,
        "call_kind": lc.call_kind,
        "turn_index": lc.turn_index,
        "logical_call_id": lc.event_id,
        "http_attempt": lc.attempts,
        "method": getattr(request, "method", None),
        "endpoint": info.endpoint,
        "stream": info.stream,
        "body_type": body_type,
        "payload_path": None,
    }
    refs = {"logical_call_id": lc.event_id, "http_attempt": lc.attempts}
    if lc.turn_index is not None:
        refs["turn_index"] = lc.turn_index
    event_id = new_event_id()
    info.event_id = event_id
    segments = lc.segments
    builder = functools.partial(_build_request, body=body, segments=segments, event_id=event_id)
    nbytes = len(body) if body is not None else 0
    ep.emit("model_request", status="started", data=data, parent=lc.parent, refs=refs,
            builder=builder, event_id=event_id, nbytes=nbytes, keep_payload=True)
    lc.last_request_id = event_id
    return info


def _after_response(info: _SendInfo, response: Any) -> None:
    status = getattr(response, "status_code", None)
    latency = time.monotonic() - info.t_send
    if info.kind == "service":
        ok = isinstance(status, int) and status < 400
        info.ctx.attempts.append({"status": status, "latency_s": round(latency, 6),
                                  "error": None if ok else f"HTTP {status}",
                                  "endpoint": info.endpoint.get("url")})
        return
    lc = info.lc
    ep = info.sink
    lc.last_status = status
    if info.ctx is not None:  # a Molmo request: also one HTTP attempt of the service call
        ok = isinstance(status, int) and status < 400
        info.ctx.attempts.append({"status": status, "latency_s": round(latency, 6),
                                  "error": None if ok else f"HTTP {status}",
                                  "endpoint": info.endpoint.get("url"),
                                  "model_event_id": info.event_id})
    if info.stream:
        headers = getattr(response, "headers", None)
        content_type = None
        try:
            content_type = headers.get("content-type") if headers is not None else None
        except TimeoutError:  # the trial alarm (section 2)
            raise
        except Exception:  # noqa: BLE001
            content_type = None
        lc.stream_info = {"http_status": status, "send_latency_s": latency,
                          "request_event_id": info.event_id, "content_type": content_type,
                          "t_send": info.t_send}
        return
    content = getattr(response, "_content", None)
    if not isinstance(content, bytes):
        content = None
    refs = {"logical_call_id": lc.event_id, "http_attempt": lc.attempts,
            "model_event_id": info.event_id}
    if lc.turn_index is not None:
        refs["turn_index"] = lc.turn_index
    event_id = new_event_id()
    base = {"role": lc.role, "call_kind": lc.call_kind, "logical_call_id": lc.event_id,
            "http_attempt": lc.attempts, "http_status": status, "latency_s": round(latency, 6)}
    if isinstance(status, int) and status >= 400:
        builder = functools.partial(_build_http_error, content=content)
        base.update({"scope": "attempt", "error_type": "HTTPError", "message": f"HTTP {status}"})
        ep.emit("model_error", status="error", data=base, parent=info.event_id, refs=refs,
                builder=builder, event_id=event_id, nbytes=len(content or b""), keep_payload=True)
        return
    base.update({"streamed": False, "n_chunks": None, "ttft_s": None})
    builder = functools.partial(_build_response, content=content, event_id=event_id)
    ep.emit("model_response", status="ok", data=base, parent=info.event_id, refs=refs,
            builder=builder, event_id=event_id, nbytes=len(content or b""), keep_payload=True)
    lc.last_response_id = event_id


RAW_ERROR = "_raw_error"  # an attempt's error text before redaction (never written; see services.py)


def _after_error(info: _SendInfo, exc: BaseException) -> None:
    """Record a failed HTTP attempt. The exception text is kept raw here and
    redacted by the writer (``payload.put_redacted``): nothing unbounded runs on
    the calling thread."""
    latency = time.monotonic() - info.t_send
    message = _safe_str(exc)
    error_text = f"{type(exc).__name__}: {message}"
    if info.kind == "service":
        info.ctx.attempts.append({"status": None, "latency_s": round(latency, 6), "error": None,
                                  RAW_ERROR: error_text, "endpoint": info.endpoint.get("url")})
        return
    lc = info.lc
    ep = info.sink
    if info.ctx is not None:
        info.ctx.attempts.append({"status": None, "latency_s": round(latency, 6), "error": None,
                                  RAW_ERROR: error_text, "endpoint": info.endpoint.get("url"),
                                  "model_event_id": info.event_id})
    if isinstance(exc, TimeoutError) and hasattr(ep, "alarm_seen"):
        ep.alarm_seen = True
    refs = {"logical_call_id": lc.event_id, "http_attempt": lc.attempts,
            "model_event_id": info.event_id}
    data = {"scope": "attempt", "role": lc.role, "call_kind": lc.call_kind,
            "logical_call_id": lc.event_id, "http_attempt": lc.attempts, "http_status": None,
            "error_type": type(exc).__name__, "message": None,  # filled in, redacted, by the writer
            "latency_s": round(latency, 6)}
    ep.emit("model_error", status="error", data=data, parent=info.event_id, refs=refs,
            builder=functools.partial(fill_message, message=message))


def fill_message(ctx: Any, *, message: str, field: str = "message", limit: int = 2000) -> None:
    """Writer side: ``ctx.data[field]`` = ``message`` redacted and cut to ``limit``."""
    put_redacted(ctx.data, field, message, limit)


def _safe_str(exc: BaseException) -> str:
    try:
        return str(exc)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return type(exc).__name__


# ---------------------------------------------------------------------------
# Writer-side builders
# ---------------------------------------------------------------------------


def _media_source(ctx: Any, media: Media, pix: str | None, used: set[str]) -> dict[str, Any]:
    """Where a sent image came from, by exact hash match only (section 6.2).

    Two harness renders of an unchanged scene have the same pixels, so one hash
    can match several snapshots. The caller assigns sources from the last media
    part to the first; each part takes the latest match that no later part of
    the same request took (a VDM diff sends history[-2], history[-1]), and the
    entry is marked ``ambiguous`` with every candidate listed. ``used`` holds
    the event ids already taken in this request.
    """
    wep = ctx.wep
    if media.kind == "video":
        entry = wep.video_encodes.get(hashlib.sha256(media.data).hexdigest())
        if entry is not None:
            return dict(entry)
        if wep.dropped_kinds.get("video_encode"):
            return {"kind": "unknown", "reason": "queue_full: the video's H12 record was dropped"}
        return {"kind": "unknown"}
    if not pix:
        return {"kind": "unknown"}
    matches = wep.pixel_index.get(pix) or []
    # an observation_snapshot (a harness render) is the spec's source; images seen
    # in API arguments or return values are used only when no render matches
    candidates = [c for c in matches if c.get("kind") == "observation_snapshot"] or matches
    if not candidates:
        return {"kind": "unknown"}
    free = [c for c in candidates if c.get("event_id") not in used]
    chosen = (free or candidates)[-1]
    if chosen.get("event_id"):
        used.add(chosen["event_id"])
    source = dict(chosen)
    if len(candidates) > 1:
        source["ambiguous"] = True
        source["candidates"] = [{k: c.get(k) for k in ("event_id", "kind", "sim_step", "wall_time")}
                                for c in candidates]
    return source


def _build_request(ctx: Any, *, body: Any, segments: tuple, event_id: str) -> None:
    data = ctx.data
    raw = body_to_bytes(body)
    if raw is None:
        data["reconstruction"] = {"verified": False, "reason": "body_not_str_or_bytes"}
        ctx.warn("payload_unparsed", "request body was not str/bytes", event_id)
        return
    body_sha = hashlib.sha256(raw).hexdigest()
    data["body_sha256"] = body_sha
    data["body_bytes"] = len(raw)
    # over the queue budget the text is still parsed and written; only the media
    # assets are not (write_asset leaves written: false, reason: queue_full)
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        data["reconstruction"] = {"verified": False, "reason": "not_json"}
        ctx.warn("payload_unparsed", "request body is not JSON", event_id)
        return
    normalized, media, external = normalize_request(obj)
    images: list[dict[str, Any]] = []
    summary_images: list[dict[str, Any]] = []
    n_images = n_videos = 0
    decoded: list[tuple[Any, Any, Any, str | None]] = []
    for m in media:
        width, height, fmt = image_dimensions(m.data) if m.kind == "image" else (None, None, None)
        pix = pixel_sha256(m.data, width, height, fmt) if m.kind == "image" else None
        decoded.append((width, height, fmt, pix))
    # sources are assigned from the last media part to the first (see _media_source)
    used: set[str] = set()
    sources: list[dict[str, Any]] = [{}] * len(media)
    for i in range(len(media) - 1, -1, -1):
        sources[i] = _media_source(ctx, media[i], decoded[i][3], used)
    for m, (width, height, _fmt, pix), source in zip(media, decoded, sources):
        ext = ext_for_mime(m.mime)
        label = "model_input" if m.kind in ("image", "video") else "payload"
        art = ctx.write_asset(m.data, ext, label=label, kind=m.kind, mime=m.mime, binary=True,
                              extra={"json_path": m.path, "order": m.order})
        sha = hashlib.sha256(m.data).hexdigest()
        fill_placeholder(m, art["path"] if art["written"] else None, sha, art["written"],
                         art.get("reason"))
        entry = {"path": m.path, "order": m.order, "kind": m.kind, "asset": art["path"],
                 "written": art["written"], "sha256": sha, "mime": m.mime, "bytes": len(m.data),
                 "width": width, "height": height, "pixel_sha256": pix, "source": source}
        if not m.canonical:
            entry["noncanonical_base64"] = True
        images.append(entry)
        if m.kind == "video":
            n_videos += 1
        elif m.kind == "image":
            n_images += 1
        summary_images.append({"path": m.path, "order": m.order, "kind": m.kind,
                               "asset": art["path"], "source": source})
    for ext_url in external:
        entry = {"path": ext_url.path, "order": ext_url.order, "kind": "external_url",
                 "url": None, "fetched": False, "source": {"kind": "unknown"}}
        images.append(entry)
        n_images += 1
        summary_images.append({"path": ext_url.path, "order": ext_url.order,
                               "kind": "external_url", "asset": None, "source": {"kind": "unknown"}})
    # reconstruction check on the unmasked normalised copy (section 6.3)
    try:
        rebuilt = reconstruct(normalized, media)
        rebuilt_bytes = rebuilt.encode("iso-8859-1") if isinstance(body, str) else rebuilt.encode("utf-8")
        verified = hashlib.sha256(rebuilt_bytes).hexdigest() == body_sha
        reason = None if verified else "roundtrip_mismatch"
    except Exception as exc:  # noqa: BLE001 - timeout-ok: writer thread
        verified = False
        reason = f"roundtrip_failed: {type(exc).__name__}"
    if any(not m.canonical for m in media) and not verified:
        reason = "noncanonical_base64"
    masked_obj, masked_paths = mask_secrets(normalized, external)
    for entry, ext_url in zip([e for e in images if e["kind"] == "external_url"], external):
        entry["url"] = sanitize_url(ext_url.url).get("url")
    nonfinite = has_nonfinite(masked_obj)
    stored = json_safe(masked_obj) if nonfinite else masked_obj
    if verified:
        reconstruction: dict[str, Any] = (
            {"verified": True, "method": "masked_compare", "masked": masked_paths}
            if masked_paths else {"verified": True, "method": "json_roundtrip"}
        )
    else:
        reconstruction = {"verified": False, "reason": reason}
        ctx.warn("payload_unparsed", f"request body reconstruction failed: {reason}", event_id)
    if nonfinite:
        reconstruction["nonfinite_values_stored_as_strings"] = True
    payload_bytes = json.dumps(stored, ensure_ascii=True, separators=(",", ":"),
                               allow_nan=False).encode("ascii")
    art = ctx.write_file(f"payloads/{event_id}.request.json", payload_bytes, label="payload",
                         kind="json", mime="application/json", binary=False,
                         extra={"normalized_for_storage": True})
    data["payload_path"] = art["path"] if art["written"] else None
    if not art["written"]:
        data["payload_not_written"] = art.get("reason")
    data["reconstruction"] = reconstruction
    data["masked_fields"] = masked_paths
    data["assets_moved"] = [m.path for m in media]
    if isinstance(stored, dict):
        data["model"] = stored.get("model") if isinstance(stored.get("model"), str) else None
        data["params"] = {k: v for k, v in stored.items() if k not in ("messages", "input")}
        messages = stored.get("messages", stored.get("input"))
        data["n_messages"] = len(messages) if isinstance(messages, list) else None
    data["images"] = images
    data["n_images"] = n_images
    data["n_videos"] = n_videos
    data["input_summary"] = compute_input_summary(segments or (), text_parts(stored), summary_images,
                                                  n_videos)


def _build_response(ctx: Any, *, content: bytes | None, event_id: str) -> None:
    data = ctx.data
    if content is None:
        data["content"] = None
        data["payload_path"] = None
        return
    data["body_bytes"] = len(content)
    data["body_sha256"] = hashlib.sha256(content).hexdigest()
    obj, fields = parse_response_body(content)
    if obj is None:
        ctx.warn("payload_unparsed", "response body is not JSON", event_id)
        data["content"] = None
        data["payload_path"] = None
        return
    masked, masked_paths = mask_secrets(obj)
    stored = json_safe(masked) if has_nonfinite(masked) else masked
    art = ctx.write_file(f"payloads/{event_id}.response.json",
                         json.dumps(stored, ensure_ascii=True, separators=(",", ":"),
                                    allow_nan=False).encode("ascii"),
                         label="payload", kind="json", mime="application/json", binary=False)
    data["payload_path"] = art["path"] if art["written"] else None
    data["masked_fields"] = masked_paths
    pending: list[Any] = []
    text = fields.get("content")
    data["content"] = text_value(text, pending, "text") if isinstance(text, str) else text
    reasoning = fields.get("reasoning")
    if isinstance(reasoning, dict):
        data["reasoning"] = {"text": text_value(reasoning["text"], pending, "text"),
                             "field": reasoning["field"]}
    else:
        data["reasoning"] = None
    data["usage"] = json_safe(fields.get("usage"))
    data["finish_reason"] = fields.get("finish_reason")
    data["response_model"] = fields.get("response_model")
    data["response_id"] = fields.get("response_id")
    ctx.fill_pending(pending)


def _build_http_error(ctx: Any, *, content: bytes | None) -> None:
    if content is None:
        ctx.data["body_excerpt"] = None
        return
    from capx_trace_recorder.payload import REDACT_WINDOW

    try:
        text = content[:REDACT_WINDOW].decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        text = ""
    put_redacted(ctx.data, "body_excerpt", text, 2000)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def make_send_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def send(self: Any, request: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        info = guard(_before, request, kwargs)
        if info is not None:
            add_overhead(info.sink or getattr(info.ctx, "episode", None), t0)
        try:
            response = original(self, request, **kwargs)
        except BaseException as exc:
            if info is not None:
                t1 = time.perf_counter()
                guard(_after_error, info, exc)
                add_overhead(info.sink or getattr(info.ctx, "episode", None), t1)
            raise
        if info is not None:
            t1 = time.perf_counter()
            guard(_after_response, info, response)
            add_overhead(info.sink or getattr(info.ctx, "episode", None), t1)
        return response

    return mark(send, original)


def install(sessions_module: Any) -> str | None:
    """Patch ``requests.sessions.Session.send``; returns an error string or None."""
    session_cls = getattr(sessions_module, "Session", None)
    if not isinstance(session_cls, type):
        return "requests.sessions.Session not found"
    original = session_cls.__dict__.get("send")
    if original is None or not callable(original):
        return "Session.send not found"
    if getattr(original, "__capx_trace_wrapper__", False):
        return None
    import inspect

    try:
        params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = []
    if params[:2] != ["self", "request"]:
        return f"Session.send has an unexpected signature {params}"
    session_cls.send = make_send_wrapper(original)
    return None


__all__ = ["install", "make_send_wrapper", "new_logical_call_id", "fill_message", "RAW_ERROR"]
