"""H5 and H6: prompt builders and parsers in ``capx.utils.launch_utils``.

Implements docs/trace-format.md sections 5.2 (``feedback_built``,
``model_output_parsed``) and 6.4. H5 wraps ``_build_multi_turn_decision_prompt``
and ``_build_multi_turn_decision_prompt_legacy``: after the original returns it
records what was assembled for the next model input (the formatted multi-turn
text, the image, the VDM text and web user feedback) with the registered
segments each part contains. H6 wraps ``_parse_multi_turn_decision`` and
``_extract_code`` and records the harness's reading of the reply; "model says
finished" is only the literal ``FINISH`` keyword, never the parser's default.

Both are rebound in the defining module right after it executes, so the aliases
bound by ``capx.envs.trial`` and ``capx.web.async_trial_runner`` are the wrappers.

Related: ``capx_trace_recorder/segments.py``, ``capx_trace_recorder/hooks/llm.py``.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import inspect
import json
import re
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import guard, mark, patch_attribute
from capx_trace_recorder.ids import new_event_id
from capx_trace_recorder.payload import image_dimensions, pixel_sha256
from capx_trace_recorder.serialize import text_value

_DATA_URL = re.compile(r"^data:([^;,]*)((?:;[^;,]*)*?);base64,")
USER_FEEDBACK_MARKER = "\n\nUser feedback: "  # capx/web/async_trial_runner.py appends this


def _link_response(ep: Any, content: Any) -> tuple[str | None, bool]:
    """``(model_event_id, linked)`` of the latest code-generation response."""
    rid, rcontent = ep.last_codegen_response
    if rid is None:
        return None, False
    if content is rcontent:
        return rid, True
    if isinstance(content, str) and isinstance(rcontent, str):
        if content == rcontent or (content and content in rcontent):
            return rid, True
    return None, False


# ---------------------------------------------------------------------------
# H6: parsers
# ---------------------------------------------------------------------------


def _after_parse(parser: str, content: Any, result: Any) -> None:
    ep = C.resolve_or_note("H6")
    if ep is None:
        return
    model_event_id, linked = _link_response(ep, content)
    content_empty = content is None or (isinstance(content, str) and not content.strip())
    pending: list[Any] = []
    data: dict[str, Any] = {
        "parser": parser,
        "model_event_id": model_event_id,
        "content_empty": content_empty,
        "code_blocks": None,
        "decision": None,
        "model_said_finish": None,
    }
    if parser == "_extract_code":
        blocks = result if isinstance(result, list) else []
        data["code_blocks"] = [text_value(b, pending, "text") if isinstance(b, str) else None
                               for b in blocks]
        if linked:
            data["code_source"] = "model"
        elif ep.config_flags.get("use_oracle_code"):
            data["code_source"] = "oracle"
        else:
            data["code_source"] = "unknown"
    else:
        decision = result[0] if isinstance(result, tuple) and result else None
        new_code = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        data["decision"] = decision
        data["harness_reading"] = (
            "REGENERATE found -> regenerate" if decision == "regenerate"
            else "no REGENERATE -> finish"
        )
        data["model_said_finish"] = bool(isinstance(content, str) and "FINISH" in content)
        data["regenerate_text"] = (text_value(new_code, pending, "text")
                                   if decision == "regenerate" and isinstance(new_code, str) else None)
        data["code_source"] = "model" if linked else "unknown"
    builder = None
    if pending:
        def builder(ctx, pending=pending):
            ctx.fill_pending(pending)
    refs = {"model_event_id": model_event_id} if model_event_id else {}
    ep.emit("model_output_parsed", data=data, parent=model_event_id, refs=refs, builder=builder)


def make_parse_wrapper(original: Any, parser: str) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        content = args[0] if args else kwargs.get("content")
        guard(_after_parse, parser, content, result)
        return result

    return mark(wrapper, original)


# ---------------------------------------------------------------------------
# H5: decision prompt builders
# ---------------------------------------------------------------------------


def _vdm_event_for(ep: Any, text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    for event_id, content in reversed(ep.vdm_responses):
        if content is text:
            return event_id
    for event_id, content in reversed(ep.vdm_responses):
        if content == text:
            return event_id
    return None


def _after_build(builder_name: str, sig: inspect.Signature, args: tuple, kwargs: dict,
                 result: Any) -> None:
    ep = C.resolve_or_note("H5")
    if ep is None:
        return
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
    except TypeError:
        return
    prompt_text = values.get("complete_multi_turn_prompt")
    visual = values.get("visual_feedback")
    vdm_text = values.get("visual_differencing_feedback")
    is_video = bool(values.get("is_video_feedback", False))
    user_feedback = None
    if ep.pending_user_feedback and isinstance(prompt_text, str):
        for text in list(ep.pending_user_feedback):
            if prompt_text.endswith(USER_FEEDBACK_MARKER + text):
                user_feedback = text
                ep.pending_user_feedback.remove(text)
                break
    role = C.current_role(ep)
    # batch: built inside _handle_multi_turn_step (role context set, turn counted);
    # web UI: built before the decision call starts, so it is for the next turn
    turn_index = role.turn_index if role is not None else ep.turn_index + 1
    dynamic = ep.segments.dynamic()
    event_id = new_event_id()
    data = {
        "builder": builder_name,
        "turn_index": turn_index,
        "vdm_model_event_id": _vdm_event_for(ep, vdm_text),
        "user_feedback": user_feedback,
        "is_video_feedback": is_video,
        "has_visual_feedback": visual is not None,
        "image_pixel_sha256": None,
        "n_messages": len(result) if isinstance(result, list) else None,
    }
    texts = {"complete_multi_turn_prompt": prompt_text if isinstance(prompt_text, str) else None,
             "visual_differencing_feedback": vdm_text if isinstance(vdm_text, str) else None,
             "user_feedback": user_feedback}
    visual_url = visual if isinstance(visual, str) else None

    def builder(ctx, texts=texts, dynamic=dynamic, visual_url=visual_url, event_id=event_id):
        _feedback_texts(ctx, texts, dynamic, event_id)
        if visual_url:
            match = _DATA_URL.match(visual_url)
            if match:
                try:
                    raw = base64.b64decode(visual_url[match.end():], validate=True)
                    w, h, fmt = image_dimensions(raw)
                    ctx.data["image_pixel_sha256"] = pixel_sha256(raw, w, h, fmt)
                except Exception:  # noqa: BLE001 - timeout-ok: writer thread
                    ctx.data["image_pixel_sha256"] = None

    def drop_builder(ctx, texts=texts, dynamic=dynamic, event_id=event_id,
                     had_image=bool(visual_url)):
        # holds no image: the queue budget drop releases it (counted as a gap)
        _feedback_texts(ctx, texts, dynamic, event_id)
        if had_image:
            ctx.data["image_pixel_sha256_reason"] = "queue_full"

    nbytes = len(visual_url) if visual_url else 0
    ep.emit("feedback_built", data=data, builder=builder, drop_builder=drop_builder,
            event_id=event_id, nbytes=nbytes)
    if user_feedback is not None:
        ep.segments.register("user_feedback", user_feedback, (event_id,), "web_ui")


def _feedback_texts(ctx: Any, texts: dict[str, Any], dynamic: Any, event_id: str) -> None:
    """Writer side of ``feedback_built``: which segments the texts contain, and the texts file."""
    segments = []
    prompt = texts["complete_multi_turn_prompt"]
    if prompt is not None:
        contains = []
        for seg in dynamic:
            if seg.text and seg.text in prompt:
                contains.append({"kind": seg.kind,
                                 "text_sha256": hashlib.sha256(seg.text.encode("utf-8", "surrogatepass")).hexdigest(),
                                 "length": len(seg.text),
                                 "source_event_ids": list(seg.source_event_ids)})
        segments.append({"kind": "multi_turn_prompt",
                         "text_sha256": hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest(),
                         "length": len(prompt), "source_event_ids": [], "contains": contains})
    vdm = texts["visual_differencing_feedback"]
    if vdm is not None:
        segments.append({"kind": "vdm_description",
                         "text_sha256": hashlib.sha256(vdm.encode("utf-8", "surrogatepass")).hexdigest(),
                         "length": len(vdm),
                         "source_event_ids": [ctx.data.get("vdm_model_event_id")]
                         if ctx.data.get("vdm_model_event_id") else []})
    user = texts["user_feedback"]
    if user is not None:
        segments.append({"kind": "user_feedback",
                         "text_sha256": hashlib.sha256(user.encode("utf-8", "surrogatepass")).hexdigest(),
                         "length": len(user), "source_event_ids": []})
    ctx.data["segments"] = segments
    art = ctx.write_file(f"payloads/{event_id}.feedback.json",
                         json.dumps(texts, ensure_ascii=True).encode("ascii"), label="text",
                         kind="json", mime="application/json", binary=False)
    ctx.data["texts_path"] = art["path"] if art["written"] else None


def make_build_wrapper(original: Any, builder_name: str) -> Any:
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        guard(_after_build, builder_name, sig, args, kwargs, result)
        return result

    return mark(wrapper, original)


def install(launch_utils: Any) -> list[str]:
    problems = []
    for name in ("_build_multi_turn_decision_prompt", "_build_multi_turn_decision_prompt_legacy"):
        err = patch_attribute(launch_utils, name, lambda orig, n=name: make_build_wrapper(orig, n))
        if err:
            problems.append(err)
    for name in ("_parse_multi_turn_decision", "_extract_code"):
        err = patch_attribute(launch_utils, name, lambda orig, n=name: make_parse_wrapper(orig, n))
        if err:
            problems.append(err)
    from capx_trace_recorder.hooks.common import rebind_aliases

    rebind_aliases(launch_utils, ("_query_model", "_query_model_streaming", "_query_model_ensemble",
                                  "_query_single_model_ensemble"))
    return problems


__all__ = ["install", "make_build_wrapper", "make_parse_wrapper", "USER_FEEDBACK_MARKER"]
