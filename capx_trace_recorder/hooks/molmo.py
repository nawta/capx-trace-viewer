"""H14: ``capx.integrations.vision.molmo._parse_points``.

Records the harness's reading of Molmo's reply (``model_output_parsed`` with
``parser = "molmo._parse_points"``), linked to the last ``perception_vlm``
response of the current object's logical call, and closes that logical call so
the next object's request starts a new one (docs/trace-format.md, sections 5.2
and 7.1). ``_parse_points`` is looked up as a module global at call time inside
``det_fn``, so rebinding it in the defining module is enough.

Related: ``capx_trace_recorder/hooks/http.py`` (creates the Molmo logical calls),
``capx_trace_recorder/hooks/services.py`` (the enclosing service call).
"""

from __future__ import annotations

import functools
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import guard, mark, patch_attribute
from capx_trace_recorder.serialize import json_safe, text_value


def _after_parse_points(text: Any, result: Any) -> None:
    ctx = C.innermost("service_call")
    if ctx is None or ctx.service != "molmo":
        return
    ep = ctx.episode
    lc = ctx.molmo_call
    ctx.molmo_call = None
    pending: list[Any] = []
    points = norm_scale = None
    if isinstance(result, tuple) and len(result) == 2:
        points, norm_scale = result
    data = {
        "parser": "molmo._parse_points",
        "model_event_id": lc.last_response_id if lc is not None else None,
        "logical_call_id": lc.event_id if lc is not None else None,
        "content_empty": not (isinstance(text, str) and text.strip()),
        "parsed_text": text_value(text, pending, "text") if isinstance(text, str) else None,
        "points": json_safe(points) if isinstance(points, list) else None,
        "norm_scale": json_safe(norm_scale),
        "code_blocks": None,
        "decision": None,
        "model_said_finish": None,
    }
    builder = None
    if pending:
        def builder(c, pending=pending):
            c.fill_pending(pending)
    refs = {"logical_call_id": lc.event_id} if lc is not None else {}
    ep.emit("model_output_parsed", data=data, parent=data["model_event_id"] or ctx.event_id,
            refs=refs, builder=builder)


def make_parse_points_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _parse_points(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        text = args[0] if args else kwargs.get("text")
        guard(_after_parse_points, text, result)
        return result

    return mark(_parse_points, original)


def install(molmo_module: Any) -> list[str]:
    err = patch_attribute(molmo_module, "_parse_points", make_parse_points_wrapper)
    return [err] if err else []


__all__ = ["install"]
