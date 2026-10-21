"""Perception outputs above the array budget are reported (docs/trace-format.md, sections 4.4 and 5.3).

SAM3 may return up to 200 masks of 512x512 per call,
and the per-call budget that was the default then (CAPX_TRACE_ARRAY_CALL_MB = 8)
held 32 of them. The default is now 64 MB, which holds 200. Masks beyond the
budget keep only shape, dtype and sha256; the service call must then carry
``array_budget_exceeded`` and an ``array_budget`` warning, so the episode is not
shown as complete.
"""

from __future__ import annotations

import json

import numpy as np

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.hooks.services import make_service_wrapper
from trace_reader import by_type, read_events


def _masks(n: int) -> list[dict]:
    out = []
    for i in range(n):
        mask = np.zeros((64, 64), dtype=bool)
        mask[i % 64, :] = True
        out.append({"mask": mask, "score": 0.5, "label": "red cube"})
    return out


def _run(work_dir, n_masks: int, call_mb: float) -> tuple[dict, list[dict]]:
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces", array_call_mb=call_mb))
    ep = C.open_episode(store, run_id="svc-run", episode_id="ep", source="test")
    ep.emit("episode_start", data={"tracer": {}})
    C.push_episode(ep)
    try:
        seg = make_service_wrapper(lambda image, text: _masks(n_masks), "sam3", "sam3_seg_fn")
        result = seg(np.zeros((4, 4, 3), dtype=np.uint8), "red cube")
        assert len(result) == n_masks
    finally:
        C.pop_episode(ep)
    ep.emit("episode_end", data={"end_reason": "completed"})
    ep.ended = True
    assert store.flush(20)
    events, _corrupt, _torn = read_events(ep.events_path)
    doc = json.loads((ep.base_dir / "episode.json").read_text())
    C.close_episode(ep)
    store.shutdown(5)
    return doc, events


def test_masks_beyond_the_call_budget_are_reported(work_dir):
    # 64x64 bool masks are 4 KiB each; a 0.01 MiB budget holds two of them
    doc, events = _run(work_dir, 5, 0.01)
    end = by_type(events, "service_call_end")[0]["data"]
    items = end["response_summary"]["items"]
    saved = [i["items"]["mask"] for i in items if "npz" in i["items"]["mask"]]
    skipped = [i["items"]["mask"] for i in items if i["items"]["mask"].get("not_saved") == "array_budget"]
    assert len(saved) == 2 and len(skipped) == 3 and all(m["sha256"] for m in skipped)
    assert end["array_budget_exceeded"] == 3
    warning = by_type(events, "trace_warning")[0]["data"]
    assert warning["code"] == "array_budget"
    assert doc["trace_complete"] is False and [g["code"] for g in doc["gaps"]] == ["array_budget"]


def test_outputs_within_the_budget_leave_the_trace_complete(work_dir):
    doc, events = _run(work_dir, 3, 1.0)
    end = by_type(events, "service_call_end")[0]["data"]
    assert "array_budget_exceeded" not in end
    assert by_type(events, "trace_warning") == []
    assert doc["trace_complete"] is True
