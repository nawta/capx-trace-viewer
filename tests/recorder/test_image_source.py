"""Model-input image sources when several renders have the same pixels (docs/trace-format.md, section 6.2).

A code block that does not move anything leaves
the scene unchanged, so the VDM's "previous" and "current" images are the same
pixels, rendered at two different times. Each sent image must still link to
its own ``observation_snapshot`` (exact hash match, assigned from the last
media part to the first) and be marked ``ambiguous`` with every candidate.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks import env as env_hooks
from test_payload import recorder, record_request  # noqa: F401 - pytest fixture
from trace_reader import by_type, read_events


def _png_url(frame: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _request(*urls: str) -> str:
    content = []
    for i, url in enumerate(urls):
        content += [{"type": "text", "text": f"Image {i}:"}, {"type": "image_url", "image_url": {"url": url}}]
    return json.dumps({"model": "m", "messages": [{"role": "user", "content": content}]})


class _Env:
    """Stands in for a CodeExecutionEnvBase: ``_snapshot`` only reads ``low_level_env``."""

    low_level_env = None


@pytest.fixture()
def bound(recorder):  # noqa: F811 - the fixture imported above
    store, ep = recorder
    C.push_episode(ep)
    try:
        yield store, ep
    finally:
        C.pop_episode(ep)


def test_identical_renders_link_to_distinct_snapshots(bound):
    store, ep = bound
    scene = np.full((24, 32, 3), 40, dtype=np.uint8)
    env_hooks._snapshot(_Env(), scene.copy(), "main", "before")
    env_hooks._snapshot(_Env(), scene.copy(), "main", "after")
    url = _png_url(scene)
    diff = record_request(store, ep, _request(url, url))
    single = record_request(store, ep, _request(url))
    events, _corrupt, _torn = read_events(ep.events_path)
    snaps = by_type(events, "observation_snapshot")
    assert len(snaps) == 2 and snaps[0]["data"]["pixel_sha256"] == snaps[1]["data"]["pixel_sha256"]
    before, after = sorted(diff["data"]["images"], key=lambda i: i["order"])
    assert before["source"]["event_id"] == snaps[0]["event_id"]
    assert after["source"]["event_id"] == snaps[1]["event_id"]
    for image in (before, after):
        assert image["source"]["kind"] == "observation_snapshot"
        assert image["source"]["ambiguous"] is True
        assert [c["event_id"] for c in image["source"]["candidates"]] == [s["event_id"] for s in snaps]
    # a request with one such image takes the latest render
    assert single["data"]["images"][0]["source"]["event_id"] == snaps[1]["event_id"]


def test_distinct_renders_are_not_ambiguous(bound):
    store, ep = bound
    first = np.full((24, 32, 3), 10, dtype=np.uint8)
    second = np.full((24, 32, 3), 200, dtype=np.uint8)
    env_hooks._snapshot(_Env(), first, "main", "before")
    env_hooks._snapshot(_Env(), second, "main", "after")
    event = record_request(store, ep, _request(_png_url(first), _png_url(second)))
    events, _corrupt, _torn = read_events(ep.events_path)
    snaps = by_type(events, "observation_snapshot")
    images = sorted(event["data"]["images"], key=lambda i: i["order"])
    assert [i["source"]["event_id"] for i in images] == [s["event_id"] for s in snaps]
    assert not any(i["source"].get("ambiguous") for i in images)
