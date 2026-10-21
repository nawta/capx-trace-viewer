"""H1 on ``requests.Session.send`` against a loopback server (docs/trace-format.md, section 6.1).

The wrapper is installed only for the duration of each test (``monkeypatch``)
and a logical model call is opened by hand, as the H3 wrapper would.
"""

from __future__ import annotations

import json

import pytest
import requests

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.hooks.http import make_send_wrapper
from trace_reader import by_type, read_events
import tracing_http_fakes as F


@pytest.fixture()
def traced_send(monkeypatch, work_dir):
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces"))
    ep = C.open_episode(store, run_id="http-run", episode_id="ep", source="test")
    ep.emit("episode_start", data={"tracer": {}})
    original = requests.Session.send
    monkeypatch.setattr(requests.Session, "send", make_send_wrapper(original))
    lc = C.LogicalCall("lc_test", ep, "code_generator", "initial_code", 0, "fake/model", None, "test")
    yield store, ep, lc
    C.pop_ctx(lc)
    C.close_episode(ep)
    store.shutdown(5)


def test_body_recorded_headers_not_stored(traced_send):
    store, ep, lc = traced_send
    llm = F.ScriptedLLM()
    with F.LoopbackServer(llm) as server:
        C.push_ctx(lc)
        body = json.dumps({"model": "fake/model", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Reply with the single word OK."}]}]})
        r = requests.post(server.url + "/chat/completions", data=body,
                          headers={"Authorization": "Bearer sk-HEADER-SECRET-123",
                                   "Content-Type": "application/json"}, timeout=10)
        C.pop_ctx(lc)
    assert r.json()["choices"][0]["message"]["content"] == "OK"
    assert store.flush(20)
    events, _, _ = read_events(ep.events_path)
    req = by_type(events, "model_request")[0]["data"]
    resp = by_type(events, "model_response")[0]["data"]
    assert req["http_attempt"] == 1 and req["logical_call_id"] == "lc_test"
    assert req["reconstruction"] == {"verified": True, "method": "json_roundtrip"}
    stored = json.loads((ep.base_dir / req["payload_path"]).read_text())
    assert stored == json.loads(body)
    assert resp["content"] == "OK" and resp["http_status"] == 200
    for path in ep.base_dir.rglob("*"):
        if path.is_file():
            assert b"sk-HEADER-SECRET" not in path.read_bytes()
            assert b"Authorization" not in path.read_bytes()


def test_exceptions_pass_through_unchanged(traced_send):
    store, ep, lc = traced_send
    marker = requests.exceptions.ConnectionError("injected failure")

    class FailingAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):
            raise marker

    session = requests.Session()
    session.mount("http://", FailingAdapter())
    C.push_ctx(lc)
    with pytest.raises(requests.exceptions.ConnectionError) as info:
        session.post("http://127.0.0.1:9/chat/completions", data="{}", timeout=1)
    C.pop_ctx(lc)
    assert info.value is marker
    assert store.flush(20)
    events, _, _ = read_events(ep.events_path)
    error = by_type(events, "model_error")[0]["data"]
    assert error["scope"] == "attempt" and error["error_type"] == "ConnectionError"
    assert error["message"] == "injected failure"


def test_streamed_body_is_not_touched(traced_send):
    store, ep, lc = traced_send
    llm = F.ScriptedLLM()
    with F.LoopbackServer(llm) as server:
        C.push_ctx(lc)
        body = json.dumps({"model": "m", "stream": True, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Reply with the single word OK."}]}]})
        with requests.post(server.url + "/chat/completions", data=body, stream=True, timeout=10) as r:
            assert r._content is False  # nothing read yet
            lines = [line for line in r.iter_lines() if line]
        C.pop_ctx(lc)
    assert lines[-1] == b"data: [DONE]" and len(lines) == 4
    assert lc.stream_info["http_status"] == 200
    assert "text/event-stream" in lc.stream_info["content_type"]
    assert store.flush(20)
    events, _, _ = read_events(ep.events_path)
    assert by_type(events, "model_request")[0]["data"]["stream"] is True
    assert by_type(events, "model_response") == []  # recorded by the H3 generator wrapper instead


def test_requests_outside_a_context_are_ignored(traced_send):
    store, ep, _lc = traced_send
    llm = F.ScriptedLLM()
    with F.LoopbackServer(llm) as server:
        requests.post(server.url + "/ik", json={"target": [0]}, timeout=10)
    assert store.flush(20)
    events, _, _ = read_events(ep.events_path)
    assert [e["event_type"] for e in events] == ["episode_start"]
