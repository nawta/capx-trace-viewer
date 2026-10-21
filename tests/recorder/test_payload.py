"""Payload normalisation, byte-exact reconstruction and masking (docs/trace-format.md, section 6).

Bodies are built by the real ``capx.llm.client`` and ``capx.integrations.vision.molmo``
code; ``requests.Session.send`` is replaced for the duration of each test by a
function that keeps the ``PreparedRequest`` body and returns a canned response,
so nothing is sent. The writer-side builder of H1 then normalises the body and
the test rebuilds the exact bytes from the stored payload and assets on its own.
"""

# All key-shaped strings in this file are synthetic test values.

from __future__ import annotations

import base64
import functools
import gzip
import hashlib
import io
import json

import numpy as np
import pytest
import requests

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.hooks.http import _build_request, _build_response
from capx_trace_recorder.payload import mask_secrets, normalize_request, redact_text, sanitize_url
from capx_trace_recorder.segments import Segment, compute_input_summary, text_parts
from trace_reader import read_events


class CaptureSend:
    """Replacement for ``requests.Session.send`` that records bodies."""

    def __init__(self, reply: dict | None = None, stream_lines: list[bytes] | None = None) -> None:
        self.bodies: list = []
        self.reply = reply or {"choices": [{"message": {"content": "ok"}}]}
        self.stream_lines = stream_lines

    def __call__(self, request, **kwargs):  # stored on the class: not bound
        self.bodies.append(request.body)
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        if kwargs.get("stream"):
            response.headers["content-type"] = "text/event-stream"
            response.raw = io.BytesIO(b"\n".join(self.stream_lines or [b"data: [DONE]"]) + b"\n")
        else:
            response.headers["content-type"] = "application/json"
            response._content = json.dumps(self.reply).encode()
        return response


def png_data_url(seed: int) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TRICKY_TEXT = "日本語のタスク 🤖 line sep\x01ctrl \"quotes\" \\ back"


def harness_prompt(video_url: str | None = None) -> list[dict]:
    content = [
        {"type": "text", "text": "Task text\nAPIs:\nget_value(name: str) -> float\n" + TRICKY_TEXT},
        {"type": "image_url", "image_url": {"url": png_data_url(1)}},
        {"type": "text", "text": "Current state:"},
        {"type": "image_url", "image_url": {"url": png_data_url(2)}},
    ]
    if video_url:
        content.append({"type": "image_url", "image_url": {"url": video_url}})
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}]


@pytest.fixture()
def recorder(work_dir):
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces"))
    ep = C.open_episode(store, run_id="payload-run", episode_id="ep", source="test")
    ep.emit("episode_start", data={"tracer": {}})
    yield store, ep
    C.close_episode(ep)
    store.shutdown(5)


def record_request(store, ep, body, segments=()):
    from capx_trace_recorder.ids import new_event_id

    eid = new_event_id()
    ep.emit("model_request", data={"payload_path": None},
            builder=functools.partial(_build_request, body=body, segments=segments, event_id=eid),
            event_id=eid, nbytes=len(body))
    assert store.flush(20)
    events, corrupt, torn = read_events(ep.events_path)
    assert corrupt == 0 and not torn
    return [e for e in events if e["event_id"] == eid][0]


def rebuild_from_storage(ep_dir, event) -> bytes:
    """Independent reconstruction: placeholders -> data URLs -> json.dumps."""
    stored = json.loads((ep_dir / event["data"]["payload_path"]).read_text())

    def walk(value):
        if isinstance(value, dict):
            if "$asset" in value:
                raw = (ep_dir / value["$asset"]).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == value["sha256"]
                prefix = value.get("prefix", f"data:{value['mime']};base64,")
                return prefix + base64.b64encode(raw).decode()
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return json.dumps(walk(stored)).encode()


@pytest.mark.parametrize("model", [
    "openai/gpt-5.4",                      # GPT: reasoning_effort + max_completion_tokens
    "anthropic/claude-opus-4-5",           # Claude: thinking.budget_tokens
    "deepseek/deepseek-v3.2",              # OSS
    "openrouter/openai/gpt-4.1",           # an upstream OpenRouter model name
    "fake/local-model",                    # default branch (local vLLM)
])
@pytest.mark.requires_capx
def test_query_model_bodies_reconstruct_byte_exact(recorder, monkeypatch, model):
    from capx.llm import client
    from capx.utils.video_utils import _encode_video_base64

    store, ep = recorder
    capture = CaptureSend()
    monkeypatch.setattr(requests.Session, "send", capture)
    video = _encode_video_base64([np.full((16, 16, 3), i * 40, np.uint8) for i in range(4)])
    args = client.ModelQueryArgs(model=model, server_url="http://127.0.0.1:9/chat/completions",
                                 temperature=1e-07, max_tokens=17)
    client.query_model(args, harness_prompt(video))
    body = capture.bodies[-1]
    assert isinstance(body, str)
    event = record_request(store, ep, body)
    data = event["data"]
    assert data["reconstruction"] == {"verified": True, "method": "json_roundtrip"}, data
    assert data["masked_fields"] == []
    assert data["n_images"] == 2 and data["n_videos"] == 1
    assert [i["order"] for i in data["images"]] == [0, 1, 2]
    assert data["images"][0]["width"] == 32 and data["images"][0]["height"] == 24
    assert data["images"][0]["pixel_sha256"]
    raw_body = body.encode("ascii")
    assert rebuild_from_storage(ep.base_dir, event) == raw_body
    assert data["body_sha256"] == hashlib.sha256(raw_body).hexdigest()
    params = data["params"]
    for key in ("max_tokens", "max_completion_tokens"):
        if key in params:
            assert params[key] == 17  # *_tokens are never masked
    if "thinking" in params:
        assert params["thinking"]["budget_tokens"] == 4096


@pytest.mark.requires_capx
def test_streaming_body_reconstructs(recorder, monkeypatch):
    from capx.llm import client

    store, ep = recorder
    chunks = [b'data: {"choices": [{"delta": {"content": "hi"}}]}', b"data: [DONE]"]
    capture = CaptureSend(stream_lines=chunks)
    monkeypatch.setattr(requests.Session, "send", capture)
    args = client.ModelQueryArgs(model="fake/local-model", server_url="http://127.0.0.1:9/x")
    out = list(client.query_model_streaming(args, harness_prompt()))
    assert out[-1]["content"] == "hi"
    event = record_request(store, ep, capture.bodies[-1])
    assert event["data"]["reconstruction"]["verified"] is True
    assert event["data"]["params"]["stream"] is True
    assert rebuild_from_storage(ep.base_dir, event) == capture.bodies[-1].encode()


@pytest.mark.requires_capx
def test_molmo_bytes_body_reconstructs(recorder, monkeypatch):
    from PIL import Image

    from capx.integrations.vision import molmo

    store, ep = recorder
    capture = CaptureSend(reply={"choices": [{"message": {"content": '<points coords="1 1 10 20">x</points>'}}]})
    monkeypatch.setattr(requests.Session, "send", capture)
    det = molmo.init_molmo(base_url="http://127.0.0.1:9/v1")
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        result = det(Image.fromarray(np.zeros((20, 30, 3), np.uint8)), ["red cube"])
    assert result["red cube"] == (0, 0)
    body = capture.bodies[-1]
    assert isinstance(body, bytes)  # json= bodies are bytes
    event = record_request(store, ep, body)
    assert event["data"]["reconstruction"] == {"verified": True, "method": "json_roundtrip"}
    assert rebuild_from_storage(ep.base_dir, event) == body


def test_masking_by_exact_key_and_external_urls(recorder):
    store, ep = recorder
    obj = {
        "model": "m", "api_key": "sk-SECRET-1", "max_tokens": 5, "budget_tokens": 7,  # gitleaks:allow
        "max_completion_tokens": 9,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png?sig=SECRET-2"}},
        ]}],
        "extra": {"Authorization": "Bearer SECRET-3", "token_count": 3, "id_token": "SECRET-4",
                  "password": "SECRET-5", "tokens": 2},
    }
    body = json.dumps(obj)
    event = record_request(store, ep, body)
    data = event["data"]
    rec = data["reconstruction"]
    assert rec["verified"] is True and rec["method"] == "masked_compare"
    assert sorted(data["masked_fields"]) == sorted([
        "api_key", "extra.Authorization", "extra.id_token", "extra.password",
        "messages[0].content[1].image_url.url (query removed)",
    ])
    stored_text = (ep.base_dir / data["payload_path"]).read_text()
    for secret in ("SECRET-1", "SECRET-2", "SECRET-3", "SECRET-4", "SECRET-5"):
        assert secret not in stored_text
    stored = json.loads(stored_text)
    assert stored["max_tokens"] == 5 and stored["budget_tokens"] == 7
    assert stored["max_completion_tokens"] == 9 and stored["extra"]["token_count"] == 3
    ext = [i for i in data["images"] if i["kind"] == "external_url"][0]
    assert ext["fetched"] is False and ext["url"] == "https://example.com/a.png"


def test_unparsable_body_is_flagged(recorder):
    store, ep = recorder
    event = record_request(store, ep, "not json at all")
    assert event["data"]["reconstruction"] == {"verified": False, "reason": "not_json"}
    assert event["data"]["payload_path"] is None


def test_response_fields_and_explicit_reasoning(recorder):
    store, ep = recorder
    body = json.dumps({"id": "r1", "model": "m", "usage": {"total_tokens": 3},
                       "choices": [{"finish_reason": "stop", "message": {
                           "content": "x" * 5000, "reasoning_content": "why"}}]}).encode()
    from capx_trace_recorder.ids import new_event_id

    eid = new_event_id()
    ep.emit("model_response", data={}, event_id=eid,
            builder=functools.partial(_build_response, content=body, event_id=eid))
    assert store.flush(20)
    event = [e for e in read_events(ep.events_path)[0] if e["event_id"] == eid][0]
    data = event["data"]
    assert data["reasoning"] == {"text": "why", "field": "choices[0].message.reasoning_content"}
    assert data["usage"] == {"total_tokens": 3} and data["finish_reason"] == "stop"
    assert data["content"]["truncated"] is True and data["content"]["length"] == 5000
    text = gzip.decompress((ep.base_dir / data["content"]["asset"]).read_bytes()).decode()
    assert text == "x" * 5000


def test_redaction_and_url_sanitising(monkeypatch):
    text = '{"error": "bad", "api_key": "sk-abcdefghijklmnopqrstuv", "max_tokens": 5} Bearer abc.def'  # gitleaks:allow
    out = redact_text(text)
    assert "sk-abc" not in out and "abc.def" not in out and '"max_tokens": 5' in out
    info = sanitize_url("http://user:pw@127.0.0.1:8110/chat/completions?key=1#frag")
    assert info == {"url": "http://127.0.0.1:8110/chat/completions", "host": "127.0.0.1", "port": 8110,
                    "path": "/chat/completions", "known_proxy": True}
    monkeypatch.setenv("CAPX_TRACE_PROXY_PORTS", "")
    assert sanitize_url("http://127.0.0.1:8110/chat/completions")["known_proxy"] is False
    monkeypatch.setenv("CAPX_TRACE_PROXY_PORTS", "9000, invalid")
    assert sanitize_url("http://localhost:9000/chat/completions")["known_proxy"] is True
    assert sanitize_url("http://example.test:9000/chat/completions")["known_proxy"] is False


@pytest.mark.parametrize("bad", ["²", "9" * 5000, "abc"])
def test_invalid_proxy_port_keeps_endpoint_and_reports_problem(tmp_path, bad):
    config = TraceConfig.from_env({"CAPX_TRACE_DIR": str(tmp_path),
                                   "CAPX_TRACE_PROXY_PORTS": f"8110,{bad}"})
    assert config is not None
    assert config.proxy_ports == frozenset({8110})
    assert any(bad in problem for problem in config.problems)
    assert config.public()["proxy_ports"] == [8110]


@pytest.mark.parametrize("bad", ["²", "9" * 5000, "abc"])
def test_invalid_proxy_port_does_not_erase_endpoint(monkeypatch, bad):
    monkeypatch.setenv("CAPX_TRACE_PROXY_PORTS", f"8110,{bad}")
    info = sanitize_url("http://localhost:8110/chat/completions")
    assert info["url"] == "http://localhost:8110/chat/completions"
    assert info["host"] == "localhost" and info["port"] == 8110
    assert info["known_proxy"] is True


def test_input_summary_positions():
    text = "TASK TEXT HERE\nAPIs:\nDOCS FOR THE API\nstdout was: printed value 42\n"
    body = {"messages": [
        {"role": "system", "content": "You are a helpful assistant that writes code."},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]}
    segments = (
        Segment("task_prompt", "TASK TEXT HERE"),
        Segment("api_docs", "DOCS FOR THE API"),
        Segment("harness_template", "You are a helpful assistant that writes code."),
        Segment("stdout", "printed value 42", ("ev_stdout",)),
        Segment("vdm_description", "a description that was not sent", ("ev_vdm",)),
        Segment("code_history", "short"),  # under 8 characters: ignored
    )
    summary = compute_input_summary(segments, text_parts(body), [], 0)
    cats = summary["categories"]
    assert summary["no_image_input"] is True
    assert cats["task_api_description"][0] == {"kind": "task_prompt", "path": "messages[1].content[0].text",
                                               "start": 0, "end": 14, "source_event_ids": []}
    assert cats["stdout_stderr"][0]["start"] == text.index("printed value 42")
    assert cats["stdout_stderr"][0]["source_event_ids"] == ["ev_stdout"]
    assert cats["other"][0]["path"] == "messages[0].content"
    unattributed = [(u["start"], u["end"]) for u in summary["unattributed"]]
    assert (14, 21) in unattributed  # "\nAPIs:\n"
    assert summary["registered_not_found"] == [
        {"kind": "vdm_description", "length": 31, "source_event_ids": ["ev_vdm"]}]


def test_snapshot_attributes_the_apis_separator():
    """The env's "\\nAPIs:\\n" separator is a known harness literal; lone newlines are flagged."""
    from capx_trace_recorder.segments import SegmentRegistry

    class Env:
        def __init__(self) -> None:
            self._task_prompt = "Stack the red cube on the green cube."
            self._system_prompt = "You are a helpful assistant that writes code."
            self._full_prompt = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": [{"type": "text",
                                              "text": self._task_prompt + "\nAPIs:\nget_value(name) -> float"}]},
            ]

    env = Env()
    text = env._full_prompt[1]["content"][0]["text"] + "\n"
    body = {"messages": [env._full_prompt[0], {"role": "user", "content": [{"type": "text", "text": text}]}]}
    summary = compute_input_summary(SegmentRegistry().snapshot(env, None), text_parts(body), [], 0)
    sep = [o for o in summary["categories"]["other"] if o.get("origin", "").endswith("_get_complete_prompt")]
    assert [(o["start"], o["end"]) for o in sep] == [(37, 44)]
    assert summary["unattributed"] == [{"path": "messages[1].content[0].text", "start": len(text) - 1,
                                        "end": len(text), "whitespace_only": True}]


def test_normalize_keeps_non_media_data_urls():
    obj = {"a": "data:text/plain,hello", "b": "data:image/png;base64,!!notbase64!!"}
    normalized, media, external = normalize_request(obj)
    assert normalized == obj and media == [] and external == []
    masked, paths = mask_secrets({"Token": "x", "tokens": 1, "nested": [{"secret": 1}]})
    assert masked == {"Token": "***masked***", "tokens": 1, "nested": [{"secret": "***masked***"}]}
    assert paths == ["Token", "nested[0].secret"]
