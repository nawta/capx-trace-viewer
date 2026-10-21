"""No secret reaches a trace file (docs/trace-format.md, sections 2 and 6.3).

Exception messages, causes and tracebacks are stored for API calls (H8),
service calls (H9), HTTP attempts (H1), model calls (H3), code blocks (H7) and
the episode (H4); hook failure reasons, recorder errors and the runner's error
texts (H16) are stored too. Requests exceptions carry the URL with userinfo and
query (``api_key=...``); generated code or an API can put a key into a message.
Every stored text goes through ``payload.redact`` (the whole text first, then
the cut to the stored length) and gets ``<field>_redacted`` only when a rule
replaced something in the stored text; the exception re-raised to the caller is
never changed. A request field under a secret key name is masked whole, also
when it holds a ``data:`` URL (it is never written as an asset).

``redact`` replaces what its four rules cover (URL userinfo and queries,
Authorization credentials, known key formats, values of secret key names; the
comment block "Redaction of stored texts" in ``capx_trace_recorder/payload.py`` states
them) and nothing else: real API, class and function names of ``capx``,
traceback lines, paths with trial ids, digests and code stay as they are. Its
work is linear in the text and bounded. ``SCOPE`` below pins the rules case by
case, directly and through the real writer.

Each test greps every file under the trace root (``.gz`` files decompressed).
"""

# All key-shaped strings in this file are synthetic test values.

from __future__ import annotations

import base64
import functools
import json
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks import http as HTTP
from capx_trace_recorder.hooks import services as SVC
from capx_trace_recorder.hooks import trial as TRIAL
from capx_trace_recorder.hooks.api import install_api, uninstall_api
from capx_trace_recorder.payload import redact_text
from inprocess import Recorder, by_type
from tracing_testlib import files_containing

URL_SECRET = "https://toy-user:toy-password@localhost:9/ik?api_key=toy-secret-key&x=1"
SECRETS = [b"toy-password", b"toy-secret-key", b"sk-AAAAAAAAAAAAAAAAAAAAAAAA", b"tok-bearer-secret-123",  # gitleaks:allow
           b"AIzaSyA1234567890abcdefghijklmnopq", b"hunter2-secret", b"pw-in-source-line-secret",  # gitleaks:allow
           b"cause-token-secret"]


def _http_error() -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 401
    response.url = URL_SECRET
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return exc
    raise AssertionError("raise_for_status did not raise")


class SecretApi:
    """Public functions that fail with secrets in their messages."""

    def __init__(self) -> None:
        self._env = None

    def functions(self):
        return {"fetch": self.fetch, "chained": self.chained, "long_failure": self.long_failure,
                "source_line": self.source_line}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def fetch(self):
        """Raise the HTTPError of a request whose URL holds userinfo and a key."""
        raise _http_error()

    def chained(self):
        """Raise with a key in the message and a token in the cause."""
        try:
            raise KeyError("token=cause-token-secret")
        except KeyError as exc:
            raise RuntimeError("upstream said api_key=sk-AAAAAAAAAAAAAAAAAAAAAAAA, "  # gitleaks:allow
                               "Authorization: Bearer tok-bearer-secret-123") from exc

    def long_failure(self):
        """A message longer than the inline limit (stored as a text asset)."""
        raise ValueError("word " * 1000 + " password=hunter2-secret key AIzaSyA1234567890abcdefghijklmnopq")  # gitleaks:allow

    def source_line(self):
        """The secret is in the source line that the traceback shows."""
        raise ValueError("password=pw-in-source-line-secret")


@pytest.fixture()
def rec(tmp_path, monkeypatch):
    r = Recorder(tmp_path / "traces", monkeypatch)
    yield r
    r.close()


def _assert_clean(root):
    for secret in SECRETS:
        assert files_containing(root, secret) == [], secret


def test_redact_text_shapes():
    text = redact_text(f"401 for url: {URL_SECRET} ; url: /ik?api_key=zzz-secret (x) "
                       "'token': 'tok-1234' \"password\": \"pw\" Bearer abc.def max_tokens=256", None)
    assert "toy-password" not in text and "toy-secret-key" not in text and "zzz-secret" not in text
    assert "https://[userinfo removed]@localhost:9/ik?[query removed]" in text
    assert "tok-1234" not in text and '"password": "***masked***"' in text
    assert "max_tokens=256" in text  # only whole secret key names are masked
    assert redact_text("plain message", None) == "plain message"


def test_api_errors_store_no_secret(rec):
    ep = rec.open()
    api = SecretApi()
    assert install_api(api)[0]
    C.push_episode(ep)
    raised = {}
    try:
        for name in ("fetch", "chained", "long_failure", "source_line"):
            with pytest.raises(Exception) as info:
                getattr(api, name)()
            raised[name] = info.value
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    # the exceptions the caller sees are the originals
    assert "toy-password" in str(raised["fetch"]) and "sk-AAAA" in str(raised["chained"])  # gitleaks:allow
    rec.end(ep)
    events = rec.events(ep)
    errors = {e["data"]["name"]: e["data"] for e in by_type(events, "api_call_error")}
    assert errors["fetch"]["message_redacted"] is True
    assert "https://[userinfo removed]@localhost:9/ik?[query removed]" in errors["fetch"]["message"]
    assert errors["chained"]["cause"]["type"] == "KeyError"
    assert "cause-token-secret" not in json.dumps(errors["chained"]["cause"])
    assert errors["long_failure"]["message"]["truncated"] is True  # a text asset (gzip)
    assert "***masked***" in errors["source_line"]["traceback"]
    _assert_clean(rec.root)


def test_service_and_http_errors_store_no_secret(rec):
    ep = rec.open()

    class Session(requests.Session):
        pass

    Session.send = HTTP.make_send_wrapper(requests.Session.send)

    def ik_solve(target):
        # a real request through H1 to a closed loopback port: the connection
        # error names the URL path and query
        Session().get(URL_SECRET, timeout=5)

    def seg(image, text):
        raise _http_error()

    C.push_episode(ep)
    try:
        for fn, service, attr in ((ik_solve, "pyroki", "ik_solve_fn"), (seg, "sam3", "sam3_seg_fn")):
            wrapped = SVC.make_service_wrapper(fn, service, attr)
            with pytest.raises(requests.RequestException):
                wrapped(np.zeros((4, 4, 3), np.uint8)) if service == "pyroki" else wrapped(np.zeros((4, 4, 3), np.uint8), "cube")
    finally:
        C.pop_episode(ep)
    rec.end(ep)
    events = rec.events(ep)
    errors = by_type(events, "service_call_error")
    assert len(errors) == 2
    attempt = errors[0]["data"]["http_attempts"][0]
    assert attempt["endpoint"] == "https://localhost:9/ik"
    _assert_clean(rec.root)


def test_trial_exception_stores_no_secret(rec, monkeypatch, tmp_path):
    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "run")
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)

    class Env:
        low_level_env = None

    class Args:
        config_path = "env_configs/fake/secret.yaml"

    seen = {}

    def _run_single_trial(env, trial, args, config, multi_turn_prompt=None, partial_artifacts=None):
        seen["ep"] = C._stack("episodes")[-1]
        raise RuntimeError(f"model endpoint failed: {URL_SECRET} api_key=sk-AAAAAAAAAAAAAAAAAAAAAAAA")  # gitleaks:allow

    wrapped = TRIAL.make_run_single_trial_wrapper(_run_single_trial)
    with pytest.raises(RuntimeError, match="toy-password"):
        wrapped(Env(), 1, Args(), {"output_dir": str(tmp_path / "h")})
    ep = seen["ep"]
    end = by_type(rec.events(ep), "episode_end")[0]["data"]
    assert end["exception"]["type"] == "RuntimeError" and end["exception"]["message_redacted"] is True
    assert rec.summary(ep)["exception"]["message"].startswith("model endpoint failed: https://[userinfo removed]@")
    _assert_clean(rec.root)


@pytest.mark.parametrize("body", [
    {"api_key": "data:text/plain;base64," + base64.b64encode(b"toy-secret-key").decode(), "messages": []},
    {"messages": [], "auth": {"token": {"url": "https://h/x?q=toy-secret-key"}}},
    {"messages": [], "secret": [{"type": "image_url",
                                 "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"toy-password").decode()}}]},
])
def test_secret_fields_are_never_written_as_assets(rec, body):
    ep = rec.open()
    raw = json.dumps(body)
    ep.emit("model_request", data={"role": "code_generator"},
            builder=functools.partial(HTTP._build_request, body=raw, segments=(), event_id="ev_toy"))
    rec.end(ep)
    request = by_type(rec.events(ep), "model_request")[0]["data"]
    assert request["masked_fields"] and request["images"] == []
    assert request["reconstruction"]["verified"] is True
    assert request["reconstruction"]["method"] == "masked_compare"
    stored = json.loads((ep.base_dir / request["payload_path"]).read_text())
    for key in ("api_key", "auth", "secret"):
        if key in body:
            text = json.dumps(stored[key])
            assert "***masked***" in text
    assert not (ep.base_dir / "assets").exists() or not list((ep.base_dir / "assets").rglob("*"))
    _assert_clean(rec.root)


def test_the_grep_finds_secrets_in_compressed_assets(tmp_path):
    """Sensitivity: a secret inside a txt.gz asset is found."""
    import gzip

    (tmp_path / "a.txt.gz").write_bytes(gzip.compress(b"prefix hunter2-secret suffix"))
    assert files_containing(tmp_path, b"hunter2-secret") == [tmp_path / "a.txt.gz"]


def test_model_call_errors_store_no_secret(rec):
    """H3 (a failed logical call) and a failed stream (partial output kept)."""
    import types

    from capx_trace_recorder.hooks import llm as LLM

    ep = rec.open()

    def query_model(args, prompt):
        raise RuntimeError(f"request failed: {URL_SECRET} Authorization: Bearer tok-bearer-secret-123")

    def query_model_streaming(args, prompt):
        yield {"type": "content_delta", "content": "partial "}
        raise ConnectionError(f"stream broke at {URL_SECRET}")

    C.push_episode(ep)
    try:
        with pytest.raises(RuntimeError, match="toy-password"):
            LLM.make_query_model_wrapper(query_model)(types.SimpleNamespace(model="m"), [])
        stream = LLM.make_streaming_wrapper(query_model_streaming)(types.SimpleNamespace(model="m"), [])
        with pytest.raises(ConnectionError, match="toy-secret-key"):
            list(stream)
    finally:
        C.pop_episode(ep)
    rec.end(ep)
    errors = [e["data"] for e in by_type(rec.events(ep), "model_error")]
    assert [e["scope"] for e in errors] == ["logical", "logical"]
    assert errors[1]["partial_content"] == "partial "
    assert "[userinfo removed]" in errors[0]["message"] and "[query removed]" in errors[1]["message"]
    _assert_clean(rec.root)


def test_runner_record_copies_have_redacted_error_texts(rec, monkeypatch, tmp_path):
    """H16 copies the runner's records; the tracebacks of failed attempts are redacted."""
    from capx_trace_recorder.hooks import runner as RUNNER

    run_dir = (tmp_path / "runner" / "run").resolve()
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(C.REGISTRY.runner, "run_dir", run_dir)
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    monkeypatch.delenv("CAPX_TRACE_RUN_ID", raising=False)
    record = {"trial": 1, "outcome": {"task_success": False, "infrastructure_error": True},
              "attempts": [{"attempt_id": 1, "status": "error",
                            "error": f"Traceback (most recent call last):\nHTTPError: 401 for url: {URL_SECRET}"}],
              "error": "last LLM call failed: api_key=sk-AAAAAAAAAAAAAAAAAAAAAAAA"}  # gitleaks:allow
    RUNNER._after_append(run_dir / "trials.jsonl", record)
    rec.flush()
    copy = json.loads((rec.root / "run" / "runner_records.jsonl").read_text())
    assert copy["outcome"] == record["outcome"] and copy["trial"] == 1
    assert "https://[userinfo removed]@localhost:9/ik?[query removed]" in copy["attempts"][0]["error"]
    _assert_clean(rec.root)


# ---------------------------------------------------------------------------
# The redaction rules: whole values, whole texts, and nothing that is not a secret
# ---------------------------------------------------------------------------

TAIL = "toy-secret-tail"


@pytest.mark.parametrize("text", [
    json.dumps({"password": 'prefix"toy-secret-tail'}),  # an escaped double quote in JSON
    str({"password": "it's toy-secret-tail"}),  # repr uses double quotes around a single quote
    str({"token": 'a\'b"toy-secret-tail'}),  # both quote kinds: repr escapes the single quote
    json.dumps({"client_secret": 'a\\"toy-secret-tail'}),  # an escaped backslash, then an escaped quote
    json.dumps({"detail": json.dumps({"api_key": 'x"y toy-secret-tail'})}),  # JSON inside a JSON string
    "ModelQueryArgs(api_key='toy-secret-tail', model='m')",  # a keyword-argument repr
    '{"api_key": ["abc123", "x]y toy-secret-tail"], "model": "m"}',  # a list under a quoted key
    "Authorization: Basic toy-secret-tail and more",  # an auth scheme: masked to the end of the line
])
def test_quoted_and_structured_values_are_masked_whole(text):
    """A secret value is read up to its real end (escapes, both quote kinds)."""
    from capx_trace_recorder import payload

    result = payload.redact(text, None)
    assert TAIL not in result.text and result.redacted is True, result.text
    assert TAIL not in redact_text(text, 2000)


def _across_cut(limit: int, prefix_len: int, secret: str, inside: int) -> str:
    """A text whose first ``limit`` stored characters end ``inside`` characters into ``secret``
    (``prefix_len``: characters the recorder puts before the text)."""
    return "x" * (limit - prefix_len - inside - 1) + " " + secret + " end"


CUT_SECRETS = [
    # (secret, characters of it that fall before the cut, bytes that must not be stored)
    ("https://toyuser:toy-password-123@proxy.local/v1", 24, b"toy-pass"),  # the cut is before '@'
    ('"password": "correct horse toy-battery staple"', 32, b"horse toy"),  # a quoted value with spaces
    ("sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2", 14, b"A1b2C3"),  # fewer than 12 key characters kept  # gitleaks:allow
    ("AIzaSyA1234567890abcdefghijklmnopq", 20, b"AIzaSyA1234"),  # fewer than 20 after AIza  # gitleaks:allow
]


@pytest.mark.parametrize("limit", [300, 400, 2000])
@pytest.mark.parametrize("secret, inside, needle", CUT_SECRETS)
def test_a_secret_across_the_cut_is_not_kept_in_part(limit, secret, inside, needle):
    text = _across_cut(limit, 0, secret, inside)
    clean = redact_text(text, limit)
    assert len(clean) <= limit
    assert needle.decode() not in clean, clean[-80:]


def test_a_text_longer_than_the_window_is_cut_without_a_partial_secret():
    from capx_trace_recorder import payload

    window = payload.REDACT_WINDOW
    # the window ends inside the userinfo: its '@' is not read
    text = "word " * (window // 5) + "https://toyuser:toy-password-123@h/x" + " tail" * 10
    text = text[: window - 10] + " https://toyuser:toy-password-123@h/x" + " tail" * 10
    target: dict = {}
    result = payload.put_redacted(target, "message", text, None)
    assert "toy-pass" not in result.text and "toyuser" not in result.text
    assert result.cut is True and target["message_cut"] is True
    assert result.text.endswith(f"only the first {window} of {len(text)} characters were read]")


# Real names, lines and texts of capx that are not secrets (regression).
def _real_texts() -> list[str]:
    import hashlib
    import traceback

    from capx.integrations.franka.control_reduced import FrankaControlApiReduced
    from capx.integrations.franka.libero import FrankaLiberoApi
    from capx.llm import client

    texts = []
    try:  # a wrong keyword on a real 40-character API name
        FrankaControlApiReduced.get_oriented_bounding_box_from_3d_points(
            object.__new__(FrankaControlApiReduced), threshold=0.5)
    except TypeError as exc:
        texts.append(str(exc))
    try:
        FrankaLiberoApi.get_object_3d_points_and_masks_from_language(object.__new__(FrankaLiberoApi))
    except TypeError as exc:
        texts.append(str(exc))
    for name in ("_build_multi_turn_decision_prompt_legacy", "target_position_above_the_red_cube_in_world"):
        try:
            eval(name, {})  # noqa: S307 - a NameError with a long generated-code name
        except NameError as exc:
            texts.append(str(exc))
    try:  # a traceback through a real function whose name is 40 characters long
        FrankaControlApiReduced.get_oriented_bounding_box_from_3d_points(
            object.__new__(FrankaControlApiReduced), None)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        assert "in get_oriented_bounding_box_from_3d_points" in tb
        texts.append(tb)
    texts.append(str(FileNotFoundError(2, "No such file or directory", "example/code.py")))
    texts.append(f"frame hash mismatch: {hashlib.sha256(b'x').hexdigest()} != "
                 f"{hashlib.sha256(b'y').hexdigest().upper()}")
    texts.append("_build_multi_turn_decision_prompt_legacy: not found")  # a hook failure reason
    texts.append("max_tokens=256 budget_tokens=1024 max_completion_tokens=12 token == other")
    texts.append("see http://127.0.0.1:8114/docs#section and what?")
    texts.append("episode demo_task__multi_turn__s1__i0__a1, 20260927T012345Z__web__ab12cd34")
    texts.append("ev_0123456789abcdef0123 Qwen2VLForConditionalGeneration Wav2Vec2PositionalConvEmbedding")
    source = Path(client.__file__).read_text().splitlines()
    code = [line for line in source if "api_key" in line or "token" in line]
    assert any("api_key=args.api_key" in line for line in code)
    texts.extend(code)  # source lines of the LLM client, as tracebacks show them
    return texts


HEADER_LINE = '        headers["Authorization"] = f"Bearer {args.api_key}"'  # capx/llm/client.py


@pytest.mark.requires_capx
def test_real_names_lines_and_code_are_not_redacted():
    """Only the one client line that rule (d) covers changes: a subscript key's
    f-string value is masked whole, also when it holds only a reference."""
    from capx_trace_recorder import payload

    changed = {}
    for text in _real_texts():
        result = payload.redact(text, None)
        if result.text != text or result.redacted:
            changed[text] = result.text
    assert changed == {HEADER_LINE: '        headers["Authorization"] = f"***masked***"'}


@pytest.mark.parametrize("unit", [
    "a.", "a-", "\\", '"', "'", "token=", '"token":"', "https://", "Bearer ", "aB3", "x-", "api_key=x&", "a@",
    "url: /",
    # the string, bracket, annotation, subscript and form readings of rule (d)
    'token=f"{', 'password=f"{x["', "token=f'{d[\"", 'token=["\\\\", "', '"token": ["a\\\\", "', "token=[\\\\\"",
    'config["token"] = ', "password: str = ", "token: a[", "token: a | ", "#token=a)", "?token=", "&token=",
    "token=f(", "token=a.b.", "--token=", "token=\"\"\"", "token=(", "Digest a=\"", "\\\"token\\\": \\\"",
])
def test_redaction_time_is_linear(unit):
    """A whole window (64 KB) of each pattern: the quadratic URL scan took 4.7 s for
    40,000 characters of 'a.'. Every pass reads each character a bounded number of
    times, so each of these takes well under a second."""
    from capx_trace_recorder.payload import REDACT_WINDOW

    text = (unit * (REDACT_WINDOW // len(unit) + 1))[:REDACT_WINDOW]
    t0 = time.perf_counter()
    redact_text(text, None)
    assert time.perf_counter() - t0 < 2.0


def test_private_key_crossing_redaction_window_is_masked():
    from capx_trace_recorder.payload import REDACT_WINDOW

    prefix = "x" * (REDACT_WINDOW - 61) + " "
    text = prefix + "-----BEGIN PRIVATE KEY-----\ntoy-secret\n" + "y" * 100 + "-----END PRIVATE KEY-----"
    result = redact_text(text, None)
    assert "toy-secret" not in result
    assert "-----BEGIN PRIVATE KEY-----" not in result


class RealNamesApi:
    """A public function with a real 40-character capx API name."""

    def __init__(self) -> None:
        self._env = None

    def functions(self):
        return {"get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def get_oriented_bounding_box_from_3d_points(self, points):
        """Call the real method with a wrong keyword (the TypeError Python makes)."""
        from capx.integrations.franka.control_reduced import FrankaControlApiReduced

        return FrankaControlApiReduced.get_oriented_bounding_box_from_3d_points(self, points, threshold=0.5)


@pytest.mark.requires_capx
def test_a_real_api_error_is_stored_verbatim(rec):
    """No flag and no change for an error that holds no secret (the later-input
    check reads message_redacted)."""
    ep = rec.open()
    api = RealNamesApi()
    assert install_api(api)[0]
    C.push_episode(ep)
    try:
        with pytest.raises(TypeError) as info:
            api.get_oriented_bounding_box_from_3d_points([[0.0, 0.0, 0.0]])
    finally:
        C.pop_episode(ep)
        uninstall_api(api)
    rec.end(ep)
    error = by_type(rec.events(ep), "api_call_error")[0]["data"]
    assert error["message"] == str(info.value)
    assert "get_oriented_bounding_box_from_3d_points() got an unexpected keyword" in error["message"]
    assert "in get_oriented_bounding_box_from_3d_points" in error["traceback"]
    assert error["traceback"].endswith(f"TypeError: {info.value}")
    assert not any(k.endswith("_redacted") for k in error), error


# ---------------------------------------------------------------------------
# Every stored text field, through the real hooks
# ---------------------------------------------------------------------------

JSON_SECRET = json.dumps({"password": 'prefix"toy-secret-tail'})


def _files_with_any(root, needles):
    return {n: files_containing(root, n) for n in needles if files_containing(root, n)}


def test_every_stored_text_field_is_redacted_and_flagged(rec, monkeypatch, tmp_path):
    """Escaped quotes in unbounded texts and secrets across every stored length."""
    import types

    from capx_trace_recorder import bootstrap
    from capx_trace_recorder.hooks import common as COMMON
    from capx_trace_recorder.hooks import env as ENV
    from capx_trace_recorder.hooks import llm as LLM

    monkeypatch.setenv("CAPX_TRACE_RUN_ID", "run")
    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    monkeypatch.setattr(C.REGISTRY, "hooks_failed", [])
    monkeypatch.setattr(C.REGISTRY, "failures_logged", 0)
    url, url_in, url_needle = CUT_SECRETS[0]
    quoted, quoted_in, quoted_needle = CUT_SECRETS[1]
    key, key_in, key_needle = CUT_SECRETS[2]
    needles = [b"toy-secret-tail", url_needle, quoted_needle, key_needle]

    class Api:
        def __init__(self) -> None:
            self._env = None

        def functions(self):
            return {"fail": self.fail}

        def combined_doc(self) -> str:
            from local_api_doc import combined_doc

            return combined_doc(self)

        def fail(self):
            """Fail with a JSON secret; the cause holds a secret across its 2000-character cut."""
            try:
                raise KeyError(_across_cut(2000, 0, url, url_in))
            except KeyError as exc:
                raise RuntimeError("upstream said " + JSON_SECRET) from exc

    class Env:
        def __init__(self) -> None:
            self.low_level_env = None
            self._apis = {}
            self._exec_globals = {}

        def step(self, code):
            raise ValueError(_across_cut(2000, 0, quoted, quoted_in))

    Env.step = ENV.make_step_wrapper(Env.__dict__["step"])

    def failing_session(message: str) -> requests.Session:
        class FailingAdapter(requests.adapters.HTTPAdapter):
            def send(self, request, **kwargs):
                raise requests.exceptions.ConnectionError(message)

        class Session(requests.Session):
            pass

        Session.send = HTTP.make_send_wrapper(requests.Session.send)  # H1
        session = Session()
        session.mount("http://", FailingAdapter())
        return session

    # a service attempt's error is "ConnectionError: <message>", stored in 300 characters
    service_session = failing_session(_across_cut(300, len("ConnectionError: "), key, key_in))
    model_session = failing_session(_across_cut(2000, 0, key, key_in))
    ep = rec.open("fields")
    api = Api()
    assert install_api(api)[0]
    C.push_episode(ep)
    try:
        with pytest.raises(RuntimeError):
            api.fail()  # H8: message, traceback, cause
        with pytest.raises(ValueError):
            Env().step("x = 1")  # H7: code_block_end.exception

        def seg(image, text):  # H9: an HTTP attempt error (300 characters) and the message
            try:
                service_session.get("http://127.0.0.1:9/sam3", timeout=1)
            except requests.exceptions.ConnectionError:
                pass
            raise RuntimeError("service said " + JSON_SECRET)

        with pytest.raises(RuntimeError):
            SVC.make_service_wrapper(seg, "sam3", "sam3_seg_fn")(np.zeros((4, 4, 3), np.uint8), "cube")
        lc = C.LogicalCall("lc_" + "0" * 20, ep, "code_generator", "initial_code", 0, "m", None, "test")
        C.push_ctx(lc)
        try:  # H1: an attempt error (2000 characters)
            with pytest.raises(requests.exceptions.ConnectionError):
                model_session.post("http://127.0.0.1:9/v1/chat/completions", data=b"{}", timeout=1)
        finally:
            C.pop_ctx(lc)

        def query_model(args, prompt):  # H3: a logical error
            raise RuntimeError(_across_cut(2000, 0, url, url_in))

        with pytest.raises(RuntimeError):
            LLM.make_query_model_wrapper(query_model)(types.SimpleNamespace(model="m"), [])
        # H1: an HTTP error body (2000 characters)
        ep.emit("model_error", data={"scope": "attempt"}, builder=functools.partial(
            HTTP._build_http_error, content=_across_cut(2000, 0, quoted, quoted_in).encode()))
        # a recorder error inside a builder (500 characters) and an internal error (400)
        prefix = len("ValueError: ")

        def failing_builder(ctx):
            raise ValueError(_across_cut(500, prefix, url, url_in))

        ep.emit("trace_warning", data={"code": "x", "counts_as_gap": False}, builder=failing_builder)
        where = "somewhere"
        COMMON.report_internal_error(RuntimeError(_across_cut(400, len(f"{where}: RuntimeError: "), key, key_in)),
                                     where)
        # a hook failure reason (300 characters), announced to the open episode
        bootstrap._record_failed("H99", "toy.target", _across_cut(300, 0, url, url_in))
    finally:
        C.pop_episode(ep)
        uninstall_api(api)

    def _trial(env, trial, args, config, multi_turn_prompt=None, partial_artifacts=None):
        seen["ep"] = C._stack("episodes")[-1]
        raise RuntimeError(_across_cut(2000, 0, key, key_in))  # H4: episode_end.exception

    seen = {}

    class Args:
        config_path = "env_configs/fake/fields.yaml"

    with pytest.raises(RuntimeError):
        TRIAL.make_run_single_trial_wrapper(_trial)(types.SimpleNamespace(low_level_env=None), 1, Args(),
                                                   {"output_dir": str(tmp_path / "h")})
    rec.end(ep)
    rec.flush()
    assert _files_with_any(rec.root, needles) == {}
    events = rec.events(ep)
    api_error = by_type(events, "api_call_error")[0]["data"]
    assert api_error["message_redacted"] is True and api_error["traceback_redacted"] is True
    assert api_error["cause"]["message_redacted"] is True
    assert by_type(events, "code_block_end")[0]["data"]["exception"]["message_redacted"] is True
    service = by_type(events, "service_call_error")[0]["data"]
    assert service["message_redacted"] is True
    assert service["http_attempts"][0]["error_redacted"] is True
    errors = by_type(events, "model_error")
    assert [e["data"].get("message_redacted") for e in errors if "message_redacted" in e["data"]] == [True, True]
    assert [e["data"]["body_excerpt_redacted"] for e in errors if "body_excerpt" in e["data"]] == [True]
    warnings = {(e["data"]["code"], bool(e["data"].get("writer_generated"))): e["data"]
                for e in by_type(events, "trace_warning")}
    assert warnings[("x", False)]["recorder_error_redacted"] is True
    assert warnings[("internal_error", False)]["message"].startswith("somewhere: RuntimeError: ")
    assert warnings[("internal_error", False)]["message_redacted"] is True
    assert warnings[("hook_install_failed", False)]["message_redacted"] is True
    trial_end = by_type(rec.events(seen["ep"]), "episode_end")[0]["data"]
    assert trial_end["exception"]["message_redacted"] is True
    assert rec.summary(seen["ep"])["exception"]["message_redacted"] is True


def test_runner_record_redaction_lists_the_changed_texts(rec, monkeypatch, tmp_path):
    from capx_trace_recorder.hooks import runner as RUNNER

    run_dir = (tmp_path / "runner" / "run").resolve()
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(C.REGISTRY.runner, "run_dir", run_dir)
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    monkeypatch.delenv("CAPX_TRACE_RUN_ID", raising=False)
    record = {"trial": 1, "attempts": [{"attempt_id": 1, "error": "HTTPError: " + JSON_SECRET},
                                       {"attempt_id": 2, "error": "FileNotFoundError: outputs/a_b/c.py"}],
              "setup_error": "setup failed"}
    RUNNER._after_append(run_dir / "trials.jsonl", record)
    rec.flush()
    copy = json.loads((rec.root / "run" / "runner_records.jsonl").read_text())
    assert copy["trace_redacted_fields"] == ["attempts[0].error"]
    assert copy["attempts"][1]["error"] == "FileNotFoundError: outputs/a_b/c.py"
    assert files_containing(rec.root, b"toy-secret-tail") == []


# ---------------------------------------------------------------------------
# The scope of the rules, case by case (redaction regression cases)
# ---------------------------------------------------------------------------

_GH = "a1" * 18
_HF = "hf_" + "abcdefghij" * 4  # gitleaks:allow

# (id, text, the stored text). Every secret holds "toy-secret" (or is "hunter", the
# example), so one grep finds any of them in the trace files.
SCOPE = [
    # (a) URLs: userinfo and the whole query go; ")", "]", "," and "." belong to a query
    ("abs_query_paren", "401 for url: https://example.test/api?api_key=toy-prefix)toy-secret-tail",
     "401 for url: https://example.test/api?[query removed]"),
    ("rel_query_paren",
     "Max retries exceeded with url: /ik?api_key=toy-prefix)toy-secret-tail&x=1 (Caused by NewConnectionError('x'))",
     "Max retries exceeded with url: /ik?[query removed] (Caused by NewConnectionError('x'))"),
    ("query_punct", "GET /v1/models?key=toy-secret-a.b,c]d HTTP/1.1", "GET /v1/models?[query removed] HTTP/1.1"),
    ("query_ends", "see <https://x.test/a?q=toy-secret> and 'https://u:toy-secret@x.test/p?q=1#frag'",
     "see <https://x.test/a?[query removed]> and 'https://[userinfo removed]@x.test/p?[query removed]#frag'"),
    ("network_path", "//toy-user:toy-secret@host.test/x", "//[userinfo removed]@host.test/x"),
    ("empty_query", "https://x.test/a? what?", "https://x.test/a? what?"),
    ("json_in_json_query", json.dumps({"detail": json.dumps({"u": "https://x.test/a?q=toy-secret"})}),
     json.dumps({"detail": json.dumps({"u": "https://x.test/a?[query removed]"})})),
    ("scheme_after_dash", "-https://u:toy-secret@x.test/a?q=1",
     "-https://[userinfo removed]@x.test/a?[query removed]"),
    ("earlier_userinfo_mark", "https://[userinfo removed]@x.test/a?q=toy-secret",
     "https://[userinfo removed]@x.test/a?[query removed]"),
    ("fragment_param", "https://x.test/cb#access_token=toy-prefix)toy-secret-tail&state=1",
     "https://x.test/cb#access_token=***masked***&state=1"),
    ("ipv6", "http://[::1]:8110/v1?key=toy-secret", "http://[::1]:8110/v1?[query removed]"),
    ("file_url", "file:///tmp/a?b=toy-secret", "file:///tmp/a?[query removed]"),
    ("quoted_query_value", 'https://x.test/a?password="toy-secret"',
     'https://x.test/a?[query removed]"***masked***"'),
    ("url_equals", "url=/x?q=toy-secret", "url=/x?[query removed]"),
    # (b) Authorization credentials
    ("bearer", "sent Bearer toy-secret.abc~+/= to", "sent Bearer ***redacted*** to"),
    ("bearer_cases", "bearer toy-secret-1 BEARER toy-secret-2", "bearer ***redacted*** BEARER ***redacted***"),
    ("basic", "Basic dG95LXNlY3JldA==", "Basic ***redacted***"),
    ("token_scheme", "Token toy-secret", "Token ***redacted***"),
    ("digest", 'Digest username="toy", realm="r", response="toy-secret", nc=00000001', "Digest ***redacted***"),
    ("digest_word", "Digest mismatch", "Digest mismatch"),
    ("challenge", 'WWW-Authenticate: Bearer realm="api", error="toy-secret"',
     "WWW-Authenticate: Bearer ***redacted***"),
    ("lower_schemes", "token toy basic toy digest toy", "token toy basic toy digest toy"),
    ("glued_scheme", "xBearer toy", "xBearer toy"),
    ("no_credential", "Bearer {api_key}", "Bearer {api_key}"),
    ("bearer_url", "Bearer /x?q=toy-secret", "Bearer ***redacted***?[query removed]"),
    # (c) known key formats
    ("openai", "key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 end", "key ***redacted*** end"),  # gitleaks:allow
    ("google", "AIzaSyA1234567890abcdefghijklmnopqrstu", "***redacted***"),  # gitleaks:allow
    ("github", f"ghp_{_GH} gho_{_GH} ghs_{_GH} github_pat_{'11ABCDEFG0' * 4}",  # gitleaks:allow
     "***redacted*** ***redacted*** ***redacted*** ***redacted***"),
    ("huggingface", _HF, "***redacted***"),
    ("slack", "xoxa-1234567890 xoxb-1234567890-abc xoxp-1234567890 xoxr-1234567890",  # gitleaks:allow
     "***redacted*** ***redacted*** ***redacted*** ***redacted***"),
    ("aws", "AKIAIOSFODNN7EXAMPLE ASIAIOSFODNN7EXAMPLE", "***redacted*** ***redacted***"),  # gitleaks:allow
    ("more_key_formats", "gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ xai-ABCDEFGHIJKLMNOPQRSTUVWXYZ r8_ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # gitleaks:allow
     "***redacted*** ***redacted*** ***redacted***"),
    ("private_key_block", "-----BEGIN PRIVATE KEY-----\ntoy-secret\n-----END PRIVATE KEY-----", "***redacted***"),  # gitleaks:allow
    ("private_key_unterminated", "before -----BEGIN PRIVATE KEY-----\ntoy-secret", "before ***redacted***"),  # gitleaks:allow
    ("permission_denied_path", "/etc/passwd: Permission denied", "/etc/passwd: ***masked*** denied"),
    ("more_secret_names", "api-key=toy-secret x-goog-api-key=toy-secret aws_secret_access_key=toy-secret private_key=toy-secret auth_token=toy-secret session_token=toy-secret passwd=toy-secret",
     "api-key=***masked*** x-goog-api-key=***masked*** aws_secret_access_key=***masked*** private_key=***masked*** auth_token=***masked*** session_token=***masked*** passwd=***masked***"),
    ("not_key_formats", "task-0123456789abcdefghijkl sk-short AKIAIOSFODNN7EXAMPLEX hf_short xoxz-1234567890",  # gitleaks:allow
     "task-0123456789abcdefghijkl sk-short AKIAIOSFODNN7EXAMPLEX hf_short xoxz-1234567890"),  # gitleaks:allow
    ("key_glued_to_flag", _HF + "--password=toy-secret", "***redacted***--password=***masked***"),
    # under a secret key a replaced key is masked with the rest of the value
    ("key_format_in_value", "api_key=sk-toysecretAAAAAAAAAAAAAAAAAAAA!toy-secret-tail", "api_key=***masked***"),  # gitleaks:allow
    ("key_format_in_json", '{"error": "busy", "api_key": "sk-toy-secret-123456789012345678"}',  # gitleaks:allow
     '{"error": "busy", "api_key": "***masked***"}'),
    ("key_format_after_scheme", "Authorization: Bearer sk-toy-secret-123456789012345678.tail",  # gitleaks:allow
     "Authorization: Bearer ***masked***"),
    # (d) keys
    ("json_key", '{"password": "toy-secret"}', '{"password": "***masked***"}'),
    ("repr_key", "{'token': 'toy-secret'}", "{'token': '***masked***'}"),
    ("escaped_key", json.dumps({"detail": json.dumps({"api_key": "toy-secret"})}),
     json.dumps({"detail": json.dumps({"api_key": "***masked***"})})),
    ("subscript_key", 'config["password"] = "toy-secret"', 'config["password"] = "***masked***"'),
    ("flag", "--password=toy-secret", "--password=***masked***"),
    ("quoted_flag", "{'--password': 'toy-secret'}", "{'--password': '***masked***'}"),
    ("attribute", "self.password = 'toy-secret'", "self.password = '***masked***'"),
    ("case", "X-Api-Key: toy-secret", "X-Api-Key: ***masked***"),
    ("env_name", "API_KEY=toy-secret", "API_KEY=***masked***"),
    ("not_keys", "max_tokens=256 budget_tokens=1024 OPENAI_API_KEY_NAME=x my-token=x token_type=x",
     "max_tokens=256 budget_tokens=1024 OPENAI_API_KEY_NAME=x my-token=x token_type=x"),
    # (d) separators, with line breaks on both sides
    ("walrus", "password := 'toy-secret'", "password := '***masked***'"),
    ("arrow", "'password' => 'toy-secret'", "'password' => '***masked***'"),
    ("comparisons", "token == other and token::x", "token == other and token::x"),
    ("newlines_both_sides", "password\n=\n'toy-secret'", "password\n=\n'***masked***'"),
    ("next_line_value", '{"password":\n"toy-secret-next-line"}', '{"password":\n"***masked***"}'),
    # (d) strings: prefixes, f-strings (nested quotes of Python 3.12 too), escapes, line breaks
    ("f_string", 'password = f"toy-secret-f-string"', 'password = f"***masked***"'),
    ("pep701_f_string", 'password = f"{d["user"]}:toy-secret"', 'password = f"***masked***"'),
    ("bytes_raw", "password = rb'toy-secret'", "password = rb'***masked***'"),
    ("triple", 'password = """toy-secret\nline2"""', 'password = """***masked***"""'),
    ("newline_in_string", 'password="first\ntoy-secret-newline"', 'password="***masked***"'),
    ("escaped_quote", json.dumps({"password": 'a"toy-secret'}), '{"password": "***masked***"}'),
    ("unterminated", "password='unterminated toy-secret\nmore", "password='***masked***"),
    # the two readings of backslashes disagree: the value runs to the end of the text
    ("backslash_ended", 'password="abc\\\\" then toy-secret', 'password="***masked***'),
    ("json_encoded_python", json.dumps("password = 'it\\'s toy-secret'"), "\"password = '***masked***"),
    # (d) brackets: to the matching bracket, across lines, skipping strings
    ("pretty_list", json.dumps({"password": ["toy-secret-pretty-list"]}, indent=2),
     '{\n  "password": ***masked***\n}'),
    ("pretty_dict", json.dumps({"password": {"a": "toy-secret-pretty-dict", "b": ["x]", "y}"]}}, indent=2),
     '{\n  "password": ***masked***\n}'),
    ("quoted_brackets", json.dumps({"detail": json.dumps({"password": ["first", "x]y toy-secret-nested"]})}),
     '{"detail": "{\\"password\\": ***masked***}"}'),
    ("unterminated_bracket", "token: [unterminated toy-secret\n more", "token: ***masked***"),
    ("parenthesis", "password=(toy-secret, other)", "password=***masked***"),
    ("backslash_in_list", '{"secret": ["a\\\\", "toy-secret"], "x": 1}', '{"secret": ***masked***'),
    ("backslash_then_next_key",
     json.dumps({"client_secret": ["a \\"], "note": {"authorization": ["b ] } )", "toy-secret"]}}),
     '{"client_secret": ***masked***'),
    # (d) Authorization schemes, annotations and bare tokens
    ("auth_bearer", "Authorization: Bearer toy-secret", "Authorization: Bearer ***masked***"),
    ("auth_digest", 'Authorization: Digest username="toy", response="toy-secret"',
     "Authorization: Digest ***masked***"),
    ("auth_token_lower", "authorization: token toy-secret", "authorization: token ***masked***"),
    ("annotation", "password: str = 'toy-secret'", "password: str = '***masked***'"),
    ("annotation_subscript", 'token: Optional[str] = "toy-secret"', 'token: Optional[str] = "***masked***"'),
    ("annotation_kept", "api_key: str | None = None", "api_key: str | None = None"),
    ("annotation_no_value", "password: str", "password: ***masked***"),
    ("bare", "password=hunter", "password=***masked***"),
    ("bare_delimiters", "password=toy-secret;next token=toy-secret,y",
     "password=***masked***;next token=***masked***,y"),
    ("bare_paren", "token=toy-secret)def", "token=***masked***)def"),
    ("form_paren", "?token=toy-prefix)toy-secret&x=1", "?token=***masked***&x=1"),
    # (d) kept values and reference expressions; a reference is never data
    ("null_words", "api_key=None, token=null, password: true, secret=undefined",
     "api_key=None, token=null, password: true, secret=undefined"),
    ("empty_values", 'password="" token=[] secret={ }', 'password="" token=[] secret={ }'),
    ("environ", 'API_KEY = os.environ["API_KEY"]', 'API_KEY = os.environ["API_KEY"]'),
    ("call", "token = get_key()", "token = get_key()"),
    ("index", "api_key = args.api_keys[0]", "api_key = args.api_keys[0]"),
    ("dotted", "api_key=args.api_key,", "api_key=args.api_key,"),
    ("mask", "password=***masked***", "password=***masked***"),
    ("jwt", "access_token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln", "access_token=***masked***"),  # gitleaks:allow
    ("form_dotted", "https://x.test/cb#id_token=toy.secret", "https://x.test/cb#id_token=***masked***"),
    ("amp_dotted", "refresh_token=toy.secret&scope=x", "refresh_token=***masked***&scope=x"),
    ("trailing_dot", "client_secret=toysecret.", "client_secret=***masked***"),
    ("inside_kept_call", 'get(token="toy-secret")', 'get(token="***masked***")'),
    # an escaped line break or tab (a JSON string, a Python repr) before a token or around a
    # separator counts as whitespace: the token at the start of an escaped line is found
    ("escaped_newline_key", json.dumps({"detail": "Config error:\npassword=toy-secret user=bob"}),
     '{"detail": "Config error:\\npassword=***masked*** user=bob"}'),
    ("escaped_newline_flag", repr("cmd failed:\n--password=toy-secret now"),
     "'cmd failed:\\n--password=***masked*** now'"),
    ("escaped_tab_key", json.dumps({"detail": "fields:\tapi_key: toy-secret end"}),
     '{"detail": "fields:\\tapi_key: ***masked*** end"}'),
    ("escaped_newline_separator", json.dumps({"detail": "password:\n  toy-secret end"}),
     '{"detail": "password:\\n  ***masked*** end"}'),
    ("escaped_newline_key_format", json.dumps({"detail": "Invalid key:\nsk-toysecretAAAAAAAAAAAAAAAAAAAA end"}),  # gitleaks:allow
     '{"detail": "Invalid key:\\n***redacted*** end"}'),
    ("escaped_crlf_scheme", json.dumps({"detail": "got:\r\nBearer toy-secret end"}),
     '{"detail": "got:\\r\\nBearer ***redacted*** end"}'),
    ("escaped_newline_path", json.dumps({"detail": "GET\n/v1/x?key=toy-secret end"}),
     '{"detail": "GET\\n/v1/x?[query removed] end"}'),
    ("escaped_in_json_in_json", json.dumps({"detail": json.dumps({"x": "a\npassword=toy-secret b"})}),
     json.dumps({"detail": json.dumps({"x": "a\npassword=***masked*** b"})})),
    ("escaped_tab_header", json.dumps({"h": "Authorization:\tBearer toy-secret"}),
     '{"h": "Authorization:\\tBearer ***masked***"}'),
    ("escaped_separator_in_json_in_json", json.dumps({"d": json.dumps({"h": "password:\n  toy-secret end"})}),
     json.dumps({"d": json.dumps({"h": "password:\n  ***masked*** end"})})),
    ("escaped_text_kept", json.dumps({"detail": "line1\nline2\tthe token count is 5\r\nnpassword=x"}),
     json.dumps({"detail": "line1\nline2\tthe token count is 5\r\nnpassword=x"})),
    # prose and code that rule (d) covers are masked too
    ("prose_token", "unexpected token: 'else'", "unexpected token: '***masked***'"),
    ("prose_password", 'password: must be set; invalid token: "else"',
     'password: ***masked*** be set; invalid token: "***masked***"'),
    ("header_line", HEADER_LINE, '        headers["Authorization"] = f"***masked***"'),
]
SCOPE_SECRETS = [b"toy-secret", b"toysecret", b"hunter", b"dG95LXNlY3JldA", b"toy-user", b"sk-proj-AbCd",  # gitleaks:allow
                 b"AIzaSyA1234567890", b"ghp_a1a1", b"gho_a1a1", b"ghs_a1a1", b"github_pat_11", b"hf_abcdefghij",  # gitleaks:allow
                 b"xoxb-1234567890", b"AKIAIOSFODNN7EXAMPLE ", b"ASIAIOSFODNN7EXAMPLE",
                 b"gsk_ABCDEFGHIJKLMNOP", b"xai-ABCDEFGHIJKLMNOP", b"r8_ABCDEFGHIJKLMNOP"]  # gitleaks:allow


@pytest.mark.parametrize("text, stored", [case[1:] for case in SCOPE], ids=[case[0] for case in SCOPE])
def test_the_rules_case_by_case(text, stored):
    from capx_trace_recorder import payload

    result = payload.redact(text, None)
    assert result.text == stored
    assert result.redacted is (stored != text)
    assert payload.redact(text, 2000) == result  # a short text: the limit changes nothing
    again = payload.redact(result.text, None)  # a stored text passes the rules unchanged
    assert again.text == result.text and again.redacted is False


class TextApi:
    """An API function that raises with a set text (H8 stores the message, unbounded,
    and the traceback). The text is not an argument: H8 stores argument summaries."""

    def __init__(self) -> None:
        self._env = None
        self.text = ""

    def functions(self):
        return {"fail": self.fail}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def fail(self):
        """Raise ValueError with the set text."""
        raise ValueError(self.text)


def _store_texts(rec, texts: dict[str, str]) -> dict[str, dict]:
    """Store each text through the real hooks and writer: as the message of the exception
    that ends an episode (H4's ``end_episode``: episode_end in events.jsonl and
    episode.json, cut to 2000 characters) and as the message of a failed API call (H8:
    the message without a cut, and the traceback). Returns what was stored per text."""
    api = TextApi()
    assert install_api(api)[0]
    api_ep = rec.open("api")
    C.push_episode(api_ep)
    try:
        for text in texts.values():
            api.text = text
            with pytest.raises(ValueError) as info:
                api.fail()
            assert str(info.value) == text  # the exception the caller sees is the original
    finally:
        C.pop_episode(api_ep)
        uninstall_api(api)
    rec.end(api_ep)
    episodes = {}
    for i, name in enumerate(texts):
        ep = rec.open(f"text{i}")
        TRIAL.end_episode(ep, end_reason="exception", exc=RuntimeError(texts[name]))
        C.close_episode(ep)
        episodes[name] = ep
    rec.flush()
    api_errors = [e["data"] for e in by_type(rec.events(api_ep), "api_call_error")]
    assert len(api_errors) == len(texts)
    stored = {}
    for (name, ep), api_error in zip(episodes.items(), api_errors):
        stored[name] = {"episode_end": by_type(rec.events(ep), "episode_end")[0]["data"]["exception"],
                        "episode_json": rec.summary(ep)["exception"], "api_call_error": api_error}
    return stored


def test_the_rules_through_the_writer(rec):
    """Every case of SCOPE stored by the real hooks: the stored message is the expected
    text in every field that holds it, flagged only when something was replaced, and
    no secret is in any file under the trace root."""
    stored = _store_texts(rec, {case[0]: case[1] for case in SCOPE})
    for name, text, expected in SCOPE:
        changed = expected != text
        for field in ("episode_end", "episode_json"):
            exc = stored[name][field]
            assert exc["message"] == expected, (name, field)
            assert exc.get("message_redacted", False) is changed, (name, field)
        api_error = stored[name]["api_call_error"]
        assert api_error["message"] == expected, name
        assert api_error.get("message_redacted", False) is changed, name
        assert api_error["traceback"].endswith(f"ValueError: {expected}"), name
        assert api_error.get("traceback_redacted", False) is changed, name
    assert _files_with_any(rec.root, SCOPE_SECRETS) == {}


# Redaction regression examples.
REGRESSION_SECRETS = {
    "pretty_scalar": json.dumps({"password": "toy-secret-pretty-scalar"}, indent=2),
    "pretty_list": json.dumps({"password": ["toy-secret-pretty-list"]}, indent=2),
    "url_parenthesis": "401 for url: https://example.test/api?api_key=toy-prefix)toy-secret-tail",
    "unquoted_password": "password=hunter",
    "literal_f_string": 'password = f"toy-secret-f-string"',
    "quoted_brackets": json.dumps({"detail": json.dumps({"password": ["first", "x]y toy-secret-nested"]})}),
    "newline_password": 'password="first\ntoy-secret-newline"',
    "next_line": '{"password":\n"toy-secret-next-line"}',
    "pretty_dict": json.dumps({"password": {"a": "toy-secret-pretty-dict", "b": ["x]", "y}"]}}, indent=2),
    "relative_url_parenthesis": "Max retries exceeded with url: /ik?api_key=toy-prefix)toy-secret-tail&x=1",
}
REGRESSION_KEPT = {  # a prior regression: these were changed and flagged
    "env_code": 'API_KEY = os.environ["API_KEY"]',
    "index_code": "api_key = args.api_keys[0]",
    "condition_id": "output: /data/demo_multi_turn__s1__i1__a1/log.txt",
}


def test_the_redaction_repros_store_no_secret(rec):
    stored = _store_texts(rec, {**REGRESSION_SECRETS, **REGRESSION_KEPT})
    for name in REGRESSION_SECRETS:
        for field in ("episode_end", "episode_json", "api_call_error"):
            assert stored[name][field]["message_redacted"] is True, (name, field)
    for name, text in REGRESSION_KEPT.items():
        for field in ("episode_end", "episode_json", "api_call_error"):
            assert stored[name][field]["message"] == text, (name, field)
            assert "message_redacted" not in stored[name][field], (name, field)
    assert _files_with_any(rec.root, [b"toy-secret", b"hunter"]) == {}


@pytest.mark.requires_capx
def test_real_texts_are_stored_verbatim(rec):
    """Trial-id paths, the os.environ reference, API, class and function names, a real
    traceback, sha256 digests and client code lines, through the real writer."""
    texts = {f"t{i}": text for i, text in enumerate(t for t in _real_texts() if t != HEADER_LINE)}
    texts.update(REGRESSION_KEPT)
    texts["traceback_with_trial_id"] = (
        "Traceback (most recent call last):\n"
        '  File "/data/demo_multi_turn__s1__i1__a1/code.py", line 3, in <module>\n'
        '    grasp_object("red cube")\n'
        "ValueError: target_position_above_the_red_cube_in_world is out of reach")
    stored = _store_texts(rec, texts)
    for name, text in texts.items():
        for field, limit in (("episode_end", 2000), ("episode_json", 2000), ("api_call_error", None)):
            exc = stored[name][field]
            message = exc["message"]
            if isinstance(message, dict):  # a long message is stored as a text asset
                continue
            assert message == text[:limit], (name, field)
            assert not any(k.endswith("_redacted") for k in exc), (name, field, exc)


def test_values_are_read_whole_whatever_the_escaping():
    """Strings and brackets under a secret key, in plain text, JSON, JSON inside JSON and
    Python reprs: the whole value goes, and a key after it is still found."""
    from capx_trace_recorder import payload

    doc = {"note": "KEEP-a", "password": ["x]", 'q"q', "it's", "back\\slash", "toy-secret-1"],
           "nested": {"token": {"a": ["toy-secret-2", {"b": "c ] } )"}]}, "next": "KEEP-b"},
           "client_secret": "toy-secret-3 \\ \" ' ] }", "tail": "KEEP-c"}
    texts = [json.dumps(doc), json.dumps(doc, indent=2), repr(doc), json.dumps({"detail": json.dumps(doc)}),
             json.dumps({"detail": json.dumps({"inner": json.dumps(doc, indent=1)})})]
    for text in texts:
        out = payload.redact(text, None).text
        assert "toy-secret" not in out, out
        assert "KEEP-a" in out and "KEEP-b" in out and "KEEP-c" in out, out


def test_a_replacement_after_the_cut_sets_no_flag():
    from capx_trace_recorder import payload

    text = "x" * 2100 + " password=toy-secret"
    result = payload.redact(text, 2000)
    assert result.text == text[:2000] and result.redacted is False
    at_the_cut = "x" * 1990 + " password=toy-secret"  # the value starts right after the cut
    result = payload.redact(at_the_cut, 2000)
    assert result.text == at_the_cut[:2000] and result.redacted is False
    across = "x" * 1985 + " password=toy-secret-across-the-cut"  # 5 characters of it before the cut
    result = payload.redact(across, 2000)
    assert result.text == "x" * 1985 + " password=***ma" and result.redacted is True


def test_the_whole_text_is_redacted_before_the_cut():
    """A value that ends far past the stored length is read to its end (up to the
    64 KB window), so what follows it is stored as it is."""
    from capx_trace_recorder import payload

    text = 'password="' + "x" * 9000 + '" then the rest of the message'
    assert payload.redact(text, 2000) == payload.Redacted(
        'password="***masked***" then the rest of the message', True, False)
