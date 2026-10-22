"""H8 API wrapping (docs/trace-format.md, sections 2, 5.3 and 7.2).

``combined_doc()`` must stay byte-identical for every ``ApiBase`` subclass
after the recorder's ``install_api``. Instances are built with
``object.__new__`` plus the constructor flags ``functions()`` reads, so no
service or simulator is touched. Wrapped calls must return the same objects
and raise the same exception objects, with or without an open episode.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import itertools
import subprocess
import sys
import types

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.hooks import api as API
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.hooks.api import install_api, uninstall_api
from tracing_testlib import PY312, REPO, clean_env
from trace_reader import by_type, read_events

FLAGS = ["use_sam3", "bimanual", "real", "is_spill_wipe", "is_peg_assembly", "multi_turn", "is_handover"]


def _api_classes():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import capx.integrations  # noqa: F401 - registers every API
    from capx.integrations.base_api import ApiBase

    def subclasses(cls):
        out = set()
        for sub in cls.__subclasses__():
            out.add(sub)
            out |= subclasses(sub)
        return out

    return sorted((c for c in subclasses(ApiBase) if not inspect.isabstract(c)
                   and c.__module__.startswith("capx.")),
                  key=lambda c: f"{c.__module__}.{c.__qualname__}")


@pytest.mark.requires_capx
def test_combined_doc_identical_for_every_api_class():
    classes = _api_classes()
    assert len(classes) >= 19
    if importlib.util.find_spec("libero") is not None:
        names = {c.__qualname__ for c in classes}
        assert {"FrankaLiberoApi", "FrankaLiberoApiReduced", "FrankaLiberoPrivilegedApi",
                "FrankaLiberoApiReducedSkillLibrary"} <= names, sorted(names)
    checked = 0
    for cls in classes:
        checked_for_class = 0
        seen_docs = set()
        for combo in itertools.product([False, True], repeat=len(FLAGS)):
            api = object.__new__(cls)
            for name, value in zip(FLAGS, combo):
                object.__setattr__(api, name, value)
            try:
                before = api.combined_doc()
                functions_before = api.functions()
            except Exception:  # noqa: BLE001 - flag combination the class cannot build
                continue
            if before in seen_docs:
                continue
            seen_docs.add(before)
            per_fn = {n: (str(inspect.signature(f)), inspect.getdoc(f), f.__name__, f.__qualname__)
                      for n, f in functions_before.items()}
            ok, reason, names = install_api(api)
            assert ok, (cls.__qualname__, combo, reason)
            # every public function is wrapped (a partial install would be a gap)
            assert reason is None, (cls.__qualname__, combo, reason)
            assert api.combined_doc() == before, cls.__qualname__
            after = api.functions()
            for n, f in after.items():
                assert isinstance(f, types.MethodType)
                assert getattr(f.__func__, "__capx_trace_wrapper__", False), (cls.__qualname__, n)
                assert (str(inspect.signature(f)), inspect.getdoc(f), f.__name__, f.__qualname__) == per_fn[n]
            ok2, _reason2, names2 = install_api(api)  # idempotent
            assert ok2 and names2 == []
            assert api.combined_doc() == before
            uninstall_api(api)
            assert api.combined_doc() == before
            assert not any(getattr(getattr(v, "__func__", None), "__capx_trace_wrapper__", False)
                           for v in vars(api).values())
            checked += 1
            checked_for_class += 1
        assert checked_for_class > 0, f"no valid configuration checked for {cls.__module__}.{cls.__qualname__}"
    assert checked >= len(classes)


class ToyEnv:
    privileged = False


class ToyApi:
    """Minimal API with the ``functions()`` / ``combined_doc()`` contract."""

    def __init__(self) -> None:
        self._env = ToyEnv()
        self.sentinel = object()
        self.buffer = np.arange(100, dtype=np.float64)

    def functions(self):
        return {"same": self.same, "boom": self.boom, "arr": self.arr, "gen": self.gen,
                "child": self.child}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def same(self, value, api_key=None):
        """Return the argument object itself."""
        return value

    def boom(self, exc):
        """Raise the given exception object."""
        raise exc

    def arr(self):
        """Return a live view of an internal buffer."""
        return self.buffer

    def gen(self):
        """Return a generator that must not be consumed by the recorder."""
        def g():
            yield 1
            yield 2
        return g()

    def child(self):
        """Call another public method through self."""
        return self.same(self.sentinel)


@pytest.fixture()
def episode(work_dir):
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces"))
    ep = C.open_episode(store, run_id="api-run", episode_id="ep", source="test")
    ep.emit("episode_start", data={"tracer": {}})
    C.push_episode(ep)
    yield store, ep
    C.pop_episode(ep)
    C.close_episode(ep)
    store.shutdown(5)


@pytest.mark.parametrize("with_episode", [False, True])
def test_identity_of_results_and_exceptions(request, with_episode):
    if with_episode:
        request.getfixturevalue("episode")
    api = ToyApi()
    ok, _, names = install_api(api)
    assert ok and sorted(names) == ["arr", "boom", "child", "gen", "same"]
    obj = object()
    assert api.same(obj) is obj
    assert api.child() is api.sentinel
    err = ValueError("specific")
    with pytest.raises(ValueError) as info:
        api.boom(err)
    assert info.value is err
    kb = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as info2:
        api.boom(kb)
    assert info2.value is kb
    alarm = TimeoutError("Trial 1 exceeded 10 seconds")
    with pytest.raises(TimeoutError) as info3:
        api.boom(alarm)
    assert info3.value is alarm
    g = api.gen()
    assert list(g) == [1, 2]  # not consumed by the recorder
    uninstall_api(api)


def test_recorded_calls_copy_arrays_and_mask_secrets(episode):
    store, ep = episode
    api = ToyApi()
    assert install_api(api)[0]
    out = api.arr()
    out[:] = -1.0  # mutating the returned array afterwards must not change the record
    api.same("value", api_key="sk-SHOULD-NOT-APPEAR")
    api.child()
    with pytest.raises(RuntimeError):
        api.boom(RuntimeError("expected failure"))
    assert store.flush(20)
    events, corrupt, torn = read_events(ep.events_path)
    assert corrupt == 0 and not torn
    arr_end = [e for e in by_type(events, "api_call_end") if e["data"]["name"] == "arr"][0]
    saved = np.load(ep.base_dir / arr_end["data"]["return"]["path"])
    assert np.array_equal(saved, np.arange(100, dtype=np.float64))
    assert arr_end["data"]["return"]["stats"] == {"min": 0.0, "max": 99.0, "mean": 49.5,
                                                   "n_finite": 100, "n_nonfinite": 0}
    same_start = [e for e in by_type(events, "api_call_start") if e["data"]["name"] == "same"][0]
    assert same_start["data"]["args"] == {"value": "value", "api_key": "***masked***"}
    text = ep.events_path.read_text()
    assert "sk-SHOULD-NOT-APPEAR" not in text
    child = [e for e in by_type(events, "api_call_start") if e["data"]["name"] == "child"][0]
    nested = [e for e in by_type(events, "api_call_start")
              if e["data"]["depth"] == 1 and e["parent_event_id"] == child["event_id"]]
    assert len(nested) == 1 and nested[0]["data"]["name"] == "same"
    error = by_type(events, "api_call_error")[0]["data"]
    assert error["error_type"] == "RuntimeError" and error["message"] == "expected failure"
    assert "capx_trace_recorder" not in (error["traceback"] or "")
    uninstall_api(api)


def test_api_error_keeps_original_type_when_raw_frames_fails(episode, monkeypatch):
    store, ep = episode
    api = ToyApi()
    assert install_api(api)[0]

    def fail_frames(_exc):
        raise RuntimeError("frame extraction failed")

    monkeypatch.setattr(API, "raw_frames", fail_frames)
    original = ValueError("original failure")
    try:
        with pytest.raises(ValueError) as info:
            api.boom(original)
        assert info.value is original
        assert store.flush(20)
        events, corrupt, torn = read_events(ep.events_path)
        assert corrupt == 0 and not torn
        errors = by_type(events, "api_call_error")
        assert len(errors) == 1 and errors[0]["data"]["error_type"] == "ValueError"
    finally:
        uninstall_api(api)


def test_install_rolls_back_when_combined_doc_changes():
    class Changing(ToyApi):
        calls = 0

        def combined_doc(self) -> str:
            Changing.calls += 1
            return f"doc {Changing.calls}"

    api = Changing()
    ok, reason, names = install_api(api)
    assert not ok and "combined_doc" in reason and names == []
    assert not any(isinstance(v, types.MethodType) for v in vars(api).values())


@pytest.mark.skipif(sys.version_info >= (3, 12) or PY312 is None or not PY312.exists(),
                    reason="set CAPX_PYTHON312 to a CaP-X Python 3.12 environment")
def test_combined_doc_on_python312_including_libero():
    proc = subprocess.run(
        [str(PY312), "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(REPO / "tests/recorder/test_api_wrap.py"), "-k", "not python312"],
        cwd=str(REPO), env=clean_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "passed" in proc.stdout


def test_ambiguous_episode_is_flagged_not_misattributed(work_dir, monkeypatch):
    """Two open episodes and no way to tell them apart: the call is not recorded,
    every candidate gets an ambiguous_episode gap per call (so neither shows as
    complete), and one warning per (hook, run, candidates) is written."""
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces"))
    monkeypatch.setattr(S, "_STORE", store)
    monkeypatch.setattr(C, "_AMBIGUOUS_NOTED", set())
    eps = []
    for name in ("a", "b"):
        ep = C.open_episode(store, run_id="amb-run", episode_id=name, source="test")
        ep.emit("episode_start", data={"tracer": {}})
        eps.append(ep)
    api = ToyApi()
    assert install_api(api)[0]
    try:
        assert api.same(1) == 1 and api.same(2) == 2
    finally:
        uninstall_api(api)
        for ep in eps:
            with ep._lock:
                ep.emit("episode_end", data={"end_reason": "completed"})
                ep.ended = True
            C.close_episode(ep)
    assert store.flush(20)
    for ep in eps:
        events, _, _ = read_events(ep.events_path)
        assert [e["event_type"] for e in events] == ["episode_start", "trace_warning", "episode_end"]
        warning = events[1]["data"]
        assert warning["code"] == "ambiguous_episode" and warning["hook"] == "H8"
        assert warning["counts_as_gap"] is False and warning["candidates"] == ["a", "b"]
        end = events[-1]["data"]
        assert end["trace_complete"] is False
        assert end["gaps"] == [{"code": "ambiguous_episode", "count": 2, "first_seq": None}]
    unscoped = list((work_dir / "traces" / "amb-run").glob("unscoped.*.jsonl"))
    events, _, _ = read_events(unscoped[0])
    assert [e["data"]["code"] for e in events] == ["ambiguous_episode"]
    assert sorted(events[0]["data"]["candidates"]) == ["a", "b"]
    store.shutdown(5)
