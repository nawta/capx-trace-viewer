"""In-process recorder fixtures for unit tests (tests/recorder).

Each test gets its own ``TraceStore`` installed as the process singleton for the
duration of the test, and everything the recorder keeps per process or per
thread (open episodes, env bindings, web sessions, attempt counters, this
thread's context stacks and pins, the ambiguity de-dup set) starts empty and is
reset afterwards, so tests cannot influence each other through
``capx_trace_recorder.context.REGISTRY``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from trace_reader import read_events


class Recorder:
    def __init__(self, root: Path, monkeypatch: Any, **config: Any) -> None:
        self.root = root
        self.store = S.TraceStore(TraceConfig(trace_dir=root, **config))
        monkeypatch.setattr(S, "_STORE", self.store)
        monkeypatch.setattr(C, "_AMBIGUOUS_NOTED", set())
        # episode ids already used by earlier tests of this process would get a
        # __dup<n> suffix here; every test starts with its own set and logs
        monkeypatch.setattr(C.REGISTRY, "used_ids", set())
        monkeypatch.setattr(C.REGISTRY, "unscoped", {})
        # an episode an earlier test left open (a bug that test reports) must not
        # change how this test's events are attributed
        monkeypatch.setattr(C.REGISTRY, "open", {})
        monkeypatch.setattr(C.REGISTRY, "env_bind", {})
        monkeypatch.setattr(C.REGISTRY, "web_sessions", {})
        monkeypatch.setattr(C.REGISTRY, "attempts", {})
        self.episodes: list[Any] = []

    def open(self, name: str = "ep", run_id: str = "run", source: str = "test", start: bool = True) -> Any:
        ep = C.open_episode(self.store, run_id=run_id, episode_id=name, source=source)
        self.episodes.append(ep)
        if start:
            ep.emit("episode_start", status="started", data={"source": source, "task": "t", "tracer": {}})
        return ep

    def end(self, ep: Any, close: bool = True) -> None:
        with ep._lock:
            ep.emit("episode_end", data={"end_reason": "completed"})
            ep.ended = True
        if close:
            C.close_episode(ep)

    def flush(self) -> None:
        assert self.store.flush(30)

    def events(self, ep: Any) -> list[dict[str, Any]]:
        self.flush()
        events, corrupt, torn = read_events(ep.events_path)
        assert corrupt == 0 and not torn
        return events

    def summary(self, ep: Any) -> dict[str, Any]:
        self.flush()
        return json.loads((ep.base_dir / "episode.json").read_text())

    def unscoped(self, run_id: str = "run") -> list[dict[str, Any]]:
        self.flush()
        out: list[dict[str, Any]] = []
        for path in sorted((self.root / run_id).glob("unscoped.*.jsonl")):
            out += read_events(path)[0]
        return out

    def close(self) -> None:
        for ep in self.episodes:
            C.close_episode(ep)
            C.REGISTRY.unbind_episode(ep)
        C.clear_stale()
        pins = getattr(C._THREAD, "env_pins", None)
        if pins:
            pins.clear()
        self.store.shutdown(5)


def by_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


def warning_codes(events: list[dict[str, Any]]) -> list[str]:
    return [e["data"]["code"] for e in events if e["event_type"] == "trace_warning"]


def thread_stacks() -> dict[str, list[Any]]:
    return {name: list(C._stack(name)) for name in ("frames", "roles", "episodes")}
