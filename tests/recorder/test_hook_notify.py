"""Hooks installed or failed while an episode is open (docs/trace-format.md, sections 4.4 and 7.1).

The web server imports ``capx.envs.trial`` and
the env classes inside the web episode, so H2, H4 and H7 are installed after
``episode_start`` listed the hooks. The install must reach ``episode.json``,
and a failure during the episode must make the trace incomplete.
"""

from __future__ import annotations

import json
import types

import pytest

from capx_trace_recorder import bootstrap
from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder.config import TraceConfig
from trace_reader import by_type, read_events


@pytest.fixture()
def open_episode(work_dir):
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces"))
    ep = C.open_episode(store, run_id="hook-run", episode_id="ep", source="web_ui")
    ep.emit("episode_start", data={"tracer": {"hooks_installed": [], "hooks_failed": []}})
    saved = (list(C.REGISTRY.hooks_installed), list(C.REGISTRY.hooks_failed))
    yield store, ep
    C.REGISTRY.hooks_installed[:] = saved[0]
    C.REGISTRY.hooks_failed[:] = saved[1]
    C.close_episode(ep)
    store.shutdown(5)


def test_install_during_episode_reaches_episode_json(open_episode):
    store, ep = open_episode
    bootstrap._record_installed("H7", "test.CodeExecutionEnvBase")
    bootstrap._record_installed("H7", "test.CodeExecutionEnvBase")  # recorded once
    assert store.flush(20)
    doc = json.loads((ep.base_dir / "episode.json").read_text())
    assert doc["hooks_installed_during_episode"] == ["H7:test.CodeExecutionEnvBase"]
    ep.emit("episode_end", data={"end_reason": "completed"})
    ep.ended = True
    assert store.flush(20)
    doc = json.loads((ep.base_dir / "episode.json").read_text())
    assert doc["trace_complete"] is True
    assert doc["hooks_installed_during_episode"] == ["H7:test.CodeExecutionEnvBase"]


def test_failure_during_episode_makes_the_trace_incomplete(open_episode):
    store, ep = open_episode
    bootstrap._record_failed("H4", "test._run_single_trial", "unexpected signature")
    ep.emit("episode_end", data={"end_reason": "completed"})
    ep.ended = True
    assert store.flush(20)
    events, _corrupt, _torn = read_events(ep.events_path)
    warning = by_type(events, "trace_warning")[0]["data"]
    assert warning["code"] == "hook_install_failed" and warning["during_episode"] is True
    assert warning["hook"] == "H4"
    doc = json.loads((ep.base_dir / "episode.json").read_text())
    assert doc["trace_complete"] is False
    assert [g["code"] for g in doc["gaps"]] == ["hook_install_failed"]


def test_optional_missing_symbol_is_listed_during_episode(open_episode):
    store, ep = open_episode
    bootstrap._record_problem("H2", "capx.envs.trial", "_patch_libero_goal: not found",
                              types.ModuleType("trial"))
    ep.emit("episode_end", data={"end_reason": "completed"})
    ep.ended = True
    assert store.flush(20)
    doc = json.loads((ep.base_dir / "episode.json").read_text())
    assert doc["hooks_skipped_during_episode"][-1]["reason"] == "_patch_libero_goal: not found"
    assert doc["trace_complete"] is True
