"""Partial hook installs and the ground-truth state probe (docs/trace-format.md, sections 4.4, 7.1 and 8.4).

* A public API function that H8 cannot wrap, or a service closure that H9
  cannot replace, is recorded as ``hook_install_failed`` (a gap): its calls are
  not recorded, so the trace must not look complete.
* Body names of a MuJoCo model are cached per model object, never per address:
  robosuite frees the model at every reset (``hard_reset``) and a model of the
  next task can get the freed address. A state snapshot that could not be taken
  (other than on an unsupported simulator) is a ``state_unavailable`` gap.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from capx_trace_recorder import bootstrap
from capx_trace_recorder import state_probe as SP
from capx_trace_recorder.hooks import api as API
from capx_trace_recorder.hooks import env as ENV
from capx_trace_recorder.hooks.api import uninstall_api
from inprocess import Recorder, by_type


@pytest.fixture()
def rec(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "_H11_OK", True)  # H8/H9 are installed only after H11
    r = Recorder(tmp_path / "traces", monkeypatch)
    yield r
    r.close()


# ---------------------------------------------------------------------------
# Partial installs
# ---------------------------------------------------------------------------


def _loose(value):
    """A function stored in the table, not a method of the instance."""
    return value


class PartialApi:
    def __init__(self) -> None:
        self._env = None

    def functions(self):
        return {"fine": self.fine, "loose": _loose}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def fine(self):
        """Wrapped normally."""
        return 1


class LockedServiceApi:
    """An API whose service closure attribute cannot be replaced."""

    def __init__(self) -> None:
        object.__setattr__(self, "_env", None)
        object.__setattr__(self, "sam3_seg_fn", lambda image, text: [])

    def __setattr__(self, name, value):
        if name.endswith("_fn"):
            raise AttributeError(f"{name} is read-only")
        object.__setattr__(self, name, value)

    def functions(self):
        return {}

    def combined_doc(self) -> str:
        return ""


def test_partial_installs_are_gaps(rec):
    partial, locked = PartialApi(), LockedServiceApi()
    env = types.SimpleNamespace(_apis={"PartialApi": partial, "LockedServiceApi": locked})
    ep = rec.open()
    try:
        ENV._install_apis(ep, env)
        assert partial.fine() == 1 and "fine" in partial.__dict__  # the rest is wrapped
    finally:
        uninstall_api(partial)
    rec.end(ep)
    events = rec.events(ep)
    warnings = [(w["data"]["hook"], w["data"]["target"], w["data"]["message"])
                for w in by_type(events, "trace_warning") if w["data"]["code"] == "hook_install_failed"]
    assert [(h, t) for h, t, _m in warnings] == [("H8", "PartialApi"), ("H9", "LockedServiceApi")]
    assert "loose" in warnings[0][2] and "sam3_seg_fn" in warnings[1][2]
    doc = rec.summary(ep)
    assert doc["trace_complete"] is False
    assert {g["code"]: g["count"] for g in doc["gaps"]} == {"hook_install_failed": 2}
    entry = {e["api"]: e for e in doc["apis_wrapped"]}
    assert entry["PartialApi"]["wrapped"] == ["fine"] and "loose" in entry["PartialApi"]["skipped"]


def test_a_complete_install_is_not_a_gap(rec):
    api = PartialApi()
    api.functions = lambda: {"fine": api.fine}  # type: ignore[method-assign]
    env = types.SimpleNamespace(_apis={"PartialApi": api})
    ep = rec.open()
    try:
        ENV._install_apis(ep, env)
    finally:
        uninstall_api(api)
    rec.end(ep)
    assert rec.summary(ep)["gaps"] == []


# ---------------------------------------------------------------------------
# State probe
# ---------------------------------------------------------------------------


class Model:
    def __init__(self, names) -> None:
        self.names = list(names)
        self.nbody = len(self.names)

    def body_id2name(self, i):
        return self.names[i]


class Data:
    def __init__(self, nbody: int) -> None:
        self.time = 0.5
        self.qpos = np.zeros(2)
        self.xpos = np.arange(nbody * 3, dtype=np.float64).reshape(nbody, 3)
        self.xquat = np.tile([1.0, 0.0, 0.0, 0.0], (nbody, 1))


def _low(model):
    sim = types.SimpleNamespace(model=model, data=Data(model.nbody))
    return types.SimpleNamespace(robosuite_env=types.SimpleNamespace(sim=sim, timestep=0))


def _names(low):
    snapshot, reason = SP.capture_state(low)
    assert reason is None, reason
    return list(snapshot.to_json()["bodies"])


def test_body_names_never_come_from_a_model_at_the_same_address():
    task_a = Model(["world", "cube_A", "table_A"])
    task_b = Model(["world", "mug_B", "bin_B"])
    assert _names(_low(task_a)) == ["world", "cube_A", "table_A"]
    # what an address reuse after robosuite's hard_reset looks like to an id() cache
    SP._BODY_NAMES[id(task_b)] = SP._BODY_NAMES[id(task_a)]
    assert _names(_low(task_b)) == ["world", "mug_B", "bin_B"]
    # the same model object with another body count is read again
    task_b.names.append("lid_B")
    task_b.nbody = 4
    assert _names(_low(task_b)) == ["world", "mug_B", "bin_B", "lid_B"]


def test_body_names_after_a_real_address_reuse():
    names_b = ["world", "mug_B", "bin_B"]
    for _ in range(500):
        a = Model(["world", "cube_A", "table_A"])
        assert _names(_low(a)) == ["world", "cube_A", "table_A"]
        freed = id(a)
        del a
        # the instance block is freed last, so the next instance usually gets it
        b = Model(names_b)
        if id(b) == freed:
            break
        del b
    else:
        pytest.skip("the freed address was not reused")
    assert _names(_low(b)) == ["world", "mug_B", "bin_B"]


class SlotModel:
    """A model object that cannot be weakly referenced: its names are never cached."""

    __slots__ = ("names", "nbody")

    def __init__(self, names) -> None:
        self.names = list(names)
        self.nbody = len(self.names)

    def body_id2name(self, i):
        return self.names[i]


def test_models_without_weak_references_are_read_every_time():
    model = SlotModel(["world", "a"])
    assert _names(_low(model)) == ["world", "a"]
    entry = SP._BODY_NAMES.get(id(model))  # a dead entry of an earlier model at this address may remain
    assert entry is None or entry[0]() is not model
    model.names[1] = "b"
    assert _names(_low(model)) == ["world", "b"]


@pytest.mark.parametrize("reason,gap", [("unexpected_state_shape", True), ("state_read_failed", True),
                                        ("unsupported_simulator", False)])
def test_missing_state_is_a_gap_except_on_unsupported_simulators(rec, reason, gap):
    ep = rec.open()
    ep.emit("code_block_end", data={"code_block_id": "ev_x"},
            builder=lambda ctx: API.state_ref_builder(ctx, reason))
    rec.end(ep)
    block = by_type(rec.events(ep), "code_block_end")[0]["data"]
    assert block["state_ref"] == {"path": None, "reason": reason}
    codes = [g["code"] for g in rec.summary(ep)["gaps"]]
    assert codes == (["state_unavailable"] if gap else [])


# ---------------------------------------------------------------------------
# Hook failures without an episode
# ---------------------------------------------------------------------------


def test_hook_failures_reach_the_run_log_of_the_runner(rec, monkeypatch):
    """An H4 failure opens no episode; in the runner it is still written (unscoped log),
    and model requests are then not labelled preflight (trial requests look the same)."""
    from capx_trace_recorder import context as C
    from capx_trace_recorder.hooks import llm as LLM

    monkeypatch.setattr(C.REGISTRY, "hooks_failed", [])
    monkeypatch.setattr(C.REGISTRY, "failures_logged", 0)
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", None)
    assert C.REGISTRY.open_episodes() == []
    bootstrap._record_failed("H4", "capx.envs.trial._run_single_trial", "unexpected signature ['a']")
    assert rec.store._writer is None  # no run known yet: nothing is queued, no thread started
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    bootstrap.log_failures_to_run()  # what H15 does when it learns the run id
    bootstrap._record_failed("H7", "capx.envs.tasks.base", "failed: api_key=sk-AAAAAAAAAAAAAAAAAAAAAAAA")
    warnings = [e["data"] for e in rec.unscoped("run") if e["event_type"] == "trace_warning"]
    assert [(w["code"], w["hook"]) for w in warnings] == [("hook_install_failed", "H4"),
                                                         ("hook_install_failed", "H7")]
    assert "sk-AAAA" not in warnings[1]["message"] and warnings[1]["run_level"] is True
    lc = LLM.start_logical(types.SimpleNamespace(model="m"), [], "query_model")
    assert lc.role == "unknown"
    monkeypatch.setattr(C.REGISTRY, "hooks_failed", [])
    assert LLM.start_logical(types.SimpleNamespace(model="m"), [], "query_model").role == "preflight"


def test_a_recorder_error_without_an_episode_reaches_the_run_log(rec, monkeypatch):
    """A failing _episode_start leaves the trial untraced; the runner's run log says so."""
    from capx_trace_recorder import context as C
    from capx_trace_recorder.hooks import common as COMMON

    monkeypatch.setattr(COMMON, "_NOTED_OUTSIDE_EPISODES", set())
    monkeypatch.setattr(C.REGISTRY.runner, "run_id", "run")
    assert C.REGISTRY.open_episodes() == []

    def _episode_start():
        raise RuntimeError("bug with password=pw-in-bug")

    assert COMMON.guard(_episode_start) is None
    assert COMMON.guard(_episode_start) is None  # recorded once per place
    warnings = [e["data"] for e in rec.unscoped("run") if e["event_type"] == "trace_warning"]
    assert [(w["code"], w["outside_episode"]) for w in warnings] == [("internal_error", True)]
    assert "_episode_start" in warnings[0]["message"] and "pw-in-bug" not in warnings[0]["message"]
