"""Summaries, frames and the state probe (docs/trace-format.md, sections 5.3 and 8)."""

from __future__ import annotations

import dataclasses
import ast
import json
import warnings
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import pytest

from capx_trace_recorder import context as C
from capx_trace_recorder import store as S
from capx_trace_recorder import serialize as SER
from capx_trace_recorder.config import TraceConfig
from capx_trace_recorder.frames import install_tap
from capx_trace_recorder.serialize import ArrayBudget, PendingArray, array_stats, summarize, summarize_arguments
from capx_trace_recorder.state_probe import capture_state, sim_clock
from trace_reader import by_type, read_events


@pytest.mark.requires_capx
def test_unit_table_names_exist_in_upstream_capx():
    import capx

    package = Path(capx.__file__).parent
    class_names = {
        node.name
        for path in package.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert {name for name, _ in SER.UNITS} <= class_names


class Explosive:
    """Any access beyond the type would be a bug in summarize()."""

    def __repr__(self):  # pragma: no cover - must never run
        raise AssertionError("repr called")

    def __getattr__(self, name):  # pragma: no cover - must never run
        raise AssertionError(f"attribute {name} read")

    def __len__(self):  # pragma: no cover - must never run
        raise AssertionError("len called")


class Weird(str):
    def __str__(self):  # pragma: no cover - must never run
        raise AssertionError("__str__ override called")


def test_summaries_never_evaluate_objects():
    def gen():
        yield 1
        raise AssertionError("generator consumed")

    g = gen()
    pending: list = []
    out = summarize({"g": g, "x": Explosive(), "it": iter([1, 2]), "mod": np, "s": Weird("ok")}, pending)
    assert out["items"]["g"] == {"type": "generator"}
    assert out["items"]["x"]["type"].endswith("Explosive")
    assert out["items"]["mod"] == {"type": "module"}
    assert out["items"]["s"] == "ok"
    assert next(g) == 1  # the generator was not advanced


def test_summary_shapes_and_limits():
    pending: list = []
    Point = namedtuple("Point", "x y")

    @dataclasses.dataclass
    class Pose:
        position: np.ndarray
        token: str

    value = {
        "nan": float("nan"), "inf": float("-inf"), "np": np.float32(1.5), "b": np.bool_(True),
        "small": np.arange(4.0), "long": "y" * 5000, "nested": [[[[[1]]]]],
        "many": list(range(120)), "od": OrderedDict(a=1), "pt": Point(1, 2),
        "pose": Pose(np.zeros(3), "secret-value"), "api_key": "sk-x",
    }
    out = summarize(value, pending)["items"]
    assert out["nan"] == "nan" and out["inf"] == "-inf" and out["np"] == 1.5 and out["b"] is True
    assert out["small"] == {"type": "ndarray", "shape": [4], "dtype": "float64", "values": [0.0, 1.0, 2.0, 3.0]}
    assert out["long"]["truncated"] is True and out["long"]["length"] == 5000
    # the value itself is depth 0; containers deeper than 4 levels are cut
    assert out["nested"]["items"][0]["items"][0]["items"][0] == {"type": "list", "truncated": "depth"}
    assert out["many"]["truncated"] == 70 and len(out["many"]["items"]) == 50
    assert out["od"]["items"] == {"a": 1}
    assert out["pt"]["fields"] == {"x": 1, "y": 2}
    assert out["pose"]["fields"]["token"] == "***masked***"
    assert out["api_key"] == "***masked***"
    args = summarize_arguments({"password": "p", "max_tokens": 5}, pending)
    assert args == {"password": "***masked***", "max_tokens": 5}


def test_array_copies_budget_and_stats():
    budget = ArrayBudget(1000, None)
    pending: list = []
    big = np.arange(200, dtype=np.float64)       # 1600 bytes: over the call budget
    ok = np.arange(100, dtype=np.float32)        # 400 bytes: copied
    out = summarize([ok, big, np.array([object()] * 70, dtype=object)], pending, budget)["items"]
    assert "path" not in out[0] and out[0]["dtype"] == "float32"
    copies = [p for p in pending if isinstance(p, PendingArray)]
    assert len(copies) == 1
    ok[:] = -1  # mutating after the call leaves the copy unchanged
    assert copies[0].array[5] == 5
    assert out[1]["not_saved"] == "array_budget" and len(out[1]["sha256"]) == 64
    assert out[2]["values_omitted"] == "object_or_structured_dtype"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert array_stats(np.array([np.nan, np.nan])) == {"min": None, "max": None, "mean": None,
                                                           "n_finite": 0, "n_nonfinite": 2}
        assert array_stats(np.array([1e308, 1e308]))["mean"] == "inf"
        assert array_stats(np.zeros((0, 3))) == {"min": None, "max": None, "mean": None,
                                                 "n_finite": 0, "n_nonfinite": 0}
        assert array_stats(np.array([True, False])) == {"min": False, "max": True, "mean": 0.5,
                                                        "n_finite": 2, "n_nonfinite": 0}
        assert array_stats(np.array(["a"])) is None


class ListLow:
    """Low-level env with a list frame buffer and a step counter."""

    save_camera_name = "cam"

    def __init__(self, dict_frames: bool = False) -> None:
        self._frame_buffer = []
        self._sim_step_count = 0
        self.dict_frames = dict_frames

    def _record_frame(self):
        img = np.full((8, 8, 3), self._sim_step_count % 255, dtype=np.uint8)
        self._frame_buffer.append({"rgb": img, "ego": img[::-1].copy()} if self.dict_frames else img)


@pytest.fixture()
def recorder(work_dir):
    store = S.TraceStore(TraceConfig(trace_dir=work_dir / "traces", frame_fps=1000.0))
    ep = C.open_episode(store, run_id="frames", episode_id="ep", source="test")
    ep.emit("episode_start", data={"tracer": {}})
    yield store, ep
    C.close_episode(ep)
    store.shutdown(5)


def test_frame_tap_epochs_backfill_and_keyframes(recorder):
    store, ep = recorder
    low = ListLow()
    low._frame_buffer.append(np.zeros((8, 8, 3), np.uint8))  # before the episode: sim_step unknown
    ep.low_env = low
    tap, reason = install_tap(low)
    assert reason is None and install_tap(low)[0] is tap
    tap.attach(ep)
    ep.frame_tap = tap
    for step in range(1, 4):
        low._sim_step_count = step
        low._record_frame()
    ref, refs = tap.keyframe("ev_x")
    assert ref["epoch"] == 0 and ref["buffer_index"] == 3 and ref["frame_sim_step"] == 3
    assert tap.buffers[0].steps == [None, 1, 2, 3]
    low._frame_buffer.clear()  # cleared outside _record_frame: a new epoch
    low._sim_step_count = 10
    low._record_frame()
    ref2, _ = tap.keyframe("ev_y")
    assert ref2["epoch"] == 1 and ref2["buffer_index"] == 0 and ref2["frame_sim_step"] == 10
    low._frame_buffer = []  # replaced list object: another epoch
    low._record_frame()
    assert tap.buffer_lengths() == {"cam": (2, 1)}
    assert store.flush(20)
    events, _, _ = read_events(ep.events_path)
    frames = by_type(events, "frame")
    assert {(e["data"]["epoch"], e["data"]["buffer_index"]) for e in frames} >= {(0, 3), (1, 0)}
    for e in frames:
        assert (ep.base_dir / e["artifacts"][0]["path"]).is_file()


def test_r1pro_dict_frames_are_saved_per_camera(recorder):
    store, ep = recorder
    low = ListLow(dict_frames=True)
    ep.low_env = low
    tap, _ = install_tap(low)
    tap.attach(ep)
    low._sim_step_count = 5
    low._record_frame()
    ref, refs = tap.keyframe(None)
    assert sorted(r["camera"] for r in refs) == ["ego", "rgb"]
    assert ref["path"].startswith("frames/")
    assert store.flush(20)
    assert (ep.base_dir / "frames/rgb/e0_0.jpg").is_file() and (ep.base_dir / "frames/ego/e0_0.jpg").is_file()


class FakeModel:
    nbody = 3

    def body_id2name(self, i):
        return ["world", "cube", ""][i]


class FakeData:
    def __init__(self):
        self.time = 1.25
        self.qpos = np.array([0.1, 0.2])
        self.xpos = np.arange(9.0).reshape(3, 3)
        self.xquat = np.tile([1.0, 0, 0, 0], (3, 1))


class FakeSim:
    def __init__(self):
        self.data = FakeData()
        self.model = FakeModel()


class RobosuiteLike:
    def __init__(self):
        self._sim_step_count = 7
        self.robosuite_env = type("R", (), {"sim": FakeSim(), "timestep": 99})()


def test_state_probe_reads_without_side_effects():
    low = RobosuiteLike()
    assert sim_clock(low) == (7, 1.25)
    snapshot, reason = capture_state(low)
    assert reason is None
    low.robosuite_env.sim.data.xpos[:] = -1  # copies were taken
    doc = snapshot.to_json()
    assert doc["sim_time"] == 1.25 and doc["qpos"] == [0.1, 0.2]
    assert set(doc["bodies"]) == {"world", "cube"}
    assert doc["bodies"]["cube"]["xpos"] == [3.0, 4.0, 5.0]
    json.dumps(doc)
    assert capture_state(object()) == (None, "unsupported_simulator")
    assert sim_clock(None) == (None, None)
