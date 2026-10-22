"""Simulator clock and ground-truth state (docs/trace-format.md, sections 8.3-8.4).

Everything here reads data without side effects: plain attributes, the MuJoCo
``sim.data`` arrays and LIBERO's ``get_current_time_s()`` (a pure computation).
Nothing calls ``step``, ``reset``, ``get_observation``, ``render`` or any video
method. State snapshots are copied on the calling thread (small arrays) and
serialised to ``json.gz`` by the writer. Every handler here re-raises
``TimeoutError`` (the trial alarm, section 2).

Body names are cached per MuJoCo model object through a weak reference and
checked against ``model.nbody`` on every use: robosuite's ``hard_reset`` frees
the model at each reset, and a new model of another task can get the freed
address, so a cache keyed by ``id(model)`` alone would hand out another task's
names (or none).

Related: ``capx_trace_recorder/frames.py`` (buffer frames), ``capx_trace_recorder/hooks/api.py``
and ``capx_trace_recorder/hooks/env.py`` (callers).
"""

from __future__ import annotations

import sys
import threading
import weakref
from typing import Any

from capx_trace_recorder.timeout import is_trial_alarm


def _attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception as exc:  # noqa: BLE001 - a property that raises
        if is_trial_alarm(exc):
            raise
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def mujoco_sim(low_env: Any) -> Any:
    """The MuJoCo ``sim`` of robosuite (``robosuite_env.sim``) or LIBERO (``handle.env.sim``)."""
    if low_env is None:
        return None
    rs = _attr(low_env, "robosuite_env")
    sim = _attr(rs, "sim") if rs is not None else None
    if sim is None:
        handle = _attr(low_env, "handle")
        env = _attr(handle, "env") if handle is not None else None
        sim = _attr(env, "sim") if env is not None else None
    if sim is None:
        return None
    data = _attr(sim, "data")
    model = _attr(sim, "model")
    if data is None or model is None:
        return None
    return sim


def sim_clock(low_env: Any) -> tuple[int | None, float | None]:
    """``(sim_step, sim_time)``; first match wins (section 8.3)."""
    if low_env is None:
        return None, None
    step = _as_int(_attr(low_env, "_sim_step_count"))
    if step is None:
        rs = _attr(low_env, "robosuite_env")
        if rs is not None:
            step = _as_int(_attr(rs, "timestep"))
    if step is None:
        env = _attr(low_env, "env")
        if env is not None:
            step = _as_int(_attr(env, "episode_steps"))
    sim_time: float | None = None
    sim = mujoco_sim(low_env)
    if sim is not None:
        data = _attr(sim, "data")
        sim_time = _as_float(_attr(data, "time"))
    if sim_time is None:
        getter = _attr(low_env, "get_current_time_s")
        if callable(getter) and type(low_env).__module__.startswith("capx.envs.simulators.libero"):
            try:
                sim_time = _as_float(getter())
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                sim_time = None
    return step, sim_time


class StateSnapshot:
    """Arrays copied at call time; the writer turns them into json.gz."""

    __slots__ = ("sim_time", "qpos", "body_names", "xpos", "xquat")

    def __init__(self, sim_time: float | None, qpos: Any, body_names: tuple[str, ...], xpos: Any,
                 xquat: Any) -> None:
        self.sim_time = sim_time
        self.qpos = qpos
        self.body_names = body_names
        self.xpos = xpos
        self.xquat = xquat

    @property
    def nbytes(self) -> int:
        return int(getattr(self.qpos, "nbytes", 0) + getattr(self.xpos, "nbytes", 0)
                   + getattr(self.xquat, "nbytes", 0))

    def to_json(self) -> dict[str, Any]:
        from capx_trace_recorder.serialize import float_value

        bodies = {}
        for i, name in enumerate(self.body_names):
            if not name:
                continue
            bodies[name] = {
                "xpos": [float_value(float(v)) for v in self.xpos[i]],
                "xquat": [float_value(float(v)) for v in self.xquat[i]],
            }
        return {
            "sim_time": self.sim_time,
            "qpos": [float_value(float(v)) for v in self.qpos],
            "bodies": bodies,
            "quaternion_order": "wxyz (MuJoCo xquat)",
            "frame": "MuJoCo world",
        }


# id(model) -> (weak reference to that model or None, nbody, names)
_BODY_NAMES: dict[int, tuple[Any, int, tuple[str, ...]]] = {}
_BODY_NAMES_LOCK = threading.Lock()


def _read_body_names(model: Any, nbody: int) -> tuple[str, ...]:
    out = []
    id2name = _attr(model, "body_id2name")
    for i in range(nbody):
        name = None
        if callable(id2name):
            try:
                name = id2name(i)
            except Exception as exc:  # noqa: BLE001
                if is_trial_alarm(exc):
                    raise
                name = None
        out.append(str(name) if name else "")
    return tuple(out)


def _body_names(sim: Any, model: Any) -> tuple[str, ...]:
    """Names of all bodies of ``model``, cached per model object (not per address)."""
    nbody = _as_int(_attr(model, "nbody")) or 0
    key = id(model)
    with _BODY_NAMES_LOCK:
        entry = _BODY_NAMES.get(key)
    if entry is not None:
        ref, cached_nbody, names = entry
        if ref is not None and ref() is model and cached_nbody == nbody and len(names) == nbody:
            return names
    names = _read_body_names(model, nbody)
    try:
        ref = weakref.ref(model)
    except TypeError:  # not weak-referenceable: never cached
        return names
    with _BODY_NAMES_LOCK:
        if len(_BODY_NAMES) > 64:
            for k in [k for k, (r, _n, _names) in _BODY_NAMES.items() if r() is None]:
                del _BODY_NAMES[k]
            if len(_BODY_NAMES) > 64:
                _BODY_NAMES.clear()
        _BODY_NAMES[key] = (ref, nbody, names)
    return names


def capture_state(low_env: Any) -> tuple[StateSnapshot | None, str | None]:
    """Copy ``sim.data`` time, ``qpos``, ``xpos``, ``xquat``; or give a reason."""
    np = sys.modules.get("numpy")
    sim = mujoco_sim(low_env)
    if sim is None or np is None:
        return None, "unsupported_simulator"
    try:
        data = sim.data
        model = sim.model
        names = _body_names(sim, model)
        qpos = np.array(data.qpos, dtype=np.float64, copy=True)
        xpos = np.array(data.xpos, dtype=np.float64, copy=True)
        xquat = np.array(data.xquat, dtype=np.float64, copy=True)
        if xpos.ndim != 2 or xquat.ndim != 2 or xpos.shape[0] != len(names):
            return None, "unexpected_state_shape"
        return StateSnapshot(_as_float(_attr(data, "time")), qpos, names, xpos, xquat), None
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None, "state_read_failed"


__all__ = ["sim_clock", "capture_state", "StateSnapshot", "mujoco_sim"]
