"""Frame tap on the video buffer, key and periodic frames (docs/trace-format.md, section 8).

H10 wraps the low-level env's ``_record_frame`` as an *instance* attribute. The
original is called exactly once per call; afterwards the tap maps every new
buffer index to the simulator step at which it was appended, and saves a
periodic frame at most ``CAPX_TRACE_FRAME_FPS`` times per wall-clock second.
Frames already in the buffer when the tap is attached are back-filled with
``sim_step: null``. The ``epoch`` of a buffer increments when it is cleared or
replaced (its length drops, or it is a new list object).

Key frames (``keyframe()``) copy the latest buffer frame at code block and
depth-0 API boundaries. Frames are read from the existing buffers by index and
copied; the recorder never renders, never toggles recording and never calls
``get_video_frames*`` (R1Pro's version records a frame through
``get_observation()``). R1Pro buffer entries are dicts of camera arrays.

Related: ``capx_trace_recorder/hooks/trial.py`` (installs H10 at episode start),
``capx_trace_recorder/hooks/env.py`` and ``capx_trace_recorder/hooks/api.py`` (key frames).
"""

from __future__ import annotations

import functools
import sys
import threading
import time
import weakref
from typing import Any

from capx_trace_recorder.ids import sanitize_id

# id(low_env) -> weak reference to its tap: the tap and the env reference each
# other through the instance attribute, so both go away together when the env is
# dropped. The weak reference has no callback (a callback, such as the one a
# WeakValueDictionary uses, runs on whatever thread drops the env, where an alarm
# inside it would be printed as "Exception ignored"); dead entries are removed
# at the next install.
_TAPS: dict[int, "weakref.ref[FrameTap]"] = {}
_TAPS_LOCK = threading.Lock()


class BufferState:
    """One buffer attribute (``_frame_buffer`` or ``_wrist_frame_buffer``)."""

    __slots__ = ("attr", "camera", "last_obj", "last_len", "epoch", "steps")

    def __init__(self, attr: str, camera: str) -> None:
        self.attr = attr
        self.camera = camera
        self.last_obj: Any = None
        self.last_len = 0
        self.epoch = 0
        self.steps: list[int | None] = []


def _camera_names(low_env: Any) -> tuple[str, str]:
    module = type(low_env).__module__
    main = getattr(low_env, "save_camera_name", None)
    if not isinstance(main, str) or not main:
        main = "agentview" if module.startswith("capx.envs.simulators.libero") else "main"
    wrist = getattr(low_env, "_wrist_camera_name", None)
    if not isinstance(wrist, str) or not wrist:
        wrist = "wrist"
    return sanitize_id(main), sanitize_id(wrist)


class FrameTap:
    """H10 state for one low-level env; re-attached to each new episode."""

    def __init__(self, low_env: Any, original: Any) -> None:
        self.low = low_env
        self.original = original
        self.episode: Any = None
        self.lock = threading.RLock()
        main, wrist = _camera_names(low_env)
        self.buffers = [BufferState("_frame_buffer", main)]
        if isinstance(low_env.__dict__.get("_wrist_frame_buffer"), list):
            self.buffers.append(BufferState("_wrist_frame_buffer", wrist))
        self.last_periodic = 0.0
        self.saved: set[tuple[int, int, str]] = set()

    # -- episode binding ------------------------------------------------------
    def attach(self, episode: Any) -> None:
        with self.lock:
            self.episode = episode
            self.saved = set()
            self.last_periodic = 0.0
            for state in self.buffers:
                buf = self._buffer(state)
                state.epoch = 0
                state.last_obj = buf
                state.last_len = len(buf) if isinstance(buf, list) else 0
                state.steps = [None] * state.last_len  # back-filled: sim_step unknown

    def detach(self, episode: Any) -> None:
        with self.lock:
            if self.episode is episode:
                self.episode = None

    # -- buffer observation ---------------------------------------------------
    def _buffer(self, state: BufferState) -> Any:
        try:
            return self.low.__dict__.get(state.attr)
        except TimeoutError:  # the trial alarm (section 2)
            raise
        except Exception:  # noqa: BLE001
            return None

    def _observe(self, state: BufferState, sim_step: int | None) -> Any:
        buf = self._buffer(state)
        if not isinstance(buf, list):
            return None
        n = len(buf)
        if buf is not state.last_obj or n < state.last_len:
            state.epoch += 1
            state.steps = []
            state.last_len = 0
            state.last_obj = buf
        if n > state.last_len:
            state.steps.extend([sim_step] * (n - state.last_len))
            state.last_len = n
        return buf

    def buffer_lengths(self) -> dict[str, tuple[int, int]]:
        """``{camera: (epoch, len)}`` without modifying anything."""
        out = {}
        with self.lock:
            for state in self.buffers:
                buf = self._buffer(state)
                if isinstance(buf, list):
                    out[state.camera] = (state.epoch, len(buf))
        return out

    # -- the wrapped _record_frame --------------------------------------------
    def before_record(self) -> None:
        ep = self.episode
        if ep is None:
            return
        with self.lock:
            for state in self.buffers:
                self._observe(state, None)

    def after_record(self) -> None:
        ep = self.episode
        if ep is None:
            return
        t0 = time.perf_counter()
        sim_step = ep.clock()[0]
        new_main = None
        with self.lock:
            for state in self.buffers:
                before = state.last_len
                buf = self._observe(state, sim_step)
                if buf is not None and state is self.buffers[0] and state.last_len > before:
                    new_main = state.last_len - 1
            fps = ep.config.frame_fps
            if new_main is not None and fps > 0:
                now = time.monotonic()
                if now - self.last_periodic >= 1.0 / fps:
                    self.last_periodic = now
                    for state in self.buffers:
                        buf = self._buffer(state)
                        if isinstance(buf, list) and buf:
                            self._save(ep, state, buf, len(buf) - 1, "periodic", None)
        ep.add_overhead(time.perf_counter() - t0)

    # -- key frames -----------------------------------------------------------
    def keyframe(self, for_event_id: str | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Copy the latest frame of each buffer; returns ``(frame_ref, all refs)``."""
        ep = self.episode
        if ep is None:
            return None, []
        refs: list[dict[str, Any]] = []
        with self.lock:
            for state in self.buffers:
                buf = self._observe(state, ep.clock()[0])
                if not isinstance(buf, list) or not buf:
                    continue
                refs.extend(self._save(ep, state, buf, len(buf) - 1, "keyframe", for_event_id))
        return (refs[0] if refs else None), refs

    def _save(self, ep: Any, state: BufferState, buf: list, index: int, reason: str,
              for_event_id: str | None) -> list[dict[str, Any]]:
        store = getattr(ep, "store", None)
        if store is not None and getattr(store, "stopped", False):
            return []  # all writing stopped (disk low): take no copy
        try:
            entry = buf[index]
        except IndexError:
            return []
        frame_step = state.steps[index] if index < len(state.steps) else None
        if isinstance(entry, dict):
            items = [(sanitize_id(str(k)), v) for k, v in entry.items()]
        else:
            items = [(state.camera, entry)]
        refs = []
        for camera, frame in items:
            path = f"frames/{camera}/e{state.epoch}_{index}.jpg"
            ref = {"epoch": state.epoch, "buffer_index": index, "camera": camera,
                   "frame_sim_step": frame_step, "path": path}
            refs.append(ref)
            key = (state.epoch, index, camera)
            if key in self.saved:
                continue
            copy = _copy_frame(frame)
            if copy is None:
                ref["reason"] = "frame_not_copyable"
                continue
            self.saved.add(key)
            quality = ep.config.jpeg_quality
            extra = {"camera": camera, "width": int(copy.shape[1]), "height": int(copy.shape[0])}

            def builder(ctx, copy=copy, path=path, extra=extra):
                from capx_trace_recorder.store import encode_jpeg

                ctx.write_file(path, lambda: encode_jpeg(copy, quality), label="debug_frame",
                               kind="image", mime="image/jpeg", binary=True, dedupe=True,
                               extra=extra)

            def drop_builder(ctx, path=path, extra=extra):
                # holds no copy: the frame is released when the queue budget drops it
                ctx.write_file(path, b"", label="debug_frame", kind="image", mime="image/jpeg",
                               binary=True, extra=extra)

            ep.emit("frame", data={"camera": camera, "epoch": state.epoch, "buffer_index": index,
                                   "sim_step": frame_step, "reason": reason,
                                   "for_event_id": for_event_id},
                    refs={"for_event_id": for_event_id} if for_event_id else None,
                    builder=builder, drop_builder=drop_builder, nbytes=int(copy.nbytes))
        return refs


def _copy_frame(frame: Any) -> Any:
    np = sys.modules.get("numpy")
    if np is None:
        return None
    try:
        arr = np.array(frame, copy=True, order="C")
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return None
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    return arr


def tap_for(low_env: Any) -> FrameTap | None:
    if low_env is None:
        return None
    with _TAPS_LOCK:
        ref = _TAPS.get(id(low_env))
    tap = ref() if ref is not None else None
    if tap is not None and tap.low is low_env:
        return tap
    return None


def install_tap(low_env: Any) -> tuple[FrameTap | None, str | None]:
    """Install H10 on ``low_env`` (idempotent); returns ``(tap, reason_if_not)``."""
    if low_env is None:
        return None, "no_low_level_env"
    existing = tap_for(low_env)
    if existing is not None:
        return existing, None
    try:
        current = low_env.__dict__.get("_record_frame")
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None, "no_instance_dict"
    if current is not None and getattr(current, "__capx_trace_wrapper__", False):
        return getattr(current, "__capx_trace_tap__", None), None
    original = getattr(low_env, "_record_frame", None)
    if not callable(original):
        return None, "no_record_frame"
    if not isinstance(low_env.__dict__.get("_frame_buffer"), list):
        return None, "no_frame_buffer"
    tap = FrameTap(low_env, original)
    func = getattr(original, "__func__", original)

    @functools.wraps(func)
    def traced_record_frame(*args: Any, **kwargs: Any) -> Any:
        try:
            tap.before_record()
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            pass
        result = original(*args, **kwargs)
        try:
            tap.after_record()
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            pass
        return result

    traced_record_frame.__capx_trace_wrapper__ = True  # type: ignore[attr-defined]
    traced_record_frame.__capx_trace_tap__ = tap  # type: ignore[attr-defined]
    try:
        low_env._record_frame = traced_record_frame
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None, "cannot_set_attribute"
    with _TAPS_LOCK:
        for key in [k for k, r in _TAPS.items() if r() is None]:
            del _TAPS[key]
        _TAPS[id(low_env)] = weakref.ref(tap)
    return tap, None


__all__ = ["FrameTap", "install_tap", "tap_for"]
