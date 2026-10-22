"""H7: ``CodeExecutionEnvBase`` class-attribute wrappers.

Implements docs/trace-format.md sections 5.2 (``code_block_*``, ``stdout``,
``stderr``, block ``evaluator_result``, ``observation_snapshot``,
``video_frames_read``, ``env_reset``) and 8. Wrapped: ``step``,
``_exec_user_code``, ``render``, ``render_wrist``, ``get_video_frames_range``,
and additionally ``get_wrist_video_frames_range`` (wrist VDM videos) and
``reset`` (needed for ``env_reset`` and to bind a web UI env). No subclass in
``capx/envs/tasks`` overrides ``step``/``_exec_user_code``/``render``/
``render_wrist``/``get_video_frames_range``; the scene-text env overrides
``reset`` but calls ``super().reset()``.

The first ``step`` of an env in an episode installs H8 and H9 on its API
instances (``combined_doc()`` checked per instance); a public function or a
service closure that cannot be wrapped is recorded as ``hook_install_failed``.
In the web UI path the env is created inside the episode, so the first
``reset``/``step`` on the env thread binds it to the episode (found through the
web session that holds the env, see ``capx_trace_recorder/context.py``) and installs
H10. An env whose episode has ended stays bound to it: its events are late
events of that episode, and nothing is re-bound or re-installed for it.

Frames the harness renders (``render``/``render_wrist``) are copied and stored
as PNG with the sha256 of their raw pixels, the only frames that can match a
model-input image. ``get_video_frames_range`` results are hashed on the writer
(they are fresh copies the harness only reads).

Related: ``capx_trace_recorder/hooks/api.py`` (H8), ``capx_trace_recorder/hooks/services.py``
(H9), ``capx_trace_recorder/frames.py`` (H10).
"""

from __future__ import annotations

import functools
import io
import sys
import threading
import time
import tokenize
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import add_overhead, guard, mark, patch_attribute
from capx_trace_recorder.ids import new_event_id
from capx_trace_recorder.serialize import ArrayBudget, json_safe, pending_builders, summarize, text_value
from capx_trace_recorder.timeout import is_trial_alarm

DIRECT_NAMES = ("env", "obs", "APIS", "INPUTS")
TRACER_PATH_MARKERS = ("capx_trace_recorder/", "capx_trace_recorder\\")


# ---------------------------------------------------------------------------
# Env binding, lazy installs
# ---------------------------------------------------------------------------


def _bind_env(ep: Any, env: Any) -> None:
    """Bind ``env`` to ``ep`` (web path) and install H10; once per env per episode."""
    if ep.ended:
        return  # late events of an ended episode bind and install nothing
    if C.REGISTRY.episode_for_env(env) is ep and ep.env is env:
        return
    from capx_trace_recorder.hooks.trial import attach_frames, env_info

    low = getattr(env, "low_level_env", None)
    C.REGISTRY.bind_env(env, ep)
    C.REGISTRY.bind_env(low, ep)
    first = ep.env is None
    ep.env = env
    ep.low_env = low
    ep.sim_thread = threading.get_ident()
    if ep.trial_thread is None:
        ep.trial_thread = threading.get_ident()
    frames = attach_frames(ep, low)
    info = env_info(env)
    info["frames"] = frames
    if first:
        def builder(ctx, info=info):
            ctx.store.update_summary(ctx.wep, {"env": info})
        ep.internal(builder)


def _install_apis(ep: Any, env: Any) -> None:
    from capx_trace_recorder import bootstrap
    from capx_trace_recorder.hooks.api import install_api, wrapped_attributes
    from capx_trace_recorder.hooks.services import install_services, wrapped_services

    if ep.ended:
        return
    apis = env.__dict__.get("_apis") if hasattr(env, "__dict__") else None
    if not isinstance(apis, dict):
        return
    done = ep.misc.setdefault("apis_done", set())
    before = len(ep.misc.get("apis_installed", []))
    for api_name, api in list(apis.items()):
        if id(api) in done:
            continue
        done.add(id(api))
        if not bootstrap.frame_hooks_allowed():
            _hook_warning(ep, "H8", api_name, "traceback protection (H11) not installed")
            continue
        ok, reason, names = install_api(api)
        if not ok:
            _hook_warning(ep, "H8", api_name, reason or "install failed")
            continue
        if reason:
            # a partial install: calls of these public functions are not recorded
            _hook_warning(ep, "H8", api_name, f"not wrapped: {reason}")
        new_services, failed_services = install_services(api)
        for failure in failed_services:
            _hook_warning(ep, "H9", api_name, failure)
        # what is wrapped now: a reused env keeps the wrappers of an earlier episode
        entry = {"api": api_name, "class": type(api).__name__, "wrapped": wrapped_attributes(api),
                 "services": wrapped_services(api),
                 "installed_in_this_episode": bool(names or new_services)}
        if reason:
            entry["skipped"] = reason
        ep.misc.setdefault("apis_installed", []).append(entry)
    installed = ep.misc.get("apis_installed", [])
    if len(installed) != before:
        snapshot = [dict(e) for e in installed]

        def builder(ctx, snapshot=snapshot):
            ctx.store.update_summary(ctx.wep, {"apis_wrapped": snapshot})

        ep.internal(builder)


def _hook_warning(ep: Any, hook: str, target: str, reason: str) -> None:
    ep.emit("trace_warning", status="warning",
            data={"code": "hook_install_failed", "hook": hook, "target": target, "message": reason})


def _episode_for_env(env: Any, where: str = "H7") -> Any:
    return C.resolve_or_note(where, env)


# ---------------------------------------------------------------------------
# step / _exec_user_code
# ---------------------------------------------------------------------------


def _direct_access(code: str) -> dict[str, Any]:
    """Which of env/obs/APIS/INPUTS the code names (``tokenize`` only)."""
    used: set[str] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.NAME and tok.string in DIRECT_NAMES:
                used.add(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {"names": sorted(used), "complete": False}
    except Exception:  # noqa: BLE001 - timeout-ok: writer thread
        return {"names": sorted(used), "complete": False}
    return {"names": sorted(used), "complete": True,
            "note": "APIS[...] calls are recorded; calls through env, obs or INPUTS are not"}


def _block_start(env: Any, action: Any) -> Any:
    """Emit ``code_block_start`` and return the block's context (the wrapper pushes it)."""
    ep = _episode_for_env(env)
    if ep is None:
        return None
    _bind_env(ep, env)
    _install_apis(ep, env)
    event_id = new_event_id()
    block = C.CodeBlockCtx(event_id, ep, action, env)
    block.block_index = ep.count("n_code_blocks") - 1
    try:
        block.exec_globals = env.__dict__.get("_exec_globals")
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        block.exec_globals = None
    block.sim_step0 = ep.clock()[0]
    from capx_trace_recorder.hooks.api import _chain, keyframe_and_state, state_parts

    frame_ref, frame_refs, state, _nbytes = keyframe_and_state(ep, event_id)
    write_state, drop_state, state_bytes = state_parts(state)
    del state
    code = action if isinstance(action, str) else None
    pending: list[Any] = []
    source_id = ep.last_codegen_response[0]
    oracle = bool(ep.config_flags.get("use_oracle_code"))
    data = {
        "code_block_id": event_id,
        "block_index": block.block_index,
        "turn_index": ep.turn_index,
        "source_model_event_id": None if oracle else source_id,
        "code_source": "oracle" if oracle else ("model" if source_id else "unknown"),
        "code": text_value(code, pending, "text") if code is not None else None,
        "code_sha256": None,
        "direct_access": None,
        "frame_ref": frame_ref,
        "state_ref": None,
    }
    if len(frame_refs) > 1:
        data["frame_refs"] = frame_refs

    def code_step(ctx, code=code):
        import hashlib

        if code is not None:
            ctx.data["code_sha256"] = hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()
            ctx.data["direct_access"] = _direct_access(code)

    builder, drop_builder, nbytes = pending_builders(
        pending, _chain(code_step, write_state), drop_after=_chain(code_step, drop_state),
        extra_bytes=state_bytes)
    refs = {"code_block_id": event_id, "turn_index": ep.turn_index}
    if source_id and not oracle:
        refs["model_event_id"] = source_id
    ep.emit("code_block_start", status="started", data=data, refs=refs, builder=builder,
            drop_builder=drop_builder, event_id=event_id, nbytes=nbytes)
    if code is not None:
        ep.segments.register("code_history", code, (event_id,), "code_block")
    return block


def _text_event(ep: Any, block: Any, etype: str, text: Any) -> str:
    pending: list[Any] = []
    data = {"code_block_id": block.event_id,
            "text": text_value(text, pending, "text") if isinstance(text, str) else None,
            "length": len(text) if isinstance(text, str) else None}
    builder = None
    if pending:
        def builder(ctx, pending=pending):
            ctx.fill_pending(pending)
    return ep.emit(etype, data=data, parent=block.event_id, refs={"code_block_id": block.event_id},
                   builder=builder)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None


def _block_end(block: Any, result: Any, exc: BaseException | None) -> None:
    ep = block.episode
    from capx_trace_recorder.hooks.api import _chain, keyframe_and_state, state_parts

    end_id = new_event_id()
    frame_ref, frame_refs, state, _nbytes = keyframe_and_state(ep, end_id)
    write_state, drop_state, state_bytes = state_parts(state)
    del state
    step1 = ep.clock()[0]
    info: dict[str, Any] = {}
    reward = terminated = truncated = None
    if exc is None and isinstance(result, tuple) and len(result) == 5:
        _obs, reward, terminated, truncated, info = result
        info = info if isinstance(info, dict) else {}
    exec_result = block.exec_result or {}
    stdout = info.get("stdout") if info else exec_result.get("stdout")
    stderr = info.get("stderr") if info else exec_result.get("stderr")
    stdout_id = stderr_id = None
    if isinstance(stdout, str):
        stdout_id = _text_event(ep, block, "stdout", stdout)
        ep.segments.register("stdout", stdout, (stdout_id, block.event_id), "block_stdout")
    if isinstance(stderr, str):
        stderr_id = _text_event(ep, block, "stderr", stderr)
        ep.segments.register("stderr", stderr, (stderr_id, block.event_id), "block_stderr")
    task_completed = _as_bool(info.get("task_completed")) if info else None
    reward_v = json_safe(reward) if reward is not None else None
    if exc is None:
        ep.emit("evaluator_result", data={
            "scope": "code_block", "code_block_id": block.event_id, "reward": reward_v,
            "task_completed": task_completed, "terminated": _as_bool(terminated),
            "truncated": _as_bool(truncated), "source": "CodeExecutionEnvBase.step",
        }, parent=block.event_id, refs={"code_block_id": block.event_id})
    data: dict[str, Any] = {
        "code_block_id": block.event_id,
        "sandbox_rc": info.get("sandbox_rc") if info else (
            (0 if exec_result.get("ok") else 1) if exec_result else None),
        "duration_s": round(time.monotonic() - block.start_mono, 6),
        "exec_duration_s": round(block.exec_duration, 6) if block.exec_duration is not None else None,
        "stdout_len": len(stdout) if isinstance(stdout, str) else None,
        "stderr_len": len(stderr) if isinstance(stderr, str) else None,
        "stdout_event_id": stdout_id,
        "stderr_event_id": stderr_id,
        "reward": reward_v,
        "terminated": _as_bool(terminated),
        "truncated": _as_bool(truncated),
        "task_completed": task_completed,
        "result_summary": exec_result.get("result_summary"),
        "frame_ref": frame_ref,
        "state_ref": None,
        "sim_step_range": [block.sim_step0, step1],
        "tracer_frame_visible": None,
    }
    if len(frame_refs) > 1:
        data["frame_refs"] = frame_refs
    exc_message = None
    if exc is not None:
        from capx_trace_recorder.hooks.api import safe_str

        exc_message = safe_str(exc)  # redacted, then cut to 2000 characters, by the writer
        data["exception"] = {"type": type(exc).__name__, "message": None}
        if is_trial_alarm(exc):
            ep.alarm_seen = True
    pending = list(exec_result.get("pending") or [])

    def text_step(ctx, stdout=stdout, stderr=stderr, exc_message=exc_message):
        visible = any(isinstance(t, str) and any(m in t for m in TRACER_PATH_MARKERS)
                      for t in (stdout, stderr))
        ctx.data["tracer_frame_visible"] = visible
        if visible:
            ctx.warn("tracer_frame_visible", "the recorder's path appears in the block's stdout/stderr",
                     ctx.data.get("code_block_id"))
        if exc_message is not None:
            from capx_trace_recorder.payload import put_redacted

            put_redacted(ctx.data["exception"], "message", exc_message, 2000)

    builder, drop_builder, nbytes = pending_builders(
        pending, _chain(text_step, write_state), drop_after=_chain(text_step, drop_state),
        extra_bytes=state_bytes)
    exceeded = exec_result.get("array_budget_exceeded") or 0
    if exceeded:
        data["array_budget_exceeded"] = exceeded
    ep.emit("code_block_end", status="error" if exc is not None else "ok", data=data,
            parent=block.event_id, refs={"code_block_id": block.event_id}, builder=builder,
            drop_builder=drop_builder, event_id=end_id, nbytes=nbytes)
    if exceeded:
        from capx_trace_recorder.hooks.api import _array_budget_warning

        _array_budget_warning(ep, end_id, exceeded)


def make_step_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        action = args[0] if args else kwargs.get("action")
        block = guard(_block_start, self, action)
        if block is None:
            return original(self, *args, **kwargs)
        add_overhead(block.episode, t0)
        try:
            C.push_ctx(block)
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            C.pop_ctx(block)
            t1 = time.perf_counter()
            guard(_block_end, block, None, exc)
            add_overhead(block.episode, t1)
            raise
        finally:
            C.pop_ctx(block)  # also when an alarm lands after the step returned
        t1 = time.perf_counter()
        guard(_block_end, block, result, None)
        add_overhead(block.episode, t1)
        return result

    return mark(step, original)


def _after_exec(block: Any, result: Any, started: float) -> None:
    block.exec_duration = time.monotonic() - started
    if not isinstance(result, dict):
        return
    pending: list[Any] = []
    budget = ArrayBudget(block.episode.config.array_call_bytes, block.episode)
    block.exec_result = {
        "ok": result.get("ok"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "result_summary": summarize(result.get("result"), pending, budget, label="value"),
        "pending": pending,
        "array_budget_exceeded": budget.exceeded,
    }


def make_exec_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _exec_user_code(self: Any, *args: Any, **kwargs: Any) -> Any:
        block = C.innermost("code_block")
        started = time.monotonic()
        result = original(self, *args, **kwargs)
        if block is not None:
            t1 = time.perf_counter()
            guard(_after_exec, block, result, started)
            add_overhead(block.episode, t1)
        return result

    return mark(_exec_user_code, original)


# ---------------------------------------------------------------------------
# render / render_wrist
# ---------------------------------------------------------------------------


def _snapshot(env: Any, frame: Any, camera_kind: str, purpose: str) -> None:
    ep = _episode_for_env(env)
    if ep is None:
        return
    np = sys.modules.get("numpy")
    if np is None or not isinstance(frame, np.ndarray):
        return
    if frame.dtype != np.uint8 or frame.ndim != 3:
        ep.emit("observation_snapshot", status="warning",
                data={"camera": camera_kind, "purpose": purpose, "stored": False,
                      "reason": f"unsupported frame dtype/shape {frame.dtype}/{list(frame.shape)}"})
        return
    if ep.store.stopped:
        return  # all writing stopped (disk low): take no copy
    copy = np.array(frame, copy=True, order="C")
    low = getattr(env, "low_level_env", None)
    from capx_trace_recorder.frames import _camera_names

    main, wrist = _camera_names(low) if low is not None else ("main", "wrist")
    camera = main if camera_kind == "main" else wrist
    data = {"camera": camera, "purpose": purpose, "pixel_sha256": None,
            "width": int(copy.shape[1]), "height": int(copy.shape[0])}

    def builder(ctx, copy=copy, camera=camera):
        from capx_trace_recorder.payload import raw_pixel_sha256

        pix = raw_pixel_sha256(copy)
        ctx.data["pixel_sha256"] = pix
        art = ctx.png_asset(copy, label="harness_render", extra={"camera": camera})
        if pix:
            ctx.wep.pixel_index.setdefault(pix, []).append({
                "kind": "observation_snapshot", "event_id": ctx.event.get("event_id"),
                "camera": camera, "sim_step": ctx.event.get("sim_step"),
                "wall_time": ctx.event.get("wall_time"), "asset": art["path"],
            })

    def drop_builder(ctx, camera=camera):
        # holds no copy; the render cannot be matched to a model-input image
        ctx.data["reason"] = "queue_full"
        ctx.write_asset(b"", "png", label="harness_render", kind="image", mime="image/png",
                        binary=True, extra={"camera": camera})

    ep.emit("observation_snapshot", data=data, builder=builder, drop_builder=drop_builder,
            nbytes=int(copy.nbytes))


def make_render_wrapper(original: Any, camera_kind: str) -> Any:
    @functools.wraps(original)
    def render(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        t1 = time.perf_counter()
        try:
            purpose = sys._getframe(1).f_code.co_name
        except Exception as exc:  # noqa: BLE001
            if is_trial_alarm(exc):
                raise
            purpose = None
        guard(_snapshot, self, result, camera_kind, purpose)
        ep = C.REGISTRY.episode_for_env(self)
        if ep is not None:
            add_overhead(ep, t1)
        return result

    return mark(render, original)


# ---------------------------------------------------------------------------
# get_video_frames_range / get_wrist_video_frames_range
# ---------------------------------------------------------------------------


def _frames_read(env: Any, start: Any, end: Any, frames: Any, camera_kind: str) -> None:
    ep = _episode_for_env(env)
    if ep is None or not isinstance(frames, list):
        return
    refs_frames = list(frames)  # fresh copies made for the harness, which only reads them
    nbytes = sum(int(getattr(f, "nbytes", 0)) for f in refs_frames)
    from capx_trace_recorder.frames import _camera_names

    low = getattr(env, "low_level_env", None)
    main, wrist = _camera_names(low) if low is not None else ("main", "wrist")
    camera_kind = main if camera_kind == "main" else wrist
    data = {"start": json_safe(start), "end": json_safe(end), "n_frames": len(refs_frames),
            "camera": camera_kind, "frame_pixel_sha256": None}

    def builder(ctx, refs_frames=refs_frames, start=start, end=end, camera_kind=camera_kind):
        from capx_trace_recorder.payload import raw_pixel_sha256

        hashes = [raw_pixel_sha256(f) for f in refs_frames]
        ctx.data["frame_pixel_sha256"] = hashes
        ctx.wep.video_reads.append({"event_id": ctx.event.get("event_id"), "start": json_safe(start),
                                    "end": json_safe(end), "camera": camera_kind, "hashes": hashes})

    def drop_builder(ctx):
        # holds no frame: the queue budget drop releases them (counted as a gap)
        ctx.data["frame_pixel_sha256"] = None
        ctx.data["reason"] = "queue_full"

    ep.emit("video_frames_read", data=data, builder=builder, drop_builder=drop_builder,
            nbytes=nbytes)


def make_frames_range_wrapper(original: Any, camera_kind: str) -> Any:
    @functools.wraps(original)
    def frames_range(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        start = args[0] if args else kwargs.get("start")
        end = args[1] if len(args) > 1 else kwargs.get("end")
        guard(_frames_read, self, start, end, result, camera_kind)
        return result

    return mark(frames_range, original)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def _before_reset(env: Any) -> Any:
    ep = _episode_for_env(env)
    if ep is None:
        return None
    _bind_env(ep, env)
    return ep


def _after_reset(ep: Any, env: Any) -> None:
    if ep.resets_seen > 0:
        with ep._lock:
            ep.sim_epoch += 1
        ep.emit("env_reset", data={"sim_epoch": ep.sim_epoch, "resets_in_episode": ep.resets_seen + 1,
                                   "env_class": type(env).__name__})
    ep.resets_seen += 1
    tap = ep.frame_tap
    if tap is not None and tap.episode is ep:
        tap.before_record()


def make_reset_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        ep = guard(_before_reset, self)
        result = original(self, *args, **kwargs)
        if ep is not None:
            guard(_after_reset, ep, self)
        return result

    return mark(reset, original)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def install(base_module: Any) -> list[str]:
    cls = getattr(base_module, "CodeExecutionEnvBase", None)
    if not isinstance(cls, type):
        return ["CodeExecutionEnvBase not found"]
    problems = []
    for name, factory in (
        ("step", make_step_wrapper),
        ("_exec_user_code", make_exec_wrapper),
        ("render", lambda orig: make_render_wrapper(orig, "main")),
        ("render_wrist", lambda orig: make_render_wrapper(orig, "wrist")),
        ("get_video_frames_range", lambda orig: make_frames_range_wrapper(orig, "main")),
        ("get_wrist_video_frames_range", lambda orig: make_frames_range_wrapper(orig, "wrist")),
        ("reset", make_reset_wrapper),
    ):
        err = patch_attribute(cls, name, factory)
        if err:
            problems.append(err)
    return problems


__all__ = ["install", "make_step_wrapper", "make_exec_wrapper", "make_render_wrapper"]
