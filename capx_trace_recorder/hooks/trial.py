"""H2 and H4: ``capx.envs.trial`` (docs/trace-format.md, sections 4.2, 5.2, 7.1, 7.3).

H2 rebinds the helpers that ``_run_single_trial`` calls as module globals and
sets the role context (``role``, ``call_kind``, ``turn_index``) for the calls
they make; ``_patch_libero_goal`` registers the task text after the LIBERO
goal is filled in.

H4 rebinds ``_run_single_trial`` right after ``capx.envs.trial`` executes, so
``capx.envs.runner`` and ``capx.experiments.run`` bind the wrapper when they
import it. The wrapper opens the episode (``episode_start``), installs H10 on
``env.low_level_env`` before the original resets the env, and on the way out
queues ``evaluator_result``, ``episode_end`` and then ``trial_videos`` and returns: nothing waits
for the writer inside the trial's alarm window. The episode (and each H2 role)
is pushed on the thread's stack as the first statement inside the wrapper's
``try``. Context entries still on the trial thread when an episode starts
(left by an earlier one) or ends (left by this one) are cleared and recorded
as ``stale_context`` warnings; the latter counts as a gap. An earlier episode
left open on the thread is ended (``end_reason: error``, ``end_note``) when the
next one starts, and an alarm that lands after the original returned still ends
the episode, as ``timeout_alarm``, before the ``TimeoutError`` leaves H4.

Episode ids: runner episodes ``<task>__<condition>__s<seed>__i<invocation>__a<attempt>``
(``invocation`` = ``len(run.json["invocations"])`` of the runner's run dir,
read once per process; ``attempt`` = 1-based count of calls for
(``config["output_dir"]``, seed) in this process); ``launch.py`` episodes
``<UTC ts>__<task>__s<seed>__<6 hex>``.

Related: ``capx_trace_recorder/context.py``, ``capx_trace_recorder/frames.py`` (H10),
``capx_trace_recorder/hooks/runner.py`` (H15 exports the runner's run id and dir).
"""

from __future__ import annotations

import dataclasses
import errno
import functools
import inspect
import json
import os
import hashlib
import re
import shutil
import stat
import threading
import time
from pathlib import Path
from typing import Any

from capx_trace_recorder import TRACER_VERSION
from capx_trace_recorder import context as C
from capx_trace_recorder.config import ENV_FALLBACK_RUN_ID, ENV_RUN_ID, ENV_RUNNER_RUN_DIR
from capx_trace_recorder.hooks.common import add_overhead, guard, mark, patch_attribute, rebind_aliases
from capx_trace_recorder.ids import fallback_run_id, join_id, random_hex, sanitize_id, utc_stamp
from capx_trace_recorder.payload import sanitize_url
from capx_trace_recorder.segments import split_prompt_text
from capx_trace_recorder.serialize import float_value, json_safe, text_value
from capx_trace_recorder.store import EpisodeRecorder, get_store, process_info
from capx_trace_recorder.timeout import is_trial_alarm

# ---------------------------------------------------------------------------
# H2: role context
# ---------------------------------------------------------------------------

_ROLE_TARGETS = {
    "_describe_initial_scene": ("vdm", "initial_scene_description", "zero"),
    "_get_visual_differencing_feedback": ("vdm", "image_diff", "current"),
    "_get_video_differencing_feedback": ("vdm", "video_diff", "current"),
    "_query_initial_code": ("code_generator", "initial_code", "zero"),
    "_handle_multi_turn_step": ("code_generator", "multi_turn_decision", "next"),
}


class _RoleToken:
    __slots__ = ("role", "ep", "previous_current", "pushed")

    def __init__(self, role: Any, ep: Any) -> None:
        self.role = role
        self.ep = ep
        self.previous_current: Any = None
        self.pushed = False

    def push(self) -> None:
        """Called as the first statement inside the wrapper's ``try``.

        ``pushed`` is set before anything changes, so ``_pop_role`` undoes
        whatever part ran even when an alarm interrupts this method.
        """
        self.previous_current = self.ep.current_role
        self.pushed = True
        self.ep.current_role = self.role  # ensemble pool threads read this (section 7.3)
        C.push_role(self.role)


def _make_role(role: str, call_kind: str, turn_mode: str) -> _RoleToken | None:
    ep, _ = C.resolve_episode()
    if ep is None:
        return None
    if turn_mode == "next":
        ep.turn_index += 1
    turn = 0 if turn_mode == "zero" else ep.turn_index
    return _RoleToken(C.RoleCtx(role, call_kind, turn, ep), ep)


def _pop_role(token: _RoleToken | None) -> None:
    if token is None:
        return
    C.pop_role(token.role)
    if token.pushed and token.ep.current_role is token.role:
        token.ep.current_role = token.previous_current


def make_role_wrapper(original: Any, role: str, call_kind: str, turn_mode: str) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = guard(_make_role, role, call_kind, turn_mode)
        try:
            if token is not None:
                token.push()
            result = original(*args, **kwargs)
        finally:
            guard(_pop_role, token)  # on every way out, an alarm included
        return result

    return mark(wrapper, original)


def _after_patch_libero_goal(env: Any, obs: Any) -> None:
    ep, _ = C.resolve_episode(env)
    if ep is None:
        return
    try:
        text = obs["full_prompt"][-1]["content"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return
    before, _after = split_prompt_text(text)
    if before is not None:
        ep.segments.register("task_prompt", before, (), "after _patch_libero_goal")


def make_patch_libero_goal_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _patch_libero_goal(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        env = args[0] if args else kwargs.get("env")
        obs = args[1] if len(args) > 1 else kwargs.get("obs")
        guard(_after_patch_libero_goal, env, obs)
        return result

    return mark(_patch_libero_goal, original)


# ---------------------------------------------------------------------------
# H4: episodes
# ---------------------------------------------------------------------------

_RUNNER_CONDITION_FIELDS = ("task", "condition", "config_path", "privileged", "apis", "multi_turn",
                            "uses_vdm", "use_visual_feedback", "use_oracle_code", "note")
_CONFIG_FLAG_KEYS = ("use_visual_feedback", "use_img_differencing", "use_video_differencing",
                     "use_wrist_camera", "use_parallel_ensemble", "use_multimodel", "record_video",
                     "use_oracle_code")


def _field(obj: Any, name: str) -> Any:
    """Read one declared field of an args object (never ``repr``/``asdict``)."""
    try:
        if isinstance(obj, dict):
            return obj.get(name)
        return obj.__dict__.get(name) if hasattr(obj, "__dict__") else None
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float_value(value)
    return str(type(value).__name__)


def resolve_run_id(runner_dir: Path | None) -> str:
    explicit = os.environ.get(ENV_RUN_ID)
    if explicit and explicit.strip():
        return sanitize_id(explicit.strip())
    runner = C.REGISTRY.runner
    if runner.run_id:
        return sanitize_id(runner.run_id)
    if runner_dir is not None:
        return sanitize_id(runner_dir.name)
    fallback = os.environ.get(ENV_FALLBACK_RUN_ID)
    if not fallback:
        fallback = fallback_run_id()
        os.environ[ENV_FALLBACK_RUN_ID] = fallback
    return sanitize_id(fallback)


def find_runner_dir(output_dir: Any) -> Path | None:
    """The runner's run dir: exported by H15, else the first ancestor of
    ``output_dir`` (at most 3 levels up) that holds ``run.json``."""
    exported = os.environ.get(ENV_RUNNER_RUN_DIR)
    if exported:
        return Path(exported)
    if C.REGISTRY.runner.run_dir is not None:
        return C.REGISTRY.runner.run_dir
    if not output_dir:
        return None
    try:
        path = Path(str(output_dir)).resolve()
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        return None
    for candidate in (path, *list(path.parents)[:3]):
        try:
            if (candidate / "run.json").is_file():
                return candidate
        except OSError as exc:
            if is_trial_alarm(exc):
                raise
            continue
    return None


def _runner_run_json(runner_dir: Path) -> dict[str, Any] | None:
    key = str(runner_dir)
    cache = C.REGISTRY.run_json_cache
    if key in cache:
        return cache[key]
    doc = None
    try:
        with open(runner_dir / "run.json", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        doc = None
    cache[key] = doc if isinstance(doc, dict) else None
    return cache[key]


def config_flags(config: Any, args: Any, multi_turn_prompt: Any) -> dict[str, Any]:
    flags: dict[str, Any] = {"multi_turn": bool(multi_turn_prompt)}
    for key in _CONFIG_FLAG_KEYS:
        flags[key] = _plain(config.get(key)) if isinstance(config, dict) else None
    flags["use_legacy_multi_turn_decision_prompt"] = _plain(
        _field(args, "use_legacy_multi_turn_decision_prompt"))
    return flags


def models_of(args: Any, flags: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    model = _field(args, "model")
    if model is not None:
        url = _field(args, "server_url")
        models["code_generator"] = {"model": _plain(model),
                                    "endpoint": sanitize_url(url) if url else None,
                                    "oracle_code": bool(flags.get("use_oracle_code"))}
    vdm = _field(args, "visual_differencing_model")
    if vdm is not None and (flags.get("use_img_differencing") or flags.get("use_video_differencing")):
        url = _field(args, "visual_differencing_model_server_url")
        models["vdm"] = {"model": _plain(vdm), "endpoint": sanitize_url(url) if url else None}
    return models


def env_info(env: Any) -> dict[str, Any]:
    low = getattr(env, "low_level_env", None)
    info: dict[str, Any] = {
        "class": f"{type(env).__module__}.{type(env).__qualname__}" if env is not None else None,
        "low_level_class": f"{type(low).__module__}.{type(low).__qualname__}" if low is not None else None,
        "apis": None,
        "privileged": None,
        "viser_port": None,
    }
    try:
        cfg = getattr(env, "cfg", None)
        apis = getattr(cfg, "apis", None)
        info["apis"] = [str(a) for a in apis] if isinstance(apis, (list, tuple)) else None
        info["privileged"] = _plain(getattr(cfg, "privileged", None))
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        pass
    try:
        server = low.__dict__.get("viser_server") if low is not None else None
        getter = getattr(server, "get_port", None)
        if callable(getter):
            port = getter()
            info["viser_port"] = int(port) if isinstance(port, int) else None
    except Exception as exc:  # noqa: BLE001
        if is_trial_alarm(exc):
            raise
        pass
    return info


def tracer_info(store: Any) -> dict[str, Any]:
    return {
        "version": TRACER_VERSION,
        "settings": store.config.public(),
        "hooks_installed": list(C.REGISTRY.hooks_installed),
        "hooks_failed": [dict(h) for h in C.REGISTRY.hooks_failed],
        "hooks_skipped": [dict(h) for h in C.REGISTRY.hooks_skipped],
    }


def attach_frames(ep: Any, low_env: Any) -> dict[str, Any]:
    """Install H10 (when allowed) and attach it to ``ep``."""
    from capx_trace_recorder import bootstrap
    from capx_trace_recorder.frames import install_tap

    if not bootstrap.frame_hooks_allowed():
        return {"installed": False, "reason": "traceback protection (H11) not installed"}
    tap, reason = install_tap(low_env)
    if tap is None:
        return {"installed": False, "reason": reason}
    tap.attach(ep)
    ep.frame_tap = tap
    return {"installed": True, "buffers": [b.attr for b in tap.buffers],
            "cameras": [b.camera for b in tap.buffers], "backfilled": tap.buffers[0].last_len}


def _start_builder(ctx: Any) -> None:
    info = process_info()
    ctx.data["host"] = info.get("host")
    ctx.data["python"] = info.get("python")
    ctx.data["git_commit"] = info.get("git_commit")
    ctx.data["process_start_time"] = info.get("process_start_time")


def _stale_warning(ep: Any, removed: list[Any], when: str, counts_as_gap: bool) -> None:
    """Record context entries that were left on the trial thread (section 7.3)."""
    kinds: dict[str, int] = {}
    for entry in removed:
        kind = "episode" if entry is not None and hasattr(entry, "frame_tap") else (
            "role" if isinstance(entry, C.RoleCtx) else getattr(entry, "kind", "?"))
        kinds[kind] = kinds.get(kind, 0) + 1
    ep.emit("trace_warning", status="warning",
            data={"code": "stale_context", "counts_as_gap": counts_as_gap, "when": when,
                  "removed": kinds,
                  "message": f"{len(removed)} context entr{'y' if len(removed) == 1 else 'ies'} "
                             f"left on the trial thread {when} were cleared"})


def _episode_start(sig: inspect.Signature, args: tuple, kwargs: dict, slot: Any) -> Any:
    """Open the episode, queue ``episode_start`` and push the episode on this thread.

    Everything from ``open_episode`` to the push is one unit with its cleanup: if
    it does not complete (the trial alarm, a recorder error), ``_abandon_start``
    takes the episode out of the open set, the thread's stack and the bindings,
    ends it if its ``episode_start`` was queued, and records what happened. After a
    recorder error the trial runs untraced (None is returned); the alarm and other
    non-``Exception`` exceptions are re-raised, and a cleanup that fails after them
    runs in its own ``guard()``, so it never replaces them. ``slot`` (``context.StartSlot``)
    holds the episode from the moment it is registered, so the wrapper can undo a
    start that an alarm interrupted anywhere.
    """
    store = get_store()
    if store is None:
        return None
    try:
        bound = sig.bind_partial(*args, **kwargs).arguments
    except TypeError:
        return None
    env = bound.get("env")
    trial = bound.get("trial")
    launch_args = bound.get("args")
    config = bound.get("config") if isinstance(bound.get("config"), dict) else {}
    multi_turn_prompt = bound.get("multi_turn_prompt")
    output_dir = config.get("output_dir")
    runner_dir = find_runner_dir(output_dir)
    task = condition = None
    source = "envs.runner"
    runner_condition = None
    invocation = None
    if runner_dir is not None and output_dir:
        try:
            rel = Path(str(output_dir)).resolve().relative_to(runner_dir.resolve())
            parts = rel.parts
        except (ValueError, OSError) as exc:
            if is_trial_alarm(exc):
                raise
            parts = ()
        if len(parts) >= 2:
            source = "experiments.run"
            task, condition = parts[0], parts[1]
            doc = _runner_run_json(runner_dir) or {}
            invocations = doc.get("invocations")
            invocation = len(invocations) if isinstance(invocations, list) else 0
            for entry in doc.get("conditions") or []:
                if isinstance(entry, dict) and entry.get("task") == task and entry.get("condition") == condition:
                    runner_condition = {k: entry.get(k) for k in _RUNNER_CONDITION_FIELDS}
                    break
    if task is None:
        cfg_path = _field(launch_args, "config_path")
        task = Path(str(cfg_path)).stem if cfg_path else type(env).__name__
    key = (str(output_dir), trial)
    with C.REGISTRY.lock:
        attempt = C.REGISTRY.attempts.get(key, 0) + 1
        C.REGISTRY.attempts[key] = attempt
    run_id = resolve_run_id(runner_dir if source == "experiments.run" else None)
    # the parts that tell trials apart come last and always survive the 200-character cut
    if source == "experiments.run":
        tail = f"__s{trial}__i{invocation}__a{attempt}"
        episode_id = join_id(f"{task}__{condition}", tail)
    else:
        tail = f"__s{trial}__{random_hex(6)}"
        episode_id = join_id(f"{utc_stamp()}__{task}", tail)
    stale = C.clear_stale()  # left on this thread by an earlier episode
    if stale:
        guard(_end_stale_episodes, stale)
    ep = C.open_episode(store, run_id=run_id, episode_id=episode_id, source=source, slot=slot, tail=tail)
    try:
        ep.trial_thread = threading.get_ident()
        ep.sim_thread = threading.get_ident()
        ep.env = env
        low = getattr(env, "low_level_env", None)
        ep.low_env = low
        ep.multi_turn_prompt = multi_turn_prompt
        flags = config_flags(config, launch_args, multi_turn_prompt)
        oracle_code = getattr(env, "oracle_code", None) if flags.get("use_oracle_code") else None
        ep.config_flags = flags
        C.REGISTRY.bind_env(env, ep)
        C.REGISTRY.bind_env(low, ep)
        frames = attach_frames(ep, low)
        data = {
            "source": source,
            "task": task,
            "condition": condition,
            "seed": _plain(trial),
            "attempt_index": attempt,
            "runner_invocation": invocation,
            "condition_label": condition,
            "runner_condition": runner_condition,
            "config_flags": flags,
            "models": models_of(launch_args, flags),
            "env": env_info(env),
            "frames": frames,
            "tracer": tracer_info(store),
            "output_dir": str(output_dir) if output_dir else None,
        }

        def builder(ctx, oracle_code=oracle_code):
            _start_builder(ctx)
            if isinstance(oracle_code, str):
                import hashlib

                ctx.data["config_flags"]["oracle_code"] = {
                    "source": "env.oracle_code",
                    "sha256": hashlib.sha256(oracle_code.encode("utf-8", "surrogatepass")).hexdigest(),
                    "length": len(oracle_code),
                }

        slot.start_queued = True  # from here on the start may be queued: undoing it ends the episode
        ep.emit("episode_start", status="started", data=data, builder=builder)
        if stale:
            # not a gap of this episode: nothing it records is affected once they are gone
            _stale_warning(ep, stale, "when this episode started (left by an earlier one)", False)
        C.push_episode(ep)
        slot.started = True
    except TimeoutError as exc:
        if is_trial_alarm(exc):
            # a failing cleanup cannot replace the alarm
            guard(_abandon_start, ep, slot, exc)
            raise
        _abandon_start(ep, slot, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - the alarm is re-raised by the handler above
        _abandon_start(ep, slot, exc)
        return None
    except BaseException as exc:
        guard(_abandon_start, ep, slot, exc)
        raise
    return ep


def _abandon_start(ep: Any, slot: Any, exc: BaseException | None, where: str = "_episode_start") -> None:
    """Undo an episode whose start did not complete, and record what happened.

    The registration is undone first (``context.abandon_episode``: the thread's
    stack, the open set, the env / low-level env / launch-args bindings and the
    web session entry; the frame tap is detached), so an interruption of the
    recording part cannot leave the episode open. Then:

    * if its ``episode_start`` was queued, the episode is ended (``timeout_alarm``
      for the alarm, ``error`` after a recorder error, which is also recorded as an
      ``internal_error`` gap of the episode) with an ``end_note``;
    * after a recorder error, or when nothing of the episode was queued, a
      ``trace_warning`` (``internal_error``, ``outside_episode: true``) goes to the
      run's ``unscoped.<pid>.jsonl``, so the attempt is not missing silently.
    """
    if slot.abandoned:
        return
    slot.abandoned = True
    alarm = is_trial_alarm(exc)
    recorder_error = not alarm and (exc is None or isinstance(exc, Exception))
    what = ("the trial alarm fired" if alarm else
            f"the recorder raised {type(exc).__name__}" if recorder_error else
            f"{type(exc).__name__} was raised") + " while the recorder was starting the episode"
    outcome = "the trial ran untraced" if recorder_error else "the trial did not start"
    try:
        C.abandon_episode(ep)
    finally:
        try:
            tap = ep.frame_tap
            if tap is not None:
                tap.detach(ep)  # no frame of the env reaches this episode any more
        finally:
            _record_abandoned(ep, slot, exc, alarm, recorder_error, f"{what}; {outcome}", where)


def _record_abandoned(ep: Any, slot: Any, exc: BaseException | None, alarm: bool, recorder_error: bool,
                      note: str, where: str) -> None:
    from capx_trace_recorder.hooks.common import internal_error_data

    if slot.start_queued and not ep.ended:
        if recorder_error:
            ep.emit("trace_warning", status="warning", data=internal_error_data(exc, where))
        end_episode(ep, end_reason="timeout_alarm" if alarm else ("error" if recorder_error else "exception"),
                    exc=None if recorder_error else exc, extra={"end_note": note})
    elif not slot.start_queued:
        ep.release_references()  # nothing was queued: the writer never sees this episode
    if not (recorder_error or not slot.start_queued):
        return
    store = get_store()
    if store is None:
        return
    if recorder_error:
        data = internal_error_data(exc, where)
    else:
        data = {"code": "internal_error", "message": note + "; nothing of this attempt was recorded"}
    data.update({"outside_episode": True, "episode_id": ep.episode_id, "alarm": alarm, "note": note})
    C.unscoped_log(store, ep.run_id).emit("trace_warning", status="warning", data=data)


def _end_stale_episodes(stale: list[Any]) -> None:
    """End an earlier episode that was left open on this thread.

    That happens only when an alarm interrupted H4's own exit path. The episode
    would otherwise stay open (and look running) for the rest of the process.
    """
    for entry in stale:
        if not isinstance(entry, EpisodeRecorder) or entry.ended:
            continue
        try:
            end_episode(entry, end_reason="error",
                        extra={"end_note": "left open on the trial thread; ended when the next "
                                           "episode started (the trial's own end was not recorded)"})
        finally:
            tap = entry.frame_tap
            if tap is not None:
                tap.detach(entry)
            C.close_episode(entry)


# What ``TrialSummary.success`` means (docs/trace-format.md, section 5.2). The two
# harnesses set it differently: ``capx/envs/trial.py`` from the last block's sandbox_rc
# after its terminated-episode override, ``capx/web/async_trial_runner.py`` from
# ``task_completed or num_finishes > 0``.
SUCCESS_MEANING_BATCH = ("sandbox_rc == 0 of the last code block that ran (0 also when its stderr says "
                         "'executing action in terminated episode'); earlier blocks may have failed; "
                         "not task success")
SUCCESS_MEANING_WEB_UI = ("task_completed or num_finishes > 0, as the web UI runner computes it (its "
                          "task_completed also counts terminated with reward > 0); sandbox_rc is not used")

_VIDEO_NAME = re.compile(r"video_(?:turn_(\d+)(?:_[\w-]+)?|combined(?:_[\w-]+)?)\.mp4\Z", re.ASCII)
_VIDEO_MTIME_TOLERANCE_S = 5.0  # filesystem timestamp rounding and short write delays


def _video_warn(ctx: Any, code: str, reason: str, name: str | None = None) -> None:
    ctx.wep.sink.emit("trace_warning", status="warning", writer_generated=True,
                      data={"code": code, "reason": reason, "file": name,
                            "message": f"video {name or 'directory'}: {reason}",
                            "counts_as_gap": False, "writer_generated": True})


def _video_copy_supported() -> bool:
    return os.name != "nt"


def _copy_trial_videos(ctx: Any, code_path: Path, block_count: int, started_wall: float) -> None:
    """Copy each verified source through a bounded buffer on the writer thread."""
    if not _video_copy_supported():
        _video_warn(ctx, "video_skipped", "unsupported_platform")
        return
    import fcntl

    if ctx.store.config.video_file_bytes <= 0:
        _video_warn(ctx, "video_size_limit", "disabled")
        return
    trial_dir = code_path.parent
    try:
        reference_mtime = code_path.stat().st_mtime
    except OSError:  # timeout-ok: video copying runs only on the writer thread
        reference_mtime = started_wall
    try:
        parent = os.open(trial_dir.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            directory = os.open(trial_dir.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
        finally:
            os.close(parent)
    except OSError:  # timeout-ok: video copying runs only on the writer thread
        _video_warn(ctx, "video_missing", "directory_missing")
        return
    try:
        try:
            files = sorted(name for name in os.listdir(directory) if _VIDEO_NAME.fullmatch(name))
        except OSError:  # timeout-ok: video copying runs only on the writer thread
            _video_warn(ctx, "video_missing", "directory_unreadable")
            return
        if not files:
            _video_warn(ctx, "video_missing", "no_video_files")
        disk_warned = False
        for name in files:
            match = _VIDEO_NAME.fullmatch(name)
            assert match is not None
            turn = int(match.group(1)) if match.group(1) is not None else None
            art = {"path": f"videos/{name}", "bytes": None, "sha256": None,
                   "mime": "video/mp4", "kind": "video", "label": "trial_video",
                   "written": False, "turn_index": turn}
            ctx.event["artifacts"].append(art)
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            except OSError as exc:  # timeout-ok: video copying runs only on the writer thread
                if exc.errno == errno.ELOOP:
                    art["reason"] = "not_regular"
                    _video_warn(ctx, "video_skipped", "not_regular", name)
                else:
                    art["reason"] = f"read_failed: {type(exc).__name__}"
                    ctx.wep.add_gap("read_failed", 1, ctx.event["seq"])
                    _video_warn(ctx, "video_skipped", str(art["reason"]), name)
                continue
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    art["reason"] = "not_regular"
                    _video_warn(ctx, "video_skipped", "not_regular", name)
                    continue
                if turn is not None and turn >= block_count:
                    art["reason"] = "turn_index_out_of_range"
                    _video_warn(ctx, "video_skipped", "turn_index_out_of_range", name)
                    continue
                if info.st_mtime < reference_mtime - _VIDEO_MTIME_TOLERANCE_S:
                    art["reason"] = "stale"
                    _video_warn(ctx, "video_skipped", "stale", name)
                    continue
                fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
                if info.st_size > ctx.store.config.video_file_bytes:
                    art["reason"] = "size_limit"
                    _video_warn(ctx, "video_size_limit", "size_limit", name)
                    continue
                target = ctx.base_dir / "videos" / name
                ok, reason = ctx.store.disk_allows(True, trigger=ctx.wep.sink)
                if not ok or shutil.disk_usage(ctx.base_dir).free - info.st_size < ctx.store.config.min_free_bytes:
                    art["reason"] = reason or "disk_low"
                else:
                    tmp = None
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_name(f".{name}.tmp.{os.getpid()}.{ctx.event['seq']}")
                        digest = hashlib.sha256()
                        copied = 0
                        last_heartbeat = 0.0
                        with open(tmp, "xb") as out:
                            while copied < info.st_size:
                                now = time.monotonic()
                                if now - last_heartbeat >= 10.0:
                                    try:
                                        os.utime(ctx.base_dir / "heartbeat", None)
                                    except OSError:  # timeout-ok: writer thread
                                        pass
                                    ctx.store.refresh_heartbeats()
                                    last_heartbeat = now
                                try:
                                    chunk = os.read(fd, min(1024 * 1024, info.st_size - copied))
                                except OSError as exc:
                                    art["reason"] = f"read_failed: {type(exc).__name__}"
                                    raise
                                if not chunk:
                                    break
                                copied += len(chunk)
                                digest.update(chunk)
                                out.write(chunk)
                        if copied != info.st_size:
                            art["reason"] = "read_failed: source_changed"
                            raise OSError("source changed during copy")
                        os.replace(tmp, target)
                        art.update(written=True, bytes=copied, sha256=digest.hexdigest())
                    except OSError as exc:  # timeout-ok: video copying runs only on the writer thread
                        art.setdefault("reason", f"write_failed: {type(exc).__name__}")
                    finally:
                        if tmp is not None:
                            try:
                                tmp.unlink(missing_ok=True)
                            except OSError:  # timeout-ok: video copying runs only on the writer thread
                                pass
                if not art["written"]:
                    gap = "disk_low" if art["reason"] == "disk_low" else ("read_failed" if str(art["reason"]).startswith("read_failed") else "write_failed")
                    ctx.wep.add_gap(gap, 1, ctx.event["seq"])
                    if art["reason"] == "disk_low":
                        if not disk_warned:
                            _video_warn(ctx, "disk_low", "disk_low", name)
                            disk_warned = True
                    _video_warn(ctx, "video_skipped", str(art["reason"]), name)
            except OSError as exc:  # timeout-ok: video copying runs only on the writer thread
                art["reason"] = f"read_failed: {type(exc).__name__}"
                ctx.wep.add_gap("read_failed", 1, ctx.event["seq"])
                _video_warn(ctx, "video_skipped", str(art["reason"]), name)
            finally:
                os.close(fd)
    finally:
        os.close(directory)


def summary_fields(summary: Any, source: str | None = None) -> tuple[dict[str, Any] | None, list[Any]]:
    """Fields of the returned ``TrialSummary`` (``log`` goes to an asset).

    ``source`` is the episode's source; a ``web_ui`` summary gets the web UI runner's
    meaning of ``success``, any other the batch harness's.
    """
    pending: list[Any] = []
    if summary is None or not dataclasses.is_dataclass(summary):
        return None, pending
    out: dict[str, Any] = {}
    for f in dataclasses.fields(summary):
        value = getattr(summary, f.name, None)
        if isinstance(value, str):
            out[f.name] = text_value(value, pending, "text")
        else:
            out[f.name] = json_safe(value)
    out["success_meaning"] = SUCCESS_MEANING_WEB_UI if source == "web_ui" else SUCCESS_MEANING_BATCH
    return out, pending


def end_episode(ep: Any, *, end_reason: str, summary: Any = None, exc: BaseException | None = None,
                extra: dict[str, Any] | None = None, video_dir: Path | None = None,
                code_path: Path | None = None) -> None:
    """Queue the result, end, and optional video event without waiting for the writer."""
    harness, pending = summary_fields(summary, getattr(ep, "source", None))
    if harness is not None:
        ep.emit("evaluator_result", data={
            "scope": "episode", "code_block_id": None,
            "reward": harness.get("reward"), "task_completed": harness.get("task_completed"),
            "terminated": harness.get("terminated"), "truncated": harness.get("truncated"),
            "source": "TrialSummary returned by the harness",
        })
    exception = None
    exc_message = None
    if exc is not None:
        from capx_trace_recorder.hooks.api import safe_str

        exc_message = safe_str(exc)  # redacted, then cut to 2000 characters, by the writer
        exception = {"type": type(exc).__name__, "message": None}
    data = {
        "end_reason": end_reason,
        "harness_summary": harness,
        "exception": exception,
        "alarm_seen": bool(ep.alarm_seen or is_trial_alarm(exc)),
        "duration_s": round(time.monotonic() - ep.started_mono, 6),
        "n_code_blocks": ep.n_code_blocks,
        "n_model_calls": ep.n_logical_calls,
        "n_http_attempts": ep.n_http_attempts,
        "tracer_overhead_s": round(ep.overhead_trial, 6),
        "tracer_overhead_all_threads_s": round(ep.overhead_all, 6),
    }
    if extra:
        data.update(extra)

    def builder(ctx, pending=pending, exc_message=exc_message):
        ctx.fill_pending(pending)
        if exc_message is not None:
            from capx_trace_recorder.payload import put_redacted

            put_redacted(ctx.data["exception"], "message", exc_message, 2000)

    status = "ok" if end_reason == "completed" else ("error" if end_reason in ("exception", "error", "timeout_alarm") else "info")
    with ep._lock:
        if video_dir is not None:
            ep.internal(lambda ctx: setattr(ctx.wep, "video_copy_pending", True))
        ep.emit("episode_end", status=status, data=data, builder=builder)
        ep.ended = True
        if video_dir is not None:
            block_count, started_wall = ep.n_code_blocks, ep.started_wall
            def copy_builder(ctx):
                _copy_trial_videos(ctx, code_path or video_dir / "code.py", block_count, started_wall)
            ep.emit("trial_videos", data={}, recorder_owned=True, builder=copy_builder)


def _episode_finish(ep: Any, summary: Any, exc: BaseException | None) -> None:
    if exc is None:
        reason = "completed"
    elif is_trial_alarm(exc):
        reason = "timeout_alarm"
    else:
        reason = "exception"
    try:
        # contexts this episode left on the trial thread (an interrupted wrapper)
        stale = C.clear_stale(("frames", "roles"))
        if stale:
            _stale_warning(ep, stale, "when this episode ended", True)
        video_dir = None
        if (exc is None and ep.source == "envs.runner" and ep.config_flags.get("record_video")
                and summary is not None):
            code_path = getattr(summary, "code_path", None)
            if isinstance(code_path, (str, os.PathLike)) and code_path:
                code_path = Path(code_path).absolute()
                video_dir = code_path.parent
        end_episode(ep, end_reason=reason, summary=summary, exc=exc, video_dir=video_dir,
                    code_path=code_path if video_dir is not None else None)
    finally:
        _close_trial_episode(ep)


def _close_trial_episode(ep: Any) -> None:
    """Take the episode off this thread's stack and out of the open set (idempotent:
    it runs again when an alarm cut the first exit short)."""
    try:
        C.pop_episode(ep)
    finally:
        try:
            tap = ep.frame_tap
            if tap is not None:
                tap.detach(ep)
        finally:
            C.close_episode(ep)


def _trial_exit(slot: Any, summary: Any, exc: BaseException | None) -> None:
    """End the episode of ``slot`` (once), or undo a start that did not complete."""
    ep = slot.ep
    if ep is None or slot.abandoned:
        return
    if not slot.started:
        _abandon_start(ep, slot, exc)  # interrupted between open_episode and the push
        return
    if ep.ended:
        _close_trial_episode(ep)  # ended, but an alarm may have cut the cleanup short
        return
    _episode_finish(ep, summary, exc)


def make_run_single_trial_wrapper(original: Any) -> Any:
    sig = inspect.signature(original)

    @functools.wraps(original)
    def _run_single_trial(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        slot = C.StartSlot()
        try:
            try:
                # the start, the trial and the end are inside these try statements, so
                # an alarm anywhere after open_episode still ends or undoes the episode
                guard(_episode_start, sig, args, kwargs, slot)
                if slot.started:
                    add_overhead(slot.ep, t0)
                summary = original(*args, **kwargs)  # untraced when the start did not complete
            except BaseException as exc:
                t1 = time.perf_counter()
                guard(_trial_exit, slot, None, exc)
                add_overhead(slot.ep, t1)
                raise
            guard(_trial_exit, slot, summary, None)
        except BaseException as late:
            # an alarm that landed after the original returned, before or inside the
            # exit above: the episode still ends (as a timeout) and leaves the stack
            guard(_trial_exit, slot, None, late)
            raise
        return summary

    return mark(_run_single_trial, original)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

_TRIAL_ALIASES = ("_query_model", "_query_model_ensemble", "_query_single_model_ensemble",
                  "_build_multi_turn_decision_prompt", "_build_multi_turn_decision_prompt_legacy",
                  "_extract_code", "_parse_multi_turn_decision", "_encode_video_base64")


def install_roles(trial_module: Any) -> list[str]:
    problems = []
    for name, (role, kind, turn_mode) in _ROLE_TARGETS.items():
        err = patch_attribute(trial_module, name,
                              lambda orig, r=role, k=kind, t=turn_mode: make_role_wrapper(orig, r, k, t))
        if err:
            problems.append(err)
    err = patch_attribute(trial_module, "_patch_libero_goal", make_patch_libero_goal_wrapper)
    if err:
        problems.append(err)
    rebind_aliases(trial_module, _TRIAL_ALIASES)
    return problems


def install_episode(trial_module: Any) -> list[str]:
    original = getattr(trial_module, "_run_single_trial", None)
    if original is None:
        return ["_run_single_trial: not found"]
    try:
        params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        params = []
    if params[:4] != ["env", "trial", "args", "config"]:
        return [f"_run_single_trial has an unexpected signature {params}"]
    err = patch_attribute(trial_module, "_run_single_trial", make_run_single_trial_wrapper)
    return [err] if err else []


def install_runner_aliases(module: Any) -> list[str]:
    rebind_aliases(module, ("_run_single_trial",))
    return []


__all__ = ["install_roles", "install_episode", "install_runner_aliases", "end_episode",
           "attach_frames", "env_info", "tracer_info", "resolve_run_id", "config_flags",
           "models_of", "summary_fields", "SUCCESS_MEANING_BATCH", "SUCCESS_MEANING_WEB_UI"]
