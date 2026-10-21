"""Activation, post-import hooks and install order (docs/trace-format.md, sections 3 and 7).

The .pth startup module calls ``activate()`` when ``CAPX_TRACE_DIR`` is set.
Viewer servers are recognised from ``sys.orig_argv``. Otherwise ``activate()``:

1. reads ``TraceConfig`` (a relative trace dir is resolved against the working
   directory and written back to ``os.environ``) and creates the store (no thread,
   no file yet);
2. exports ``CAPX_TRACE_FALLBACK_RUN_ID`` once, in the first process, so that
   spawned workers of a ``launch.py`` run share one run id;
3. installs H11 on the stdlib ``traceback`` module and runs its self-test;
4. inserts ``PostImportFinder`` at the front of ``sys.meta_path``. For an
   explicit set of module names it delegates to the other finders and wraps
   the per-module loader's ``exec_module`` so that the patch runs right after
   the module body executes and before any other module can bind names from
   it. Every other import is left alone. A target already in ``sys.modules`` is
   patched at once. ``activate()`` never imports a capx hook target.

Hooks that put recorder frames on the stack while generated code runs (H1, H4,
H7 and the lazily installed H8, H9, H10) are installed only when H11 passed.
The spec names H1, H8, H9 and H10; H4 and H7 are added here because their
frames are also on the stack during ``exec`` (visible to ``print_stack``).

Related: every module in ``capx_trace_recorder/hooks/``; ``capx_trace_recorder/tbfilter.py``.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import threading
from typing import Any, Callable

from capx_trace_recorder import context as C
from capx_trace_recorder.config import ENV_FALLBACK_RUN_ID, TraceConfig
from capx_trace_recorder.ids import fallback_run_id
from capx_trace_recorder.store import init_store

_ACTIVATED = False
_H11_OK = False
_FINDER: "PostImportFinder | None" = None
_CAPX_FINDER: "_FirstCapxFinder | None" = None
_PATCH_LOCK = threading.RLock()
_ACTIVE_RECORDER = "inactive"
_STAND_DOWN_REASON = ""
# H2's LIBERO goal helper is absent in CaP-X variants without that task.
# Its absence does not affect the other trial roles. All other symbols are
# required once their target module is imported. Experiment modules are
# optional as modules and are never patched unless imported.
OPTIONAL_SYMBOLS: dict[tuple[str, str], frozenset[str]] = {
    ("H2", "capx.envs.trial"): frozenset({"_patch_libero_goal"}),
}


def _record_choice(active: str, reason: str) -> None:
    """Keep the process decision in memory without writing at interpreter startup."""
    global _ACTIVE_RECORDER, _STAND_DOWN_REASON
    _ACTIVE_RECORDER, _STAND_DOWN_REASON = active, reason


def _write_status() -> None:
    """Write the active public state after CaP-X is requested for import."""
    if _ACTIVE_RECORDER != "public":
        return
    try:
        import json
        from pathlib import Path

        config = TraceConfig.from_env(write_back=False)
        if config is None:
            return
        root = Path(config.trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"recorder_status.{os.getpid()}.json").write_text(
            json.dumps({"active": _ACTIVE_RECORDER, "reason": _STAND_DOWN_REASON}, ensure_ascii=True) + "\n",
            encoding="utf-8")
    except TimeoutError:
        raise
    except (KeyError, OSError, ValueError):
        pass


def frame_hooks_allowed() -> bool:
    """H1/H4/H7/H8/H9/H10 may be installed only when H11 is in place."""
    return _H11_OK


def _record_installed(hook: str, target: str) -> None:
    with C.REGISTRY.lock:
        entry = f"{hook}:{target}"
        if entry in C.REGISTRY.hooks_installed:
            return
        C.REGISTRY.hooks_installed.append(entry)
    _notify_open_episodes(entry, None)


def _record_partial(hook: str, target: str, module: Any, names: tuple[str, ...]) -> None:
    """Report wrappers that did install when another symbol in the target failed."""
    for name in names:
        obj = module
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except (AttributeError, TypeError):
            continue
        if getattr(obj, "__capx_trace_wrapper__", False):
            _record_installed(hook, f"{target}.{name}")


def _record_failed(hook: str, target: str, reason: str) -> None:
    from capx_trace_recorder.payload import put_redacted

    failure: dict[str, Any] = {"hook": hook, "target": target}
    put_redacted(failure, "reason", str(reason), 300)  # adds reason_redacted when a secret was replaced
    with C.REGISTRY.lock:
        C.REGISTRY.hooks_failed.append(failure)
    _notify_open_episodes(None, failure)
    log_failures_to_run()
    if hook == "H4":
        _write_orphan_h4_warning(failure)


def _write_orphan_h4_warning(failure: dict[str, Any]) -> None:
    """Explain an empty trace root when the episode-opening hook cannot install."""
    try:
        import json
        from pathlib import Path

        root = os.environ.get("CAPX_TRACE_DIR", "").strip()
        if not root:
            return
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        warning = path / f"recorder_warnings.{os.getpid()}.jsonl"
        with warning.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"code": "hook_install_failed", **failure}, ensure_ascii=True) + "\n")
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001 - a warning write cannot break the target import
        pass


def _record_problem(hook: str, target: str, reason: str, module: Any) -> None:
    """Classify an absent allow-listed symbol separately from an install failure."""
    symbol = reason.split(":", 1)[0]
    obj = module
    missing = False
    for part in symbol.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            missing = True
            break
    if not (missing and symbol in OPTIONAL_SYMBOLS.get((hook, target), frozenset())):
        _record_failed(hook, target, reason)
        return
    from capx_trace_recorder.payload import put_redacted

    skipped: dict[str, Any] = {"hook": hook, "target": target}
    put_redacted(skipped, "reason", str(reason), 300)
    with C.REGISTRY.lock:
        C.REGISTRY.hooks_skipped.append(skipped)
    try:
        episodes = C.REGISTRY.open_episodes()
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001 - never interrupt the triggering import
        return
    for ep in episodes:
        try:
            def builder(ctx, entry=skipped):
                done = list(ctx.wep.summary.get("hooks_skipped_during_episode") or [])
                done.append(dict(entry))
                ctx.store.update_summary(ctx.wep, {"hooks_skipped_during_episode": done})

            ep.internal(builder)
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            continue


def log_failures_to_run() -> None:
    """Write hook failures to the run's ``unscoped.<pid>.jsonl`` (section 2, honesty).

    ``episode_start`` lists the failures known when an episode starts, but a
    failure that prevents every episode (H4 not installed) would otherwise
    reach no file. In the experiment runner, whose run id H15 learns before the
    preflight, each failure is written once to the run's unscoped log; H15
    calls this again when it learns the run id. Other processes (a perception
    server that inherited ``CAPX_TRACE_DIR``) have no run and write nothing.
    """
    try:
        sink = C.unscoped_sink_for([])
        if sink is None:
            return
        with C.REGISTRY.lock:
            pending = C.REGISTRY.hooks_failed[C.REGISTRY.failures_logged:]
            C.REGISTRY.failures_logged = len(C.REGISTRY.hooks_failed)
        for failure in pending:
            data = {"code": "hook_install_failed", "hook": failure["hook"], "target": failure["target"],
                    "message": failure["reason"], "run_level": True}
            if failure.get("reason_redacted"):
                data["message_redacted"] = True
            sink.emit("trace_warning", status="warning", data=data)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001 - never break the import that triggered this
        return


def _notify_open_episodes(installed: str | None, failure: dict[str, str] | None) -> None:
    """A hook installed or failed while episodes are open (section 7.1).

    ``episode_start`` lists the hooks known when the episode began. In the web
    UI server the trial module and the env classes are first imported inside
    the episode, so H2, H4 and H7 are installed after that list was taken. An
    install is added to ``episode.json`` (``hooks_installed_during_episode``); a
    failure becomes a ``hook_install_failed`` warning, so the episode is not
    shown as complete.
    """
    try:
        episodes = C.REGISTRY.open_episodes()
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return
    for ep in episodes:
        try:
            if failure is not None:
                data = {"code": "hook_install_failed", "hook": failure["hook"], "target": failure["target"],
                        "message": failure["reason"], "during_episode": True}
                if failure.get("reason_redacted"):
                    data["message_redacted"] = True
                ep.emit("trace_warning", status="warning", data=data)
                continue

            def builder(ctx, entry=installed):
                done = list(ctx.wep.summary.get("hooks_installed_during_episode") or [])
                if entry not in done:
                    done.append(entry)
                ctx.store.update_summary(ctx.wep, {"hooks_installed_during_episode": done})

            ep.internal(builder)
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 - never break the import that triggered this
            continue


# ---------------------------------------------------------------------------
# Patch functions (one per target module)
# ---------------------------------------------------------------------------


def _requires_h11(hook: str, target: str) -> bool:
    if _H11_OK:
        return True
    _record_failed(hook, target, "traceback protection (H11) not installed")
    return False


def _patch_requests(module: Any) -> None:
    if not _requires_h11("H1", "requests.sessions.Session.send"):
        return
    from capx_trace_recorder.hooks import http

    err = http.install(module)
    if err:
        _record_failed("H1", "requests.sessions.Session.send", err)
    else:
        _record_installed("H1", "requests.sessions.Session.send")


def _patch_client(module: Any) -> None:
    from capx_trace_recorder.hooks import llm

    problems = llm.install(module)
    for p in problems:
        _record_problem("H3", "capx.llm.client", p, module)
    if problems:
        _record_partial("H3", "capx.llm.client", module,
                        ("query_model", "query_model_streaming", "query_model_ensemble",
                         "query_single_model_ensemble"))
    if not problems:
        _record_installed("H3", "capx.llm.client.query_model/query_model_streaming")


def _patch_launch_utils(module: Any) -> None:
    from capx_trace_recorder.hooks import prompts

    problems = prompts.install(module)
    for p in problems:
        _record_problem("H5/H6", "capx.utils.launch_utils", p, module)
    if problems:
        _record_partial("H5/H6", "capx.utils.launch_utils", module,
                        ("_build_multi_turn_decision_prompt",
                         "_build_multi_turn_decision_prompt_legacy",
                         "_parse_multi_turn_decision", "_extract_code"))
    if not problems:
        _record_installed("H5", "capx.utils.launch_utils._build_multi_turn_decision_prompt(+_legacy)")
        _record_installed("H6", "capx.utils.launch_utils._parse_multi_turn_decision/_extract_code")


def _patch_video_utils(module: Any) -> None:
    from capx_trace_recorder.hooks import video

    problems = video.install(module)
    for p in problems:
        _record_problem("H12", "capx.utils.video_utils._encode_video_base64", p, module)
    if not problems:
        _record_installed("H12", "capx.utils.video_utils._encode_video_base64")


def _patch_trial(module: Any) -> None:
    from capx_trace_recorder.hooks import trial

    problems = trial.install_roles(module)
    for p in problems:
        _record_problem("H2", "capx.envs.trial", p, module)
    if problems:
        _record_partial("H2", "capx.envs.trial", module,
                        (*tuple(trial._ROLE_TARGETS), "_patch_libero_goal"))
    if not problems:
        _record_installed("H2", "capx.envs.trial role helpers and _patch_libero_goal")
    if _requires_h11("H4", "capx.envs.trial._run_single_trial"):
        problems = trial.install_episode(module)
        for p in problems:
            _record_problem("H4", "capx.envs.trial._run_single_trial", p, module)
        if not problems:
            _record_installed("H4", "capx.envs.trial._run_single_trial")


def _patch_runner_aliases(module: Any) -> None:
    from capx_trace_recorder.hooks import trial

    trial.install_runner_aliases(module)


def _patch_env_base(module: Any) -> None:
    if not _requires_h11("H7", "capx.envs.tasks.base.CodeExecutionEnvBase"):
        _record_failed("H8/H9/H10", "API methods, service closures, _record_frame",
                       "traceback protection (H11) not installed")
        return
    from capx_trace_recorder.hooks import env

    problems = env.install(module)
    for p in problems:
        _record_problem("H7", "capx.envs.tasks.base.CodeExecutionEnvBase", p, module)
    if problems:
        _record_partial("H7", "capx.envs.tasks.base", module,
                        tuple("CodeExecutionEnvBase." + name for name in
                              ("step", "_exec_user_code", "render", "render_wrist",
                               "get_video_frames_range", "get_wrist_video_frames_range", "reset")))
    if not problems:
        _record_installed("H7", "capx.envs.tasks.base.CodeExecutionEnvBase")


def _patch_web_runner(module: Any) -> None:
    from capx_trace_recorder.hooks import web

    problems = web.install_runner(module)
    for p in problems:
        _record_problem("H13", "capx.web.async_trial_runner", p, module)
    if problems:
        _record_partial("H13", "capx.web.async_trial_runner", module,
                        ("run_trial_async", "_query_model", "_query_model_streaming"))
    if not problems:
        _record_installed("H13", "capx.web.async_trial_runner.run_trial_async")


def _patch_session_manager(module: Any) -> None:
    from capx_trace_recorder.hooks import web

    problems = web.install_session_manager(module)
    for p in problems:
        _record_problem("H13", "capx.web.session_manager", p, module)
    if problems:
        _record_partial("H13", "capx.web.session_manager", module,
                        ("SessionManager.inject_prompt", "SessionManager.stop_session"))
    if not problems:
        _record_installed("H13", "capx.web.session_manager.SessionManager.inject_prompt/stop_session")


def _patch_molmo(module: Any) -> None:
    from capx_trace_recorder.hooks import molmo

    problems = molmo.install(module)
    for p in problems:
        _record_problem("H14", "capx.integrations.vision.molmo._parse_points", p, module)
    if not problems:
        _record_installed("H14", "capx.integrations.vision.molmo._parse_points")


def _patch_revision(module: Any) -> None:
    from capx_trace_recorder.hooks import runner

    problems = runner.install_revision(module)
    for p in problems:
        _record_problem("H15", "capx.experiments.revision.fingerprint", p, module)
    if not problems:
        _record_installed("H15", "capx.experiments.revision.fingerprint")


def _patch_manifest(module: Any) -> None:
    from capx_trace_recorder.hooks import runner

    problems = runner.install_manifest(module)
    for p in problems:
        _record_problem("H15", "capx.experiments.manifest.load_manifest", p, module)
    if not problems:
        _record_installed("H15", "capx.experiments.manifest.load_manifest")


def _patch_records(module: Any) -> None:
    from capx_trace_recorder.hooks import runner

    problems = runner.install_records(module)
    for p in problems:
        _record_problem("H16", "capx.experiments.records.append_jsonl", p, module)
    if not problems:
        _record_installed("H16", "capx.experiments.records.append_jsonl")


TARGETS: dict[str, Callable[[Any], None]] = {
    "requests.sessions": _patch_requests,
    "capx.llm.client": _patch_client,
    "capx.utils.launch_utils": _patch_launch_utils,
    "capx.utils.video_utils": _patch_video_utils,
    "capx.envs.trial": _patch_trial,
    "capx.envs.runner": _patch_runner_aliases,
    "capx.envs.tasks.base": _patch_env_base,
    "capx.web.async_trial_runner": _patch_web_runner,
    "capx.web.session_manager": _patch_session_manager,
    "capx.integrations.vision.molmo": _patch_molmo,
    "capx.experiments.revision": _patch_revision,
    "capx.experiments.manifest": _patch_manifest,
    "capx.experiments.records": _patch_records,
}


def run_patch(name: str, module: Any) -> None:
    """Apply the patch for ``name`` to ``module``.

    Every patch is idempotent (wrappers are marked and never wrapped twice), so
    running it again after ``importlib.reload`` wraps the fresh functions.
    """
    patch = TARGETS.get(name)
    if patch is None:
        return
    if _bundled_active():
        return
    try:
        with _PATCH_LOCK:
            patch(module)
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - never break the import
        _record_failed("install", name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Post-import finder
# ---------------------------------------------------------------------------


class _LoaderProxy:
    """Fallback for loaders that are not per-module ``FileLoader`` instances."""

    def __init__(self, loader: Any, fullname: str) -> None:
        self._capx_trace_loader = loader
        self._capx_trace_name = fullname

    def create_module(self, spec: Any) -> Any:
        create = getattr(self._capx_trace_loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self._capx_trace_loader.exec_module(module)
        run_patch(self._capx_trace_name, module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capx_trace_loader, name)


_PER_MODULE_LOADERS = (
    importlib.machinery.SourceFileLoader,
    importlib.machinery.SourcelessFileLoader,
    importlib.machinery.ExtensionFileLoader,
)


def _hook_loader(spec: Any, fullname: str) -> None:
    """Run the patch right after ``exec_module`` of this module's loader."""
    loader = spec.loader
    exec_module = getattr(loader, "exec_module", None)
    if exec_module is None or isinstance(loader, _LoaderProxy):
        return
    if isinstance(loader, _PER_MODULE_LOADERS) and "exec_module" not in vars(loader):
        # FileFinder and the editable-install finder create one loader per
        # module, so an instance attribute affects this module only.
        def patched_exec_module(module: Any, _exec=exec_module, _name=fullname) -> None:
            _exec(module)
            run_patch(_name, module)

        loader.exec_module = patched_exec_module
    else:
        spec.loader = _LoaderProxy(loader, fullname)


class PostImportFinder(importlib.abc.MetaPathFinder):
    """Patch an explicit set of modules right after their body executes."""

    def __init__(self, names: set[str]) -> None:
        self._names = frozenset(names)
        self._local = threading.local()

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in self._names:
            return None
        active = self._local.__dict__.setdefault("active", set())
        if fullname in active:
            return None
        active.add(fullname)
        try:
            spec = self._delegate(fullname, path, target)
        finally:
            active.discard(fullname)
        if spec is None or spec.loader is None:
            return spec
        try:
            _hook_loader(spec, fullname)
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - the import itself must go on unchanged
            _record_failed("install", fullname, f"post-import hook not set: {type(exc).__name__}")
        return spec

    def _delegate(self, fullname: str, path: Any, target: Any) -> Any:
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is not None:
                spec = find(fullname, path, target)
                if spec is not None:
                    return spec
        return None

    def invalidate_caches(self) -> None:
        return None


# ---------------------------------------------------------------------------
# activate()
# ---------------------------------------------------------------------------


def _bundled_active() -> bool:
    module = sys.modules.get("capx.tracing.bootstrap")
    return bool(module is not None and callable(getattr(module, "is_active", None))
                and module.is_active())


def _has_bundled_recorder(origin: str | None) -> bool:
    if not origin:
        return False
    package = os.path.dirname(origin)
    return os.path.isdir(os.path.join(package, "tracing"))


class _FirstCapxFinder(importlib.abc.MetaPathFinder):
    """Resolve a late CaP-X path before enabling public hooks."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "capx":
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        global _CAPX_FINDER
        _CAPX_FINDER = None
        spec = importlib.util.find_spec("capx")
        if spec is not None:
            if _ACTIVE_RECORDER == "public":
                _write_status()
            elif _has_bundled_recorder(spec.origin):
                _record_choice("bundled", "public recorder stood down: capx has a bundled recorder")
            else:
                _start_public(write_status=True)
        return spec


def is_viewer_only_process(argv: list[str] | None = None) -> bool:
    """A standalone viewer process (``capx-trace-viewer`` or ``capx_trace_viewer.server``).

    The viewer does not activate the recorder, even when ``CAPX_TRACE_DIR`` is
    exported in the shell that starts it. The
    interpreter's own command line (``sys.orig_argv``, Python 3.10+) is read,
    because ``import capx`` runs before the module parses its arguments.
    """
    args = list(argv if argv is not None else (getattr(sys, "orig_argv", None) or []))
    script = args[1] if len(args) > 1 and os.path.basename(args[0]).startswith("python") else (
        args[0] if args else "")
    if os.path.basename(script) == "capx-trace-viewer":
        return True
    if any(a == "-m" and i + 1 < len(args) and args[i + 1] == "capx_trace_viewer.server"
           or a == "-mcapx_trace_viewer.server" or
           a.replace("\\", "/").endswith("capx_trace_viewer/server.py")
           for i, a in enumerate(args)):
        return True
    return False


def activate() -> None:
    """Enable tracing for this process (idempotent; never raises)."""
    global _ACTIVATED, _CAPX_FINDER
    if _ACTIVATED or not os.environ.get("CAPX_TRACE_DIR", "").strip():
        return
    try:
        if is_viewer_only_process():
            return
        if _bundled_active():
            _record_choice("bundled", "public recorder stood down: bundled recorder already active")
            return
        module = sys.modules.get("capx")
        if module is None:
            _watch_first_capx_import()
            return
        if _has_bundled_recorder(getattr(module, "__file__", None)):
            _record_choice("bundled", "public recorder stood down: capx has a bundled recorder")
            return
        _start_public(write_status=True)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001 - tracing must never break ``import capx``
        pass


def _watch_first_capx_import() -> None:
    global _CAPX_FINDER
    if _CAPX_FINDER is None:
        _CAPX_FINDER = _FirstCapxFinder()
        sys.meta_path.insert(0, _CAPX_FINDER)


def _start_public(*, write_status: bool = False) -> None:
    global _ACTIVATED
    if _ACTIVATED:
        return
    _ACTIVATED = True
    _activate()
    if is_active():
        _record_choice("public", "public recorder active: capx has no bundled recorder")
        if write_status:
            _write_status()


def _activate() -> None:
    global _H11_OK, _FINDER
    config = TraceConfig.from_env()
    if config is None:
        return
    init_store(config)
    if not os.environ.get(ENV_FALLBACK_RUN_ID):
        os.environ[ENV_FALLBACK_RUN_ID] = fallback_run_id()
    from capx_trace_recorder import tbfilter
    from capx_trace_recorder.hooks.common import selftest_wrap

    ok, reason = tbfilter.install(selftest_wrap)
    _H11_OK = ok
    if ok:
        _record_installed("H11", "traceback.walk_tb/walk_stack/_walk_tb_with_full_positions")
    else:
        _record_failed("H11", "traceback", reason)
    finder = PostImportFinder(set(TARGETS))
    _FINDER = finder
    sys.meta_path.insert(0, finder)
    for name in TARGETS:
        module = sys.modules.get(name)
        if module is not None:
            run_patch(name, module)


def is_active() -> bool:
    return _FINDER is not None


def recorder_status() -> tuple[str, str]:
    return _ACTIVE_RECORDER, _STAND_DOWN_REASON


__all__ = ["activate", "frame_hooks_allowed", "PostImportFinder", "TARGETS", "run_patch",
           "is_active", "recorder_status"]
