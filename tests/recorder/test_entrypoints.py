"""Recorder startup and public command tests."""

import sys
import types
import os
import subprocess
from pathlib import Path

import pytest

from capx_trace_recorder import __main__ as cli
from capx_trace_recorder import bootstrap
from capx_trace_recorder import context
from capx_trace_recorder import store


def test_pth_off_and_never_raises(monkeypatch, capsys):
    monkeypatch.delenv("CAPX_TRACE_DIR", raising=False)
    sys.modules.pop("capx_trace_recorder.pth", None)
    import capx_trace_recorder.pth as pth

    assert capsys.readouterr().err == ""
    monkeypatch.setenv("CAPX_TRACE_DIR", "trace")
    monkeypatch.setattr(bootstrap, "activate", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    pth.start()
    assert "activation failed: RuntimeError" in capsys.readouterr().err
    monkeypatch.setattr(bootstrap, "activate", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        pth.start()


def test_real_pth_is_silent_when_off_and_after_uninstall(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "capx_trace_recorder.pth").write_text(cli.PTH_LINE)
    # site.addsitedir processes .pth files in a temporary directory; no shared
    # interpreter installation is changed.
    code = f"import site; site.addsitedir({str(site)!r}); print('ok')"
    env = os.environ.copy()
    env.pop("CAPX_TRACE_DIR", None)
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], env=env, text=True,
                          capture_output=True, timeout=10)
    assert proc.returncode == 0 and proc.stdout == "ok\n" and proc.stderr == ""
    # The package is absent in isolated mode; an orphaned hook prints one line.
    env["CAPX_TRACE_DIR"] = str(tmp_path / "trace")
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], env=env, text=True,
                          capture_output=True, timeout=10)
    assert proc.returncode == 0 and proc.stdout == "ok\n"
    assert proc.stderr.count("capx_trace_recorder: startup failed: ModuleNotFoundError") == 1


@pytest.mark.skipif(sys.platform != "linux", reason="lib64 symlink test requires Linux")
def test_orphaned_pth_reports_once_through_lib64_symlink(tmp_path):
    site = tmp_path / "lib" / "site-packages"
    site.mkdir(parents=True)
    (tmp_path / "lib64").symlink_to(tmp_path / "lib", target_is_directory=True)
    (site / "capx_trace_recorder.pth").write_text(cli.PTH_LINE)
    code = f"import site; site.addsitedir({str(site)!r}); site.addsitedir({str(tmp_path / 'lib64' / 'site-packages')!r})"
    env = os.environ.copy()
    env["CAPX_TRACE_DIR"] = str(tmp_path / "trace")
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code], env=env,
                          text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0
    assert proc.stderr.count("capx_trace_recorder: startup failed: ModuleNotFoundError") == 1
    assert not (tmp_path / "trace").exists()


def test_install_uninstall_and_status(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    path = tmp_path / "capx_trace_recorder.pth"
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "install-pth"])
    cli.main()
    assert path.read_text() == cli.PTH_LINE
    cli.main()
    assert path.read_text() == cli.PTH_LINE
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "status"])
    cli.main()
    output = capsys.readouterr().out
    assert "installed" in output and "this process: active=" in output
    path.write_text("custom hook\n")
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "status"])
    cli.main()
    assert "modified" in capsys.readouterr().out
    for command in ("install-pth", "uninstall-pth"):
        monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", command])
        with pytest.raises(SystemExit):
            cli.main()
    assert path.read_text() == "custom hook\n"
    path.write_text(cli.PTH_LINE)
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "uninstall-pth"])
    cli.main()
    assert not path.exists()


def test_legacy_pth_is_recognized_replaced_and_removed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    path = tmp_path / "capx_trace_recorder.pth"
    path.write_text(cli.LEGACY_PTH_LINE)
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "status"])
    cli.main()
    assert "legacy" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "install-pth"])
    cli.main()
    assert path.read_text() == cli.PTH_LINE
    path.write_text(cli.LEGACY_PTH_LINE)
    monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", "uninstall-pth"])
    cli.main()
    assert not path.exists()


def test_pth_error_report_handles_broken_stderr(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "capx_trace_recorder.pth").write_text(cli.PTH_LINE)
    code = ("import site,sys\n"
            "class Broken:\n"
            " def write(self, value): raise OSError('broken stderr')\n"
            " def flush(self): pass\n"
            "sys.stderr = Broken()\n"
            f"site.addsitedir({str(site)!r})\n"
            "print('survived')\n")
    env = os.environ.copy()
    env["CAPX_TRACE_DIR"] = str(tmp_path / "trace")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], env=env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0 and "survived" in result.stdout


def test_directory_at_pth_path_is_a_parser_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    (tmp_path / "capx_trace_recorder.pth").mkdir()
    for command in ("status", "install-pth", "uninstall-pth"):
        monkeypatch.setattr(sys, "argv", ["capx-trace-recorder", command])
        with pytest.raises(SystemExit) as raised:
            cli.main()
        assert raised.value.code == 2
        assert "hook path is a directory" in capsys.readouterr().err


def test_first_capx_finder_tolerates_prior_removal(monkeypatch):
    finder = bootstrap._FirstCapxFinder()
    monkeypatch.setattr(bootstrap, "_CAPX_FINDER", finder)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: None)
    assert finder.find_spec("capx") is None


def test_run_module_and_script(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(bootstrap, "activate", lambda: called.append("activated"))
    monkeypatch.setattr(cli.runpy, "run_module", lambda *a, **k: called.append(tuple(sys.argv)))
    monkeypatch.setattr(cli.runpy, "run_path", lambda *a, **k: called.append(tuple(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["recorder", "run", "-m", "target_module", "--flag", "one"])
    cli.main()
    script = tmp_path / "target_script.py"
    monkeypatch.setattr(sys, "argv", ["recorder", "run", str(script), "two"])
    cli.main()
    assert called == ["activated", ("target_module", "--flag", "one"),
                      "activated", (str(script), "two")]


def test_viewer_server_guard():
    assert bootstrap.is_viewer_only_process(["python", "-m", "capx_trace_viewer.server"])
    assert bootstrap.is_viewer_only_process(["/venv/bin/capx-trace-viewer"])
    assert bootstrap.is_viewer_only_process(["python", "/venv/bin/capx-trace-viewer"])
    assert not bootstrap.is_viewer_only_process(["python", "trial.py", "--output-dir", "capx-trace-viewer"])


@pytest.mark.parametrize("order", ["bundled-imported-before-activation", "bundled-visible-at-activation", "bundled-imported-after-late-path", "bundled-imported-after-path-override", "unbundled-visible-at-activation"])
def test_bundled_recorder_is_untouched_and_public_stands_down(tmp_path, order):
    package = tmp_path / "fork" / "capx"
    package.mkdir(parents=True)
    bundled_case = order != "unbundled-visible-at-activation"
    if bundled_case:
        (package / "tracing").mkdir()
    (package / "__init__.py").write_text(
        "import os\n"
        + ("from capx.tracing.bootstrap import activate\n"
           "if os.environ.get('CAPX_TRACE_DIR', '').strip(): activate()\n" if bundled_case else ""))
    if bundled_case:
        (package / "tracing" / "__init__.py").write_text("")
        (package / "tracing" / "bootstrap.py").write_text(
            "ACTIVE = False\nCALLS = 0\n"
            "def activate():\n global ACTIVE,CALLS\n ACTIVE = True\n CALLS += 1\n"
            "def is_active(): return ACTIVE\n")
    plain = tmp_path / "plain" / "capx"
    plain.mkdir(parents=True)
    (plain / "__init__.py").write_text("")
    root = Path(__file__).resolve().parents[2]
    code = (
        "import importlib,json,os,sys\n"
        f"fork={str(package.parent)!r}\n"
        "from capx_trace_recorder import bootstrap as public\n"
        f"order={order!r}\n"
        "if order == 'bundled-imported-before-activation':\n"
        " sys.path.insert(0,fork); import capx; public.activate()\n"
        "elif order in ('bundled-visible-at-activation','unbundled-visible-at-activation'):\n"
        " sys.path.insert(0,fork); public.activate(); assert not os.path.exists(os.environ['CAPX_TRACE_DIR']); import capx\n"
        "else:\n"
        f" if order == 'bundled-imported-after-path-override': sys.path.insert(0,{str(plain.parent)!r})\n"
        " public.activate()\n"
        " sys.path.insert(0,fork); import capx\n"
        "bundled=sys.modules.get('capx.tracing.bootstrap')\n"
        "print('STATE='+json.dumps([public.is_active(),bundled.is_active() if bundled else False,"
        "bundled.CALLS if bundled else 0,public.recorder_status()]))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["CAPX_TRACE_DIR"] = str(tmp_path / "traces")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    state = next(line for line in proc.stdout.splitlines() if line.startswith("STATE="))
    public_active, bundled_active, calls, (chosen, reason) = __import__("json").loads(state[6:])
    if bundled_case:
        assert not public_active and bundled_active and calls == 1
        assert chosen == "bundled" and "stood down" in reason
        assert not (tmp_path / "traces").exists()
    else:
        assert public_active and not bundled_active and calls == 0
        assert chosen == "public"
        status = __import__("json").loads(next((tmp_path / "traces").glob("recorder_status.*.json")).read_text())
        assert status == {"active": chosen, "reason": reason}


def test_active_bundled_recorder_prevents_double_hooks(monkeypatch, tmp_path):
    original = bootstrap._ACTIVATED
    try:
        bootstrap._ACTIVATED = False
        monkeypatch.setenv("CAPX_TRACE_DIR", str(tmp_path / "traces"))
        module = types.ModuleType("capx.tracing.bootstrap")
        module.is_active = lambda: True
        monkeypatch.setitem(sys.modules, "capx.tracing.bootstrap", module)
        before = list(sys.meta_path)
        bootstrap.activate()
        assert sys.meta_path == before
    finally:
        bootstrap._ACTIVATED = original


def test_missing_optional_symbol_is_reported_as_skipped(monkeypatch):
    monkeypatch.setattr(context.REGISTRY, "hooks_skipped", [])
    bootstrap._record_problem("H2", "capx.envs.trial", "_patch_libero_goal: not found",
                              types.ModuleType("trial"))
    assert context.REGISTRY.hooks_skipped == [{"hook": "H2", "target": "capx.envs.trial",
                                                "reason": "_patch_libero_goal: not found"}]


def test_required_symbol_and_wrong_type_are_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPX_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(context.REGISTRY, "hooks_failed", [])
    monkeypatch.setattr(context.REGISTRY, "hooks_skipped", [])
    bootstrap._record_problem("H4", "capx.envs.trial._run_single_trial",
                              "_run_single_trial: not found", types.ModuleType("trial"))
    bootstrap._record_problem("H13", "capx.web.async_trial_runner",
                              "run_trial_async: not found or not a coroutine function",
                              types.SimpleNamespace(run_trial_async=lambda: None))
    assert [f["hook"] for f in context.REGISTRY.hooks_failed] == ["H4", "H13"]
    assert context.REGISTRY.hooks_skipped == []
    assert context.REGISTRY.h4_failed()
    assert "hook_install_failed" in next(tmp_path.glob("recorder_warnings.*.jsonl")).read_text()


def test_late_environment_setting_can_activate(monkeypatch):
    original = bootstrap._ACTIVATED
    try:
        bootstrap._ACTIVATED = False
        calls = []
        monkeypatch.delenv("CAPX_TRACE_DIR", raising=False)
        monkeypatch.setattr(bootstrap, "_activate", lambda: calls.append("active"))
        monkeypatch.setitem(sys.modules, "capx", types.SimpleNamespace(__file__="/tmp/plain-capx/capx/__init__.py"))
        bootstrap.activate()
        assert not calls
        monkeypatch.setenv("CAPX_TRACE_DIR", "trace")
        bootstrap.activate()
        assert calls == ["active"]
    finally:
        bootstrap._ACTIVATED = original


def test_git_commit_comes_from_loaded_capx_not_working_directory(monkeypatch, tmp_path):
    from capx_trace_recorder.config import capx_checkout_root

    checkout = tmp_path / "checkout"
    package = checkout / "capx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    git = checkout / ".git"
    git.mkdir()
    (git / "HEAD").write_text("test-capx-commit\n")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setitem(sys.modules, "capx", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(store, "_PROCESS_INFO", None)
    monkeypatch.setattr(store, "_PROCESS_INFO_CAPX_SEEN", False)
    assert capx_checkout_root() == checkout
    assert store.process_info()["git_commit"] == "test-capx-commit"


def test_git_commit_is_null_for_site_packages_in_unrelated_checkout(monkeypatch, tmp_path):
    from capx_trace_recorder.config import capx_checkout_root

    other = tmp_path / "unrelated"
    package = other / ".venv" / "lib" / "python3.12" / "site-packages" / "capx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (other / ".git").mkdir()
    monkeypatch.setitem(sys.modules, "capx", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(store, "_PROCESS_INFO", None)
    monkeypatch.setattr(store, "_PROCESS_INFO_CAPX_SEEN", False)
    assert capx_checkout_root() is None
    assert store.process_info()["git_commit"] is None


def test_git_commit_accepts_src_layout_checkout(monkeypatch, tmp_path):
    from capx_trace_recorder.config import capx_checkout_root

    checkout = tmp_path / "checkout"
    package = checkout / "src" / "capx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (checkout / ".git").mkdir()
    monkeypatch.setitem(sys.modules, "capx", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    assert capx_checkout_root() == checkout


def test_run_script_resolves_sibling_import(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "activate", lambda: None)
    (tmp_path / "sibling.py").write_text("VALUE = 'ok'\n")
    script = tmp_path / "target.py"
    result = tmp_path / "result.txt"
    script.write_text(f"import sibling\nfrom pathlib import Path\nPath({str(result)!r}).write_text(sibling.VALUE)\n")
    monkeypatch.setattr(sys, "argv", ["recorder", "run", str(script)])
    original_path = sys.path[:]
    try:
        cli.main()
    finally:
        sys.path[:] = original_path
    assert result.read_text() == "ok"


def test_run_module_and_script_match_python_paths(tmp_path):
    package = tmp_path / "demo_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    result = tmp_path / "result.json"
    body = (
        "import json,sys\n"
        f"from pathlib import Path\nPath({str(result)!r}).write_text(json.dumps({{"
        "'name': __name__, 'file': __file__, 'path': sys.path[0], 'argv': sys.argv}))\n"
    )
    (package / "__main__.py").write_text(body)
    script = tmp_path / "script.py"
    script.write_text(body)
    env = os.environ.copy()
    env.pop("CAPX_TRACE_DIR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for arg, expected_path in ((["-m", "demo_pkg"], str(tmp_path)),
                               ([str(script)], str(tmp_path))):
        proc = subprocess.run([sys.executable, "-m", "capx_trace_recorder", "run", *arg, "value"],
                              cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15)
        assert proc.returncode == 0, proc.stderr
        doc = __import__("json").loads(result.read_text())
        assert doc["name"] == "__main__"
        assert doc["path"] == expected_path
        assert Path(doc["file"]).is_absolute()
        assert doc["argv"][-1] == "value"


def test_run_viewer_server_does_not_activate(tmp_path):
    env = os.environ.copy()
    env["CAPX_TRACE_DIR"] = str(tmp_path / "traces")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    proc = subprocess.run(
        [sys.executable, "-m", "capx_trace_recorder", "run", "-m",
         "capx_trace_viewer.server", "--help"],
        env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "traces").exists()
