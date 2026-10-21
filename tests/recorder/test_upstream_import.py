"""Import-level compatibility check against an explicitly chosen upstream checkout."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "requests.sessions": ["Session.send"],
    "capx.llm.client": ["query_model", "query_model_streaming", "query_model_ensemble",
                         "query_single_model_ensemble"],
    "capx.utils.launch_utils": ["_build_multi_turn_decision_prompt",
                                 "_build_multi_turn_decision_prompt_legacy",
                                 "_parse_multi_turn_decision", "_extract_code"],
    "capx.utils.video_utils": ["_encode_video_base64"],
    "capx.envs.trial": ["_describe_initial_scene", "_get_visual_differencing_feedback",
                        "_get_video_differencing_feedback", "_query_initial_code",
                        "_handle_multi_turn_step", "_patch_libero_goal", "_run_single_trial"],
    "capx.envs.runner": [],
    "capx.envs.tasks.base": ["CodeExecutionEnvBase.step", "CodeExecutionEnvBase._exec_user_code",
                             "CodeExecutionEnvBase.render", "CodeExecutionEnvBase.render_wrist",
                             "CodeExecutionEnvBase.get_video_frames_range",
                             "CodeExecutionEnvBase.get_wrist_video_frames_range", "CodeExecutionEnvBase.reset"],
    "capx.web.async_trial_runner": ["run_trial_async", "_query_model", "_query_model_streaming"],
    "capx.web.session_manager": ["SessionManager.inject_prompt", "SessionManager.stop_session"],
    "capx.integrations.vision.molmo": ["_parse_points"],
}


def _run(code, upstream, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(upstream), str(ROOT)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("CAPX_TRACE_DIR", None)
    return subprocess.run([os.environ.get("CAPX_PYTHON", sys.executable), "-c", code],
                          cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)


def _result(proc, prefix):
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    line = next((line for line in proc.stdout.splitlines() if line.startswith(prefix)), None)
    assert line is not None, proc.stdout[-2000:]
    return json.loads(line[len(prefix):])


def test_upstream_targets_import_with_recorder(tmp_path):
    raw = os.environ.get("CAPX_UPSTREAM_DIR")
    if not raw:
        pytest.skip("CAPX_UPSTREAM_DIR is unset; provide an upstream CaP-X checkout")
    upstream = Path(raw).resolve()
    assert (upstream / "capx" / "__init__.py").is_file(), upstream
    preflight = _result(_run(
        "import importlib,json\n"
        f"names={list(EXPECTED)!r}\n"
        "out={}\n"
        "for name in names:\n"
        " try: importlib.import_module(name); out[name]='ok'\n"
        " except ImportError as exc: out[name]=str(exc)\n"
        "print('PREFLIGHT='+json.dumps(out))\n", upstream, tmp_path), "PREFLIGHT=")
    missing = {name: reason for name, reason in preflight.items() if reason != "ok"}
    if missing:
        pytest.skip(f"upstream dependencies missing before recorder activation: {missing}")
    code = (
        "import importlib,json,os\n"
        "from capx_trace_recorder import bootstrap as b\n"
        "os.environ['CAPX_TRACE_DIR']=os.path.abspath('traces')\n"
        "b.activate()\n"
        f"wrappers={EXPECTED!r}\n"
        "mods={name:importlib.import_module(name) for name in wrappers}\n"
        "checks={}\n"
        "for name,paths in wrappers.items():\n"
        " for path in paths:\n"
        "  obj=mods[name]\n"
        "  for part in path.split('.'): obj=getattr(obj,part)\n"
        "  checks[name+':'+path]=bool(getattr(obj,'__capx_trace_wrapper__',False))\n"
        "from capx_trace_recorder.context import REGISTRY\n"
        "print('SMOKE_RESULTS='+json.dumps({'checks':checks,'failed':REGISTRY.hooks_failed,"
        "'skipped':REGISTRY.hooks_skipped,'installed':REGISTRY.hooks_installed}))\n"
    )
    result = _result(_run(code, upstream, tmp_path), "SMOKE_RESULTS=")
    assert all(result["checks"].values()), result["checks"]
    assert result["failed"] == [], result["failed"]
    assert result["skipped"] == [], result["skipped"]
    installed = {entry.partition(":")[0] for entry in result["installed"]}
    assert {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H11", "H12", "H13", "H14"} <= installed
