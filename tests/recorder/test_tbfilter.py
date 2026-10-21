"""H11 traceback protection (docs/trace-format.md, section 7.4).

Generated code runs through the real ``CodeExecutionEnvBase._exec_user_code``
(``base.py`` prints the traceback of anything it catches into the stderr that
multi-turn prompts copy). With the H8 wrappers on the API and H11 installed,
stdout and stderr must equal the run without any wrapper and without H11,
whoever prints: ``base.py``, the generated code's own ``traceback`` calls
(including ``limit`` values that count frames from either end), ``logging``, or
a stack printed inside an API. The module is re-run under Python 3.12.
"""

from __future__ import annotations

import contextlib
import io
import logging
import subprocess
import sys
import traceback

import pytest

from capx_trace_recorder import tbfilter
from capx_trace_recorder.hooks.api import install_api, uninstall_api
from capx_trace_recorder.hooks.common import selftest_wrap
from tracing_testlib import PY312, REPO, clean_env


class StackApi:
    """API whose methods fail, recurse and print stacks (lines matter)."""

    def __init__(self) -> None:
        self._env = None

    def functions(self):
        return {"fail": self.fail, "recurse": self.recurse, "stack": self.stack,
                "nested_fail": self.nested_fail, "takes_one": self.takes_one}

    def combined_doc(self) -> str:
        from local_api_doc import combined_doc

        return combined_doc(self)

    def fail(self, msg: str) -> None:
        """Raise a ValueError chained to a KeyError."""
        try:
            raise KeyError(msg)
        except KeyError as exc:
            raise ValueError(f"failed: {msg}") from exc

    def recurse(self, n: int) -> int:
        """Recurse through the public API and fail at the bottom."""
        if n <= 0:
            raise RuntimeError("bottom")
        return self.recurse(n - 1)

    def stack(self) -> str:
        """Print and return the current stack."""
        traceback.print_stack(file=sys.stdout, limit=5)
        names = [f.name for f in traceback.StackSummary.extract(traceback.walk_stack(None), limit=4)]
        return ",".join(names) + "|" + "".join(traceback.format_stack(limit=3))

    def nested_fail(self) -> None:
        """Fail inside a nested public call (context, not cause)."""
        try:
            self.fail("inner")
        except ValueError:
            raise LookupError("while handling")

    def takes_one(self, x: int) -> int:
        """Accept exactly one argument."""
        return x


CASES = {
    "base_print_exc": 'fail("x")\n',
    "own_print_exc": (
        "import traceback\n"
        "try:\n    fail('y')\nexcept ValueError:\n    traceback.print_exc()\n"
        "    print(traceback.format_exc(limit=1))\n"
        "    traceback.print_exception(*__import__('sys').exc_info(), limit=-2)\n"
    ),
    "logging_exception": (
        "import logging\n"
        "try:\n    nested_fail()\nexcept LookupError:\n    logging.getLogger('gen').exception('oops')\n"
    ),
    "stacks": "print(stack())\nimport traceback\ntraceback.print_stack(limit=4)\n",
    "recursion": (
        "import traceback\n"
        "def helper(n):\n    return recurse(n)\n"
        "try:\n    helper(8)\nexcept RuntimeError:\n    traceback.print_exc()\n"
        "    print(''.join(traceback.TracebackException(*__import__('sys').exc_info()).format()))\n"
        "recurse(3)\n"
    ),
    "wrong_arguments": "takes_one(1, 2)\n",
    "apis_dict": "APIS['toy'].nested_fail()\n",
    "limits": (
        "import sys, traceback\n"
        "try:\n    recurse(5)\nexcept RuntimeError:\n"
        "    print(traceback.format_exc(limit=1))\n"
        "    print(traceback.format_exc(limit=-1))\n"
        "    traceback.print_exception(*sys.exc_info(), limit=-3)\n"
        "    print(''.join(traceback.format_exception(*sys.exc_info(), limit=2, chain=False)))\n"
        "traceback.print_stack(limit=-2)\n"
        "print([f.name for f in traceback.StackSummary.extract(traceback.walk_stack(None), limit=4)])\n"
        "print([f.name for f in traceback.extract_stack(limit=3)])\n"
    ),
}


class FakeLow:
    def get_observation(self):
        return {}


class CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time, like ``logging.lastResort``."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def run_block(code: str, wrapped: bool) -> tuple[str, str, bool]:
    from capx.envs.tasks.base import CodeExecutionEnvBase

    api = StackApi()
    if wrapped:
        ok, reason, _ = install_api(api)
        assert ok, reason
    env = object.__new__(CodeExecutionEnvBase)
    env.low_level_env = FakeLow()
    env._apis = {"toy": api}
    env._full_prompt = []
    env._exec_globals = {"__name__": "__main__", "APIS": env._apis}
    gen_logger = logging.getLogger("gen")
    handler = CurrentStderrHandler()
    gen_logger.addHandler(handler)
    gen_logger.propagate = False  # pytest captures records that reach the root logger
    sink_out, sink_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            result = CodeExecutionEnvBase._exec_user_code(env, code)
    finally:
        gen_logger.removeHandler(handler)
        gen_logger.propagate = True
    if wrapped:
        uninstall_api(api)
    return result["stdout"], result["stderr"], result["ok"]


@pytest.fixture()
def h11():
    ok, reason = tbfilter.install(selftest_wrap)
    assert ok, reason
    yield
    tbfilter.uninstall()


def test_self_test_passes(h11):
    assert tbfilter.installed()
    assert tbfilter.self_test(selftest_wrap) == (True, "ok")


def run_arm(code: str, traced: bool) -> tuple[str, str, bool]:
    """The plain arm runs with the stdlib ``traceback`` module untouched; the
    traced arm with H11 and the H8 wrappers. Both call ``run_block`` from the same
    line (printed stacks include this frame)."""
    if traced:
        ok, reason = tbfilter.install(selftest_wrap)
        assert ok, reason
    else:
        assert not tbfilter.installed()
        assert not getattr(traceback.walk_tb, "__capx_trace_wrapper__", False)
    try:
        return run_block(code, wrapped=traced)
    finally:
        if traced:
            tbfilter.uninstall()


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.requires_capx
def test_output_identical_with_wrappers(case):
    # both arms from the same source line: printed stacks include this frame
    plain, traced = [run_arm(CASES[case], traced=t) for t in (False, True)]
    assert traced == plain
    assert "capx_trace_recorder" not in traced[0] + traced[1]
    if case != "stacks":
        assert traced[1], "the case must produce a traceback"


@pytest.mark.requires_capx
def test_wrappers_are_visible_without_h11():
    """Sensitivity check: without H11 the same cases do show recorder frames."""
    tbfilter.uninstall()
    plain, traced = [run_block(CASES["base_print_exc"], wrapped=w) for w in (False, True)]
    assert traced != plain and "capx_trace_recorder" in traced[1]


def test_walk_stack_none_start_frame_matches_original(h11):
    original = tbfilter._ORIGINALS["walk_stack"]

    def first(walker):
        for frame, lineno in walker(None):
            return frame.f_code.co_name, lineno

    def via(fn, walker):
        return fn(walker)

    def outer(walker, call):
        return call(lambda w: first(w), walker)

    reference = outer(original, via)
    assert outer(traceback.walk_stack, via) == reference
    assert outer(traceback.walk_stack, selftest_wrap(via)) == reference


def test_internal_error_yields_original_sequence(h11, monkeypatch):
    def broken(frame):
        raise RuntimeError("filter bug")

    monkeypatch.setattr(tbfilter, "is_recorder_frame", broken)
    try:
        raise ValueError("x")
    except ValueError:
        text = traceback.format_exc()
    assert "ValueError: x" in text and "test_internal_error_yields_original_sequence" in text


@pytest.mark.skipif(sys.version_info >= (3, 12) or PY312 is None or not PY312.exists(),
                    reason="set CAPX_PYTHON312 to a CaP-X Python 3.12 environment")
def test_same_checks_on_python312():
    proc = subprocess.run(
        [str(PY312), "-m", "pytest", "-q", "-p", "no:cacheprovider", str(REPO / "tests/recorder/test_tbfilter.py"),
         "-k", "not python312"],
        cwd=str(REPO), env=clean_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "passed" in proc.stdout
