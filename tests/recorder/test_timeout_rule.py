"""Every recorder handler that can catch the trial alarm re-raises it (docs/trace-format.md, section 2).

The alarm is a ``TimeoutError`` without an errno, raised by a SIGALRM handler.
An OS ``ETIMEDOUT`` is also a ``TimeoutError`` and must be handled as recorder
I/O failure. Any ``except Exception`` / ``except OSError``
/ bare ``except`` / ``except BaseException`` in code that runs on the trial
thread would swallow it and let the trial run past its limit.

This test parses every module of the recorder (``capx_trace_recorder`` without
the startup entry points, which can report installation errors) and fails on a handler that
can catch ``TimeoutError`` unless one of these holds:

* an earlier handler of the same ``try`` catches exactly ``TimeoutError`` and
  conditionally re-raises the alarm (not enough for handlers that also catch non-``Exception``
  exceptions);
* an ordinary handler first checks ``is_trial_alarm(exc)`` and re-raises the alarm;
* the handler re-raises: its body ends with a bare ``raise`` and has no
  ``return`` (and no ``break``/``continue`` out of it) before that;
* the ``except`` line says ``timeout-ok: <reason>`` (code that only runs on the
  writer thread, which never receives the alarm, or that forwards the
  exception into a generator).

"Can catch" is decided from the names in the ``except`` clause:
``Exception``, ``BaseException``, ``OSError`` and its aliases
(``EnvironmentError``, ``IOError``, ``socket.error``, ``select.error``,
``os.error``), ``TimeoutError`` and its aliases (``socket.timeout``,
``asyncio.TimeoutError``, ...), a bare ``except``, and any name that is not in
the explicit list ``SAFE`` of classes that cannot be a base of ``TimeoutError``
(an alias such as ``except _ERRS:`` is assumed to catch it). ``try`` and
``try``/``except*`` statements are both checked. A handler of a
non-``Exception`` class (``GeneratorExit``, ``KeyboardInterrupt``,
``SystemExit``) must re-raise too (section 2).
An ``except*`` handler receives an exception group, so its alarm check is
insufficient; it must end in a bare ``raise`` or carry ``timeout-ok``.

Reviewed stdlib calls that swallow an ``OSError`` internally, so a
``TimeoutError`` raised inside them is lost: ``linecache.checkcache`` (reached
from ``traceback.StackSummary.extract``) and ``os.path.realpath`` (reached from
``Path.resolve``). The recorder keeps the first off the trial thread (the API
traceback is formatted by the writer, ``hooks/api.format_frames``); the second
runs in H4's and H15's setup, before the alarm is armed.
"""

from __future__ import annotations

import ast
import errno
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx_trace_recorder.hooks.common import guard

REPO = Path(__file__).resolve().parents[2]
MODULES = sorted(p for p in (REPO / "capx_trace_recorder").rglob("*.py")
                 if p.name not in {"pth.py", "__main__.py"})

# names (last component of a dotted name) that catch TimeoutError: the class, its
# bases and their aliases
CATCHES_TIMEOUT = {"Exception", "BaseException", "OSError", "EnvironmentError", "IOError", "TimeoutError",
                   "timeout", "error"}
# classes that are not Exception subclasses: their handlers must re-raise as well
NON_EXCEPTION = {"BaseException", "GeneratorExit", "KeyboardInterrupt", "SystemExit", "CancelledError"}
# classes that cannot catch TimeoutError (none of them is a base of it)
SAFE = {
    "ArithmeticError", "AssertionError", "AttributeError", "BlockingIOError", "BrokenPipeError", "BufferError",
    "ChildProcessError", "ConnectionError", "EOFError", "Empty", "Error", "FileExistsError",
    "FileNotFoundError", "FloatingPointError", "GeneratorExit", "ImportError", "IndentationError",
    "IndexError", "InterruptedError", "IsADirectoryError", "JSONDecodeError", "KeyError", "KeyboardInterrupt",
    "LookupError", "MemoryError", "ModuleNotFoundError", "NameError", "NotADirectoryError",
    "NotImplementedError", "OverflowError", "PermissionError", "ProcessLookupError", "RecursionError",
    "ReferenceError", "RuntimeError", "StopIteration", "StopAsyncIteration", "SyntaxError", "SystemExit",
    "TabError", "TokenError", "TypeError", "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError",
    "UnicodeError", "ValueError", "ZeroDivisionError", "CancelledError",
}
MARKER = "timeout-ok:"
TRY_NODES = tuple(t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None)


def _names(node: ast.expr | None) -> set[str]:
    if node is None:
        return {"<bare>"}
    if isinstance(node, ast.Tuple):
        out: set[str] = set()
        for elt in node.elts:
            out |= _names(elt)
        return out
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return {"<expr>"}


def _can_catch_timeout(names: set[str]) -> bool:
    return bool(names & CATCHES_TIMEOUT) or any(n not in SAFE for n in names)


def _can_catch_non_exception(names: set[str]) -> bool:
    return bool(names & NON_EXCEPTION) or any(n not in SAFE and n not in CATCHES_TIMEOUT for n in names)


def _exits_early(body: list[ast.stmt]) -> bool:
    """A ``return`` anywhere in ``body`` (not in a nested function or class), or a
    ``break``/``continue`` that leaves the handler (not inside a nested loop)."""

    def visit(node: ast.AST, in_loop: bool) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return False
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, (ast.Break, ast.Continue)) and not in_loop:
            return True
        loop = isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        for child in ast.iter_child_nodes(node):
            # the loop's own body and else are inside it; its target/iter/test are not statements
            if visit(child, in_loop or loop):
                return True
        return False

    return any(visit(stmt, False) for stmt in body)


def _reraises(handler: ast.ExceptHandler) -> bool:
    body = handler.body
    return bool(body) and isinstance(body[-1], ast.Raise) and body[-1].exc is None and not _exits_early(body)


def _reraises_timeout(handler: ast.ExceptHandler) -> bool:
    return _names(handler.type) == {"TimeoutError"} and (_reraises(handler) or _guards_alarm(handler))


def _guards_alarm(handler: ast.ExceptHandler) -> bool:
    if not handler.name or not handler.body or not isinstance(handler.body[0], ast.If):
        return False
    check = handler.body[0]
    return (isinstance(check.test, ast.Call) and isinstance(check.test.func, ast.Name)
            and check.test.func.id == "is_trial_alarm" and len(check.test.args) == 1
            and isinstance(check.test.args[0], ast.Name) and check.test.args[0].id == handler.name
            and not check.orelse and bool(check.body)
            and isinstance(check.body[-1], ast.Raise) and check.body[-1].exc is None
            and not _exits_early(check.body))


def violations(path: Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, TRY_NODES):
            continue
        timeout_reraised = False
        for handler in node.handlers:
            names = _names(handler.type)
            try_star = isinstance(node, getattr(ast, "TryStar", ()))
            if not try_star and _reraises_timeout(handler):
                timeout_reraised = True
                continue
            catches_timeout = _can_catch_timeout(names)
            catches_non_exception = _can_catch_non_exception(names)
            if not catches_timeout and not catches_non_exception:
                continue
            if MARKER in lines[handler.lineno - 1]:
                continue
            if _reraises(handler) or (not try_star and not catches_non_exception and _guards_alarm(handler)):
                continue
            if timeout_reraised and not catches_non_exception:
                continue
            rel = path.relative_to(REPO)
            what = "the trial alarm" if catches_timeout else "a non-Exception exception"
            out.append(f"{rel}:{handler.lineno}: except {sorted(names)} can swallow {what}")
    return out


def test_modules_found():
    assert len(MODULES) > 20
    assert all(p.is_file() for p in MODULES)


def test_guard_distinguishes_trial_alarm_from_io_timeout(monkeypatch):
    with pytest.raises(TimeoutError, match="Trial 0 exceeded 1 seconds"):
        guard(lambda: (_ for _ in ()).throw(TimeoutError("Trial 0 exceeded 1 seconds")))
    recorded = []
    monkeypatch.setattr("capx_trace_recorder.hooks.common.report_internal_error",
                        lambda exc, where: recorded.append(exc))
    guard(lambda: (_ for _ in ()).throw(OSError(errno.ETIMEDOUT, "timed out")))
    assert len(recorded) == 1 and recorded[0].errno == errno.ETIMEDOUT


def test_bad_exception_str_keeps_original_error():
    from capx_trace_recorder.hooks.common import internal_error_data
    from capx_trace_recorder.hooks.http import _safe_str
    from capx_trace_recorder.hooks.llm import StreamState, end_logical_error

    class Bad(Exception):
        def __str__(self):
            raise ValueError("str failed")

    bad = Bad()
    assert internal_error_data(bad, "w")["message"] == "w: Bad: "
    assert _safe_str(bad) == "Bad"
    emitted = []
    ep = SimpleNamespace(emit=lambda *args, **kwargs: emitted.append((args, kwargs)))
    lc = SimpleNamespace(episode=ep, role="r", call_kind="k", event_id="id",
                         attempts=0, last_status=None, turn_index=None,
                         last_request_id=None, stream_info=None)
    end_logical_error(lc, bad)
    assert emitted[-1][1]["data"]["error_type"] == "Bad"
    assert emitted[-1][1]["builder"].keywords["message"] == "Bad"
    StreamState(lc).failed(bad)
    assert emitted[-1][1]["data"]["error_type"] == "Bad"


def test_find_runner_dir_handles_file_io_timeout(tmp_path, monkeypatch):
    from capx_trace_recorder import context as C
    from capx_trace_recorder.hooks.trial import find_runner_dir

    monkeypatch.delenv("CAPX_TRACE_RUNNER_RUN_DIR", raising=False)
    monkeypatch.setattr(C.REGISTRY.runner, "run_dir", None)
    original = Path.is_file
    def timeout(self):
        if self.name == "run.json":
            raise OSError(errno.ETIMEDOUT, "timed out")
        return original(self)
    monkeypatch.setattr(Path, "is_file", timeout)
    assert find_runner_dir(tmp_path) is None


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_handler_swallows_the_alarm(path):
    assert violations(path) == []


def _lines_flagged(tmp_path: Path, source: str) -> list[str]:
    bad = tmp_path / "bad.py"
    bad.write_text(source)
    global REPO
    saved = REPO
    try:
        REPO = tmp_path
        found = violations(bad)
    finally:
        REPO = saved
    return [line.split(":")[1] for line in found]


def test_the_checker_finds_a_swallowing_handler(tmp_path):
    """Sensitivity: the checker flags the patterns the rule forbids."""
    assert _lines_flagged(
        tmp_path,
        "def a():\n    try:\n        f()\n    except Exception:\n        pass\n"
        "def b():\n    try:\n        f()\n    except (ValueError, OSError):\n        pass\n"
        "def c():\n    try:\n        f()\n    except TimeoutError:\n        raise\n"
        "    except BaseException:\n        pass\n"
        "def d():\n    try:\n        f()\n    except TimeoutError:\n        raise\n"
        "    except Exception:  # fine\n        pass\n"
        "def e():\n    try:\n        f()\n    except Exception:  # timeout-ok: writer thread\n        pass\n"
        "def g():\n    try:\n        f()\n    except BaseException:\n        cleanup()\n        raise\n",
    ) == ["4", "9", "16"]


def test_the_checker_accepts_conditional_alarm_reraise(tmp_path):
    assert _lines_flagged(tmp_path,
        "def good():\n    try:\n        f()\n    except Exception as exc:\n"
        "        if is_trial_alarm(exc):\n            raise\n        pass\n") == []
    assert _lines_flagged(tmp_path,
        "def bad():\n    try:\n        f()\n    except Exception as exc:\n"
        "        if is_trial_alarm(exc):\n            return\n        pass\n") == ["4"]


def test_the_checker_knows_the_aliases_and_early_exits(tmp_path):
    """Aliases of OSError/TimeoutError, unknown names, and handlers that can leave
    before their final ``raise`` are flagged; safe classes and real re-raises are not."""
    source = (
        "import socket, os, select\n"
        "_ERRS = (OSError, ValueError)\n"
        "def a():\n    try:\n        f()\n    except socket.timeout:\n        pass\n"  # line 6
        "def b():\n    try:\n        f()\n    except os.error:\n        pass\n"  # 11
        "def c():\n    try:\n        f()\n    except (socket.error, select.error):\n        pass\n"  # 16
        "def d():\n    try:\n        f()\n    except _ERRS:\n        pass\n"  # 21: an unknown alias
        "def e():\n    try:\n        f()\n    except BaseException:\n        if x:\n            return None\n"
        "        raise\n"  # 26: returns before the raise
        "def g():\n    for i in y:\n        try:\n            f()\n        except Exception:\n"
        "            if x:\n                break\n            raise\n"  # 34: breaks out of the handler
        "def h():\n    try:\n        f()\n    except GeneratorExit:\n        pass\n"  # 41: not re-raised
        "def i():\n    try:\n        f()\n    except (ValueError, KeyError, queue.Empty):\n        pass\n"  # safe
        "def j():\n    try:\n        f()\n    except TimeoutError:\n        cleanup()\n        raise\n"
        "    except Exception:\n        pass\n"  # fine: TimeoutError re-raised by the handler above
        "def k():\n    try:\n        f()\n    except TimeoutError:\n        if x:\n            return\n"
        "        raise\n    except Exception:\n        pass\n"  # 59 and 63: the first handler can return
        "def m():\n    try:\n        f()\n    except BaseException:\n        for z in w:\n"
        "            if z:\n                continue\n        raise\n"  # fine: continue stays in its loop
        "def n():\n    try:\n        f()\n    except Exception:\n        def later():\n            return 1\n"
        "        raise\n"  # fine: the return is in a nested function
    )
    flagged = sorted(_lines_flagged(tmp_path, source), key=int)
    assert flagged == ["6", "11", "16", "21", "26", "34", "41", "59", "63"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* needs Python 3.11")
def test_the_checker_visits_try_star(tmp_path):
    assert _lines_flagged(
        tmp_path, "def a():\n    try:\n        f()\n    except* OSError:\n        pass\n") == ["4"]
    assert _lines_flagged(
        tmp_path, "def a():\n    try:\n        f()\n    except* OSError as exc:\n"
        "        if is_trial_alarm(exc):\n            raise\n") == ["4"]


def shadowed_parameter_handlers(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []

    class Check(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            self._check_function(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def _check_function(self, node):
            params = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                          *node.args.kwonlyargs)}
            if node.args.vararg:
                params.add(node.args.vararg.arg)
            if node.args.kwarg:
                params.add(node.args.kwarg.arg)

            class Handlers(ast.NodeVisitor):
                def visit_FunctionDef(self, nested):
                    pass

                visit_AsyncFunctionDef = visit_FunctionDef
                visit_Lambda = visit_FunctionDef
                visit_ClassDef = visit_FunctionDef

                def visit_ExceptHandler(self, handler):
                    if handler.name in params:
                        found.append(f"{path}:{handler.lineno}: {handler.name}")
                    self.generic_visit(handler)

            checker = Handlers()
            for stmt in node.body:
                checker.visit(stmt)
            for stmt in node.body:
                self.visit(stmt)

    Check().visit(tree)
    return found


def test_handlers_do_not_shadow_function_parameters(tmp_path):
    for path in MODULES:
        assert shadowed_parameter_handlers(path) == []
    bad = tmp_path / "bad.py"
    bad.write_text("def f(exc):\n    try:\n        work()\n    except Exception as exc:\n        pass\n"
                   "    return exc\n")
    assert len(shadowed_parameter_handlers(bad)) == 1
