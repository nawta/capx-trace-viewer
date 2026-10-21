"""H11: hide recorder frames from tracebacks and stacks (docs/trace-format.md, section 7.4).

Sandbox stdout/stderr, including full tracebacks, is copied into multi-turn
prompts. The recorder's wrappers add frames, so the stdlib ``traceback``
module's ``walk_tb``, ``walk_stack`` and (3.11+) ``_walk_tb_with_full_positions``
are replaced by generators that skip frames whose code belongs to
``capx_trace_recorder``. Every formatting function (``print_exc``, ``format_exc``,
``print_exception``, ``TracebackException``, ``extract_tb``, ``print_stack``,
``extract_stack``, and ``logging``) takes its frames from these generators, so
``limit`` counts the same frames as in an untraced run. On any internal error a
replacement yields the original sequence; a ``TimeoutError`` (the trial alarm)
that lands inside a replacement is re-raised, as it would be from the stdlib
code in an untraced run, and never turns into an unfiltered walk.

``walk_stack(None)`` is delegated to the original generator, which computes
its start frame relative to its own frame; delegating adds exactly one frame
(this module's generator) to that chain, so the first frame it yields is
dropped. This keeps the behaviour identical on 3.10 (two ``f_back`` hops) and
3.12 (four).

``install()`` runs ``self_test()``; when the wrapped and unwrapped outputs
differ the originals are restored and hooks that add frames are not installed.

Related: ``capx_trace_recorder/bootstrap.py`` (install order).
"""

from __future__ import annotations

import os
import sys
from typing import Any

_TRACING_DIR = os.path.dirname(os.path.abspath(__file__))
_TRACING_DIR_REAL = os.path.realpath(_TRACING_DIR)
_ORIGINALS: dict[str, Any] = {}
_INSTALLED = False


def is_recorder_frame(frame: Any) -> bool:
    """True for frames of code that lives in ``capx_trace_recorder``."""
    try:
        name = frame.f_globals.get("__name__")
        if isinstance(name, str) and (name == "capx_trace_recorder" or name.startswith("capx_trace_recorder.")):
            return True
        filename = frame.f_code.co_filename
        if isinstance(filename, str) and filename.startswith((_TRACING_DIR + os.sep,
                                                              _TRACING_DIR_REAL + os.sep)):
            return True
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return False
    return False


def _filtered_walk_tb(tb: Any):
    original = _ORIGINALS["walk_tb"]
    try:
        items = list(original(tb))
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        yield from original(tb)
        return
    try:
        kept = [item for item in items if not is_recorder_frame(item[0])]
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        kept = items
    yield from kept


def _filtered_walk_tb_with_full_positions(tb: Any):
    original = _ORIGINALS["_walk_tb_with_full_positions"]
    try:
        items = list(original(tb))
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        yield from original(tb)
        return
    try:
        kept = [item for item in items if not is_recorder_frame(item[0])]
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        kept = items
    yield from kept


def _start_frame(consumer: Any) -> Any:
    """The frame ``walk_stack(None)`` starts at, counting only visible frames.

    The original computes ``sys._getframe()`` plus ``_HOPS`` ``f_back`` hops from
    its own generator frame (2 on 3.10, 4 on 3.12). The first hop is the frame
    that resumed the generator (``consumer``); recorder frames between the
    remaining hops are not counted, so the start is the frame an untraced run
    would start at.
    """
    cur = consumer
    remaining = (_HOPS or 2) - 1
    while remaining > 0 and cur is not None:
        cur = cur.f_back
        while cur is not None and is_recorder_frame(cur):
            cur = cur.f_back
        remaining -= 1
    return cur


def _filtered_walk_stack(f: Any):
    original = _ORIGINALS["walk_stack"]
    if f is None:
        try:
            f = _start_frame(sys._getframe(1))
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            yield from original(None)
            return
        if f is None:
            return
    for item in original(f):
        try:
            hidden = is_recorder_frame(item[0])
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            hidden = False
        if not hidden:
            yield item


def _measure_hops(original: Any) -> int | None:
    """How many ``f_back`` hops the running ``walk_stack(None)`` makes."""

    def probe() -> int | None:
        gen = original(None)
        try:
            frame = next(gen)[0]
        except StopIteration:
            return None
        finally:
            gen.close()
        cur = sys._getframe()
        k = 0
        while cur is not None and cur is not frame:
            cur = cur.f_back
            k += 1
        return k + 1 if cur is frame else None

    return probe()


_HOPS: int | None = None


_REPLACEMENTS = {
    "walk_tb": _filtered_walk_tb,
    "walk_stack": _filtered_walk_stack,
    "_walk_tb_with_full_positions": _filtered_walk_tb_with_full_positions,
}


def installed() -> bool:
    return _INSTALLED


def _apply(traceback_module: Any) -> list[str]:
    names = []
    for name, replacement in _REPLACEMENTS.items():
        current = getattr(traceback_module, name, None)
        if current is None:
            continue
        if getattr(current, "__capx_trace_wrapper__", False):
            names.append(name)
            continue
        _ORIGINALS[name] = current
        replacement.__capx_trace_wrapper__ = True  # type: ignore[attr-defined]
        replacement.__wrapped__ = current  # type: ignore[attr-defined]
        replacement.__doc__ = current.__doc__
        setattr(traceback_module, name, replacement)
        names.append(name)
    return names


def uninstall() -> None:
    """Restore the originals (used when the self-test fails, and by tests)."""
    global _INSTALLED
    tb_mod = sys.modules.get("traceback")
    if tb_mod is not None:
        for name, original in list(_ORIGINALS.items()):
            if getattr(getattr(tb_mod, name, None), "__capx_trace_wrapper__", False):
                setattr(tb_mod, name, original)
    _INSTALLED = False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_TOY_SOURCE = '''
import traceback as _tb

def inner(n):
    if n <= 0:
        raise ValueError("capx trace self-test")
    return inner(n - 1)

def recurse(fn, n):
    return fn(n)

def caller(fn):
    try:
        fn(4)
    except ValueError as exc:
        raise RuntimeError("chained") from exc

def format_error(fn):
    try:
        caller(fn)
    except RuntimeError:
        return _tb.format_exc() + _tb.format_exc(limit=2) + "".join(
            _tb.format_exception(*__import__("sys").exc_info(), limit=-3))

def walk_names(walker):
    out = []
    for frame, lineno in walker(None):
        out.append((frame.f_code.co_name, lineno))
        if len(out) >= 4:
            break
    return out

def probe():
    return "".join(_tb.format_stack(limit=6)) + repr(walk_names(_tb.walk_stack))

def call_probe(fn):
    return fn()

def first_walked(walker):
    for frame, lineno in walker(None):
        return (frame.f_code.co_name, lineno)

def c6(walker):
    return first_walked(walker)

def direct(fn, walker):
    return fn(walker)

def c5(walker, call):
    return call(c6, walker)

def c4(walker, call):
    return c5(walker, call)

def c3(walker, call):
    return c4(walker, call)

def c2(walker, call):
    return c3(walker, call)

def c1(walker, call):
    return c2(walker, call)
'''


def self_test(wrap: Any) -> tuple[bool, str]:
    """Wrapped and unwrapped calls must print the same text (tracebacks and stacks)."""
    try:
        namespace: dict[str, Any] = {"__name__": "capx_trace_selftest"}
        code = compile(_TOY_SOURCE, "<capx-trace-selftest>", "exec")
        exec(code, namespace, namespace)  # noqa: S102 - fixed source, not model output
        inner = namespace["inner"]
        probe = namespace["probe"]
        wrapped_inner = wrap(inner)
        wrapped_probe = wrap(probe)
        outputs = []
        for fn in (inner, wrapped_inner):
            outputs.append(namespace["format_error"](fn))
        if outputs[0] != outputs[1]:
            return False, "traceback text differs with the wrapper"
        stacks = []
        for fn in (probe, wrapped_probe):
            stacks.append(namespace["call_probe"](fn))
        if stacks[0] != stacks[1]:
            return False, "stack text differs with the wrapper"
        if "capx_trace_recorder" in outputs[1] or "capx_trace_recorder" in stacks[1]:
            return False, "recorder path visible"
        # walk_stack(None) called directly must start where the original starts
        # in an untraced run, also when a wrapper frame lies within its hops
        c1, direct = namespace["c1"], namespace["direct"]
        reference = c1(_ORIGINALS["walk_stack"], direct)
        if reference is None or reference[0] not in ("c1", "c2", "c3", "c4", "c5", "c6", "direct"):
            return False, "walk_stack(None) reference start frame not in the toy chain"
        if c1(_filtered_walk_stack, direct) != reference or c1(_filtered_walk_stack, wrap(direct)) != reference:
            return False, "walk_stack(None) start frame differs"
        return True, "ok"
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        return False, f"self-test raised {type(exc).__name__}"


def install(wrap: Any) -> tuple[bool, str]:
    """Install H11 on the stdlib ``traceback`` module and self-test it."""
    global _INSTALLED
    try:
        import traceback as tb_mod  # stdlib; imported by logging/requests anyway
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import traceback: {type(exc).__name__}"
    global _HOPS
    original_walk_stack = getattr(tb_mod, "walk_stack", None)
    if getattr(original_walk_stack, "__capx_trace_wrapper__", False):
        original_walk_stack = _ORIGINALS.get("walk_stack")
    hops = _measure_hops(original_walk_stack) if original_walk_stack is not None else None
    if hops is None or not 1 <= hops <= 8:
        return False, "cannot measure traceback.walk_stack(None) start frame"
    _HOPS = hops
    names = _apply(tb_mod)
    if "walk_tb" not in names or "walk_stack" not in names:
        uninstall()
        return False, "traceback.walk_tb / walk_stack not found"
    if sys.version_info >= (3, 11) and "_walk_tb_with_full_positions" not in names:
        uninstall()
        return False, "traceback._walk_tb_with_full_positions not found"
    try:
        ok, reason = self_test(wrap)
    except BaseException:  # an alarm during the self-test: restore, then re-raise
        uninstall()
        raise
    if not ok:
        uninstall()
        return False, reason
    _INSTALLED = True
    return True, reason


__all__ = ["install", "uninstall", "installed", "is_recorder_frame", "self_test"]
