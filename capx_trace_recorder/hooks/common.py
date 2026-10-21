"""Shared helpers for the hooks (docs/trace-format.md, sections 2 and 7.1).

Wrapper rules (section 2): each wrapper calls the original exactly once, in its
own ``try``/``except BaseException: raise`` that passes the result or the very
same exception object through; recorder code runs in ``guard()``, which
swallows ``Exception`` but always re-raises ``TimeoutError`` (the trial alarm)
and every ``BaseException`` that is not an ``Exception``. Every other handler in
recorder code that can run on a trial thread re-raises ``TimeoutError`` too
(``tests/recorder/test_timeout_rule.py``).

A wrapper that opens a context (API call, code block, logical model call,
service call, role, episode) builds it inside ``guard()`` without pushing it,
then pushes it as the first statement inside its own ``try`` and pops it in
that ``try``'s ``finally`` (and in its handler, before recording the error): an
alarm that lands between the push and the ``try``, or right after the original
returned, cannot leave a stale context. H4 clears what is left anyway at the
next episode boundary.

Related: every module in ``capx_trace_recorder/hooks/``; ``capx_trace_recorder/tbfilter.py``
(self-test uses ``selftest_wrap``).
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from capx_trace_recorder import context as C

WRAPPER_FLAG = "__capx_trace_wrapper__"
_WRAPPERS: dict[int, tuple[Any, Any]] = {}  # id(original) -> (original, wrapper)


def is_wrapper(obj: Any) -> bool:
    try:
        return bool(getattr(obj, WRAPPER_FLAG, False))
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        return False


def mark(wrapper: Any, original: Any) -> Any:
    setattr(wrapper, WRAPPER_FLAG, True)
    setattr(wrapper, "__capx_trace_original__", original)
    _WRAPPERS[id(original)] = (original, wrapper)
    return wrapper


def mark_only(wrapper: Any, original: Any) -> Any:
    """Flag ``wrapper`` without registering it as *the* wrapper of ``original``."""
    setattr(wrapper, WRAPPER_FLAG, True)
    setattr(wrapper, "__capx_trace_original__", original)
    return wrapper


def wrapper_for(original: Any) -> Any:
    entry = _WRAPPERS.get(id(original))
    if entry is not None and entry[0] is original:
        return entry[1]
    return None


_NOTED_OUTSIDE_EPISODES: set[str] = set()


def report_internal_error(exc: BaseException, where: str) -> None:
    """A recorder bug: one ``trace_warning(internal_error)`` per place per episode.

    Without an episode (for example ``_episode_start`` itself failed, so the trial
    runs untraced) the warning goes once per place to the run's unscoped log, when
    the process has a run (the experiment runner); otherwise nowhere.
    """
    try:
        ep, candidates = C.resolve_episode()
        data = internal_error_data(exc, where)
        if ep is None:
            sink = C.unscoped_sink_for(candidates)
            if sink is None or where in _NOTED_OUTSIDE_EPISODES:
                return
            _NOTED_OUTSIDE_EPISODES.add(where)
            data["outside_episode"] = True
            sink.emit("trace_warning", status="warning", data=data)
            return
        key = f"internal_error:{where}"
        if key in ep.misc:
            return
        ep.misc[key] = True
        ep.emit("trace_warning", status="warning", data=data)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        pass


def internal_error_data(exc: BaseException, where: str) -> dict[str, Any]:
    """``trace_warning`` data of a recorder error: the message is redacted and cut
    to 400 characters (``redact`` reads at most ``REDACT_WINDOW`` characters, in
    linear time, so this is cheap on any thread)."""
    from capx_trace_recorder.payload import put_redacted

    try:
        text = str(exc)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        text = ""
    data: dict[str, Any] = {"code": "internal_error"}
    put_redacted(data, "message", f"{where}: {type(exc).__name__}: {text}", 400)
    return data


def guard(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run recorder code: ``TimeoutError`` and non-``Exception`` propagate, the rest is recorded."""
    try:
        return fn(*args, **kwargs)
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        report_internal_error(exc, getattr(fn, "__qualname__", getattr(fn, "__name__", "?")))
        return None


def add_overhead(ep: Any, t0: float) -> None:
    add = getattr(ep, "add_overhead", None)
    if add is not None:
        add(time.perf_counter() - t0)


def patch_attribute(owner: Any, name: str, make_wrapper: Callable[[Any], Any]) -> str | None:
    """Replace ``owner.name`` by ``make_wrapper(original)``; ``None`` on success."""
    try:
        current = owner.__dict__.get(name) if isinstance(owner, type) else getattr(owner, name, None)
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        return f"{name}: {type(exc).__name__}"
    if current is None:
        return f"{name}: not found"
    if is_wrapper(current):
        return None
    if not callable(current):
        return f"{name}: not callable"
    wrapper = wrapper_for(current) or make_wrapper(current)
    setattr(owner, name, wrapper)
    return None


def rebind_aliases(module: Any, names: tuple[str, ...]) -> list[str]:
    """Point ``module.<name>`` at the wrapper of the function it still holds."""
    fixed = []
    for name in names:
        current = getattr(module, name, None)
        if current is None or is_wrapper(current):
            continue
        wrapper = wrapper_for(current)
        if wrapper is not None:
            setattr(module, name, wrapper)
            fixed.append(name)
    return fixed


def _selftest_before() -> object:
    return object()


def _selftest_after(token: object, value: Any) -> None:
    return None


def selftest_wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A wrapper with the control flow of the API wrapper (H8), for the H11 self-test."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = guard(_selftest_before)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            guard(_selftest_after, token, exc)
            raise
        guard(_selftest_after, token, result)
        return result

    setattr(wrapper, WRAPPER_FLAG, True)
    return wrapper


__all__ = [
    "is_wrapper",
    "mark",
    "mark_only",
    "wrapper_for",
    "guard",
    "report_internal_error",
    "internal_error_data",
    "add_overhead",
    "patch_attribute",
    "rebind_aliases",
    "selftest_wrap",
]
