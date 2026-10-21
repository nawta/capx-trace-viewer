"""Id generation and sanitising (docs/trace-format.md, section 4.2).

Ids match ``^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$``: other characters become
``-`` and a leading character that is not alphanumeric becomes ``x``. Random
parts come from ``os.urandom`` only (section 2: no ``random`` module).

Related: ``capx_trace_recorder/store.py`` (event ids), ``capx_trace_recorder/hooks/trial.py``
and ``capx_trace_recorder/hooks/web.py`` (episode ids).
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import socket

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
MAX_ID_CHARS = 200
_BAD_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_id(value: object, *, fallback: str = "x") -> str:
    """Return ``value`` as a valid id (never raises, except the trial alarm)."""
    try:
        text = str(value)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001 - str() of an arbitrary object
        text = ""
    text = _BAD_CHARS.sub("-", text)[:MAX_ID_CHARS]
    if not text:
        return fallback
    if not text[0].isalnum() or not text[0].isascii():
        text = "x" + text[1:]
    return text


def join_id(head: object, tail: object) -> str:
    """``head + tail`` as one id; ``head`` is shortened so ``tail`` always survives.

    Episode ids end in the parts that tell trials apart (``__s<seed>__i..__a..``
    or a random suffix); cutting a long id at 200 characters must not remove them.
    """
    tail_s = _BAD_CHARS.sub("-", str(tail))[: MAX_ID_CHARS - 1]
    head_s = sanitize_id(head)[: MAX_ID_CHARS - len(tail_s)]
    return sanitize_id(head_s + tail_s)


def with_suffix(base: str, suffix: str, tail: str = "") -> str:
    """``base + suffix`` in 200 characters (section 4.2 ``__dup<n>``).

    When ``base`` ends in ``tail`` (the parts that tell trials apart, as given to
    ``join_id``), the part before the tail is shortened and ``tail + suffix``
    survives; otherwise the end of ``base`` is cut.
    """
    tail_s = _BAD_CHARS.sub("-", str(tail)) if tail else ""
    if tail_s and base.endswith(tail_s) and len(tail_s) < len(base):
        return join_id(base[: len(base) - len(tail_s)], tail_s + suffix)
    return sanitize_id(base[: max(1, MAX_ID_CHARS - len(suffix))] + suffix)


def is_valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value or ""))


def new_event_id() -> str:
    """``ev_`` + 20 hex characters from ``os.urandom``."""
    return "ev_" + os.urandom(10).hex()


def random_hex(n_chars: int) -> str:
    return os.urandom((n_chars + 1) // 2).hex()[:n_chars]


def utc_stamp(ts: float | None = None) -> str:
    """UTC time as ``yyyymmddTHHMMSSZ``."""
    moment = (
        _dt.datetime.now(_dt.timezone.utc)
        if ts is None
        else _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)
    )
    return moment.strftime("%Y%m%dT%H%M%SZ")


def utc_iso(ts: float | None = None) -> str:
    moment = (
        _dt.datetime.now(_dt.timezone.utc)
        if ts is None
        else _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)
    )
    return moment.isoformat(timespec="milliseconds")


def hostname() -> str:
    try:
        return socket.gethostname()
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return "unknown-host"


def fallback_run_id(ts: float | None = None) -> str:
    """``<UTC yyyymmddTHHMMSSZ>_<host>_<pid>`` (section 4.2)."""
    return sanitize_id(f"{utc_stamp(ts)}_{hostname()}_{os.getpid()}")


__all__ = [
    "ID_PATTERN",
    "MAX_ID_CHARS",
    "sanitize_id",
    "join_id",
    "with_suffix",
    "is_valid_id",
    "new_event_id",
    "random_hex",
    "utc_stamp",
    "utc_iso",
    "hostname",
    "fallback_run_id",
]
