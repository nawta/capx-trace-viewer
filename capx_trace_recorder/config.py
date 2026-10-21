"""Trace settings read from the environment (docs/trace-format.md, section 3).

``TraceConfig.from_env()`` returns ``None`` when ``CAPX_TRACE_DIR`` is unset,
which means tracing is off. A relative trace directory is resolved against the
working directory and written back to ``os.environ`` so that spawned workers use
the same root. Invalid numeric values fall back to the defaults and are listed
in ``TraceConfig.problems`` (recorded in ``episode_start.data.tracer``).

Related: ``capx_trace_recorder/bootstrap.py`` (calls ``from_env``),
``capx_trace_recorder/store.py`` (budgets, disk floor, heartbeat).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ENV_TRACE_DIR = "CAPX_TRACE_DIR"
ENV_RUN_ID = "CAPX_TRACE_RUN_ID"
ENV_FRAME_FPS = "CAPX_TRACE_FRAME_FPS"
ENV_JPEG_QUALITY = "CAPX_TRACE_JPEG_QUALITY"
ENV_STATE = "CAPX_TRACE_STATE"
ENV_MIN_FREE_GB = "CAPX_TRACE_MIN_FREE_GB"
ENV_MAX_QUEUE_MB = "CAPX_TRACE_MAX_QUEUE_MB"
ENV_ARRAY_CALL_MB = "CAPX_TRACE_ARRAY_CALL_MB"
ENV_ARRAY_EPISODE_MB = "CAPX_TRACE_ARRAY_EPISODE_MB"
ENV_VIDEO_FILE_MB = "CAPX_TRACE_VIDEO_FILE_MB"
ENV_PROXY_PORTS = "CAPX_TRACE_PROXY_PORTS"
# Written by the recorder itself (not user settings):
ENV_FALLBACK_RUN_ID = "CAPX_TRACE_FALLBACK_RUN_ID"  # derived once by the first process
ENV_RUNNER_RUN_DIR = "CAPX_TRACE_RUNNER_RUN_DIR"  # exported by hook H15

# Exact, case-insensitive key names whose values are masked (section 6.3).
# Substrings never match, so max_tokens / budget_tokens are kept.
SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "aws_secret_access_key",
        "private_key",
        "auth_token",
        "session_token",
        "passwd",
        "authorization",
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "client_secret",
        "secret",
        "password",
    }
)
MASK_VALUE = "***masked***"

HEARTBEAT_INTERVAL_S = 10.0
ATEXIT_WAIT_S = 10.0
INLINE_TEXT_LIMIT = 4000  # characters; longer text goes to a txt.gz asset
MAX_EVENT_LINE_BYTES = 256 * 1024


def repo_root() -> Path:
    """Resolve relative trace paths from the process working directory."""
    return Path.cwd()


def capx_checkout_root() -> Path | None:
    """Find the checkout of the CaP-X package actually loaded by this process."""
    import sys

    module = sys.modules.get("capx")
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    package = Path(origin).resolve().parent
    for directory in package.parents:
        if directory.name in {"site-packages", "dist-packages"}:
            break
        # The candidate must contain the actual imported package, including
        # checkouts that use a src/capx layout.
        if ((directory / ".git").exists() and (package / "__init__.py").is_file()
                and package.is_relative_to(directory)):
            return directory
    return None


def _parse_float(environ: Mapping[str, str], name: str, default: float, problems: list[str],
                 minimum: float = 0.0) -> float:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{name}={raw!r} is not a number; using {default}")
        return default
    if value != value or value < minimum or value in (float("inf"), float("-inf")):
        problems.append(f"{name}={raw!r} is out of range; using {default}")
        return default
    return value


def _parse_proxy_ports(environ: Mapping[str, str], problems: list[str]) -> frozenset[int]:
    ports: set[int] = set()
    for part in environ.get(ENV_PROXY_PORTS, "8110").split(","):
        value = part.strip()
        if not value:
            continue
        try:
            port = int(value) if value.isdecimal() else 0
        except (ValueError, OverflowError):
            port = 0
        if 0 < port < 65536:
            ports.add(port)
        else:
            problems.append(f"{ENV_PROXY_PORTS}: invalid port {value!r}; ignoring it")
    return frozenset(ports)


@dataclass(frozen=True)
class TraceConfig:
    """Effective recorder settings (one instance per process)."""

    trace_dir: Path
    run_id: str | None = None
    frame_fps: float = 4.0
    jpeg_quality: int = 85
    state: bool = True
    min_free_gb: float = 2.0
    max_queue_mb: float = 512.0
    array_call_mb: float = 64.0
    array_episode_mb: float = 2048.0
    video_file_mb: float = 256.0
    proxy_ports: frozenset[int] = field(default_factory=lambda: frozenset({8110}))
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_queue_bytes(self) -> int:
        return int(self.max_queue_mb * 1024 * 1024)

    @property
    def array_call_bytes(self) -> int:
        return int(self.array_call_mb * 1024 * 1024)

    @property
    def array_episode_bytes(self) -> int:
        return int(self.array_episode_mb * 1024 * 1024)

    @property
    def min_free_bytes(self) -> int:
        return int(self.min_free_gb * 1024 ** 3)

    @property
    def video_file_bytes(self) -> int:
        return int(self.video_file_mb * 1024 * 1024)

    def public(self) -> dict[str, Any]:
        """Settings recorded in ``episode_start.data.tracer.settings``."""
        return {
            "trace_dir": str(self.trace_dir),
            "frame_fps": self.frame_fps,
            "jpeg_quality": self.jpeg_quality,
            "state": self.state,
            "min_free_gb": self.min_free_gb,
            "max_queue_mb": self.max_queue_mb,
            "array_call_mb": self.array_call_mb,
            "array_episode_mb": self.array_episode_mb,
            "video_file_mb": self.video_file_mb,
            "proxy_ports": sorted(self.proxy_ports),
            "problems": list(self.problems),
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, *,
                 write_back: bool = True) -> "TraceConfig | None":
        """Read the settings; ``None`` means tracing is off.

        ``write_back`` stores the absolute trace directory in ``os.environ``
        (only when ``environ`` is the real process environment).
        """
        env = os.environ if environ is None else environ
        raw_dir = env.get(ENV_TRACE_DIR, "")
        if not raw_dir or not raw_dir.strip():
            return None
        problems: list[str] = []
        path = Path(os.path.expanduser(raw_dir.strip()))
        if not path.is_absolute():
            path = repo_root() / path
        path = Path(os.path.abspath(str(path)))
        if write_back and environ is None:
            os.environ[ENV_TRACE_DIR] = str(path)
        fps = _parse_float(env, ENV_FRAME_FPS, 4.0, problems)
        quality = int(_parse_float(env, ENV_JPEG_QUALITY, 85.0, problems, minimum=1.0))
        if quality > 100:
            problems.append(f"{ENV_JPEG_QUALITY}={quality} is above 100; using 100")
            quality = 100
        state_raw = env.get(ENV_STATE, "1").strip().lower()
        state = state_raw not in ("0", "false", "no", "off", "")
        run_id = env.get(ENV_RUN_ID) or None
        return cls(
            trace_dir=path,
            run_id=run_id.strip() if run_id and run_id.strip() else None,
            frame_fps=fps,
            jpeg_quality=quality,
            state=state,
            min_free_gb=_parse_float(env, ENV_MIN_FREE_GB, 2.0, problems),
            max_queue_mb=_parse_float(env, ENV_MAX_QUEUE_MB, 512.0, problems),
            array_call_mb=_parse_float(env, ENV_ARRAY_CALL_MB, 64.0, problems),
            array_episode_mb=_parse_float(env, ENV_ARRAY_EPISODE_MB, 2048.0, problems),
            video_file_mb=_parse_float(env, ENV_VIDEO_FILE_MB, 256.0, problems),
            proxy_ports=_parse_proxy_ports(env, problems),
            problems=tuple(problems),
        )


__all__ = [
    "TraceConfig",
    "SECRET_KEYS",
    "MASK_VALUE",
    "HEARTBEAT_INTERVAL_S",
    "ATEXIT_WAIT_S",
    "INLINE_TEXT_LIMIT",
    "MAX_EVENT_LINE_BYTES",
    "repo_root",
]
