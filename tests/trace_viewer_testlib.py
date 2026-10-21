"""Shared helpers of the trace viewer backend tests (docs/trace-format.md section 9).

``raw_get`` sends a request with an already decoded path, the way uvicorn hands
``%2E%2E`` and ``%2F`` to the app, so literal ``..`` segments reach the router (httpx
would normalise them away). ``load_demo_module`` loads scripts/make_demo_trace.py.

The trace API answers only requests whose ``Host`` header names a loopback host, so
``raw_get`` and ``LOOPBACK_BASE_URL`` (for ``TestClient(base_url=...)``) use
``127.0.0.1:8765``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "scripts" / "make_demo_trace.py"
RUN_ID = "demo_trace_viewer"
EPISODE_ENDED = "cube_stack__M4_skill__s1__i1__a1"
EPISODE_RUNNING = "20260926T060000Z__web__5e1f0a2c"
EPISODE_INCOMPLETE = "cube_lift__S3__s2__i1__a1"
LOOPBACK_HOST = "127.0.0.1:8765"
LOOPBACK_BASE_URL = f"http://{LOOPBACK_HOST}"


def load_demo_module() -> Any:
    spec = importlib.util.spec_from_file_location("make_demo_trace", DEMO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_demo_trace"] = module
    spec.loader.exec_module(module)
    return module


def raw_get(
    app: Any,
    path: str,
    *,
    client: tuple[str, int] | None = ("127.0.0.1", 50000),
    query: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    """GET with a decoded path, bypassing httpx URL normalisation.

    ``headers`` replaces the default ``Host: 127.0.0.1:8765`` header (pass ``[]`` to send
    no header at all).
    """

    async def run() -> list[dict[str, Any]]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8", "surrogateescape"),
            "query_string": query,
            "headers": [(b"host", LOOPBACK_HOST.encode())] if headers is None else headers,
            "client": client,
            "server": ("127.0.0.1", 8765),
            "root_path": "",
        }
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        return messages

    messages = asyncio.run(run())
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    headers_out = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], body, headers_out


def damaged_npz_bytes() -> bytes:
    """An ``np.savez_compressed`` file (member ``a0``: float32 120 x 160, plus a bit-packed
    mask ``a1``) with one byte of ``a0``'s deflate data flipped so that inflating it raises
    ``zlib.error`` (the damaged-member case)."""
    import io
    import zipfile
    import zlib

    import numpy as np

    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    mask = rng.random((120, 160)) > 0.5
    np.savez_compressed(buf, a0=rng.random((120, 160)).astype(np.float32),
                        a1__packbits=np.packbits(mask.ravel()), a1__shape=np.array(mask.shape))
    good = buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(good)) as zf:
        info = zf.getinfo("a0.npy")
    offset = info.header_offset
    start = offset + 30 + int.from_bytes(good[offset + 26: offset + 28], "little") + int.from_bytes(good[offset + 28: offset + 30], "little")
    for pos in range(info.compress_size):
        bad = bytearray(good)
        bad[start + pos] ^= 0xFF
        with zipfile.ZipFile(io.BytesIO(bytes(bad))) as zf:
            try:
                zf.read("a0.npy")
            except zlib.error:
                return bytes(bad)
            except Exception:  # noqa: BLE001 - another kind of damage; try the next byte
                continue
    raise AssertionError("no byte flip produced zlib.error")
