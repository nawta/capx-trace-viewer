"""H12: ``capx.utils.video_utils._encode_video_base64``.

Implements the VDM-video part of docs/trace-format.md section 6.2. The wrapper
keeps references to the frames passed in (fresh copies returned by
``get_video_frames_range``, which the harness only reads) and to the returned
data URL (an immutable ``str``). On the writer thread it hashes every frame's
raw pixels, decodes the data URL and indexes the MP4 bytes by sha256, and
matches the frame hashes against the ``video_frames_read`` events of the
episode. The ``model_request`` builder looks the video up by the sha256 of the
MP4 it finds in the request body, so ``images[].source`` names the exact buffer
frame range, or ``{kind: "unknown"}`` when no exact match exists.

Related: ``capx_trace_recorder/hooks/env.py`` (``video_frames_read``),
``capx_trace_recorder/hooks/http.py`` (``_media_source``).
"""

from __future__ import annotations

import base64
import functools
import hashlib
import re
from typing import Any

from capx_trace_recorder import context as C
from capx_trace_recorder.hooks.common import guard, mark, patch_attribute

_DATA_URL = re.compile(r"^data:([^;,]*)((?:;[^;,]*)*?);base64,")


def _after_encode(frames: Any, result: Any) -> None:
    ep = C.resolve_or_note("H12")
    if ep is None or not hasattr(ep, "frame_tap") or not isinstance(result, str):
        return
    refs = list(frames) if isinstance(frames, (list, tuple)) else []
    nbytes = sum(int(getattr(f, "nbytes", 0)) for f in refs) + len(result)

    def builder(ctx, refs=refs, url=result):
        from capx_trace_recorder.payload import raw_pixel_sha256

        match = _DATA_URL.match(url)
        if not match:
            return
        try:
            mp4 = base64.b64decode(url[match.end():], validate=True)
        except Exception:  # noqa: BLE001 - timeout-ok: writer thread
            return
        hashes = [raw_pixel_sha256(f) for f in refs]
        reason = "no video_frames_read with the same frames"
        if ctx.wep.dropped_kinds.get("video_frames_read"):
            reason = "queue_full: a video_frames_read of this episode was dropped"
        source: dict[str, Any] = {"kind": "unknown", "n_frames": len(hashes), "reason": reason}
        for read in reversed(ctx.wep.video_reads):
            if read.get("hashes") == hashes and hashes:
                source = {"kind": "video_frames_read", "event_id": read.get("event_id"),
                          "frame_range": [read.get("start"), read.get("end")],
                          "camera": read.get("camera"), "n_frames": len(hashes),
                          "match": "all frame pixel hashes equal"}
                break
        ctx.wep.video_encodes[hashlib.sha256(mp4).hexdigest()] = source

    def drop_builder(ctx):
        # holds neither the frames nor the data URL; the VDM request's video
        # source then says why it is unknown (counted as a queue_full gap)
        ctx.note_dropped("video_encode")

    ep.internal(builder, nbytes=nbytes, drop_builder=drop_builder)


def make_encode_wrapper(original: Any) -> Any:
    @functools.wraps(original)
    def _encode_video_base64(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        frames = args[0] if args else kwargs.get("frames")
        guard(_after_encode, frames, result)
        return result

    return mark(_encode_video_base64, original)


def install(video_utils: Any) -> list[str]:
    err = patch_attribute(video_utils, "_encode_video_base64", make_encode_wrapper)
    return [err] if err else []


__all__ = ["install", "make_encode_wrapper"]
