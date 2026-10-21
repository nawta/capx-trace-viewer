"""Test-side reader for traces (docs/trace-format.md, section 4.5).

A small stand-in for ``capx_trace_viewer/reader.py`` (owned by the viewer backend):
splits ``events.jsonl`` on ``b"\\n"``, skips lines that do not parse and counts
them, and treats a last line without a newline as a torn tail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_events(path: Path) -> tuple[list[dict[str, Any]], int, bool]:
    """``(events, corrupt_lines, torn_tail)``."""
    raw = Path(path).read_bytes()
    parts = raw.split(b"\n")
    torn = bool(parts and parts[-1])
    lines = parts[:-1]
    events = []
    corrupt = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            corrupt += 1
    return events, corrupt, torn


def episode_dirs(trace_root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(trace_root).glob("*/*/events.jsonl"))


def load_episode(ep_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events, corrupt, torn = read_events(ep_dir / "events.jsonl")
    assert corrupt == 0 and not torn, (corrupt, torn)
    summary = json.loads((ep_dir / "episode.json").read_text())
    return summary, events


def by_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event_type"] == event_type]


def artifact_paths(events: list[dict[str, Any]]) -> list[str]:
    """Every file path an event references (artifacts, payloads, frame/state refs)."""
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "known_proxy" in value:  # an endpoint: its "path" is a URL path
                return
            for key, item in value.items():
                if key in ("path", "payload_path", "asset", "texts_path", "npz", "data_asset") and \
                        isinstance(item, str) and ("/" in item) and not item.startswith(("messages", "input", "/")):
                    out.append(item)
                elif key == "$asset" and isinstance(item, str):
                    out.append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for event in events:
        walk(event.get("artifacts"))
        walk(event.get("data"))
    return out
