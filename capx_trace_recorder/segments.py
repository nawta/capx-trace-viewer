"""Text segments with known sources and ``input_summary`` (docs/trace-format.md, section 6.4).

Producer side (trial thread): ``SegmentRegistry`` keeps the dynamic segments an
episode produced (executed code, block stdout/stderr, VDM descriptions, user
feedback, the task text actually sent). ``SegmentRegistry.snapshot()`` adds the
env-derived segments *at the moment of the request* (task prompt, API docs,
system prompt, the literal parts of ``multi_turn_prompt``) and the harness's
own fixed strings, read from the code objects of the functions that build
prompts (so the list always matches the running code). The snapshot is an
immutable tuple handed to the writer.

Writer side: ``compute_input_summary()`` searches every segment of at least 8
characters as an exact substring in every text part of the request. The one
shorter literal searched is the ``"\\nAPIs:\\n"`` separator the env puts between
the task text and the API docs. Nothing is matched by meaning, numbers or
coordinates. Ranges no segment covers are listed as ``unattributed``; those
holding only whitespace (the newlines that join collapsed text parts) carry
``whitespace_only: true``.

Related: ``capx_trace_recorder/hooks/llm.py`` (takes snapshots), ``capx_trace_recorder/payload.py``.
"""

from __future__ import annotations

import inspect
import string
import sys
import threading
from typing import Any, NamedTuple

MIN_SEGMENT_CHARS = 8
MAX_OCCURRENCES = 20
MAX_UNATTRIBUTED = 200
APIS_MARKER = "\nAPIs:\n"
# The one harness literal shorter than MIN_SEGMENT_CHARS that is still matched:
# the separator CodeExecutionEnvBase._get_complete_prompt puts between the task
# text and the API docs (it would otherwise show as "unknown origin").
SHORT_LITERALS = frozenset({APIS_MARKER})

DYNAMIC_KINDS = frozenset({"code_history", "stdout", "stderr", "vdm_description", "user_feedback"})
CATEGORY_OF = {
    "task_prompt": "task_api_description",
    "api_docs": "task_api_description",
    "code_history": "code_history",
    "stdout": "stdout_stderr",
    "stderr": "stdout_stderr",
    "vdm_description": "vdm_description",
    "harness_template": "other",
    "user_feedback": "other",
}
CATEGORIES = ("task_api_description", "code_history", "stdout_stderr", "vdm_description", "other")


class Segment(NamedTuple):
    kind: str
    text: str
    source_event_ids: tuple[str, ...] = ()
    origin: str = ""


# ---------------------------------------------------------------------------
# Harness literals from code objects
# ---------------------------------------------------------------------------

# (module, dotted attribute path) of functions whose string constants are
# prompt text the harness sends. Read from sys.modules only (never imported).
_LITERAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("capx.utils.launch_utils", "_build_multi_turn_decision_prompt"),
    ("capx.utils.launch_utils", "_build_multi_turn_decision_prompt_legacy"),
    ("capx.envs.trial", "_capture_initial_visual_feedback"),
    ("capx.envs.trial", "_describe_initial_scene"),
    ("capx.envs.trial", "_get_visual_differencing_feedback"),
    ("capx.envs.trial", "_get_video_differencing_feedback"),
    ("capx.llm.client", "query_model_ensemble"),
    ("capx.llm.client", "query_single_model_ensemble"),
    ("capx.web.async_trial_runner", "run_trial_async"),
)
_LITERAL_CACHE: dict[int, tuple[Segment, ...]] = {}
_LITERAL_LOCK = threading.Lock()


def _code_strings(code: Any, out: list[str], depth: int = 0) -> None:
    if depth > 8:
        return
    for const in getattr(code, "co_consts", ()):
        if isinstance(const, str):
            out.append(const)
        elif hasattr(const, "co_consts"):
            _code_strings(const, out, depth + 1)


def literals_of(func: Any, origin: str) -> tuple[Segment, ...]:
    """Prose-like string constants (>= 8 chars, containing whitespace) of ``func``."""
    try:
        original = inspect.unwrap(func)
    except TimeoutError:  # the trial alarm (section 2)
        raise
    except Exception:  # noqa: BLE001
        original = func
    code = getattr(original, "__code__", None)
    if code is None:
        return ()
    key = id(code)
    with _LITERAL_LOCK:
        cached = _LITERAL_CACHE.get(key)
    if cached is not None:
        return cached
    strings: list[str] = []
    _code_strings(code, strings)
    doc = getattr(original, "__doc__", None)
    seen: set[str] = set()
    out = []
    for s in strings:
        if s is doc or s in seen or len(s) < MIN_SEGMENT_CHARS:
            continue
        if not any(ch.isspace() for ch in s):
            continue
        seen.add(s)
        out.append(Segment("harness_template", s, (), origin))
    result = tuple(out)
    with _LITERAL_LOCK:
        _LITERAL_CACHE[key] = result
    return result


def harness_literals() -> tuple[Segment, ...]:
    out: list[Segment] = []
    for module_name, attr in _LITERAL_SOURCES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        func = getattr(module, attr, None)
        if func is None:
            continue
        out.extend(literals_of(func, f"{module_name}.{attr}"))
    return tuple(out)


def template_literals(template: Any) -> tuple[Segment, ...]:
    """Literal parts of a ``str.format`` template such as ``multi_turn_prompt``."""
    if not isinstance(template, str) or not template:
        return ()
    out = []
    try:
        for literal, _field, _spec, _conv in string.Formatter().parse(template):
            if literal and len(literal) >= MIN_SEGMENT_CHARS:
                out.append(Segment("harness_template", literal, (), "multi_turn_prompt"))
    except ValueError:
        return ()
    return tuple(out)


# ---------------------------------------------------------------------------
# Producer-side registry
# ---------------------------------------------------------------------------


def split_prompt_text(text: Any) -> tuple[str | None, str | None]:
    """Text before and after ``"\\nAPIs:\\n"``; ``(None, None)`` without the marker."""
    if not isinstance(text, str):
        return None, None
    idx = text.find(APIS_MARKER)
    if idx < 0:
        return None, None
    return text[:idx], text[idx + len(APIS_MARKER):]


def _first_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    return None


def find_task_text(prompt: Any) -> tuple[str | None, str | None]:
    """Task text of a code-generation request: the text before ``"\\nAPIs:\\n"``
    in the last user message (``(text, None)`` or ``(None, reason)``)."""
    if not isinstance(prompt, list):
        return None, "prompt_not_a_list"
    last_user = None
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = message
    if last_user is None:
        return None, "no_user_message"
    content = last_user.get("content")
    parts = [content] if isinstance(content, str) else [
        p.get("text") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
    ] if isinstance(content, list) else []
    for text in parts:
        before, _after = split_prompt_text(text)
        if before is not None:
            return before, None
    return None, "no_apis_marker"


class SegmentRegistry:
    """Dynamic segments of one episode (appends from any thread)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dynamic: list[Segment] = []

    def register(self, kind: str, text: Any, source_event_ids: tuple[str, ...] | list[str] = (),
                 origin: str = "") -> None:
        if not isinstance(text, str) or len(text) < MIN_SEGMENT_CHARS:
            return
        seg = Segment(kind, text, tuple(i for i in source_event_ids if i), origin)
        with self._lock:
            self._dynamic.append(seg)

    def dynamic(self) -> tuple[Segment, ...]:
        with self._lock:
            return tuple(self._dynamic)

    def snapshot(self, env: Any = None, multi_turn_prompt: Any = None) -> tuple[Segment, ...]:
        """All segments known right now (called right before a request)."""
        out: list[Segment] = []
        if env is not None:
            task_prompt = _safe_attr(env, "_task_prompt")
            if isinstance(task_prompt, str):
                out.append(Segment("task_prompt", task_prompt, (), "env._task_prompt"))
            system_prompt = _safe_attr(env, "_system_prompt")
            if isinstance(system_prompt, str):
                out.append(Segment("harness_template", system_prompt, (), "env._system_prompt"))
            full = _safe_attr(env, "_full_prompt")
            try:
                text = _first_text(full[-1]["content"]) if isinstance(full, list) and full else None
            except TimeoutError:
                raise
            except Exception:  # noqa: BLE001
                text = None
            before, after = split_prompt_text(text)
            if before is not None:
                out.append(Segment("task_prompt", before, (), "env._full_prompt"))
                out.append(Segment("harness_template", APIS_MARKER, (),
                                   "capx.envs.tasks.base.CodeExecutionEnvBase._get_complete_prompt"))
            if after is not None:
                out.append(Segment("api_docs", after, (), "env._full_prompt"))
        out.extend(template_literals(multi_turn_prompt))
        out.extend(harness_literals())
        out.extend(self.dynamic())
        return tuple(s for s in out if _searchable(s.text))


def _searchable(text: Any) -> bool:
    return isinstance(text, str) and (len(text) >= MIN_SEGMENT_CHARS or text in SHORT_LITERALS)


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return obj.__dict__.get(name) if hasattr(obj, "__dict__") else None
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Writer side: input_summary
# ---------------------------------------------------------------------------


def text_parts(body: Any) -> list[tuple[str, str]]:
    """``(json path, text)`` for every text part of the request messages."""
    out: list[tuple[str, str]] = []
    if not isinstance(body, dict):
        return out
    for key in ("messages", "input"):
        messages = body.get(key)
        if not isinstance(messages, list):
            continue
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                out.append((f"{key}[{i}].content", content))
            elif isinstance(content, list):
                for j, part in enumerate(content):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        out.append((f"{key}[{i}].content[{j}].text", part["text"]))
    return out


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def compute_input_summary(snapshot: tuple[Segment, ...], parts: list[tuple[str, str]],
                          images: list[dict[str, Any]], n_videos: int) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
    covered: dict[str, list[tuple[int, int]]] = {path: [] for path, _ in parts}
    # merge identical (kind, text) segments so each is searched once
    merged: dict[tuple[str, str], list[str]] = {}
    origins: dict[tuple[str, str], str] = {}
    for seg in snapshot:
        if not _searchable(seg.text):
            continue
        key = (seg.kind, seg.text)
        ids = merged.setdefault(key, [])
        for sid in seg.source_event_ids:
            if sid not in ids:
                ids.append(sid)
        origins.setdefault(key, seg.origin)
    not_found: list[dict[str, Any]] = []
    for (kind, text), ids in merged.items():
        found = False
        for path, part in parts:
            start = part.find(text)
            n = 0
            while start >= 0 and n < MAX_OCCURRENCES:
                end = start + len(text)
                entry = {"kind": kind, "path": path, "start": start, "end": end,
                         "source_event_ids": list(ids)}
                if kind == "harness_template" and origins.get((kind, text)):
                    entry["origin"] = origins[(kind, text)]
                categories[CATEGORY_OF.get(kind, "other")].append(entry)
                covered[path].append((start, end))
                found = True
                n += 1
                start = part.find(text, end)
        if not found and kind in DYNAMIC_KINDS:
            not_found.append({"kind": kind, "length": len(text), "source_event_ids": list(ids)})
    unattributed: list[dict[str, Any]] = []
    truncated = False

    def gap(path: str, part: str, start: int, end: int) -> dict[str, Any]:
        entry: dict[str, Any] = {"path": path, "start": start, "end": end}
        if not part[start:end].strip():
            entry["whitespace_only"] = True  # e.g. the "\n" joins of collapse_text_image_inputs
        return entry

    for path, part in parts:
        pos = 0
        for start, end in _merge(covered[path]):
            if start > pos:
                if len(unattributed) >= MAX_UNATTRIBUTED:
                    truncated = True
                    break
                unattributed.append(gap(path, part, pos, start))
            pos = max(pos, end)
        if pos < len(part):
            if len(unattributed) >= MAX_UNATTRIBUTED:
                truncated = True
            else:
                unattributed.append(gap(path, part, pos, len(part)))
    for entries in categories.values():
        entries.sort(key=lambda e: (e["path"], e["start"]))
    # ``images`` lists every media part in payload order; a video counts as a
    # video, not an image (section 6.4), and other data URLs count as neither
    n_images = sum(1 for i in images if i.get("kind", "image") in ("image", "external_url"))
    summary: dict[str, Any] = {
        "n_images": n_images,
        "n_videos": n_videos,
        "no_image_input": n_images == 0 and n_videos == 0,
        "categories": {"images": images, **categories},
        "unattributed": unattributed,
        "registered_not_found": not_found,
    }
    if truncated:
        summary["unattributed_truncated"] = True
    return summary


__all__ = [
    "Segment",
    "SegmentRegistry",
    "MIN_SEGMENT_CHARS",
    "APIS_MARKER",
    "find_task_text",
    "split_prompt_text",
    "text_parts",
    "compute_input_summary",
    "harness_literals",
    "template_literals",
    "literals_of",
]
