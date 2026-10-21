"""Write a synthetic experiment trace for the trace viewer.

Implements the storage layout of docs/trace-format.md section 4.1 and the event schema
of sections 5 and 6 with real files, so the viewer (web-ui/src/trace) and the trace API
(capx_trace_viewer/trace_api.py, which reads through capx_trace_viewer/reader.py) can be developed and
tested without running an experiment. Nothing here comes from a real run: run.json and
every episode.json carry "synthetic": true.

The run "demo_trace_viewer" holds three episodes:

- cube_stack__M4__s1__i1__a1 (ended with gaps): a multi-turn condition with VDM
  image differencing and visual feedback. It has an HTTP 503 retry, a VDM request with
  two images in order, a perception VLM (Molmo) request nested under an API call, a
  nested API call, an API call that raises, a degraded SAM3 call, a skipped frame
  (disk_low), stdout long enough to be stored as a txt.gz asset, a multi-turn decision
  request whose input summary finds the previous block's stdout, a registered segment
  that a later input no longer contains, and events that arrive after episode_end. Its
  second block also has an api_call_end whose data exceeded the 256 KB line limit and was
  moved to a json.gz asset ("oversize": true, "data_asset", as the trace recorder
  does when an event line would exceed 256 KB), and an api_call_error whose message (about 5900 characters) and
  traceback are stored as text assets ({"truncated", "preview", "length", "asset",
  "sha256", "written"}), caught by the generated code.
- 20260926T060000Z__web__5e1f0a2c (running): a web UI episode with streamed responses,
  API-provided reasoning, a VDM video request, user feedback, a block timeout followed
  by env_reset, a service_call_error, and open start events. Its heartbeat is fresh when
  the script ends; --live keeps it fresh and keeps appending events.
- cube_lift__S3__s2__i1__a1 (incomplete): a single-turn run with video recording and
  state snapshots off, a masked body field, a corrupt line in the middle of events.jsonl,
  a torn last line and a stale heartbeat.

The run directory also gets runner_records.jsonl and an unscoped log with the runner's
preflight request and response (payloads under unscoped.412300.files/, as the recorder
writes them), and the trace root gets an _unscoped/ file (which the run listing must
ignore).

Field formats follow the recorder where the spec leaves them open (checked against
the trace recorder format): values are summarised as
serialize.summarize() does (scalars and short strings inline; containers as {"type",
"length", "items"}; arrays as {"type": "ndarray", "shape", "dtype"} plus "values" or a
saved npy "path" with "stats"; a long string as {"truncated", "preview", "length",
"asset", "sha256", "written"}, whose txt.gz is not listed in "artifacts", as
fill_pending writes it). Service results keep their arrays in one npz (keys a0, a1, ...; bool arrays as
<key>__packbits + <key>__shape) referenced by "npz" and "npz_key". End events sit under
their start event. feedback_built keeps the builder's text arguments
(complete_multi_turn_prompt, visual_differencing_feedback, user_feedback) in the JSON file
named by "texts_path"; web UI user text is appended to the multi-turn prompt with
"\\n\\nUser feedback: ", as the runner that wrote the trace does. episode.json has the
recorder's fields (started_wall_time, ended_wall_time, env.viser_port, status "running"
until episode_end).

Usage:
    .venv/bin/python scripts/make_demo_trace.py --out outputs/traces_demo
    .venv/bin/python scripts/make_demo_trace.py --out outputs/traces_demo --live 600
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "capx-trace/1"
RUN_ID = "demo_trace_viewer"
MARKER = ".capx_demo_trace"
REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_INLINE_LIMIT = 4000
MAX_EVENT_LINE_BYTES = 256 * 1024  # the trace recorder: longer lines move "data" to an asset
SUBSAMPLE = 5  # control steps per buffer frame (_record_frame)
CONTROL_DT = 0.05  # robosuite control period, seconds of simulated time per step

PROXY_URL = "http://127.0.0.1:8110/chat/completions"
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
MOLMO_URL = "http://127.0.0.1:8122/v1/chat/completions"
GEMINI = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Prompt texts (taken from the real harness; the API docs are shortened)
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = "You are a helpful assistant that generates Python code to directly solve the task."

STACK_TASK_PROMPT = (
    "You are controlling a Franka Emika robot with API described below.\n"
    "Goal: place the red cube on top of the green cube and then open the gripper.\n"
    "You may write python code comments for reasoning but ONLY write the executable Python "
    "code and do not write it in code fences.\n"
    "After this code executes, you will be able to get a new observation of the environment "
    "and write new code to complete the task if needed.\n"
    "The functions (APIs) below are already imported to the environment. If you want to use "
    "numpy, you need to import it explicitly.\n"
)

LIFT_TASK_PROMPT = (
    "You are controlling a Franka Emika robot with API described below.\n"
    "Goal: pick up the red cube and lift it 20 cm above the table.\n"
    "Write only executable Python code; the functions (APIs) below are already imported.\n"
)

API_DOCS = """get_observation()
  Doc:
    Get the current observation of the robot's cameras.
    Returns:
        dict with key "robot0_robotview": {"images": {"rgb": (H, W, 3) uint8,
        "depth": (H, W, 1) float32 metres}, "intrinsics": (3, 3), "pose_mat": (4, 4)}.

segment_sam3_text_prompt(rgb: numpy.ndarray, text_prompt: str) -> list[dict[str, typing.Any]]
  Doc:
    Run SAM3 segmentation on an RGB image conditioned on a text prompt.
    Returns:
        A list of dicts with "mask" (H, W) bool, "box" [x0, y0, x1, y1] pixels and "score".

point_prompt_molmo(image: numpy.ndarray, text_prompt: str) -> dict[str, tuple[int | None, int | None]]
  Doc:
    Use Molmo to point to a coordinate in the image based on a text prompt.
    Returns:
        Pixel coordinates for each object query; (None, None) if parsing failed.

plan_grasp(depth: numpy.ndarray, intrinsics: numpy.ndarray, segmentation: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
  Doc:
    Plan grasp candidates using Contact-GraspNet for a single instance.
    Returns:
        grasp_poses (K, 4, 4) in the camera frame and scores (K,).

solve_ik(position: numpy.ndarray, quaternion_wxyz: numpy.ndarray) -> numpy.ndarray
  Doc:
    Solve inverse kinematics for the panda_hand link.
    Args:
        position: Target position in world frame. Shape: (3,), metres.
        quaternion_wxyz: Target orientation (w, x, y, z) in world frame.
    Returns:
        Joint positions (7,) in radians.

move_to_joints(joints: numpy.ndarray) -> None
  Doc:
    Move the arm to the given joint positions (radians).

open_gripper() -> None
  Doc:
    Open the gripper fully.

close_gripper() -> None
  Doc:
    Close the gripper.

mask_to_world_points(mask: numpy.ndarray, depth: numpy.ndarray, intrinsics: numpy.ndarray, cam_pose: numpy.ndarray) -> numpy.ndarray
  Doc:
    Back-project the masked depth pixels to world-frame points (N, 3) in metres.

pixel_to_world_point(pixel: tuple[int, int], depth: numpy.ndarray, intrinsics: numpy.ndarray, cam_pose: numpy.ndarray) -> numpy.ndarray
  Doc:
    Back-project one pixel to a world-frame point (3,) in metres.

depth_to_point_cloud(depth_img: numpy.ndarray, intrinsics: numpy.ndarray) -> numpy.ndarray
  Doc:
    Convert a depth image to a camera-frame point cloud (H*W, 3).

select_top_down_grasp(grasp_poses: numpy.ndarray, scores: numpy.ndarray) -> numpy.ndarray
  Doc:
    Pick the highest-scoring grasp whose approach axis points down. Returns a (4, 4) pose.

summarize_workspace() -> dict[str, typing.Any]
  Doc:
    Describe every body in the workspace: name, position (world frame, metres), faces and
    mesh vertices.

verify_demo_alignment(top: str, bottom: str) -> None
  Doc:
    Check that the contact points between two stacked objects are within 2 mm.
    Raises ValueError listing every contact point outside the tolerance."""

MULTI_TURN_TEMPLATE = (
    "The following code blocks were just executed:\n"
    "```python\n"
    "{executed_code}\n"
    "```\n"
    "The current console stdout is:\n"
    "```\n"
    "{console_stdout}\n"
    "```\n"
    "The current console stderr is:\n"
    "```\n"
    "{console_stderr}\n"
    "```\n"
    "If there were no errors during compilation or execution, the console stderr will be "
    "empty. If there were errors, consider regenerating the code.\n"
    "Please respond with EXACTLY ONE of the following:\n"
    "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block "
    "(```python...```) if you want to modify the code.\n"
    "- The word 'FINISH' if the task appears to be complete\n"
)

# The traced web runner appends injected user text to the multi-turn prompt with this marker.
USER_FEEDBACK_MARKER = "\n\nUser feedback: "


def multi_turn_prompt(executed_code: str, stdout: str, stderr: str, user_feedback: str | None = None) -> str:
    """The `complete_multi_turn_prompt` the harness passes to the decision prompt builder."""
    text = MULTI_TURN_TEMPLATE.format(executed_code=executed_code, console_stdout=stdout, console_stderr=stderr)
    if user_feedback:
        text += USER_FEEDBACK_MARKER + user_feedback
    return text


DECISION_IMAGE_TEXT = (
    "Included below is an image of the current state of the environment (after the code "
    "above was executed)."
)
DECISION_DIFF_HEADER = (
    "Included below is the observed differences between the current state of the "
    "environment (afer the code above was executed) and the previous state of the "
    "environment (before the code above was executed):"
)
DECISION_VIDEO_HEADER = (
    "Included below is a description of the robot's execution based on video observation "
    "of this turn:"
)
DECISION_INSTRUCTION = (
    "Based on the code output, potential error messages, and the observation made above, "
    "carefully reason about the following:\nPlease respond with EXACTLY ONE of the "
    "following:\n- The word 'REGENERATE' followed immediately by new Python code in a fenced "
    "code block (```python...```) if you want to modify the code.\n- The word 'FINISH' if "
    "the task appears to be complete"
)

VDM_INITIAL_SYSTEM = (
    "You are a helpful assistant that describes the initial state of the environment with the "
    "goal of the task in mind. You should try to provide objective information and no "
    "assumptions. Do *NOT* write any code."
)
VDM_INITIAL_TEXT = (
    "Describe the initial state of the environment with the goal of the task in mind. You "
    "should try to provide objective information and no assumptions. Do *NOT* write any code."
)
VDM_DIFF_SYSTEM = (
    "You are a helpful assistant that describes the difference between the current state of "
    "the environment and the previous state of the environment with the goal of the task in "
    "mind and whether the task has been completed. You should try to provide objective "
    "information and no assumptions. Do *NOT* write any code."
)
VDM_DIFF_TEXT = (
    "Describe the difference between the current state of the environment and the previous "
    "state of the environment with the goal of the task in mind and whether the task has been "
    "completed. You should try to provide objective information and no assumptions. Do *NOT* "
    "write any code.."
)
VDM_VIDEO_SYSTEM = (
    "You are a helpful assistant that analyzes robot execution videos. You describe what "
    "happened during the robot's code execution, what actions were taken, how the environment "
    "changed, and whether the task appears to have been completed. Provide objective "
    "information and no assumptions. Do *NOT* write any code."
)
VDM_VIDEO_TEXT = (
    "The following video shows the robot executing code in the environment from the main "
    "camera view. Describe what happened during execution, including what actions the robot "
    "took, how the objects in the scene changed, and whether the task appears to have been "
    "completed. Provide objective information and no assumptions. Do *NOT* write any code."
)

TEMPLATE_SEGMENTS = [
    CODEGEN_SYSTEM,
    DECISION_IMAGE_TEXT,
    DECISION_DIFF_HEADER,
    DECISION_VIDEO_HEADER,
    DECISION_INSTRUCTION,
    VDM_INITIAL_SYSTEM,
    VDM_INITIAL_TEXT,
    VDM_DIFF_SYSTEM,
    VDM_DIFF_TEXT,
    VDM_VIDEO_SYSTEM,
    VDM_VIDEO_TEXT,
    "Main camera view:",
    "Previous state (main camera):",
    "Current state (main camera):",
    "Main camera video:",
    # literal parts of the multi-turn template around its three placeholders
    *[
        part
        for chunk in MULTI_TURN_TEMPLATE.split("{executed_code}")
        for sub in chunk.split("{console_stdout}")
        for part in sub.split("{console_stderr}")
    ],
]

CONTENT_KINDS = {"code_history", "stdout", "stderr", "vdm_description", "user_feedback"}
CATEGORY_OF_KIND = {
    "task_prompt": "task_api_description",
    "api_docs": "task_api_description",
    "harness_template": "other",
    "code_history": "code_history",
    "stdout": "stdout_stderr",
    "stderr": "stdout_stderr",
    "vdm_description": "vdm_description",
    "user_feedback": "other",
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def endpoint_info(url: str, *, known_proxy: bool) -> dict[str, Any]:
    rest = url.split("://", 1)[1]
    hostport, _, path = rest.partition("/")
    host, _, port = hostport.partition(":")
    return {
        "url": url,
        "host": host,
        "port": int(port) if port else None,
        "path": "/" + path,
        "known_proxy": known_proxy,
    }


def png_bytes(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="png")
    return buf.getvalue()


def jpeg_bytes(pixels: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="jpeg", quality=quality)
    return buf.getvalue()


def npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def mp4_bytes(frames: list[np.ndarray], fps: int = 10) -> bytes:
    import imageio  # Used to encode the synthetic video.

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "turn.mp4")
        with imageio.get_writer(path, fps=fps, format="FFMPEG", codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.ascontiguousarray(frame))
        return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# Synthetic scene rendering
# ---------------------------------------------------------------------------

TABLE_Z = 0.82
CUBE_HALF = 0.02


@dataclass
class Scene:
    gripper: np.ndarray
    red: np.ndarray
    green: np.ndarray
    gripper_open: bool = True
    held: bool = False

    def copy(self) -> Scene:
        return Scene(
            self.gripper.copy(), self.red.copy(), self.green.copy(), self.gripper_open, self.held
        )


def _project(camera: str, p: np.ndarray, size: int) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in p)
    if camera == "robot0_robotview":  # the perception camera sits a little to the side
        y += 0.03
    depth = 1.0 + (0.75 - x)
    scale = size * 1.6 / depth
    u = size / 2 + y * scale
    v = size * 0.62 + (x - 0.55) * size * 1.1 - (z - TABLE_Z) * scale
    return u, v, scale


def render(scene: Scene, camera: str, size: int) -> np.ndarray:
    img = Image.new("RGB", (size, size), (70, 76, 86))
    draw = ImageDraw.Draw(img)
    if camera == "robot0_eye_in_hand":
        draw.rectangle([0, 0, size, size], fill=(150, 118, 88))
        g = scene.gripper
        for color, cube in (((40, 150, 60), scene.green), ((200, 40, 40), scene.red)):
            height = max(float(g[2] - cube[2]), 0.03)
            half = CUBE_HALF / height * size * 0.9
            u = size / 2 + float(cube[1] - g[1]) / height * size * 0.9
            v = size / 2 + float(cube[0] - g[0]) / height * size * 0.9
            draw.rectangle([u - half, v - half, u + half, v + half], fill=color)
        gap = size * (0.32 if scene.gripper_open else 0.16)
        draw.rectangle([size / 2 - gap - 6, size - 34, size / 2 - gap + 6, size], fill=(35, 35, 40))
        draw.rectangle([size / 2 + gap - 6, size - 34, size / 2 + gap + 6, size], fill=(35, 35, 40))
        return np.asarray(img, dtype=np.uint8)

    draw.polygon(
        [(0, size * 0.45), (size, size * 0.45), (size, size), (0, size)], fill=(150, 118, 88)
    )
    items = [("green", scene.green), ("red", scene.red)]
    items.sort(key=lambda it: -float(it[1][0]))
    for name, cube in items:
        u, v, scale = _project(camera, cube, size)
        su, sv, _ = _project(camera, np.array([cube[0], cube[1], TABLE_Z]), size)
        half = CUBE_HALF * scale
        draw.ellipse([su - half * 1.2, sv - half * 0.4, su + half * 1.2, sv + half * 0.4], fill=(110, 86, 64))
        color = (200, 40, 40) if name == "red" else (40, 150, 60)
        draw.rectangle([u - half, v - 2 * half, u + half, v], fill=color)
    gu, gv, gscale = _project(camera, scene.gripper, size)
    gap = (0.04 if scene.gripper_open else 0.022) * gscale
    finger_h = 0.05 * gscale
    draw.line([gu, 0, gu, gv - finger_h], fill=(210, 210, 215), width=max(2, size // 40))
    draw.rectangle([gu - gap - 3, gv - finger_h, gu - gap + 3, gv], fill=(35, 35, 40))
    draw.rectangle([gu + gap - 3, gv - finger_h, gu + gap + 3, gv], fill=(35, 35, 40))
    return np.asarray(img, dtype=np.uint8)


def depth_map(scene: Scene, size: int) -> np.ndarray:
    """A synthetic (H, W) float32 depth image in metres with NaN holes."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    depth = 1.1 + 0.4 * (yy / size)
    for cube in (scene.red, scene.green):
        u, v, scale = _project("robot0_robotview", cube, size)
        half = CUBE_HALF * scale
        inside = (np.abs(xx - u) < half) & (yy > v - 2 * half) & (yy < v)
        depth[inside] -= 0.05
    depth[:6, :] = np.nan
    return depth.astype(np.float32)


def cube_mask(scene: Scene, cube: np.ndarray, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    u, v, scale = _project("robot0_robotview", cube, size)
    half = CUBE_HALF * scale
    return (np.abs(xx - u) < half) & (yy > v - 2 * half) & (yy < v)


# ---------------------------------------------------------------------------
# Value summaries (section 5.3)
# ---------------------------------------------------------------------------


def _float(value: float) -> float | str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def array_stats(arr: np.ndarray) -> dict[str, Any] | None:
    """Compute min, max and mean over finite values, as the recorder's array_stats() does."""
    if arr.dtype.kind not in "biuf":
        return None
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)] if arr.dtype.kind == "f" else flat
    n_nonfinite = int(flat.size - finite.size)
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "n_finite": 0, "n_nonfinite": n_nonfinite}
    mn, mx = finite.min().item(), finite.max().item()
    return {
        "min": _float(mn) if isinstance(mn, float) else mn,
        "max": _float(mx) if isinstance(mx, float) else mx,
        "mean": _float(float(finite.mean(dtype=np.float64))),
        "n_finite": int(finite.size),
        "n_nonfinite": n_nonfinite,
    }


class Summarizer:
    """Encodes values as the trace recorder summarize() does (section 5.3).

    Saved files are filled in as the trace recorder does it: ``asset``, ``sha256`` and
    ``written`` for long text; ``stats``, ``path`` and ``written`` for arrays.
    """

    def __init__(self, episode: EpisodeBuilder, label: str) -> None:
        self.episode = episode
        self.label = label
        self.artifacts: list[dict[str, Any]] = []

    def __call__(self, value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, bool) or type(value) is int:
            return value
        if isinstance(value, np.ndarray):
            return self.array(value)
        if isinstance(value, np.generic):
            return self(value.item(), depth)
        if isinstance(value, float):
            return _float(value)
        if isinstance(value, str):
            if len(value) <= TEXT_INLINE_LIMIT:
                return value
            return self.episode.long_text(value, self.artifacts)
        if depth >= 4:
            return {"type": type(value).__name__, "truncated": "depth"}
        if isinstance(value, dict):
            items = {str(k): self(v, depth + 1) for k, v in list(value.items())[:50]}
            out: dict[str, Any] = {"type": "dict", "length": len(value), "items": items}
            if len(value) > 50:
                out["truncated"] = len(value) - 50
            return out
        if isinstance(value, list | tuple):
            out = {
                "type": type(value).__name__,
                "length": len(value),
                "items": [self(v, depth + 1) for v in value[:50]],
            }
            if len(value) > 50:
                out["truncated"] = len(value) - 50
            return out
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}

    def array(self, value: np.ndarray) -> dict[str, Any]:
        node: dict[str, Any] = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if value.size <= 64:
            node["values"] = [_float(v) if isinstance(v, float) else v for v in value.ravel().tolist()]
            if value.ndim > 1:
                node["values"] = np.asarray(node["values"], dtype=object).reshape(value.shape).tolist()
            return node
        ref = self.episode.asset(npy_bytes(value), "npy")
        node.update({"stats": array_stats(value), "path": ref["path"], "written": True})
        self.artifacts.append(
            {
                **ref,
                "kind": "array",
                "mime": "application/octet-stream",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "label": self.label,
                "written": True,
            }
        )
        return node


# Units and frames as in the trace recorder table (only where the API docstring
# states them; everything else is shown as unknown).
UNITS = {
    "depth_to_point_cloud": {
        "args": {"depth_img": {"unit": "m", "frame": None}},
        "returns": {"points": {"unit": "m", "frame": "camera"}},
    },
    "mask_to_world_points": {"returns": {"points": {"unit": "m", "frame": "world"}}},
    "pixel_to_world_point": {
        "args": {"u": {"unit": "pixel (column)", "frame": "image"}, "v": {"unit": "pixel (row)", "frame": "image"}},
        "returns": {"point": {"unit": None, "frame": "world"}},
    },
}


# ---------------------------------------------------------------------------
# Episode builder
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    kind: str
    text: str
    source_event_ids: list[str] = field(default_factory=list)


class EpisodeBuilder:
    """Accumulates the events of one episode and writes its files."""

    def __init__(
        self,
        run_dir: Path,
        episode_id: str,
        *,
        pid: int,
        t0: float,
        mono0: float,
        cameras: tuple[str, ...],
        record_video: bool = True,
        record_state: bool = True,
        env_thread: str = "MainThread",
        model_thread: str = "MainThread",
    ) -> None:
        self.run_dir = run_dir
        self.episode_id = episode_id
        self.dir = run_dir / episode_id
        self.dir.mkdir(parents=True)
        self.pid = pid
        self.wall = t0
        self.mono = mono0
        self.cameras = cameras
        self.record_video = record_video
        self.record_state = record_state
        self.env_thread = env_thread
        self.model_thread = model_thread
        self.counter = 0
        self.seq = 0
        self.events: list[dict[str, Any]] = []
        self.raw_lines: list[bytes] = []
        self.sim_step = 0
        self.sim_epoch = 0
        self.buffer: list[tuple[int | None, Scene]] = []
        self.saved_frames: dict[tuple[int, int, str], dict[str, Any]] = {}
        self.segments: list[Segment] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.video_reads: list[tuple[dict[str, Any], list[str]]] = []
        self.skip_next_frame_reason: str | None = None
        self.periodic_every = 4
        self.ended = False
        self.api_stack: list[dict[str, Any]] = []
        self.stdout = ""
        self.stderr = ""
        self.video_frames: dict[str, list[np.ndarray]] = {}
        self._pending_keyframe: int | None = None

    def events_of(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event_type"] == event_type]

    # -- ids and files -----------------------------------------------------
    def new_id(self, prefix: str = "ev_", n: int = 20) -> str:
        self.counter += 1
        return prefix + sha256_hex(f"{self.episode_id}:{prefix}:{self.counter}".encode())[:n]

    def write_file(self, rel: str, data: bytes) -> None:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def long_text(self, text: str, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """A text over 4000 characters: preview plus a txt.gz asset (serialize.text_value).

        As the recorder's fill_pending writes it (``text_asset(..., add=False)``), the asset
        is referenced only by the returned node, not listed in the event's artifacts.
        """
        ref = self.asset(gzip.compress(text.encode("utf-8"), mtime=0), "txt.gz")
        return {
            "truncated": True,
            "preview": text[:TEXT_INLINE_LIMIT],
            "length": len(text),
            "asset": ref["path"],
            "sha256": sha256_hex(text.encode("utf-8")),
            "written": True,
        }

    def asset(self, data: bytes, ext: str) -> dict[str, Any]:
        digest = sha256_hex(data)
        rel = f"assets/{digest[:2]}/{digest}.{ext}"
        if not (self.dir / rel).exists():
            self.write_file(rel, data)
        return {"path": rel, "sha256": digest, "bytes": len(data)}

    # -- clocks ------------------------------------------------------------
    def tick(self, seconds: float) -> None:
        self.wall += seconds
        self.mono += seconds

    def sim_fields(self, thread: str) -> dict[str, Any]:
        if thread != self.env_thread:
            return {"sim_step": None, "sim_time": None, "sim_epoch": None}
        return {
            "sim_step": self.sim_step,
            "sim_time": round(self.sim_step * CONTROL_DT, 4),
            "sim_epoch": self.sim_epoch,
        }

    # -- events ------------------------------------------------------------
    def emit(
        self,
        event_type: str,
        status: str,
        data: dict[str, Any],
        *,
        parent: dict[str, Any] | None = None,
        refs: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        thread: str | None = None,
        dt: float = 0.02,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        self.tick(dt)
        thread = thread or self.env_thread
        event = {
            "schema_version": SCHEMA,
            "run_id": RUN_ID,
            "episode_id": self.episode_id,
            "event_id": event_id or self.new_id(),
            "parent_event_id": parent["event_id"] if parent else None,
            "seq": self.seq,
            "wall_time": round(self.wall, 6),
            "monotonic_time": round(self.mono, 6),
            "pid": self.pid,
            "thread": thread,
            **self.sim_fields(thread),
            "event_type": event_type,
            "status": status,
            "late": self.ended,
            "refs": refs or {},
            "artifacts": artifacts or [],
            "data": data,
        }
        self.seq += 1
        line = event_line(event)
        if len(line) > MAX_EVENT_LINE_BYTES:
            line = self.move_data_to_asset(event)
        self.events.append(event)
        self.raw_lines.append(line)
        return event

    def move_data_to_asset(self, event: dict[str, Any]) -> bytes:
        """Move data to an asset when the recorder's event line would exceed 256 KB.

        The whole ``data`` goes to a json.gz asset (label "payload", listed in artifacts);
        the line keeps only scalars and strings of at most 200 characters, plus
        ``oversize``, ``data_asset`` and ``data_asset_written``.
        """
        data = event.get("data", {})
        raw = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        ref = self.asset(gzip.compress(raw, mtime=0), "json.gz")
        event["artifacts"].append(
            {**ref, "kind": "json", "mime": "application/gzip", "label": "payload", "written": True}
        )
        small = {
            k: v
            for k, v in data.items()
            if v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200)
        }
        small["oversize"] = True
        small["data_asset"] = ref["path"]
        small["data_asset_written"] = True
        event["data"] = small
        line = event_line(event)
        if len(line) > MAX_EVENT_LINE_BYTES:
            raise RuntimeError(f"event line over 256 KB even without its data: {event['event_type']}")
        return line

    def _lines_since(self, event: dict[str, Any]) -> int:
        """How many lines were written after ``event`` (to rewrite its line in place)."""
        return len(self.raw_lines) - 1 - next(
            i for i in range(len(self.raw_lines) - 1, -1, -1)
            if json.loads(self.raw_lines[i]).get("event_id") == event["event_id"]
        )

    def lose_seq(self, raw: bytes) -> None:
        """Consume one seq for a line that was damaged on disk (corrupt-line demo)."""
        self.seq += 1
        self.raw_lines.append(raw)

    # -- frames ------------------------------------------------------------
    def record_buffer_frame(self, scene: Scene, *, backfilled: bool = False) -> None:
        if not self.record_video:
            return
        self.buffer.append((None if backfilled else self.sim_step, scene.copy()))
        index = len(self.buffer) - 1
        if index % self.periodic_every == 0:
            self.save_frame(index, reason="periodic")

    def save_frame(
        self, index: int, *, reason: str, for_event: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        frame_step, scene = self.buffer[index]
        first_ref = None
        for camera in self.cameras:
            key = (self.sim_epoch, index, camera)
            if key in self.saved_frames:
                first_ref = first_ref or self.saved_frames[key]
                continue
            size = 256 if camera == "agentview" else 128
            pixels = render(scene, camera, size)
            rel = f"frames/{camera}/e{self.sim_epoch}_{index}.jpg"
            artifact: dict[str, Any] = {
                "path": rel,
                "kind": "image",
                "mime": "image/jpeg",
                "width": size,
                "height": size,
                "camera": camera,
                "label": "debug_frame",
            }
            if self.skip_next_frame_reason:
                artifact.update(
                    {"written": False, "reason": self.skip_next_frame_reason, "sha256": None, "bytes": None}
                )
            else:
                data = jpeg_bytes(pixels)
                self.write_file(rel, data)
                artifact.update({"written": True, "sha256": sha256_hex(data), "bytes": len(data)})
            ref = {
                "epoch": self.sim_epoch,
                "buffer_index": index,
                "camera": camera,
                "frame_sim_step": frame_step,
                "path": rel,
                "written": artifact["written"],
            }
            if not artifact["written"]:
                ref["reason"] = artifact["reason"]
            self.saved_frames[key] = ref
            first_ref = first_ref or ref
            frame_event = self.emit(
                "frame",
                "ok" if artifact["written"] else "warning",
                {
                    "camera": camera,
                    "epoch": self.sim_epoch,
                    "buffer_index": index,
                    "sim_step": frame_step,
                    "reason": reason,
                    "for_event_id": for_event["event_id"] if for_event else None,
                },
                refs={"for_event_id": for_event["event_id"]} if for_event else {},
                artifacts=[artifact],
                dt=0.005,
            )
            if not artifact["written"]:
                self.warning(
                    self.skip_next_frame_reason or "write_failed",
                    "Free space 1.7 GB is below CAPX_TRACE_MIN_FREE_GB=2: the frame file was "
                    "skipped. Event lines are still written.",
                    frame_event,
                )
                self.skip_next_frame_reason = None
        return first_ref

    def keyframe(self, for_event: dict[str, Any]) -> dict[str, Any] | None:
        if not self.record_video or not self.buffer:
            return None
        ref = self.save_frame(len(self.buffer) - 1, reason="keyframe", for_event=for_event)
        return dict(ref) if ref else None

    def state_ref(self, scene: Scene) -> dict[str, Any] | None:
        if not self.record_state:
            return None  # CAPX_TRACE_STATE=0
        state = {
            "sim_step": self.sim_step,
            "sim_time": round(self.sim_step * CONTROL_DT, 4),
            "qpos": [round(float(v), 5) for v in np.linspace(-0.4, 0.4, 9) + self.sim_step * 1e-4],
            "body_xpos": {
                "gripper0_right_eef": [round(float(v), 5) for v in scene.gripper],
                "cubeA_main": [round(float(v), 5) for v in scene.red],
                "cubeB_main": [round(float(v), 5) for v in scene.green],
            },
            "body_xquat": {
                "gripper0_right_eef": [0.0, 1.0, 0.0, 0.0],
                "cubeA_main": [1.0, 0.0, 0.0, 0.0],
                "cubeB_main": [0.9998, 0.0, 0.0, 0.0175],
            },
        }
        ref = self.asset(gzip.compress(json.dumps(state).encode(), mtime=0), "json.gz")
        return {"path": ref["path"], "written": True, "sim_time": state["sim_time"]}

    def advance(self, scene: Scene, target: np.ndarray, steps: int, seconds: float) -> tuple[int, int]:
        """Move the gripper to target over `steps` control steps; returns buffer [first, last]."""
        start = scene.gripper.copy()
        first = len(self.buffer)
        for k in range(1, steps + 1):
            self.sim_step += 1
            scene.gripper = start + (target - start) * (k / steps)
            if scene.held:
                scene.red = scene.gripper - np.array([0.0, 0.0, 0.045])
            if self.sim_step % SUBSAMPLE == 0:
                self.record_buffer_frame(scene)
            self.tick(seconds / steps)
        return first, len(self.buffer) - 1

    def warning(self, code: str, message: str, related: dict[str, Any] | None) -> dict[str, Any]:
        return self.emit(
            "trace_warning",
            "warning",
            {"code": code, "message": message, "related_event_id": related["event_id"] if related else None},
            refs={"for_event_id": related["event_id"]} if related else {},
            dt=0.001,
        )

    # -- model calls -------------------------------------------------------
    def register(self, kind: str, text: str, sources: list[str] | None = None) -> None:
        if len(text) >= 8:
            self.segments.append(Segment(kind, text, list(sources or [])))

    def model_request(
        self,
        *,
        role: str,
        call_kind: str,
        turn_index: int | None,
        url: str,
        payload: dict[str, Any],
        logical_call_id: str,
        http_attempt: int,
        known_proxy: bool,
        send_as_bytes: bool = False,
        parent: dict[str, Any] | None = None,
        refs_extra: dict[str, Any] | None = None,
        mask_keys: tuple[str, ...] = (),
        thread: str | None = None,
    ) -> dict[str, Any]:
        event_id = self.new_id()
        body = json.dumps(payload)
        body_bytes = body.encode("utf-8") if send_as_bytes else body.encode("ascii")
        normalized, media = normalize_payload(payload, self)
        masked: list[str] = []
        for key in mask_keys:
            if key in normalized:
                normalized[key] = "***masked***"
                masked.append(key)
        # reconstruction check, as the recorder does it (before masking)
        restored = restore_payload(normalize_payload(payload, self)[0], self)
        if json.dumps(restored).encode() != body_bytes:
            raise RuntimeError("demo payload does not round-trip")
        reconstruction: dict[str, Any] = (
            {"verified": True, "method": "masked_compare", "masked": masked}
            if masked
            else {"verified": True, "method": "json_roundtrip"}
        )
        payload_rel = f"payloads/{event_id}.request.json"
        payload_data = json.dumps(normalized, indent=1).encode()
        self.write_file(payload_rel, payload_data)
        images = []
        for order, item in enumerate(media):
            source = self.image_source(item)
            images.append({**item, "order": order, "source": source})
        messages = payload.get("messages", [])
        data = {
            "role": role,
            "call_kind": call_kind,
            "turn_index": turn_index,
            "logical_call_id": logical_call_id,
            "http_attempt": http_attempt,
            "endpoint": endpoint_info(url, known_proxy=known_proxy),
            "model": payload.get("model"),
            "params": {k: v for k, v in normalized.items() if k != "messages"},
            "stream": bool(payload.get("stream", False)),
            "payload_path": payload_rel,
            "body_sha256": sha256_hex(body_bytes),
            "body_bytes": len(body_bytes),
            "reconstruction": reconstruction,
            "masked_fields": masked,
            "images": images,
            "n_messages": len(messages),
            "n_images": sum(1 for m in images if m["mime"].startswith("image/")),
            "n_videos": sum(1 for m in images if m["mime"].startswith("video/")),
            "input_summary": self.input_summary(messages, images),
        }
        artifacts = [
            {
                "path": payload_rel,
                "sha256": sha256_hex(payload_data),
                "kind": "payload",
                "mime": "application/json",
                "bytes": len(payload_data),
                "label": "payload",
                "written": True,
            }
        ]
        for item in images:
            artifacts.append(
                {
                    "path": item["asset"],
                    "sha256": item["sha256"],
                    "kind": "video" if item["mime"].startswith("video/") else "image",
                    "mime": item["mime"],
                    "bytes": item["bytes"],
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "label": "perception_input" if role == "perception_vlm" else "model_input",
                    "written": True,
                }
            )
        refs = {"logical_call_id": logical_call_id, "http_attempt": http_attempt}
        if turn_index is not None:
            refs["turn_index"] = turn_index
        refs.update(refs_extra or {})
        return self.emit(
            "model_request",
            "started",
            data,
            parent=parent,
            refs=refs,
            artifacts=artifacts,
            thread=thread or self.model_thread,
            event_id=event_id,
        )

    def model_response(
        self,
        request: dict[str, Any],
        content: str,
        *,
        latency_s: float,
        usage: dict[str, Any] | None,
        reasoning: dict[str, Any] | None = None,
        streamed: bool = False,
        n_chunks: int | None = None,
        ttft_s: float | None = None,
        response_model: str | None = None,
    ) -> dict[str, Any]:
        req = request["data"]
        event_id = self.new_id()
        response_id = "gen-" + sha256_hex(event_id.encode())[:24]
        if streamed:
            body: dict[str, Any] = {
                "streamed": True,
                "n_chunks": n_chunks,
                "assembled": {"content": content, "reasoning": (reasoning or {}).get("text")},
                "finish_reason": "stop",
            }
        else:
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                message[reasoning["field"]] = reasoning["text"]
            body = {
                "id": response_id,
                "object": "chat.completion",
                "model": response_model or req["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": usage,
            }
        payload_rel = f"payloads/{event_id}.response.json"
        payload_data = json.dumps(body, indent=1).encode()
        self.write_file(payload_rel, payload_data)
        refs = {k: v for k, v in request["refs"].items()}
        refs["model_event_id"] = request["event_id"]
        return self.emit(
            "model_response",
            "ok",
            {
                "role": req["role"],
                "call_kind": req["call_kind"],
                "logical_call_id": req["logical_call_id"],
                "http_attempt": req["http_attempt"],
                "http_status": 200,
                "latency_s": latency_s,
                "streamed": streamed,
                "n_chunks": n_chunks,
                "ttft_s": ttft_s,
                "content": content,
                "reasoning": reasoning,
                "usage": usage,
                "finish_reason": "stop",
                "response_model": response_model or req["model"],
                "response_id": response_id,
                "payload_path": payload_rel,
            },
            parent=request,
            refs=refs,
            artifacts=[
                {
                    "path": payload_rel,
                    "sha256": sha256_hex(payload_data),
                    "kind": "payload",
                    "mime": "application/json",
                    "bytes": len(payload_data),
                    "label": "payload",
                    "written": True,
                }
            ],
            thread=request["thread"],
            dt=latency_s,
            event_id=event_id,
        )

    def image_source(self, item: dict[str, Any]) -> dict[str, Any]:
        pixel = item.get("pixel_sha256")
        if pixel and pixel in self.snapshots:
            snap = self.snapshots[pixel]
            return {
                "kind": "observation_snapshot",
                "event_id": snap["event_id"],
                "camera": snap["data"]["camera"],
                "sim_step": snap["sim_step"],
                "wall_time": snap["wall_time"],
            }
        video_frames = item.pop("_frame_hashes", None)
        if video_frames:
            for read_event, hashes in self.video_reads:
                if hashes == video_frames:
                    return {
                        "kind": "video_frames_read",
                        "event_id": read_event["event_id"],
                        "frame_range": [read_event["data"]["start"], read_event["data"]["end"]],
                    }
        return {"kind": "unknown"}

    def input_summary(self, messages: list[dict[str, Any]], images: list[dict[str, Any]]) -> dict[str, Any]:
        categories: dict[str, list[dict[str, Any]]] = {
            "images": [
                {"path": m["path"], "order": m["order"], "kind": m["kind"], "asset": m["asset"], "source": m["source"]}
                for m in images
            ],
            "task_api_description": [],
            "code_history": [],
            "stdout_stderr": [],
            "vdm_description": [],
            "other": [],
        }
        unattributed: list[dict[str, Any]] = []
        found: set[int] = set()
        for path, text in text_parts(messages):
            covered: list[tuple[int, int]] = []
            for index, segment in enumerate(self.segments):
                start = text.find(segment.text)
                hits = 0
                while start != -1 and hits < 20:
                    end = start + len(segment.text)
                    categories[CATEGORY_OF_KIND[segment.kind]].append(
                        {
                            "kind": segment.kind,
                            "path": path,
                            "start": start,
                            "end": end,
                            "source_event_ids": segment.source_event_ids,
                        }
                    )
                    covered.append((start, end))
                    found.add(index)
                    hits += 1
                    start = text.find(segment.text, end)
            position = 0
            for start, end in sorted(covered):
                if start > position:
                    unattributed.append({"path": path, "start": position, "end": start})
                position = max(position, end)
            if position < len(text):
                unattributed.append({"path": path, "start": position, "end": len(text)})
        not_found = [
            {"kind": s.kind, "length": len(s.text), "source_event_ids": s.source_event_ids}
            for i, s in enumerate(self.segments)
            if i not in found and s.kind in CONTENT_KINDS
        ]
        n_images = sum(1 for m in images if m["mime"].startswith("image/"))
        n_videos = sum(1 for m in images if m["mime"].startswith("video/"))
        return {
            "n_images": n_images,
            "n_videos": n_videos,
            "no_image_input": n_images == 0 and n_videos == 0,
            "categories": categories,
            "unattributed": unattributed,
            "registered_not_found": not_found,
        }

    def parsed(
        self,
        response: dict[str, Any],
        *,
        parser: str,
        code_blocks: list[str],
        decision: str | None,
        model_said_finish: bool | None,
    ) -> dict[str, Any]:
        return self.emit(
            "model_output_parsed",
            "info",
            {
                "model_event_id": response["event_id"],
                "parser": parser,
                "code_blocks": code_blocks,
                "decision": decision,
                "model_said_finish": model_said_finish,
                "content_empty": not response["data"]["content"],
                "code_source": "model",
            },
            parent=response,
            refs={"model_event_id": response["event_id"], "turn_index": response["refs"].get("turn_index")},
            thread=response["thread"],
            dt=0.001,
        )

    def observation_snapshot(self, scene: Scene, purpose: str) -> tuple[dict[str, Any], str]:
        pixels = render(scene, "agentview", 256)
        data = png_bytes(pixels)
        ref = self.asset(data, "png")
        pixel_sha = sha256_hex(pixels.tobytes())
        event = self.emit(
            "observation_snapshot",
            "ok",
            {
                "camera": "agentview",
                "purpose": purpose,
                "pixel_sha256": pixel_sha,
                "width": 256,
                "height": 256,
            },
            artifacts=[
                {
                    **ref,
                    "kind": "image",
                    "mime": "image/png",
                    "width": 256,
                    "height": 256,
                    "camera": "agentview",
                    "label": "harness_render",
                    "written": True,
                }
            ],
            dt=0.03,
        )
        self.snapshots[pixel_sha] = event
        url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        return event, url

    # -- code blocks and API calls -----------------------------------------
    def code_block_start(
        self, *, block_index: int, turn_index: int, code: str, source: dict[str, Any], scene: Scene
    ) -> dict[str, Any]:
        self.stdout, self.stderr = "", ""
        event_id = self.new_id()
        probe = {"event_id": event_id}
        frame_ref = self.keyframe_after(probe)
        event = self.emit(
            "code_block_start",
            "started",
            {
                "code_block_id": event_id,
                "block_index": block_index,
                "turn_index": turn_index,
                "source_model_event_id": source["event_id"],
                "code_source": "model",
                "code": code,
                "code_sha256": sha256_hex(code.encode()),
                # Match the recorder's direct access summary shape.
                "direct_access": {
                    "names": [name for name in ("env", "obs", "APIS", "INPUTS") if f"{name}[" in code or f"{name}." in code],
                    "complete": True,
                    "note": "APIS[...] calls are recorded; calls through env, obs or INPUTS are not",
                },
                "frame_ref": frame_ref,
                "state_ref": self.state_ref(scene),
            },
            refs={"code_block_id": event_id, "turn_index": turn_index, "model_event_id": source["event_id"]},
            event_id=event_id,
        )
        self.flush_pending_keyframe(event)
        return event

    def keyframe_after(self, probe: dict[str, Any]) -> dict[str, Any] | None:
        """Frame ref for an event that is emitted next; its frame events follow it."""
        if not self.record_video or not self.buffer:
            return None  # no buffer frames: the viewer shows 「未取得」
        index = len(self.buffer) - 1
        self._pending_keyframe = index
        _step, _scene = self.buffer[index]
        camera = self.cameras[0]
        key = (self.sim_epoch, index, camera)
        if key in self.saved_frames:
            return dict(self.saved_frames[key])
        return {
            "epoch": self.sim_epoch,
            "buffer_index": index,
            "camera": camera,
            "frame_sim_step": _step,
            "path": f"frames/{camera}/e{self.sim_epoch}_{index}.jpg",
            "written": self.skip_next_frame_reason is None,
            **({"reason": self.skip_next_frame_reason} if self.skip_next_frame_reason else {}),
        }

    def flush_pending_keyframe(self, event: dict[str, Any]) -> None:
        index = self._pending_keyframe
        if index is not None:
            self.save_frame(index, reason="keyframe", for_event=event)
            self._pending_keyframe = None

    def print_out(self, text: str) -> None:
        self.stdout += text

    def api_start(
        self,
        name: str,
        args: dict[str, Any],
        *,
        block: dict[str, Any],
        scene: Scene,
        line: int | None = None,
        source_line: str | None = None,
        qualname_owner: str = "FrankaControlApiReduced",
    ) -> dict[str, Any]:
        depth = len(self.api_stack)
        parent = self.api_stack[-1] if self.api_stack else block
        summarizer = Summarizer(self, "perception_input")
        event_id = self.new_id()
        frame_ref = self.keyframe_after({"event_id": event_id}) if depth == 0 else None
        data = {
            "api_call_id": event_id,
            "name": name,
            "qualname": f"{qualname_owner}.{name}",
            "api_class": "FrankaControlApiReducedSkillLibrary",
            "depth": depth,
            "privileged": False,
            "args": {k: summarizer(v) for k, v in args.items()},
            "call_site": (
                {"code_block_id": block["event_id"], "line": line, "source_line": source_line}
                if depth == 0
                else None
            ),
            "stdout_offset": len(self.stdout),
            "stderr_offset": len(self.stderr),
            "frame_ref": frame_ref,
            "state_ref": self.state_ref(scene) if depth == 0 else None,
            "units": UNITS.get(name),
        }
        event = self.emit(
            "api_call_start",
            "started",
            data,
            parent=parent,
            refs={"code_block_id": block["event_id"], "api_call_id": event_id},
            artifacts=summarizer.artifacts,
            event_id=event_id,
        )
        if depth == 0:
            self.flush_pending_keyframe(event)
        event["_sim_start"] = self.sim_step
        event["_buffer_start"] = len(self.buffer)
        event["_block"] = block
        self.api_stack.append(event)
        return event

    def api_end(
        self,
        start: dict[str, Any],
        *,
        scene: Scene,
        result: Any = None,
        error: tuple[str, str, str] | None = None,
        duration_s: float,
        alarm_timeout: bool = False,
    ) -> dict[str, Any]:
        self.api_stack.pop()
        depth = start["data"]["depth"]
        block = start["_block"]
        summarizer = Summarizer(self, "perception_output")
        so, se = start["data"]["stdout_offset"], start["data"]["stderr_offset"]
        event_id = self.new_id()
        frame_ref = self.keyframe_after({"event_id": event_id}) if depth == 0 else None
        first_buffer = start["_buffer_start"]
        last_buffer = len(self.buffer) - 1
        data: dict[str, Any] = {
            "api_call_id": start["event_id"],
            "name": start["data"]["name"],
            "depth": depth,
            "duration_s": duration_s,
        }
        if error is None:
            data["return"] = summarizer(result)
        else:
            # the recorder stores message and traceback with text_value(): over 4000
            # characters they become a text node with a txt.gz asset
            data.update(
                {
                    "error_type": error[0],
                    "message": summarizer(error[1]),
                    "traceback": summarizer(error[2]),
                    "alarm_timeout": alarm_timeout,
                }
            )
        data.update(
            {
                "stdout_slice": {"start": so, "end": len(self.stdout), "text": summarizer(self.stdout[so:])},
                "stderr_slice": {"start": se, "end": len(self.stderr), "text": summarizer(self.stderr[se:])},
                "frame_ref": frame_ref,
                "state_ref": self.state_ref(scene) if depth == 0 else None,
                "sim_step_delta": self.sim_step - start["_sim_start"],
                "buffer_frames": [first_buffer, last_buffer] if last_buffer >= first_buffer else None,
            }
        )
        event = self.emit(
            "api_call_error" if error else "api_call_end",
            "error" if error else "ok",
            data,
            parent=start,
            refs={"code_block_id": block["event_id"], "api_call_id": start["event_id"]},
            artifacts=summarizer.artifacts,
            dt=0.01,
            event_id=event_id,
        )
        if depth == 0:
            self.flush_pending_keyframe(event)
        return event

    def service_call(
        self,
        api: dict[str, Any],
        *,
        service: str,
        function: str,
        url: str,
        request: dict[str, Any],
        result: Any,
        latency_s: float,
        http_attempts: list[dict[str, Any]],
        error: tuple[str, str] | None = None,
        open_ended: bool = False,
    ) -> dict[str, Any]:
        """Build service_call_start/end/error as the trace recorder records them.

        Images in ``request`` are stored as PNG (label perception_input) inside the
        summary node; arrays of ``result`` go into one npz (label perception_output) with
        keys a0, a1, ...; bool arrays are bit-packed as <key>__packbits + <key>__shape.
        """
        start_artifacts: list[dict[str, Any]] = []
        request_summary = {}
        for key, value in request.items():
            if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3:
                ref = self.asset(png_bytes(value), "png")
                image = {
                    "label": "perception_input",
                    "path": ref["path"],
                    "sha256": ref["sha256"],
                    "written": True,
                    "width": int(value.shape[1]),
                    "height": int(value.shape[0]),
                    "pixel_sha256": sha256_hex(value.tobytes()),
                }
                request_summary[key] = {"type": "ndarray", "shape": list(value.shape), "dtype": "uint8", "image": image}
                start_artifacts.append(
                    {**ref, "kind": "image", "mime": "image/png", "width": image["width"],
                     "height": image["height"], "label": "perception_input", "written": True}
                )
            else:
                request_summary[key] = Summarizer(self, "perception_input")(value)
        start = self.emit(
            "service_call_start",
            "started",
            {"service": service, "function": function, "request_summary": request_summary, "endpoint": None},
            parent=api,
            refs={"api_call_id": api["event_id"]},
            artifacts=start_artifacts,
            dt=0.002,
        )
        if open_ended:
            return start
        data: dict[str, Any] = {
            "service": service,
            "function": function,
            "endpoint": endpoint_info(url, known_proxy=False),
            "latency_s": latency_s,
            "http_attempts": [{**a, "endpoint": url} for a in http_attempts],
        }
        end_artifacts: list[dict[str, Any]] = []
        failed = [a for a in http_attempts if a.get("error")]
        if error is None:
            data["response_summary"] = self.service_result(result, end_artifacts)
            status = "degraded" if failed else "ok"
            if failed:
                data["degraded_reason"] = "an HTTP attempt failed but the closure returned"
        else:
            data.update({"error_type": error[0], "message": error[1]})
            status = "error"
        return self.emit(
            "service_call_error" if error else "service_call_end",
            status,
            data,
            parent=start,
            refs={},
            artifacts=end_artifacts,
            dt=latency_s,
        )

    def service_result(self, result: Any, artifacts: list[dict[str, Any]]) -> Any:
        """Summary of a service result whose arrays are stored in one npz file."""
        arrays: list[tuple[str, np.ndarray, dict[str, Any]]] = []

        def walk(value: Any) -> Any:
            if isinstance(value, np.ndarray) and value.size > 64:
                node = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
                arrays.append((f"a{len(arrays)}", value, node))
                return node
            if isinstance(value, dict):
                return {"type": "dict", "length": len(value), "items": {str(k): walk(v) for k, v in value.items()}}
            if isinstance(value, list | tuple):
                return {"type": type(value).__name__, "length": len(value), "items": [walk(v) for v in value]}
            return Summarizer(self, "perception_output")(value)

        summary = walk(result)
        if arrays:
            payload: dict[str, np.ndarray] = {}
            for key, arr, node in arrays:
                node["stats"] = array_stats(arr)
                if arr.dtype == np.bool_:
                    payload[key + "__packbits"] = np.packbits(arr.reshape(-1))
                    payload[key + "__shape"] = np.asarray(arr.shape, dtype=np.int64)
                    node.update({"npz_key": key + "__packbits", "packbits": True, "is_mask": True})
                else:
                    payload[key] = arr
                    node["npz_key"] = key
            ref = self.asset(npz_bytes(**payload), "npz")
            for _key, _arr, node in arrays:
                node.update({"npz": ref["path"], "written": True})
            artifacts.append(
                {**ref, "kind": "array", "mime": "application/octet-stream", "label": "perception_output",
                 "written": True, "names": sorted(payload)}
            )
        return summary

    def code_block_end(
        self,
        start: dict[str, Any],
        *,
        scene: Scene,
        sandbox_rc: int,
        reward: float,
        task_completed: bool,
        terminated: bool,
        sim_start: int,
        duration_s: float,
    ) -> dict[str, Any]:
        event_id = self.new_id()
        frame_ref = self.keyframe_after({"event_id": event_id})
        block_id = start["event_id"]
        turn = start["data"]["turn_index"]
        end = self.emit(
            "code_block_end",
            "ok" if sandbox_rc == 0 else "error",
            {
                "code_block_id": block_id,
                "sandbox_rc": sandbox_rc,
                "duration_s": duration_s,
                "stdout_len": len(self.stdout),
                "stderr_len": len(self.stderr),
                "reward": reward,
                "terminated": terminated,
                "truncated": False,
                "task_completed": task_completed,
                "result_summary": None,
                "frame_ref": frame_ref,
                "state_ref": self.state_ref(scene),
                "sim_step_range": [sim_start, self.sim_step],
                "tracer_frame_visible": False,
            },
            parent=start,
            refs={"code_block_id": block_id, "turn_index": turn},
            event_id=event_id,
        )
        self.flush_pending_keyframe(end)
        streams = {}
        stream_ids = {"stdout": self.new_id(), "stderr": self.new_id()}
        end["data"]["stdout_event_id"] = stream_ids["stdout"]
        end["data"]["stderr_event_id"] = stream_ids["stderr"]
        self.raw_lines[-1 - self._lines_since(end)] = event_line(end)
        for name, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            artifacts: list[dict[str, Any]] = []
            value: Any = text if len(text) <= TEXT_INLINE_LIMIT else self.long_text(text, artifacts)
            streams[name] = self.emit(
                name,
                "info",
                {"code_block_id": block_id, "text": value, "length": len(text)},
                parent=start,
                refs={"code_block_id": block_id, "turn_index": turn},
                artifacts=artifacts,
                dt=0.001,
                event_id=stream_ids[name],
            )
            if text:
                self.register(name, text, [streams[name]["event_id"]])
        self.emit(
            "evaluator_result",
            "info",
            {
                "scope": "code_block",
                "code_block_id": block_id,
                "reward": reward,
                "task_completed": task_completed,
                "terminated": terminated,
                "truncated": False,
                "source": "env.step info",
            },
            parent=start,
            refs={"code_block_id": block_id},
            dt=0.001,
        )
        return end

    # -- files -------------------------------------------------------------
    def write_events(self, *, torn_tail: bytes | None = None) -> None:
        data = b"".join(self.raw_lines) + (torn_tail or b"")
        (self.dir / "events.jsonl").write_bytes(data)

    def write_episode_json(self, summary: dict[str, Any]) -> None:
        self.write_file("episode.json", json.dumps(summary, indent=1).encode())

    def touch_heartbeat(self, mtime: float) -> None:
        path = self.dir / "heartbeat"
        path.write_bytes(b"")
        os.utime(path, (mtime, mtime))


def event_line(event: dict[str, Any]) -> bytes:
    """One ASCII JSON line (EpisodeBuilder.emit moves the data of an over-long line to an asset)."""
    clean = {k: v for k, v in event.items() if not k.startswith("_")}
    return json.dumps(clean, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"


def text_parts(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, str):
            parts.append((f"messages[{i}].content", content))
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if part.get("type") == "text":
                    parts.append((f"messages[{i}].content[{j}].text", part["text"]))
    return parts


def normalize_payload(payload: dict[str, Any], episode: EpisodeBuilder) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace data: URLs by $asset placeholders (section 6.2)."""
    media: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {k: walk(v, f"{path}.{k}" if path else k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            header, b64 = value.split(",", 1)
            mime = header[5:].split(";", 1)[0]
            raw = base64.b64decode(b64)
            ext = {"image/png": "png", "image/jpeg": "jpg", "video/mp4": "mp4"}[mime]
            ref = episode.asset(raw, ext)
            item: dict[str, Any] = {
                "path": path,
                "kind": mime.split("/", 1)[0],
                "written": True,
                "asset": ref["path"],
                "sha256": ref["sha256"],
                "mime": mime,
                "bytes": ref["bytes"],
            }
            if mime.startswith("image/"):
                pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                item.update(
                    {
                        "width": int(pixels.shape[1]),
                        "height": int(pixels.shape[0]),
                        "pixel_sha256": sha256_hex(pixels.tobytes()),
                    }
                )
            else:
                frames = episode.video_frames.get(ref["sha256"], [])
                item.update(
                    {
                        "width": int(frames[0].shape[1]) if frames else None,
                        "height": int(frames[0].shape[0]) if frames else None,
                        "pixel_sha256": None,
                        "_frame_hashes": [sha256_hex(f.tobytes()) for f in frames],
                    }
                )
            media.append(item)
            return {"$asset": ref["path"], "sha256": ref["sha256"], "mime": mime, "bytes": ref["bytes"]}
        return value

    return walk(payload, ""), media


def restore_payload(normalized: Any, episode: EpisodeBuilder) -> Any:
    if isinstance(normalized, dict):
        if "$asset" in normalized:
            raw = (episode.dir / normalized["$asset"]).read_bytes()
            return f"data:{normalized['mime']};base64,{base64.b64encode(raw).decode('ascii')}"
        return {k: restore_payload(v, episode) for k, v in normalized.items()}
    if isinstance(normalized, list):
        return [restore_payload(v, episode) for v in normalized]
    return normalized


def codegen_messages(task_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": CODEGEN_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": f"{task_prompt}\nAPIs:\n\n{API_DOCS}"}]},
    ]


def collapse(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """capx.llm.client.collapse_text_image_inputs."""
    out: list[dict[str, Any]] = []
    text = ""
    for part in parts:
        if part["type"] == "text":
            text += part["text"] + "\n"
        else:
            if text:
                out.append({"type": "text", "text": text})
                text = ""
            out.append(part)
    if text:
        out.append({"type": "text", "text": text})
    return out


def decision_messages(
    task_prompt: str,
    executed_code: str,
    stdout: str,
    stderr: str,
    *,
    image_url: str | None,
    feedback: str | None,
    video_feedback: bool = False,
    user_feedback: str | None = None,
) -> list[dict[str, Any]]:
    messages = codegen_messages(task_prompt)
    parts = list(messages[-1]["content"])
    # User text injected in the web UI is part of the multi-turn prompt text, as the web runner sends it.
    parts.append({"type": "text", "text": multi_turn_prompt(executed_code, stdout, stderr, user_feedback)})
    if image_url is not None:
        parts.append({"type": "text", "text": DECISION_IMAGE_TEXT})
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if feedback is not None:
        header = DECISION_VIDEO_HEADER if video_feedback else DECISION_DIFF_HEADER
        parts.append({"type": "text", "text": f"{header}\n{feedback}"})
    parts.append({"type": "text", "text": DECISION_INSTRUCTION})
    messages[-1]["content"] = collapse(parts)
    return messages


def task_text_data(text: str, request: dict[str, Any]) -> dict[str, Any]:
    """Build task_text data as the trace recorder records it."""
    return {
        "text": text,
        "reason": None,
        "logical_call_id": request["data"]["logical_call_id"],
        "source": 'text before "\\nAPIs:\\n" in the last user message of the first code-generation request',
    }


def register_templates(ep: EpisodeBuilder, task_prompt: str) -> None:
    ep.register("task_prompt", task_prompt)
    ep.register("api_docs", f"\n{API_DOCS}")
    for text in TEMPLATE_SEGMENTS:
        ep.register("harness_template", text)


def tracer_info(state: bool, video: bool) -> dict[str, Any]:
    return {
        "version": SCHEMA,
        "settings": {
            "frame_fps": 4 if video else 0,
            "jpeg_quality": 85,
            "state": state,
            "min_free_gb": 2.0,
            "max_queue_mb": 512,
            "array_call_mb": 8,
            "array_episode_mb": 2048,
        },
        "hooks_installed": [f"H{i}" for i in range(1, 17)],
        "hooks_failed": [],
    }


TRACEBACK_IK = """Traceback (most recent call last):
  File "/path/to/cap-x/capx/envs/tasks/base.py", line 172, in _exec_user_code
    exec(code, self._exec_globals, self._exec_globals)
  File "<string>", line 24, in <module>
  File "/path/to/cap-x/capx/integrations/franka/control_reduced.py", line 482, in solve_ik
    raise RuntimeError(
RuntimeError: IK failed to converge: position error 0.0412 m exceeds tolerance 0.005 m (target out of reach?)
"""

# ---------------------------------------------------------------------------
# Episode A: multi-turn with VDM, ended with gaps
# ---------------------------------------------------------------------------

BLOCK0_CODE = """import numpy as np

obs = get_observation()
view = obs["robot0_robotview"]
rgb = view["images"]["rgb"]
depth = view["images"]["depth"][:, :, 0]

red = segment_sam3_text_prompt(rgb, "red cube")
best = max(red, key=lambda r: r["score"])
print("red cube score:", round(best["score"], 3))

green_px = point_prompt_molmo(rgb, "green cube")
print("green cube pixel:", green_px["green cube"])

red_points = mask_to_world_points(best["mask"], depth, view["intrinsics"], view["pose_mat"])
print("red cube centre:", np.round(red_points.mean(axis=0), 4))

grasps, scores = plan_grasp(depth, view["intrinsics"], best["mask"].astype(np.uint8))
grasp = select_top_down_grasp(grasps, scores)
q = solve_ik(grasp[:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q)
close_gripper()

place = np.array([0.60, 0.10, 1.35])
q = solve_ik(place, np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q)
open_gripper()"""

BLOCK1_CODE = """import numpy as np

obs = get_observation()
view = obs["robot0_robotview"]
rgb = view["images"]["rgb"]
depth = view["images"]["depth"][:, :, 0]

green = segment_sam3_text_prompt(rgb, "green cube")
if not green:
    print("SAM3 returned no mask for the green cube; using the Molmo pixel from the last turn")
    green_world = pixel_to_world_point((141, 118), depth, view["intrinsics"], view["pose_mat"])
else:
    green_world = mask_to_world_points(green[0]["mask"], depth, view["intrinsics"], view["pose_mat"]).mean(axis=0)
print("green cube centre:", np.round(green_world, 4))

place = green_world + np.array([0.0, 0.0, 0.065])
q = solve_ik(place, np.array([0.0, 1.0, 0.0, 0.0]))
for k, row in enumerate(np.linspace(0.0, 1.0, 120)):
    print(f"waypoint {k:03d}: blend={row:.4f} target={np.round(place, 4)}")
move_to_joints(q)
open_gripper()
print("released the red cube above the green cube")
report = summarize_workspace()
print("workspace objects:", report["n_objects"])
try:
    verify_demo_alignment("red cube", "green cube")
except ValueError as exc:
    print("alignment check failed:", str(exc)[:60])"""


def block1_line(text: str) -> int:
    """1-based line of ``text`` in BLOCK1_CODE (the call sites of block 1)."""
    return BLOCK1_CODE.splitlines().index(text) + 1


def workspace_report() -> dict[str, Any]:
    """summarize_workspace()'s return value: 36 objects with two texts over 4000 characters
    each, so the summarised api_call_end is over 256 KB and its data goes to data_asset."""
    objects = []
    for i in range(36):
        name = f"body_{i:02d}"
        pos = np.round(np.array([0.40 + 0.01 * i, -0.2 + 0.011 * i, 0.84]), 4)
        faces = " ".join(
            f"{name} face {k}: normal ({(k % 3) - 1:+d}, {((k // 3) % 3) - 1:+d}, {k % 2:d}) area {0.0004 + 1e-5 * k:.6f} m2;"
            for k in range(90)
        )
        vertices = " ".join(
            f"{name} vertex {k}: ({pos[0] + 0.001 * k:.4f}, {pos[1] - 0.001 * k:.4f}, {pos[2] + 0.0005 * k:.4f});"
            for k in range(110)
        )
        objects.append({"name": name, "position": pos, "faces": faces, "mesh_vertices": vertices})
    return {"frame": "world", "n_objects": len(objects), "objects": objects}


ALIGNMENT_MESSAGE = (
    "stack alignment check failed: 96 of 96 contact points between 'red cube' and 'green cube' "
    "exceed the 0.002 m tolerance: "
    + "; ".join(
        f"contact {k:03d} at ({0.58 + 0.0004 * k:.4f}, {0.095 + 0.0003 * k:.4f}, {0.884 + 0.0001 * k:.4f}) "
        f"residual {0.0031 + 0.00002 * k:.5f} m"
        for k in range(96)
    )
)
# As the recorder stores an API's traceback: only the frames below the API wrapper (the
# generated code's frame is not in exc.__traceback__ yet), then "<type>: <message>".
TRACEBACK_ALIGNMENT = (
    "Traceback (most recent call last):\n"
    '  File "/synthetic_demo/alignment.py", line 311, in verify_demo_alignment\n'
    "    raise ValueError(\n"
    "ValueError: " + ALIGNMENT_MESSAGE
)


def build_episode_a(run_dir: Path, now: float) -> EpisodeBuilder:
    ep = EpisodeBuilder(
        run_dir,
        "cube_stack__M4__s1__i1__a1",
        pid=412345,
        t0=now - 7200.0,
        mono0=91234.5,
        cameras=("agentview", "robot0_eye_in_hand"),
    )
    scene = Scene(
        gripper=np.array([0.45, 0.0, 1.05]),
        red=np.array([0.58, -0.08, 0.84]),
        green=np.array([0.60, 0.10, 0.84]),
    )
    ep.buffer.append((None, scene.copy()))  # frame 0 was in the buffer before the frame hook (H10) was installed
    config_path = "env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml"
    start = ep.emit(
        "episode_start",
        "info",
        {
            "source": "experiments.run",
            "task": "cube_stack",
            "condition": "M4",
            "seed": 1,
            "attempt_index": 1,
            "runner_invocation": 1,
            "condition_label": "M4",
            "runner_condition": {
                "label": "M4",
                "config_path": config_path,
                "privileged": False,
                "apis": ["FrankaControlApiReducedSkillLibrary"],
                "multi_turn": True,
                "uses_vdm": True,
                "use_visual_feedback": True,
                "use_oracle_code": False,
                "note": "synthetic demo condition",
            },
            "config_flags": {
                "multi_turn": True,
                "use_visual_feedback": True,
                "use_img_differencing": True,
                "use_video_differencing": False,
                "use_wrist_camera": False,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
                "use_legacy_multi_turn_decision_prompt": True,
                "record_video": True,
                "use_oracle_code": False,
                "oracle_file": None,
                "await_user_input_each_turn": None,
            },
            "models": {
                "code_generator": {"model": GEMINI, "endpoint": endpoint_info(PROXY_URL, known_proxy=True)},
                "vdm": {"model": GEMINI, "endpoint": endpoint_info(PROXY_URL, known_proxy=True)},
                "perception_vlm": {"model": "allenai/Molmo2-8B", "endpoint": endpoint_info(MOLMO_URL, known_proxy=False)},
            },
            "env": {
                "class": "FrankaPickPlaceCodeEnv",
                "low_level_class": "FrankaRobosuiteCubesLowLevel",
                "apis": ["FrankaControlApiReducedSkillLibrary"],
                "viser_port": None,
            },
            "tracer": tracer_info(state=True, video=True),
            "host": "demo-host",
            "python": "3.10.19",
            "git_commit": "15ffdf6a0c1e5b0e9d7c3f1b2a4e6d8c0b9a7f5e",
            "process_start_time": 18423311,
        },
        dt=0.0,
    )
    ep.save_frame(0, reason="periodic")  # back-filled: sim_step is null
    register_templates(ep, STACK_TASK_PROMPT)
    snap0, url0 = ep.observation_snapshot(scene, "_capture_initial_visual_feedback")
    task_description = f"{STACK_TASK_PROMPT}\nAPIs:\n\n{API_DOCS}"

    # initial scene description (VDM)
    lc = ep.new_id("lc_", 16)
    vdm0 = ep.model_request(
        role="vdm",
        call_kind="initial_scene_description",
        turn_index=0,
        url=PROXY_URL,
        known_proxy=True,
        logical_call_id=lc,
        http_attempt=1,
        payload={
            "model": GEMINI,
            "messages": [
                {"role": "system", "content": VDM_INITIAL_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": VDM_INITIAL_TEXT},
                        {"type": "text", "text": "Main camera view:"},
                        {"type": "image_url", "image_url": {"url": url0}},
                    ],
                },
            ],
            "temperature": 1.0,
            "max_tokens": 20480,
        },
    )
    vdm0_text = (
        "The table holds two cubes of similar size. A red cube is on the left side of the "
        "table and a green cube is on the right, about 18 cm apart. The gripper is open and "
        "sits above the middle of the table, clear of both cubes. The red cube is not on top "
        "of the green cube, so the task is not complete."
    )
    vdm0_resp = ep.model_response(
        vdm0, vdm0_text, latency_s=4.81, usage={"prompt_tokens": 1874, "completion_tokens": 71, "total_tokens": 1945}
    )
    ep.register("vdm_description", vdm0_text, [vdm0_resp["event_id"]])

    # initial code: HTTP 503 on the first attempt, then success
    lc = ep.new_id("lc_", 16)
    payload = {"model": GEMINI, "messages": codegen_messages(STACK_TASK_PROMPT), "temperature": 1.0, "max_tokens": 20480}
    req1 = ep.model_request(
        role="code_generator", call_kind="initial_code", turn_index=0, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1, payload=payload,
    )
    ep.emit(
        "task_text",
        "info",
        task_text_data(STACK_TASK_PROMPT, req1),
        refs={"logical_call_id": req1["data"]["logical_call_id"]},
        dt=0.0,
    )
    ep.emit(
        "model_error",
        "error",
        {
            "scope": "attempt",
            "role": "code_generator",
            "call_kind": "initial_code",
            "logical_call_id": lc,
            "http_attempt": 1,
            "http_status": 503,
            "error_type": "HTTPStatus",
            "message": "503 Service Unavailable from the local proxy; the client retries",
            "body_excerpt": '{"error": {"code": 503, "message": "The model is overloaded. Please try again later.", "status": "UNAVAILABLE"}}',
        },
        parent=req1,
        refs={"logical_call_id": lc, "http_attempt": 1, "turn_index": 0, "model_event_id": req1["event_id"]},
        dt=2.4,
    )
    ep.tick(5.0)  # CAPX_LLM_RETRY_SLEEP_S
    req2 = ep.model_request(
        role="code_generator", call_kind="initial_code", turn_index=0, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=2, payload=payload,
    )
    resp2 = ep.model_response(
        req2,
        "```python\n" + BLOCK0_CODE + "\n```",
        latency_s=11.37,
        usage={"prompt_tokens": 1523, "completion_tokens": 402, "total_tokens": 1925},
        reasoning={"text": "Segment the red cube, grasp it top-down, then carry it above the green cube.", "field": "reasoning_content"},
    )
    ep.parsed(resp2, parser="_extract_code", code_blocks=[BLOCK0_CODE], decision=None, model_said_finish=None)

    # ---- code block 0 --------------------------------------------------------
    block0 = ep.code_block_start(block_index=0, turn_index=0, code=BLOCK0_CODE, source=resp2, scene=scene)
    sim0 = ep.sim_step
    rgb = render(scene, "robot0_robotview", 128)
    depth = depth_map(scene, 128)
    intr = np.array([[110.9, 0.0, 64.0], [0.0, 110.9, 64.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = [1.2, 0.0, 1.4]
    obs_value = {"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[:, :, None]}, "intrinsics": intr, "pose_mat": pose}}

    call = ep.api_start("get_observation", {}, block=block0, scene=scene, line=3, source_line="obs = get_observation()")
    ep.api_end(call, scene=scene, result=obs_value, duration_s=0.041)

    call = ep.api_start(
        "segment_sam3_text_prompt", {"rgb": rgb, "text_prompt": "red cube"}, block=block0, scene=scene,
        line=8, source_line='red = segment_sam3_text_prompt(rgb, "red cube")',
    )
    red_mask = cube_mask(scene, scene.red, 128)
    other_mask = cube_mask(scene, scene.green, 128)
    masks = np.stack([red_mask, other_mask])
    ep.service_call(
        call, service="sam3", function="sam3_seg_fn", url="http://127.0.0.1:8114/segment",
        request={"image": rgb, "text_prompt": "red cube"},
        result=[
            {"mask": masks[0], "box": [40.0, 60.0, 58.0, 78.0], "score": 0.912},
            {"mask": masks[1], "box": [80.0, 58.0, 98.0, 76.0], "score": 0.231},
        ],
        latency_s=0.84, http_attempts=[{"status": 200, "latency_s": 0.84, "error": None}],
    )
    ep.print_out("red cube score: 0.912\n")
    ep.api_end(
        call, scene=scene, duration_s=0.87,
        result=[{"mask": red_mask, "box": [40.0, 60.0, 58.0, 78.0], "score": 0.912}, {"mask": other_mask, "box": [80.0, 58.0, 98.0, 76.0], "score": 0.231}],
    )

    call = ep.api_start(
        "point_prompt_molmo", {"image": rgb, "text_prompt": "green cube"}, block=block0, scene=scene,
        line=12, source_line='green_px = point_prompt_molmo(rgb, "green cube")',
    )
    svc = ep.emit(
        "service_call_start",
        "started",
        {"service": "molmo", "function": "molmo_point_fn", "request_summary": {"objects": ["green cube"]}, "endpoint": None},
        parent=call,
        refs={"api_call_id": call["event_id"]},
        dt=0.002,
    )
    molmo_url = "data:image/png;base64," + base64.b64encode(png_bytes(rgb)).decode("ascii")
    lc = ep.new_id("lc_", 16)
    molmo_req = ep.model_request(
        role="perception_vlm", call_kind="point", turn_index=None, url=MOLMO_URL, known_proxy=False,
        logical_call_id=lc, http_attempt=1, send_as_bytes=True, parent=call,
        refs_extra={"code_block_id": block0["event_id"], "api_call_id": call["event_id"]},
        payload={
            "model": "allenai/Molmo2-8B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Point at green cube"},
                        {"type": "image_url", "image_url": {"url": molmo_url}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stop": ["<|endoftext|>"],
        },
        thread=ep.env_thread,
    )
    molmo_text = '<point x="55.1" y="46.2" alt="green cube">green cube</point>'
    ep.model_response(molmo_req, molmo_text, latency_s=0.63, usage={"prompt_tokens": 312, "completion_tokens": 21, "total_tokens": 333})
    ep.print_out(f"Generated text for 'green cube': {molmo_text}\n")
    ep.emit(
        "service_call_end",
        "ok",
        {
            "service": "molmo",
            "function": "molmo_point_fn",
            "endpoint": endpoint_info(MOLMO_URL, known_proxy=False),
            "latency_s": 0.66,
            "http_attempts": [{"status": 200, "latency_s": 0.63, "error": None, "endpoint": MOLMO_URL}],
            "model_logical_call_ids": [lc],
            "response_summary": {"type": "dict", "length": 1, "items": {"green cube": {"type": "tuple", "length": 2, "items": [141, 118]}}},
        },
        parent=svc,
        refs={},
        dt=0.01,
    )
    ep.api_end(call, scene=scene, result={"green cube": (141, 118)}, duration_s=0.69)
    ep.print_out("green cube pixel: (141, 118)\n")

    call = ep.api_start(
        "mask_to_world_points",
        {"mask": red_mask, "depth": depth, "intrinsics": intr, "cam_pose": pose},
        block=block0, scene=scene, line=15,
        source_line='red_points = mask_to_world_points(best["mask"], depth, view["intrinsics"], view["pose_mat"])',
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    nested = ep.api_start(
        "depth_to_point_cloud", {"depth_img": depth, "intrinsics": intr}, block=block0, scene=scene,
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    cloud = np.stack(np.meshgrid(np.linspace(-0.3, 0.3, 128), np.linspace(-0.3, 0.3, 128)), -1).reshape(-1, 2)
    cloud = np.concatenate([cloud, depth.reshape(-1, 1)], axis=1).astype(np.float32)
    ep.api_end(nested, scene=scene, result=cloud, duration_s=0.012)
    red_points = (scene.red + np.random.default_rng(3).normal(0, 0.004, size=(96, 3))).astype(np.float32)
    ep.api_end(call, scene=scene, result=red_points, duration_s=0.019)
    ep.print_out(f"red cube centre: {np.round(red_points.mean(axis=0), 4)}\n")

    call = ep.api_start(
        "plan_grasp", {"depth": depth, "intrinsics": intr, "segmentation": red_mask.astype(np.uint8)},
        block=block0, scene=scene, line=18,
        source_line='grasps, scores = plan_grasp(depth, view["intrinsics"], best["mask"].astype(np.uint8))',
    )
    grasps = np.tile(np.eye(4, dtype=np.float32), (20, 1, 1))
    grasps[:, :3, 3] = scene.red + np.linspace(-0.01, 0.01, 20)[:, None]
    scores = np.linspace(0.93, 0.41, 20).astype(np.float32)
    ep.service_call(
        call, service="graspnet", function="grasp_net_plan_fn", url="http://127.0.0.1:8115/plan",
        request={"depth": depth, "segmentation": red_mask.astype(np.uint8)},
        result=(grasps, scores, grasps[:, :3, 3].copy()),
        latency_s=1.92, http_attempts=[{"status": 200, "latency_s": 1.92, "error": None}],
    )
    ep.api_end(call, scene=scene, result=(grasps, scores), duration_s=1.95)

    call = ep.api_start(
        "select_top_down_grasp", {"grasp_poses": grasps, "scores": scores}, block=block0, scene=scene,
        line=19, source_line="grasp = select_top_down_grasp(grasps, scores)",
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    ep.api_end(call, scene=scene, result=grasps[0], duration_s=0.001)

    call = ep.api_start(
        "solve_ik", {"position": scene.red.copy(), "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])},
        block=block0, scene=scene, line=20, source_line="q = solve_ik(grasp[:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))",
    )
    q_grasp = np.array([0.012, 0.431, -0.018, -2.104, 0.009, 2.531, 0.781])
    ep.service_call(
        call, service="pyroki", function="ik_solve_fn", url="http://127.0.0.1:8116/ik",
        request={"target_link": "panda_hand", "position": scene.red.copy()},
        result=q_grasp,
        latency_s=0.31, http_attempts=[{"status": 200, "latency_s": 0.31, "error": None}],
    )
    ep.api_end(call, scene=scene, result=q_grasp, duration_s=0.33)

    call = ep.api_start("move_to_joints", {"joints": q_grasp}, block=block0, scene=scene, line=21, source_line="move_to_joints(q)")
    ep.advance(scene, scene.red + np.array([0.0, 0.0, 0.045]), steps=90, seconds=3.1)
    ep.api_end(call, scene=scene, result=None, duration_s=3.14)

    call = ep.api_start("close_gripper", {}, block=block0, scene=scene, line=22, source_line="close_gripper()")
    scene.gripper_open = False
    scene.held = True
    # 22 steps: the motion ends between buffer frames (every SUBSAMPLE steps), so the key
    # frames after it are 2 steps older than their events, as in a real run.
    ep.advance(scene, scene.gripper.copy(), steps=22, seconds=0.7)
    ep.api_end(call, scene=scene, result=None, duration_s=0.71)

    call = ep.api_start(
        "solve_ik", {"position": np.array([0.60, 0.10, 1.35]), "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])},
        block=block0, scene=scene, line=25, source_line="q = solve_ik(place, np.array([0.0, 1.0, 0.0, 0.0]))",
    )
    ep.service_call(
        call, service="pyroki", function="ik_solve_fn", url="http://127.0.0.1:8116/ik",
        request={"target_link": "panda_hand", "position": np.array([0.6, 0.1, 1.35])},
        result=np.array([0.2, -0.61, 0.1, -0.35, 0.05, 1.2, 0.8]),
        latency_s=0.35, http_attempts=[{"status": 200, "latency_s": 0.35, "error": None}],
    )
    message = "IK failed to converge: position error 0.0412 m exceeds tolerance 0.005 m (target out of reach?)"
    ep.api_end(call, scene=scene, error=("RuntimeError", message, TRACEBACK_IK), duration_s=0.36)
    ep.stderr += TRACEBACK_IK  # printed by base.py after exec() raised, not by the API call
    ep.tick(0.05)
    ep.code_block_end(block0, scene=scene, sandbox_rc=1, reward=0.0, task_completed=False, terminated=False, sim_start=sim0, duration_s=9.8)
    ep.register("code_history", BLOCK0_CODE, [block0["event_id"]])
    stdout0, stderr0 = ep.stdout, ep.stderr

    # ---- turn 1: VDM image differencing, then the decision request -----------
    snap1, url1 = ep.observation_snapshot(scene, "_get_visual_feedback")
    lc = ep.new_id("lc_", 16)
    vdm1 = ep.model_request(
        role="vdm", call_kind="image_diff", turn_index=1, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={
            "model": GEMINI,
            "messages": [
                {"role": "system", "content": VDM_DIFF_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": VDM_DIFF_TEXT},
                        {"type": "text", "text": "Previous state (main camera):"},
                        {"type": "image_url", "image_url": {"url": url0}},
                        {"type": "text", "text": "Current state (main camera):"},
                        {"type": "image_url", "image_url": {"url": url1}},
                    ],
                },
            ],
            "temperature": 1.0,
            "max_tokens": 20480,
        },
    )
    vdm1_text = (
        "The gripper has closed on the red cube and holds it about 5 cm above its original "
        "position. The green cube has not moved. The red cube is not on the green cube, so the "
        "task is not complete."
    )
    vdm1_resp = ep.model_response(vdm1, vdm1_text, latency_s=5.12, usage={"prompt_tokens": 3120, "completion_tokens": 52, "total_tokens": 3172})
    ep.register("vdm_description", vdm1_text, [vdm1_resp["event_id"]])

    messages = decision_messages(STACK_TASK_PROMPT, BLOCK0_CODE, stdout0, stderr0, image_url=url1, feedback=vdm1_text)
    feedback = build_feedback(ep, 1, "_build_multi_turn_decision_prompt_legacy", snap1, vdm1_resp, [
        ("code_history", BLOCK0_CODE, [block0["event_id"]]),
        ("stdout", stdout0, [ep.events_of("stdout")[-1]["event_id"]]),
        ("stderr", stderr0, [ep.events_of("stderr")[-1]["event_id"]]),
        ("vdm_description", vdm1_text, [vdm1_resp["event_id"]]),
    ], prompt_text=multi_turn_prompt(BLOCK0_CODE, stdout0, stderr0), user_feedback=None)
    lc = ep.new_id("lc_", 16)
    req3 = ep.model_request(
        role="code_generator", call_kind="multi_turn_decision", turn_index=1, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={"model": GEMINI, "messages": messages, "temperature": 1.0, "max_tokens": 20480},
        refs_extra={"for_event_id": feedback["event_id"]},
    )
    resp3 = ep.model_response(
        req3,
        "The IK call failed because the placement target was 51 cm above the table. Lower the "
        "target to just above the green cube.\nREGENERATE\n```python\n" + BLOCK1_CODE + "\n```",
        latency_s=9.64,
        usage={"prompt_tokens": 4410, "completion_tokens": 388, "total_tokens": 4798},
    )
    ep.parsed(resp3, parser="_parse_multi_turn_decision", code_blocks=[BLOCK1_CODE], decision="regenerate", model_said_finish=False)

    # ---- code block 1 --------------------------------------------------------
    block1 = ep.code_block_start(block_index=1, turn_index=1, code=BLOCK1_CODE, source=resp3, scene=scene)
    sim1 = ep.sim_step
    rgb = render(scene, "robot0_robotview", 128)
    depth = depth_map(scene, 128)
    obs_value = {"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[:, :, None]}, "intrinsics": intr, "pose_mat": pose}}
    call = ep.api_start("get_observation", {}, block=block1, scene=scene, line=3, source_line="obs = get_observation()")
    ep.api_end(call, scene=scene, result=obs_value, duration_s=0.039)
    call = ep.api_start(
        "segment_sam3_text_prompt", {"rgb": rgb, "text_prompt": "green cube"}, block=block1, scene=scene,
        line=8, source_line='green = segment_sam3_text_prompt(rgb, "green cube")',
    )
    ep.service_call(
        call, service="sam3", function="sam3_seg_fn", url="http://127.0.0.1:8114/segment",
        request={"image": rgb, "text_prompt": "green cube"},
        result=[],  # the closure caught the transport error and returned an empty list
        latency_s=30.02,
        http_attempts=[{"status": None, "latency_s": 30.0, "error": "ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=8114): Read timed out. (read timeout=30)"}],
    )
    ep.api_end(call, scene=scene, result=[], duration_s=30.05)
    ep.print_out("SAM3 returned no mask for the green cube; using the Molmo pixel from the last turn\n")
    call = ep.api_start(
        "pixel_to_world_point", {"pixel": (141, 118), "depth": depth, "intrinsics": intr, "cam_pose": pose},
        block=block1, scene=scene,
        line=block1_line('    green_world = pixel_to_world_point((141, 118), depth, view["intrinsics"], view["pose_mat"])'),
        # the recorder keeps the line as it is in the code, indentation included
        source_line='    green_world = pixel_to_world_point((141, 118), depth, view["intrinsics"], view["pose_mat"])',
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    green_world = np.array([0.6012, 0.0987, 0.8395])
    ep.api_end(call, scene=scene, result=green_world, duration_s=0.004)
    ep.print_out(f"green cube centre: {np.round(green_world, 4)}\n")
    place = green_world + np.array([0.0, 0.0, 0.065])
    call = ep.api_start(
        "solve_ik", {"position": place, "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])},
        block=block1, scene=scene, line=block1_line("q = solve_ik(place, np.array([0.0, 1.0, 0.0, 0.0]))"),
        source_line="q = solve_ik(place, np.array([0.0, 1.0, 0.0, 0.0]))",
    )
    q_place = np.array([0.211, 0.402, 0.095, -2.051, -0.031, 2.462, 0.998])
    ep.service_call(
        call, service="pyroki", function="ik_solve_fn", url="http://127.0.0.1:8116/ik",
        request={"target_link": "panda_hand", "position": place},
        result=q_place,
        latency_s=0.29, http_attempts=[{"status": 200, "latency_s": 0.29, "error": None}],
    )
    ep.api_end(call, scene=scene, result=q_place, duration_s=0.3)
    for k, blend in enumerate(np.linspace(0.0, 1.0, 120)):
        ep.print_out(f"waypoint {k:03d}: blend={blend:.4f} target={np.round(place, 4)}\n")
    call = ep.api_start("move_to_joints", {"joints": q_place}, block=block1, scene=scene, line=block1_line("move_to_joints(q)"), source_line="move_to_joints(q)")
    ep.advance(scene, np.array([0.60, 0.10, 0.93]), steps=40, seconds=1.4)
    ep.skip_next_frame_reason = "disk_low"  # the next periodic frame is skipped (gap)
    ep.advance(scene, place + np.array([0.0, 0.0, 0.045]), steps=60, seconds=2.1)
    ep.api_end(call, scene=scene, result=None, duration_s=3.52)
    call = ep.api_start("open_gripper", {}, block=block1, scene=scene, line=block1_line("open_gripper()"), source_line="open_gripper()")
    scene.gripper_open = True
    scene.held = False
    scene.red = scene.green + np.array([0.0, 0.0, 0.04])
    ep.advance(scene, scene.gripper + np.array([0.0, 0.0, 0.08]), steps=27, seconds=0.9)
    ep.api_end(call, scene=scene, result=None, duration_s=0.92)
    ep.print_out("released the red cube above the green cube\n")
    # a return value so large that the recorder moves the event's data to data_asset
    call = ep.api_start(
        "summarize_workspace", {}, block=block1, scene=scene,
        line=block1_line("report = summarize_workspace()"), source_line="report = summarize_workspace()",
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    report = workspace_report()
    ep.api_end(call, scene=scene, result=report, duration_s=0.21)
    ep.print_out(f"workspace objects: {report['n_objects']}\n")
    # an exception message over 4000 characters (a text asset), caught by the generated code
    call = ep.api_start(
        "verify_demo_alignment", {"top": "red cube", "bottom": "green cube"}, block=block1, scene=scene,
        line=block1_line('    verify_demo_alignment("red cube", "green cube")'),
        source_line='    verify_demo_alignment("red cube", "green cube")',
        qualname_owner="FrankaControlApiReducedSkillLibrary",
    )
    ep.api_end(call, scene=scene, error=("ValueError", ALIGNMENT_MESSAGE, TRACEBACK_ALIGNMENT), duration_s=0.05)
    ep.print_out(f"alignment check failed: {ALIGNMENT_MESSAGE[:60]}\n")
    ep.code_block_end(block1, scene=scene, sandbox_rc=0, reward=1.0, task_completed=True, terminated=True, sim_start=sim1, duration_s=37.1)
    ep.register("code_history", BLOCK1_CODE, [block1["event_id"]])
    stdout1, stderr1 = ep.stdout, ep.stderr

    # ---- turn 2: VDM again, then FINISH ---------------------------------------
    snap2, url2 = ep.observation_snapshot(scene, "_get_visual_feedback")
    lc = ep.new_id("lc_", 16)
    vdm2 = ep.model_request(
        role="vdm", call_kind="image_diff", turn_index=2, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={
            "model": GEMINI,
            "messages": [
                {"role": "system", "content": VDM_DIFF_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": VDM_DIFF_TEXT},
                        {"type": "text", "text": "Previous state (main camera):"},
                        {"type": "image_url", "image_url": {"url": url1}},
                        {"type": "text", "text": "Current state (main camera):"},
                        {"type": "image_url", "image_url": {"url": url2}},
                    ],
                },
            ],
            "temperature": 1.0,
            "max_tokens": 20480,
        },
    )
    vdm2_text = (
        "The red cube now rests on top of the green cube and the gripper is open above them. "
        "The task appears to be complete."
    )
    vdm2_resp = ep.model_response(vdm2, vdm2_text, latency_s=4.4, usage={"prompt_tokens": 3120, "completion_tokens": 29, "total_tokens": 3149})
    ep.register("vdm_description", vdm2_text, [vdm2_resp["event_id"]])
    executed = "\n".join([BLOCK0_CODE, BLOCK1_CODE])
    messages = decision_messages(STACK_TASK_PROMPT, executed, stdout1, stderr1, image_url=url2, feedback=vdm2_text)
    feedback = build_feedback(ep, 2, "_build_multi_turn_decision_prompt_legacy", snap2, vdm2_resp, [
        ("code_history", executed, [block0["event_id"], block1["event_id"]]),
        ("stdout", stdout1, [ep.events_of("stdout")[-1]["event_id"]]),
        ("vdm_description", vdm2_text, [vdm2_resp["event_id"]]),
    ], prompt_text=multi_turn_prompt(executed, stdout1, stderr1), user_feedback=None)
    ep.register("code_history", executed, [block0["event_id"], block1["event_id"]])
    lc = ep.new_id("lc_", 16)
    req4 = ep.model_request(
        role="code_generator", call_kind="multi_turn_decision", turn_index=2, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={"model": GEMINI, "messages": messages, "temperature": 1.0, "max_tokens": 20480},
        refs_extra={"for_event_id": feedback["event_id"]},
    )
    resp4 = ep.model_response(req4, "FINISH", latency_s=3.02, usage={"prompt_tokens": 7310, "completion_tokens": 2, "total_tokens": 7312})
    ep.parsed(resp4, parser="_parse_multi_turn_decision", code_blocks=[], decision="finish", model_said_finish=True)

    ep.emit(
        "evaluator_result",
        "info",
        {"scope": "episode", "code_block_id": None, "reward": 1.0, "task_completed": True, "terminated": True, "truncated": False, "source": "TrialSummary"},
        dt=0.05,
    )
    disk_warning = ep.events_of("trace_warning")[0]
    end = ep.emit(
        "episode_end",
        "info",
        {
            "end_reason": "completed",
            "harness_summary": {
                "success": True,
                "success_meaning": ("sandbox_rc == 0 of the last code block that ran (0 also when its "
                                    "stderr says 'executing action in terminated episode'); earlier "
                                    "blocks may have failed; not task success"),
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
                "sandbox_rc": 0,
                "task_completed": True,
                "num_regenerations": 1,
                "num_finishes": 1,
                "num_code_blocks": 2,
            },
            "exception": None,
            "alarm_seen": False,
            "duration_s": round(ep.wall - start["wall_time"], 3),
            "n_code_blocks": 2,
            "n_model_calls": 7,
            "tracer_overhead_s": 0.184,
            "trace_complete": False,
            "gaps": [{"code": "disk_low", "count": 1, "first_seq": disk_warning["seq"]}],
            "dropped_events": 0,
            "late_events": 0,
            "n_warnings": 1,
        },
        dt=0.02,
    )
    ep.ended = True
    ep.tick(0.4)
    ep.sim_step += SUBSAMPLE
    ep.buffer.append((ep.sim_step, scene.copy()))
    ep.save_frame(len(ep.buffer) - 1, reason="periodic")
    late = [e for e in ep.events if e["late"]]
    late_frame = late[0]
    late_warning = ep.warning(
        "late_event",
        f"{len(late)} frame events arrived after episode_end; they are kept and marked late.",
        late_frame,
    )
    ep.write_events()
    ep.write_episode_json(
        episode_summary(
            ep, start, end=end,
            gaps=[
                {"code": "disk_low", "count": 1, "first_seq": disk_warning["seq"]},
                {"code": "late_event", "count": len(late), "first_seq": late_frame["seq"]},
            ],
            late_events=len(late), n_warnings=2, last_event=late_warning,
        )
    )
    ep.touch_heartbeat(end["wall_time"])
    return ep


def build_feedback(
    ep: EpisodeBuilder,
    turn: int,
    builder: str,
    snapshot: dict[str, Any] | None,
    vdm_response: dict[str, Any] | None,
    segments: list[tuple[str, str, list[str]]],
    *,
    prompt_text: str,
    user_feedback: str | None,
    video_feedback: bool = False,
) -> dict[str, Any]:
    """feedback_built as the trace recorder records it.

    `segments` are the registered texts (kind, text, source ids) the builder may have
    used; `prompt_text` is the complete_multi_turn_prompt it received. The texts file holds
    the builder's text arguments; `segments` lists the prompt (with the registered texts it
    contains), the VDM description and the user feedback.
    """
    event_id = ep.new_id()
    payload_rel = f"payloads/{event_id}.feedback.json"
    vdm_text = next((text for kind, text, _ids in segments if kind == "vdm_description"), None) if vdm_response else None
    stored = {"complete_multi_turn_prompt": prompt_text, "visual_differencing_feedback": vdm_text, "user_feedback": user_feedback}
    data = json.dumps(stored, ensure_ascii=True).encode("ascii")
    ep.write_file(payload_rel, data)

    def seg(kind: str, text: str, ids: list[str]) -> dict[str, Any]:
        return {"kind": kind, "text_sha256": sha256_hex(text.encode()), "length": len(text), "source_event_ids": ids}

    # The recorder registers the user text after the builder ran, so it is not in `contains`.
    contains = [seg(kind, text, ids) for kind, text, ids in segments if kind != "user_feedback" and text and text in prompt_text]
    recorded = [{**seg("multi_turn_prompt", prompt_text, []), "contains": contains}]
    if vdm_text is not None:
        recorded.append(seg("vdm_description", vdm_text, [vdm_response["event_id"]]))
    if user_feedback is not None:
        recorded.append(seg("user_feedback", user_feedback, []))
    return ep.emit(
        "feedback_built",
        "info",
        {
            "builder": builder,
            "turn_index": turn,
            "segments": recorded,
            "image_pixel_sha256": snapshot["data"]["pixel_sha256"] if snapshot else None,
            "vdm_model_event_id": vdm_response["event_id"] if vdm_response else None,
            "user_feedback": user_feedback,
            "is_video_feedback": video_feedback,
            "has_visual_feedback": snapshot is not None,
            "n_messages": 2,
            "texts_path": payload_rel,
        },
        refs={"turn_index": turn, **({"model_event_id": vdm_response["event_id"]} if vdm_response else {})},
        artifacts=[
            {"path": payload_rel, "sha256": sha256_hex(data), "kind": "json", "mime": "application/json", "bytes": len(data), "label": "text", "written": True}
        ],
        thread=ep.model_thread,
        event_id=event_id,
        dt=0.003,
    )


def episode_summary(
    ep: EpisodeBuilder,
    start: dict[str, Any],
    *,
    end: dict[str, Any] | None,
    gaps: list[dict[str, Any]],
    late_events: int,
    n_warnings: int,
    last_event: dict[str, Any],
) -> dict[str, Any]:
    """Build episode.json in the format the trace recorder writes."""
    d = start["data"]
    doc: dict[str, Any] = {
        "schema_version": SCHEMA,
        "tracer_version": "1.0.0",
        "synthetic": True,
        "run_id": RUN_ID,
        "episode_id": ep.episode_id,
        "status": "ended" if end else "running",
        "episode_end_written": end is not None,
        "trace_complete": (not gaps) if end else None,
        "gaps": gaps,
        "dropped_events": 0,
        "late_events": late_events,
        "n_warnings": n_warnings,
        "n_events": len(ep.events),
        "last_seq": last_event["seq"],
        "events_bytes": (ep.dir / "events.jsonl").stat().st_size,
        "heartbeat_interval_s": 10.0,
        "updated_wall_time": last_event["wall_time"],
        "host": d["host"],
    }
    for key in ("source", "task", "condition", "seed", "attempt_index", "runner_invocation",
                "condition_label", "models", "config_flags", "env"):
        doc[key] = d.get(key)
    doc["started_wall_time"] = start["wall_time"]
    doc["pid"] = start["pid"]
    if end is not None:
        e = end["data"]
        doc["ended_wall_time"] = end["wall_time"]
        for key in ("end_reason", "duration_s", "alarm_seen", "n_code_blocks", "n_model_calls", "tracer_overhead_s"):
            doc[key] = e.get(key)
        hs = e.get("harness_summary") or {}
        doc["harness_summary"] = {
            k: hs.get(k)
            for k in ("reward", "task_completed", "sandbox_rc", "success", "num_code_blocks", "num_regenerations", "num_finishes")
        }
        doc["exception"] = e.get("exception")
    return doc


# ---------------------------------------------------------------------------
# Episode B: web UI, still running
# ---------------------------------------------------------------------------

WEB_BLOCK0 = """obs = get_observation()
view = obs["robot0_robotview"]
depth = view["images"]["depth"][:, :, 0]
red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")
grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))
q = solve_ik(grasps[0][:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q)
close_gripper()"""

WEB_BLOCK1 = """import numpy as np
q_up = solve_ik(np.array([0.58, -0.08, 1.10]), np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q_up)"""

WEB_BLOCK2 = """import numpy as np
obs = get_observation()
view = obs["robot0_robotview"]
depth = view["images"]["depth"][:, :, 0]
red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")
for attempt in range(3):
    try:
        grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))
        break
    except Exception as exc:
        print("plan_grasp failed:", exc)"""


def build_episode_b(run_dir: Path, now: float) -> tuple[EpisodeBuilder, Scene]:
    env_thread = "ThreadPoolExecutor-1_0"
    ep = EpisodeBuilder(
        run_dir,
        "20260926T060000Z__web__5e1f0a2c",
        pid=420001,
        t0=now - 420.0,
        mono0=5021.25,
        cameras=("agentview",),
        env_thread=env_thread,
        model_thread="ThreadPoolExecutor-0_0",
    )
    ep.periodic_every = 3
    scene = Scene(
        gripper=np.array([0.45, 0.0, 1.05]),
        red=np.array([0.57, -0.07, 0.84]),
        green=np.array([0.61, 0.11, 0.84]),
    )
    ep.buffer.append((None, scene.copy()))
    start = ep.emit(
        "episode_start",
        "info",
        {
            "source": "web_ui",
            "task": "franka_robosuite_cube_stack_multiturn_vdm",
            "condition": "web",
            "seed": None,
            "attempt_index": 1,
            "runner_invocation": None,
            "condition_label": None,
            "runner_condition": None,
            "config_flags": {
                "multi_turn": True,
                "use_visual_feedback": False,
                "use_img_differencing": False,
                "use_video_differencing": True,
                "use_wrist_camera": False,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
                "use_legacy_multi_turn_decision_prompt": False,
                "record_video": True,
                "use_oracle_code": False,
                "oracle_file": None,
                "await_user_input_each_turn": True,
            },
            "models": {
                "code_generator": {"model": GEMINI, "endpoint": endpoint_info(PROXY_URL, known_proxy=True)},
                "vdm": {"model": GEMINI, "endpoint": endpoint_info(PROXY_URL, known_proxy=True)},
            },
            "env": {
                "class": "FrankaPickPlaceCodeEnv",
                "low_level_class": "FrankaRobosuiteCubesLowLevel",
                "apis": ["FrankaControlApiReduced"],
                "viser_port": 8081,
            },
            "tracer": tracer_info(state=True, video=True),
            "host": "demo-host",
            "python": "3.10.19",
            "git_commit": "15ffdf6a0c1e5b0e9d7c3f1b2a4e6d8c0b9a7f5e",
            "process_start_time": 19855120,
        },
        thread="MainThread",
        dt=0.0,
    )
    ep.save_frame(0, reason="periodic")
    register_templates(ep, STACK_TASK_PROMPT)
    lc = ep.new_id("lc_", 16)
    payload = {
        "model": GEMINI,
        "messages": codegen_messages(STACK_TASK_PROMPT),
        "temperature": 1.0,
        "max_tokens": 20480,
        "stream": True,
    }
    req = ep.model_request(
        role="code_generator", call_kind="initial_code", turn_index=0, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1, payload=payload,
    )
    ep.emit(
        "task_text",
        "info",
        task_text_data(STACK_TASK_PROMPT, req),
        refs={"logical_call_id": req["data"]["logical_call_id"]}, thread="MainThread", dt=0.0)
    resp = ep.model_response(
        req, "```python\n" + WEB_BLOCK0 + "\n```", latency_s=8.9, streamed=True, n_chunks=64, ttft_s=2.31,
        usage=None,
        reasoning={"text": "Grasp the red cube first; the placement comes in the next turn after looking at the result.", "field": "delta.reasoning"},
    )
    ep.parsed(resp, parser="_extract_code", code_blocks=[WEB_BLOCK0], decision=None, model_said_finish=None)

    block0 = ep.code_block_start(block_index=0, turn_index=0, code=WEB_BLOCK0, source=resp, scene=scene)
    sim0 = ep.sim_step
    rgb = render(scene, "robot0_robotview", 128)
    depth = depth_map(scene, 128)
    intr = np.array([[110.9, 0.0, 64.0], [0.0, 110.9, 64.0], [0.0, 0.0, 1.0]])
    call = ep.api_start("get_observation", {}, block=block0, scene=scene, line=1, source_line="obs = get_observation()")
    ep.api_end(call, scene=scene, result={"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[:, :, None]}, "intrinsics": intr}}, duration_s=0.05)
    call = ep.api_start("segment_sam3_text_prompt", {"rgb": rgb, "text_prompt": "red cube"}, block=block0, scene=scene, line=4, source_line='red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")')
    red_mask = cube_mask(scene, scene.red, 128)
    ep.service_call(
        call, service="sam3", function="sam3_seg_fn", url="http://127.0.0.1:8114/segment",
        request={"image": rgb, "text_prompt": "red cube"},
        result=[{"mask": red_mask, "box": [40.0, 61.0, 57.0, 78.0], "score": 0.94}],
        latency_s=0.8, http_attempts=[{"status": 200, "latency_s": 0.8, "error": None}],
    )
    ep.api_end(call, scene=scene, result=[{"mask": red_mask, "box": [40.0, 61.0, 57.0, 78.0], "score": 0.94}], duration_s=0.83)
    call = ep.api_start("plan_grasp", {"depth": depth, "intrinsics": intr, "segmentation": red_mask.astype(np.uint8)}, block=block0, scene=scene, line=5, source_line='grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))')
    grasps = np.tile(np.eye(4, dtype=np.float32), (12, 1, 1))
    grasps[:, :3, 3] = scene.red
    scores = np.linspace(0.9, 0.5, 12).astype(np.float32)
    ep.service_call(
        call, service="graspnet", function="grasp_net_plan_fn", url="http://127.0.0.1:8115/plan",
        request={"depth": depth, "segmentation": red_mask.astype(np.uint8)},
        result=(grasps, scores),
        latency_s=1.7, http_attempts=[{"status": 200, "latency_s": 1.7, "error": None}],
    )
    ep.api_end(call, scene=scene, result=(grasps, scores), duration_s=1.72)
    call = ep.api_start("solve_ik", {"position": scene.red.copy(), "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])}, block=block0, scene=scene, line=6, source_line="q = solve_ik(grasps[0][:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))")
    q = np.array([0.01, 0.43, -0.02, -2.1, 0.01, 2.53, 0.78])
    ep.service_call(
        call, service="pyroki", function="ik_solve_fn", url="http://127.0.0.1:8116/ik",
        request={"target_link": "panda_hand", "position": scene.red.copy()},
        result=q,
        latency_s=0.3, http_attempts=[{"status": 200, "latency_s": 0.3, "error": None}],
    )
    ep.api_end(call, scene=scene, result=q, duration_s=0.31)
    call = ep.api_start("move_to_joints", {"joints": q}, block=block0, scene=scene, line=7, source_line="move_to_joints(q)")
    ep.advance(scene, scene.red + np.array([0.0, 0.0, 0.045]), steps=80, seconds=2.8)
    ep.api_end(call, scene=scene, result=None, duration_s=2.83)
    call = ep.api_start("close_gripper", {}, block=block0, scene=scene, line=8, source_line="close_gripper()")
    scene.gripper_open = False
    scene.held = True
    ep.advance(scene, scene.gripper.copy(), steps=23, seconds=0.7)  # ends 3 steps after a buffer frame
    ep.api_end(call, scene=scene, result=None, duration_s=0.7)
    ep.code_block_end(block0, scene=scene, sandbox_rc=0, reward=0.0, task_completed=False, terminated=False, sim_start=sim0, duration_s=6.9)
    ep.register("code_history", WEB_BLOCK0, [block0["event_id"]])

    # video of the turn -> VDM
    turn_frames = [render(s, "agentview", 128) for _step, s in ep.buffer]
    frame_hashes = [sha256_hex(f.tobytes()) for f in turn_frames]
    read = ep.emit(
        "video_frames_read",
        "ok",
        # the fields the trace recorder writes for a get_video_frames_range call
        {"start": 0, "end": len(turn_frames) - 1, "n_frames": len(turn_frames), "camera": "agentview",
         "frame_pixel_sha256": frame_hashes},
        dt=0.02,
    )
    ep.video_reads.append((read, frame_hashes))
    video = mp4_bytes(turn_frames)
    ep.video_frames = {sha256_hex(video): turn_frames}
    video_url = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
    task_description = f"{STACK_TASK_PROMPT}\nAPIs:\n\n{API_DOCS}"
    lc = ep.new_id("lc_", 16)
    vdm = ep.model_request(
        role="vdm", call_kind="video_diff", turn_index=1, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={
            "model": GEMINI,
            "messages": [
                {"role": "system", "content": VDM_VIDEO_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_description},
                        {"type": "text", "text": VDM_VIDEO_TEXT},
                        {"type": "text", "text": "Main camera video:"},
                        {"type": "image_url", "image_url": {"url": video_url}},
                    ],
                },
            ],
            "temperature": 1.0,
            "max_tokens": 20480,
        },
    )
    vdm_text = (
        "The robot moved down to the red cube, closed the gripper around it and stayed at "
        "table height. The red cube is held but has not been lifted. The task is not complete."
    )
    vdm_resp = ep.model_response(vdm, vdm_text, latency_s=6.2, usage={"prompt_tokens": 2400, "completion_tokens": 44, "total_tokens": 2444})
    ep.register("vdm_description", vdm_text, [vdm_resp["event_id"]])
    user_text = "Lift the red cube higher before moving it over the green cube."
    ep.register("user_feedback", user_text)
    feedback = build_feedback(ep, 1, "_build_multi_turn_decision_prompt", None, vdm_resp, [
        ("code_history", WEB_BLOCK0, [block0["event_id"]]),
        ("vdm_description", vdm_text, [vdm_resp["event_id"]]),
        ("user_feedback", user_text, []),
    ], prompt_text=multi_turn_prompt(WEB_BLOCK0, "", "", user_text), user_feedback=user_text, video_feedback=True)
    messages = decision_messages(STACK_TASK_PROMPT, WEB_BLOCK0, "", "", image_url=None, feedback=vdm_text, video_feedback=True, user_feedback=user_text)
    lc = ep.new_id("lc_", 16)
    req2 = ep.model_request(
        role="code_generator", call_kind="multi_turn_decision", turn_index=1, url=PROXY_URL, known_proxy=True,
        logical_call_id=lc, http_attempt=1,
        payload={"model": GEMINI, "messages": messages, "temperature": 1.0, "max_tokens": 20480, "stream": True},
        refs_extra={"for_event_id": feedback["event_id"]},
    )
    resp2 = ep.model_response(
        req2, "REGENERATE\n```python\n" + WEB_BLOCK1 + "\n```", latency_s=5.5, streamed=True, n_chunks=31, ttft_s=1.8, usage=None,
    )
    ep.parsed(resp2, parser="_parse_multi_turn_decision", code_blocks=[WEB_BLOCK1], decision="regenerate", model_said_finish=False)

    # block 1 hits the web UI block timeout; the env is reset (new sim epoch)
    block1 = ep.code_block_start(block_index=1, turn_index=1, code=WEB_BLOCK1, source=resp2, scene=scene)
    sim1 = ep.sim_step
    call = ep.api_start("solve_ik", {"position": np.array([0.58, -0.08, 1.10]), "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])}, block=block1, scene=scene, line=2, source_line="q_up = solve_ik(np.array([0.58, -0.08, 1.10]), np.array([0.0, 1.0, 0.0, 0.0]))")
    ep.api_end(call, scene=scene, result=np.array([0.02, 0.1, -0.02, -1.6, 0.0, 1.9, 0.79]), duration_s=0.3)
    call = ep.api_start("move_to_joints", {"joints": np.array([0.02, 0.1, -0.02, -1.6, 0.0, 1.9, 0.79])}, block=block1, scene=scene, line=3, source_line="move_to_joints(q_up)")
    ep.advance(scene, scene.gripper + np.array([0.0, 0.0, 0.02]), steps=30, seconds=179.5)
    tb = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 3, in <module>\n'
        "TimeoutError: code block exceeded the web UI execution timeout (180 s)\n"
    )
    ep.api_end(call, scene=scene, error=("TimeoutError", "code block exceeded the web UI execution timeout (180 s)", tb), duration_s=180.0)
    ep.stderr += tb
    ep.code_block_end(block1, scene=scene, sandbox_rc=1, reward=0.0, task_completed=False, terminated=False, sim_start=sim1, duration_s=180.2)
    ep.sim_epoch += 1
    ep.sim_step = 0
    ep.buffer = []
    scene = Scene(gripper=np.array([0.45, 0.0, 1.05]), red=np.array([0.57, -0.07, 0.84]), green=np.array([0.61, 0.11, 0.84]))
    ep.emit("env_reset", "info", {"sim_epoch": ep.sim_epoch, "reason": "web_block_timeout"}, dt=1.2)
    ep.record_buffer_frame(scene)

    ep.tick(max(0.0, (now - 25.0) - ep.wall))
    block2 = ep.code_block_start(block_index=2, turn_index=1, code=WEB_BLOCK2, source=resp2, scene=scene)
    rgb = render(scene, "robot0_robotview", 128)
    depth = depth_map(scene, 128)
    call = ep.api_start("get_observation", {}, block=block2, scene=scene, line=2, source_line="obs = get_observation()")
    ep.api_end(call, scene=scene, result={"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[:, :, None]}, "intrinsics": intr}}, duration_s=0.05)
    call = ep.api_start("segment_sam3_text_prompt", {"rgb": rgb, "text_prompt": "red cube"}, block=block2, scene=scene, line=5, source_line='red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")')
    ep.service_call(
        call, service="sam3", function="sam3_seg_fn", url="http://127.0.0.1:8114/segment",
        request={"image": rgb, "text_prompt": "red cube"},
        result=[{"mask": red_mask, "box": [40.0, 61.0, 57.0, 78.0], "score": 0.95}],
        latency_s=0.8, http_attempts=[{"status": 200, "latency_s": 0.8, "error": None}],
    )
    ep.api_end(call, scene=scene, result=[{"mask": red_mask, "box": [40.0, 61.0, 57.0, 78.0], "score": 0.95}], duration_s=0.82)
    call = ep.api_start("plan_grasp", {"depth": depth, "intrinsics": intr, "segmentation": red_mask.astype(np.uint8)}, block=block2, scene=scene, line=8, source_line='        grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))')
    ep.service_call(
        call, service="graspnet", function="grasp_net_plan_fn", url="http://127.0.0.1:8115/plan",
        request={"depth": depth, "segmentation": red_mask.astype(np.uint8)},
        result=None,
        latency_s=0.05, http_attempts=[{"status": 503, "latency_s": 0.05, "error": "HTTP 503"}],
        error=("HTTPError", "503 Server Error: Service Unavailable for url: http://127.0.0.1:8115/plan"),
    )
    ep.print_out("plan_grasp failed: 503 Server Error: Service Unavailable for url: http://127.0.0.1:8115/plan\n")
    ep.api_end(call, scene=scene, error=("HTTPError", "503 Server Error: Service Unavailable for url: http://127.0.0.1:8115/plan", "Traceback (most recent call last):\n  ...\nrequests.exceptions.HTTPError: 503 Server Error: Service Unavailable for url: http://127.0.0.1:8115/plan\n"), duration_s=0.06)
    call = ep.api_start("plan_grasp", {"depth": depth, "intrinsics": intr, "segmentation": red_mask.astype(np.uint8)}, block=block2, scene=scene, line=8, source_line='        grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))')
    ep.service_call(
        call, service="graspnet", function="grasp_net_plan_fn", url="http://127.0.0.1:8115/plan",
        request={"depth": depth, "segmentation": red_mask.astype(np.uint8)},
        result=None, latency_s=0.0, http_attempts=[], open_ended=True,
    )
    ep.record_buffer_frame(scene)
    ep.write_events()
    ep.write_episode_json(
        episode_summary(ep, start, end=None, gaps=[], late_events=0, n_warnings=0, last_event=ep.events[-1])
    )
    ep.touch_heartbeat(time.time())
    return ep, scene


# ---------------------------------------------------------------------------
# Episode C: single turn, process killed mid-block
# ---------------------------------------------------------------------------

LIFT_CODE = """import numpy as np
obs = get_observation()
view = obs["robot0_robotview"]
depth = view["images"]["depth"][:, :, 0]
red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")
grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))
q = solve_ik(grasps[0][:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q)
close_gripper()
q = solve_ik(grasps[0][:3, 3] + np.array([0.0, 0.0, 0.2]), np.array([0.0, 1.0, 0.0, 0.0]))
move_to_joints(q)"""


def build_episode_c(run_dir: Path, now: float) -> EpisodeBuilder:
    ep = EpisodeBuilder(
        run_dir,
        "cube_lift__S3__s2__i1__a1",
        pid=412999,
        t0=now - 4000.0,
        mono0=87211.0,
        cameras=("agentview",),
        record_video=False,
        record_state=False,
    )
    scene = Scene(gripper=np.array([0.45, 0.0, 1.05]), red=np.array([0.56, 0.02, 0.84]), green=np.array([0.9, 0.4, 0.84]))
    vllm_model = "Qwen/Qwen3-VL-8B-Instruct"
    start = ep.emit(
        "episode_start",
        "info",
        {
            "source": "experiments.run",
            "task": "cube_lift",
            "condition": "S3",
            "seed": 2,
            "attempt_index": 1,
            "runner_invocation": 1,
            "condition_label": "S3",
            "runner_condition": {
                "label": "S3",
                "config_path": "env_configs/cube_lifting/franka_robosuite_cube_lifting_reduced_api.yaml",
                "privileged": False,
                "apis": ["FrankaControlApiReduced"],
                "multi_turn": False,
                "uses_vdm": False,
                "use_visual_feedback": False,
                "use_oracle_code": False,
                "note": "synthetic demo condition",
            },
            "config_flags": {
                "multi_turn": False,
                "use_visual_feedback": False,
                "use_img_differencing": False,
                "use_video_differencing": False,
                "use_wrist_camera": False,
                "use_parallel_ensemble": False,
                "use_multimodel": False,
                "use_legacy_multi_turn_decision_prompt": False,
                "record_video": False,
                "use_oracle_code": False,
                "oracle_file": None,
                "await_user_input_each_turn": None,
            },
            "models": {"code_generator": {"model": vllm_model, "endpoint": endpoint_info(VLLM_URL, known_proxy=False)}, "vdm": {"model": vllm_model, "endpoint": endpoint_info(VLLM_URL, known_proxy=False)}},
            "env": {"class": "FrankaLiftCodeEnv", "low_level_class": "FrankaRobosuiteCubeLiftLowLevel", "apis": ["FrankaControlApiReduced"], "viser_port": None},
            "tracer": tracer_info(state=False, video=False),
            "host": "demo-host",
            "python": "3.10.19",
            "git_commit": "15ffdf6a0c1e5b0e9d7c3f1b2a4e6d8c0b9a7f5e",
            "process_start_time": 18423900,
        },
        dt=0.0,
    )
    ep.register("task_prompt", LIFT_TASK_PROMPT)
    ep.register("api_docs", f"\n{API_DOCS}")
    ep.register("harness_template", CODEGEN_SYSTEM)
    lc = ep.new_id("lc_", 16)
    # The demo adds an api_key body field to show masking; CaP-X bodies carry no secrets.
    payload = {
        "model": vllm_model,
        "messages": codegen_messages(LIFT_TASK_PROMPT),
        "temperature": 0.7,
        "max_tokens": 4096,
        "api_key": "placeholder-not-a-key",
    }
    req = ep.model_request(
        role="code_generator", call_kind="initial_code", turn_index=0, url=VLLM_URL, known_proxy=False,
        logical_call_id=lc, http_attempt=1, payload=payload, mask_keys=("api_key",),
    )
    ep.emit(
        "task_text",
        "info",
        task_text_data(LIFT_TASK_PROMPT, req),
        refs={"logical_call_id": req["data"]["logical_call_id"]}, dt=0.0)
    resp = ep.model_response(req, LIFT_CODE, latency_s=14.2, usage={"prompt_tokens": 1301, "completion_tokens": 233, "total_tokens": 1534}, response_model=vllm_model)
    ep.parsed(resp, parser="_extract_code", code_blocks=[LIFT_CODE], decision=None, model_said_finish=None)
    block = ep.code_block_start(block_index=0, turn_index=0, code=LIFT_CODE, source=resp, scene=scene)
    rgb = render(scene, "robot0_robotview", 128)
    depth = depth_map(scene, 128)
    intr = np.array([[110.9, 0.0, 64.0], [0.0, 110.9, 64.0], [0.0, 0.0, 1.0]])
    call = ep.api_start("get_observation", {}, block=block, scene=scene, line=2, source_line="obs = get_observation()")
    ep.api_end(call, scene=scene, result={"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[:, :, None]}, "intrinsics": intr}}, duration_s=0.04)
    call = ep.api_start("segment_sam3_text_prompt", {"rgb": rgb, "text_prompt": "red cube"}, block=block, scene=scene, line=5, source_line='red = segment_sam3_text_prompt(view["images"]["rgb"], "red cube")')
    red_mask = cube_mask(scene, scene.red, 128)
    ep.service_call(
        call, service="sam3", function="sam3_seg_fn", url="http://127.0.0.1:8114/segment",
        request={"image": rgb, "text_prompt": "red cube"},
        result=[{"mask": red_mask, "box": [38.0, 60.0, 56.0, 78.0], "score": 0.97}],
        latency_s=0.9, http_attempts=[{"status": 200, "latency_s": 0.9, "error": None}],
    )
    ep.api_end(call, scene=scene, result=[{"mask": red_mask, "box": [38.0, 60.0, 56.0, 78.0], "score": 0.97}], duration_s=0.93)
    # a line that was damaged on disk; its seq is lost
    ep.lose_seq(b'{"schema_version":"capx-trace/1","run_id":"demo_trace_viewer","episode_id":"cube_lift__S3__s2__i1__a1","event_id":"ev_\x00\x00\x00\x00\x00\x00\x00\x00\n')
    call = ep.api_start("plan_grasp", {"depth": depth, "intrinsics": intr, "segmentation": red_mask.astype(np.uint8)}, block=block, scene=scene, line=6, source_line='grasps, scores = plan_grasp(depth, view["intrinsics"], red[0]["mask"].astype(np.uint8))')
    grasps = np.tile(np.eye(4, dtype=np.float32), (8, 1, 1))
    grasps[:, :3, 3] = scene.red
    ep.service_call(
        call, service="graspnet", function="grasp_net_plan_fn", url="http://127.0.0.1:8115/plan",
        request={"depth": depth, "segmentation": red_mask.astype(np.uint8)},
        result=(grasps, np.linspace(0.88, 0.5, 8).astype(np.float32)),
        latency_s=1.8, http_attempts=[{"status": 200, "latency_s": 1.8, "error": None}],
    )
    ep.api_end(call, scene=scene, result=(grasps, np.linspace(0.88, 0.5, 8).astype(np.float32)), duration_s=1.83)
    call = ep.api_start("solve_ik", {"position": scene.red.copy(), "quaternion_wxyz": np.array([0.0, 1.0, 0.0, 0.0])}, block=block, scene=scene, line=7, source_line="q = solve_ik(grasps[0][:3, 3], np.array([0.0, 1.0, 0.0, 0.0]))")
    q = np.array([0.03, 0.44, -0.01, -2.09, 0.0, 2.52, 0.8])
    ep.service_call(
        call, service="pyroki", function="ik_solve_fn", url="http://127.0.0.1:8116/ik",
        request={"target_link": "panda_hand", "position": scene.red.copy()},
        result=q,
        latency_s=0.3, http_attempts=[{"status": 200, "latency_s": 0.3, "error": None}],
    )
    ep.api_end(call, scene=scene, result=q, duration_s=0.31)
    ep.api_start("move_to_joints", {"joints": q}, block=block, scene=scene, line=8, source_line="move_to_joints(q)")
    ep.sim_step += 40
    ep.tick(1.2)
    torn = (
        b'{"schema_version":"capx-trace/1","run_id":"demo_trace_viewer","episode_id":"cube_lift__S3__s2__i1__a1",'
        b'"event_id":"ev_7d2a91c04be3f5a61d20","parent_event_id":null,"seq":' + str(ep.seq).encode() + b',"wall_time":'
    )
    ep.write_events(torn_tail=torn)
    ep.write_episode_json(
        episode_summary(ep, start, end=None, gaps=[], late_events=0, n_warnings=0, last_event=ep.events[-1])
    )
    ep.touch_heartbeat(ep.wall + 4.0)
    return ep


# ---------------------------------------------------------------------------
# Run level files
# ---------------------------------------------------------------------------


def runner_record(task: str, condition: str, seed: int, *, status: str, error: str | None, outcome: dict[str, Any], reward: float | None, started: float, duration: float) -> dict[str, Any]:
    def iso(t: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(t))

    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "task": task,
        "condition": condition,
        "trial": seed,
        "seed": seed,
        "init_state_id": f"seed:{seed}",
        "attempts": [
            {"attempt_id": 1, "status": status, "started_at": iso(started), "ended_at": iso(started + duration), "duration_s": duration, "error": error}
        ],
        "num_attempts": 1,
        "timed_out_attempts": 0,
        "outcome": outcome,
        "reward": reward,
        "synthetic": True,
    }


def write_run_files(run_dir: Path, now: float, a: EpisodeBuilder, c: EpisodeBuilder) -> None:
    run_json = {
        "schema_version": SCHEMA,
        "synthetic": True,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now - 7300)),
        "generator": "scripts/make_demo_trace.py",
        "note": "Synthetic demo trace for the trace viewer; not a real experiment.",
        "host": "demo-host",
        "tracer": tracer_info(state=True, video=True),
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=1))
    records = [
        runner_record(
            "cube_stack", "M4", 1, status="completed", error=None,
            outcome={"task_success": True, "code_execution_success": True, "agent_finish": True, "infrastructure_error": False, "timeout": False, "horizon_exhausted": False, "termination_reason": "agent_finish"},
            reward=1.0, started=a.events[0]["wall_time"], duration=round(a.wall - a.events[0]["wall_time"], 3),
        ),
        runner_record(
            "cube_lift", "S3", 2, status="error", error="worker process exited with code -9 (SIGKILL)",
            outcome={"task_success": None, "code_execution_success": None, "agent_finish": None, "infrastructure_error": True, "timeout": False, "horizon_exhausted": False, "termination_reason": "infrastructure_error"},
            reward=None, started=c.events[0]["wall_time"], duration=round(c.wall - c.events[0]["wall_time"], 3),
        ),
    ]
    (run_dir / "runner_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    # The runner's preflight request and its response, as the recorder writes them into the
    # run's unscoped log: payload paths are relative to the run dir, under
    # unscoped.<pid>.files/ (the trace recorder).
    files_rel = "unscoped.412300.files/payloads"
    (run_dir / files_rel).mkdir(parents=True)
    request_id = "ev_" + sha256_hex(b"preflight")[:20]
    response_id = "ev_" + sha256_hex(b"preflight-response")[:20]
    body = {"model": GEMINI, "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply with the single word OK."}]}], "max_tokens": 8}
    body_text = json.dumps(body)
    request_rel = f"{files_rel}/{request_id}.request.json"
    (run_dir / request_rel).write_text(body_text)
    reply = {"id": "gen-preflight", "model": GEMINI, "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}}
    reply_text = json.dumps(reply)
    response_rel = f"{files_rel}/{response_id}.response.json"
    (run_dir / response_rel).write_text(reply_text)
    envelope = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "episode_id": None,
        "monotonic_time": 91100.0,
        "pid": 412300,
        "thread": "MainThread",
        "sim_step": None,
        "sim_time": None,
        "sim_epoch": None,
        "late": False,
    }
    lcid = "lc_" + sha256_hex(b"preflight-call")[:20]
    preflight = {
        **envelope,
        "event_id": request_id,
        "parent_event_id": None,
        "seq": 1,
        "wall_time": round(now - 7250.0, 6),
        "event_type": "model_request",
        "status": "started",
        "refs": {"logical_call_id": lcid, "http_attempt": 1},
        "artifacts": [{"path": request_rel, "sha256": sha256_hex(body_text.encode()), "kind": "json", "mime": "application/json", "bytes": len(body_text), "label": "payload", "written": True}],
        "data": {
            "role": "preflight", "call_kind": None, "turn_index": None, "logical_call_id": lcid, "http_attempt": 1,
            "model": GEMINI, "endpoint": endpoint_info(PROXY_URL, known_proxy=True), "stream": False,
            "payload_path": request_rel, "body_sha256": sha256_hex(body_text.encode()), "body_bytes": len(body_text),
            "reconstruction": {"verified": True, "method": "json_roundtrip"}, "masked_fields": [],
            "params": {"model": GEMINI, "max_tokens": 8}, "n_messages": 1, "images": [], "n_images": 0, "n_videos": 0,
            "input_summary": {
                "n_images": 0, "n_videos": 0, "no_image_input": True,
                "categories": {"images": [], "task_api_description": [], "code_history": [], "stdout_stderr": [], "vdm_description": [], "other": []},
                "unattributed": [{"path": "messages[0].content[0].text", "start": 0, "end": 30}],
                "registered_not_found": [],
            },
        },
    }
    response = {
        **envelope,
        "event_id": response_id,
        "parent_event_id": request_id,
        "seq": 2,
        "wall_time": round(now - 7249.4, 6),
        "event_type": "model_response",
        "status": "ok",
        "refs": {"logical_call_id": lcid, "http_attempt": 1, "model_event_id": request_id},
        "artifacts": [{"path": response_rel, "sha256": sha256_hex(reply_text.encode()), "kind": "json", "mime": "application/json", "bytes": len(reply_text), "label": "payload", "written": True}],
        "data": {
            "role": "preflight", "call_kind": None, "logical_call_id": lcid, "http_attempt": 1, "http_status": 200,
            "latency_s": 0.6, "streamed": False, "n_chunks": None, "ttft_s": None, "payload_path": response_rel,
            "content": "OK", "reasoning": None, "usage": reply["usage"], "finish_reason": "stop", "response_model": GEMINI, "response_id": "gen-preflight",
        },
    }
    (run_dir / "unscoped.412300.jsonl").write_text(json.dumps(preflight) + "\n" + json.dumps(response) + "\n")


def keep_live(run_dir: Path, episode: EpisodeBuilder, scene: Scene, seconds: float) -> None:
    """Keep episode B running: refresh the heartbeat and append frames, sometimes in two parts."""
    events_path = episode.dir / "events.jsonl"
    deadline = time.time() + seconds
    k = 0
    print(f"keeping {episode.episode_id} live for {seconds:.0f} s (Ctrl-C to stop)")
    try:
        while time.time() < deadline:
            time.sleep(2.0)
            k += 1
            episode.wall = time.time()
            episode.mono += 2.0
            episode.raw_lines = []
            scene.gripper = scene.gripper + np.array([0.0, 0.004 * np.sin(k / 3), 0.0])
            episode.sim_step += SUBSAMPLE
            episode.buffer.append((episode.sim_step, scene.copy()))
            episode.save_frame(len(episode.buffer) - 1, reason="periodic")
            data = b"".join(episode.raw_lines)
            with open(events_path, "ab") as fh:
                if k % 5 == 0 and len(data) > 40:
                    fh.write(data[:40])  # a torn tail for about a second
                    fh.flush()
                    time.sleep(1.0)
                    fh.write(data[40:])
                else:
                    fh.write(data)
            (episode.dir / "heartbeat").touch()
    except KeyboardInterrupt:
        pass
    print("stopped; the episode will show as incomplete once its heartbeat is 30 s old")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="outputs/traces_demo", help="trace root to write into")
    parser.add_argument("--live", type=float, default=0.0, help="keep the running episode live for this many seconds")
    args = parser.parse_args(argv)

    root = Path(args.out)
    if not root.is_absolute():
        root = Path.cwd() / root
    run_dir = root / RUN_ID
    if run_dir.exists():
        if not (run_dir / MARKER).exists():
            print(f"{run_dir} exists and was not written by this script; refusing to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / MARKER).write_text("written by scripts/make_demo_trace.py\n")
    now = time.time()
    a = build_episode_a(run_dir, now)
    b, b_scene = build_episode_b(run_dir, now)
    c = build_episode_c(run_dir, now)
    write_run_files(run_dir, now, a, c)
    unscoped = root / "_unscoped"
    unscoped.mkdir(exist_ok=True)
    (unscoped / "unscoped.412301.jsonl").write_text(
        json.dumps({"schema_version": SCHEMA, "run_id": None, "episode_id": None, "event_id": "ev_" + sha256_hex(b"orphan")[:20], "seq": 0, "event_type": "trace_warning", "status": "warning", "data": {"code": "ambiguous_episode", "message": "synthetic example", "related_event_id": None}}) + "\n"
    )
    for ep in (a, b, c):
        print(f"{ep.episode_id}: {len(ep.events)} events -> {ep.dir.relative_to(Path.cwd()) if ep.dir.is_relative_to(Path.cwd()) else ep.dir}")
    if args.live > 0:
        keep_live(run_dir, b, b_scene, args.live)
    return 0


if __name__ == "__main__":
    sys.exit(main())
