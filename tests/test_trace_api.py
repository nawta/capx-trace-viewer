"""Tests of capx_trace_viewer/trace_api.py (docs/trace-format.md sections 9.2 and 9.4).

Every endpoint is exercised on the synthetic demo trace (scripts/make_demo_trace.py)
and on a small fake runner root. Path traversal ("." "..", %2E%2E, %2F, NUL, symlinks) is
tried on every endpoint, including the legacy ones; loopback gating, array limits and
the section 2 rule (no output, no warnings) are checked too.
"""

from __future__ import annotations

import errno
import gzip
import io
import json
import os
import shutil
import threading
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from trace_viewer_testlib import EPISODE_ENDED, EPISODE_INCOMPLETE, EPISODE_RUNNING, LOOPBACK_BASE_URL, RUN_ID, raw_get

from capx_trace_viewer import reader
from capx_trace_viewer import trace_api, trace_arrays
from capx_trace_viewer.server import create_app

SECRET = "TOP-SECRET-OUTSIDE"
SECRET_NUMBER = 424242
LOCAL = ("127.0.0.1", 50000)
TRIAL_DIR = "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"

SUMMARY_TXT = "\n".join(
    [
        "-" * 100,
        "Generated program:",
        "# Code block 0\nprint('hello')\nprint('world')",
        "\n\nEnvironment response:",
        "  Sandbox failed: 0",
        "  Stdout: hello\nworld\n",
        "  Stderr: Traceback (most recent call last):\n  File \"<string>\", line 1\nValueError: boom\n",
        "  Reward: 0.536",
        "  Task Completed: False",
        "  Terminated: True, Truncated: False",
        "  Num Regenerations: 2",
        "  Num Finishes: 1",
        "  Num Code Blocks: 3",
        "-" * 100,
    ]
)


def _png(size: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8)).save(buf, format="png")
    return buf.getvalue()


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory, demo_root: Path) -> dict[str, Any]:
    """A copy of the demo trace root with symlink traps, a legacy runner root and secrets outside."""
    base = tmp_path_factory.mktemp("trace_api")
    root = base / "traces"
    shutil.copytree(demo_root, root, symlinks=True)
    (base / ".env").write_text(f"EXAMPLE_API_KEY={SECRET}\n")
    (base / "secret.txt").write_text(SECRET)
    (base / "secret.json").write_text(json.dumps({"status": "ended", "trace_complete": True, "task": SECRET}))
    np.save(base / "secret.npy", np.arange(100, dtype=np.int64) + SECRET_NUMBER)
    outside_ep = base / "outside_run" / "ep1"
    outside_ep.mkdir(parents=True)
    (outside_ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True, "task": SECRET}))
    (outside_ep / "events.jsonl").write_text(json.dumps({"seq": 0, "event_type": "episode_start", "data": {"task": SECRET}}) + "\n")
    (outside_ep / "secret.txt").write_text(SECRET)
    np.save(outside_ep / "secret.npy", np.arange(100, dtype=np.int64) + SECRET_NUMBER)

    # symlinks inside the trace root that lead outside
    (root / "escape_run").symlink_to(base / "outside_run")
    (root / RUN_ID / "escape_ep").symlink_to(outside_ep)
    ep = root / RUN_ID / EPISODE_ENDED
    (ep / "assets" / "leak.txt").symlink_to(base / "secret.txt")
    (ep / "assets" / "leak.npy").symlink_to(base / "secret.npy")
    real_png = sorted((ep / "assets").rglob("*.png"))[0]
    (ep / "assets" / "alias.png").symlink_to(real_png)  # stays inside: refused too (no symlink is followed)
    # the run's unscoped log: a symlink trap in its files dir
    (root / RUN_ID / "unscoped.412300.files" / "leak.txt").symlink_to(base / "secret.txt")

    # a run with episodes whose fixed-name files are symlinks to outside, and test arrays
    trap_run = root / "trap_run"
    trap1 = trap_run / "trap_episode_json"
    trap1.mkdir(parents=True)
    (trap1 / "episode.json").symlink_to(base / "secret.json")
    (trap1 / "events.jsonl").write_text("")
    trap2 = trap_run / "trap_events"
    trap2.mkdir()
    (trap2 / "episode.json").write_text(json.dumps({"status": "open"}))
    (trap2 / "events.jsonl").symlink_to(outside_ep / "events.jsonl")
    arrays = trap_run / "arrays_ep"
    (arrays / "arrays").mkdir(parents=True)
    (arrays / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    np.save(arrays / "arrays" / "object.npy", np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
    np.save(arrays / "arrays" / "structured.npy", np.zeros(3, dtype=[("x", "f4"), ("y", "i4")]))
    np.save(arrays / "arrays" / "complex.npy", np.zeros(3, dtype=np.complex64))
    (arrays / "arrays" / "pickle.npy").write_bytes(b"\x80\x04\x95" + b"cos\nsystem\n" * 5)
    (arrays / "arrays" / "corrupt.npz").write_bytes(b"PK\x03\x04 this is not a zip")
    with zipfile.ZipFile(arrays / "arrays" / "odd_member.npz", "w") as zf:
        zf.writestr("readme.txt", "not an array")
    np.savez(arrays / "arrays" / "two.npz", a=np.arange(10), b=np.ones((3, 3)))
    stack = np.zeros((3, 5, 7), dtype=bool)
    stack[2, 1:4, 2:6] = True
    np.savez(arrays / "arrays" / "stack.npz", m__packbits=np.packbits(stack.ravel()), m__shape=np.array(stack.shape), v=np.ones(3))
    np.save(arrays / "arrays" / "empty.npy", np.zeros((0, 4), dtype=np.float32))
    np.save(arrays / "arrays" / "nonfinite.npy", np.array([[np.nan, np.inf], [-np.inf, 1e308]], dtype=np.float64))
    np.save(arrays / "arrays" / "all_nan.npy", np.full((5, 5), np.nan, dtype=np.float32))
    np.save(arrays / "arrays" / "fortran.npy", np.asfortranarray(np.arange(12, dtype=np.int16).reshape(3, 4)))
    np.save(arrays / "arrays" / "big_endian.npy", np.arange(6, dtype=">f4"))
    np.save(arrays / "arrays" / "scalar.npy", np.float64(2.5))
    np.save(arrays / "arrays" / "depth.npy", np.linspace(0.5, 2.0, 64 * 48, dtype=np.float32).reshape(64, 48))
    np.save(arrays / "arrays" / "huge_side.npy", np.zeros((3000, 10), dtype=np.bool_))
    # a SAM3 call: 50 bit-packed masks in one npz (100 members), as the trace recorder writes it
    masks = {}
    for i in range(50):
        m = np.zeros((32, 40), dtype=bool)
        m[i % 32, :] = True
        masks[f"a{i}__packbits"] = np.packbits(m.ravel())
        masks[f"a{i}__shape"] = np.array(m.shape)
    np.savez_compressed(arrays / "arrays" / "sam3_masks.npz", **masks)

    # legacy runner root
    legacy = base / "runner"
    run = legacy / "smoke_run"
    trial = run / "cube_lift" / "S2" / TRIAL_DIR
    (trial / "prompts_and_responses").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"run_id": "smoke_run", "created_at": "2026-01-01T00:00:00+00:00", "manifest_path": "fixtures/example.yaml", "manifest": {"description": "legacy test run"}})
    )
    record = {
        "schema_version": 1, "run_id": "smoke_run", "task": "cube_lift", "condition": "S2", "trial": 1, "seed": 1,
        "attempts": [{"attempt_id": 1, "status": "completed"}], "outcome": {"task_success": True, "termination_reason": "code_finished"},
        "reward": 1.0, "artifacts": {"trial_dir": str(trial)},
    }
    (run / "trials.jsonl").write_text(json.dumps(record) + "\nthis line is corrupt\n")
    (trial / "code.py").write_text("print('hello')\n")
    (trial / "summary.txt").write_text(SUMMARY_TXT)
    (trial / "trial_record.json").write_text(json.dumps({**record, "sandbox_rc": 0}))
    (trial / "video_combined.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    (trial / "visual_feedback_00.png").write_bytes(_png())
    (trial / "prompts_and_responses" / "initial_prompt.txt").write_text("harness prompt text")
    (trial / "all_responses.json").write_text("[]")
    (trial / "leak.txt").symlink_to(base / "secret.txt")
    outside_legacy = base / "outside_legacy"
    (outside_legacy / "cube_lift" / "S2" / TRIAL_DIR).mkdir(parents=True)
    (outside_legacy / "run.json").write_text(json.dumps({"run_id": SECRET}))
    (outside_legacy / "cube_lift" / "S2" / TRIAL_DIR / "code.py").write_text(SECRET)
    (legacy / "escape_run").symlink_to(outside_legacy)
    (legacy / "smoke_run" / "cube_lift" / "escape_cond").symlink_to(outside_legacy / "cube_lift" / "S2")

    dist = base / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>INDEX</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(viewer_only=True, trace_root=root, runner_roots=[legacy], frontend_dist=dist)
    return {"base": base, "root": root, "legacy": legacy, "app": app, "client": TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL)}


def _get(env: dict[str, Any], url: str, **kw: Any) -> Any:
    return env["client"].get(url, **kw)


def _strict_json(body: bytes) -> Any:
    def reject(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    return json.loads(body, parse_constant=reject)


def _events(env: dict[str, Any], episode: str) -> list[dict[str, Any]]:
    return list(reader.iter_events(env["root"] / RUN_ID / episode / "events.jsonl"))


def _base(episode: str = EPISODE_ENDED) -> str:
    return f"/api/traces/runs/{RUN_ID}/episodes/{episode}"


# ---------------------------------------------------------------------------
# endpoints on the demo trace
# ---------------------------------------------------------------------------


def test_config(env: dict[str, Any]) -> None:
    body = _get(env, "/api/traces/config").json()
    assert body["trace_root"] == str(env["root"].resolve())
    assert body["exists"] is True and body["viewer_only"] is True and body["expose"] is False
    assert body["runner_roots"] == [str(env["legacy"])]


def test_exposed_config_keeps_paths_for_trial_video_lookup(env: dict[str, Any]) -> None:
    app = create_app(trace_root=env["root"], expose_traces=True,
                     runner_roots=[env["legacy"]], frontend_dist=env["base"] / "dist")
    client = TestClient(app)
    config = client.get("/api/traces/config").json()
    assert config["trace_root"] == str(env["root"].resolve())
    assert config["runner_roots"] == [str(env["legacy"])]
    roots = client.get("/api/traces/legacy/runs").json()["roots"]
    assert roots[0]["path"] == str(env["legacy"])
    from pathlib import Path
    detail = client.get(f"/api/traces/legacy/trials/smoke_run/cube_lift/S2/{TRIAL_DIR}").json()
    trial_dir = Path(detail["trial_record"]["artifacts"]["trial_dir"])
    assert trial_dir.is_relative_to(Path(config["runner_roots"][0]))
    video = client.get(f"/api/traces/legacy/file/smoke_run/cube_lift/S2/{TRIAL_DIR}/video_combined.mp4")
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"


def test_runs_list(env: dict[str, Any]) -> None:
    resp = _get(env, "/api/traces/runs")
    assert resp.status_code == 200 and resp.headers["x-content-type-options"] == "nosniff"
    runs = {r["run_id"]: r for r in resp.json()["runs"]}
    assert set(runs) == {RUN_ID, "trap_run"}
    demo = runs[RUN_ID]
    assert demo["n_episodes"] == 3
    assert sum(demo["status_counts"].values()) == 3 and demo["status_counts"]["ended_with_gaps"] == 1
    assert SECRET not in resp.text


def test_run_detail(env: dict[str, Any]) -> None:
    resp = _get(env, f"/api/traces/runs/{RUN_ID}")
    body = _strict_json(resp.content)
    assert body["run_id"] == RUN_ID and body["run"]["synthetic"] is True
    assert [e["episode_id"] for e in body["episodes"]] == [EPISODE_ENDED, EPISODE_INCOMPLETE, EPISODE_RUNNING]
    assert len(body["runner_records"]) == 2 and body["runner_records_info"]["corrupt_lines"] == 0
    assert body["unscoped_files"][0]["file"] == "unscoped.412300.jsonl"
    assert SECRET not in resp.text


def test_episode_summary(env: dict[str, Any]) -> None:
    ended = _get(env, _base()).json()
    assert ended["status"] == "ended_with_gaps" and ended["task"] == "cube_stack" and ended["seed"] == 1
    assert ended["trace_complete"] is False and ended["end_reason"] == "completed"
    incomplete = _get(env, _base(EPISODE_INCOMPLETE)).json()
    assert incomplete["status"] == "incomplete" and incomplete["status_detail"].startswith("no episode_end")
    web = _get(env, _base(EPISODE_RUNNING)).json()
    assert web["source"] == "web_ui" and web["viser_port"] == 8081


@pytest.mark.parametrize("episode", [EPISODE_ENDED, EPISODE_RUNNING, EPISODE_INCOMPLETE])
@pytest.mark.parametrize(("max_bytes", "max_events"), [(2_000_000, 5000), (3000, 5000), (2_000_000, 7)])
def test_events_paging_by_offset(env: dict[str, Any], episode: str, max_bytes: int, max_events: int) -> None:
    expected = [e["event_id"] for e in _events(env, episode)]
    offset, got, torn, corrupt, pages = 0, [], False, 0, 0
    while True:
        resp = _get(env, f"{_base(episode)}/events", params={"after_offset": offset, "max_bytes": max_bytes, "max_events": max_events})
        assert resp.status_code == 200 and resp.headers["cache-control"] == "no-store"
        page = _strict_json(resp.content)
        got += [e["event_id"] for e in page["events"]]
        corrupt += page["corrupt_lines"]
        torn = page["torn_tail"]
        pages += 1
        if page["eof"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert got == expected
    if episode == EPISODE_INCOMPLETE:
        assert torn and corrupt == 1
    else:
        assert not torn and corrupt == 0
    if max_bytes == 3000 or max_events == 7:
        assert pages > 1


def test_events_live_append_is_seen_by_the_next_poll(env: dict[str, Any]) -> None:
    ep_dir = env["root"] / RUN_ID / EPISODE_RUNNING
    first = _get(env, f"{_base(EPISODE_RUNNING)}/events").json()
    assert first["eof"]
    line = json.dumps({"seq": 10**6, "event_id": "ev_live", "event_type": "frame", "data": {}}).encode() + b"\n"
    with open(ep_dir / "events.jsonl", "ab") as fh:
        fh.write(line[:10])
    torn = _get(env, f"{_base(EPISODE_RUNNING)}/events", params={"after_offset": first["next_offset"]}).json()
    assert torn["events"] == [] and torn["torn_tail"] and torn["next_offset"] == first["next_offset"]
    with open(ep_dir / "events.jsonl", "ab") as fh:
        fh.write(line[10:])
    done = _get(env, f"{_base(EPISODE_RUNNING)}/events", params={"after_offset": first["next_offset"]}).json()
    assert [e["event_id"] for e in done["events"]] == ["ev_live"] and not done["torn_tail"]


@pytest.mark.parametrize("offset", [1, 17, 10**9])
def test_events_bad_offset_is_409(env: dict[str, Any], offset: int) -> None:
    resp = _get(env, f"{_base()}/events", params={"after_offset": offset})
    assert resp.status_code == 409


def test_events_query_validation(env: dict[str, Any]) -> None:
    assert _get(env, f"{_base()}/events", params={"after_offset": -1}).status_code == 422
    assert _get(env, f"{_base()}/events", params={"max_bytes": 0}).status_code == 422
    huge = _get(env, f"{_base()}/events", params={"max_bytes": 10**12, "max_events": 10**9})
    assert huge.status_code == 200 and huge.json()["eof"]


def test_file_types_and_headers(env: dict[str, Any]) -> None:
    events = _events(env, EPISODE_ENDED)
    request = next(e for e in events if e["event_type"] == "model_request")
    payload = _get(env, f"{_base()}/file/{request['data']['payload_path']}")
    assert payload.status_code == 200 and payload.headers["content-type"] == "application/json"
    assert payload.headers["x-content-type-options"] == "nosniff" and payload.headers["content-security-policy"] == "sandbox"
    assert payload.headers["cache-control"] == "no-cache"  # revalidated, never served stale
    image = request["data"]["images"][0]["asset"]
    png = _get(env, f"{_base()}/file/{image}")
    assert png.headers["content-type"] == "image/png" and png.content.startswith(b"\x89PNG")
    frame = next(e for e in events if e["event_type"] == "frame" and e["artifacts"][0]["written"])
    jpg = _get(env, f"{_base()}/file/{frame['artifacts'][0]['path']}")
    assert jpg.headers["content-type"] == "image/jpeg"
    stdout = next(e for e in events if e["event_type"] == "stdout" and isinstance(e["data"]["text"], dict))
    text = _get(env, f"{_base()}/file/{stdout['data']['text']['asset']}")
    assert text.headers["content-encoding"] == "gzip" and text.headers["content-type"].startswith("text/plain")
    assert len(text.text) == stdout["data"]["text"]["length"] and text.text.startswith(stdout["data"]["text"]["preview"][:100])
    state = next(e for e in events if e["event_type"] == "code_block_start")["data"]["state_ref"]["path"]
    assert "qpos" in _get(env, f"{_base()}/file/{state}").json()
    npy = next(a for e in events for a in e["artifacts"] if a["path"].endswith(".npy"))
    download = _get(env, f"{_base()}/file/{npy['path']}")
    assert download.headers["content-type"] == "application/octet-stream" and download.headers["content-disposition"] == "attachment"
    video_ep = _events(env, EPISODE_RUNNING)
    video = next(i for e in video_ep if e["event_type"] == "model_request" for i in e["data"]["images"] if i["mime"] == "video/mp4")
    mp4 = _get(env, _base(EPISODE_RUNNING) + f"/file/{video['asset']}")
    assert mp4.status_code == 200 and mp4.headers["content-type"] == "video/mp4"
    real_png = sorted((env["root"] / RUN_ID / EPISODE_ENDED / "assets").rglob("*.png"))[0]
    rel_png = real_png.relative_to(env["root"] / RUN_ID / EPISODE_ENDED).as_posix()
    assert _get(env, f"{_base()}/file/{rel_png}").content == real_png.read_bytes()
    # no symlink below the trace root is followed, not even one that stays inside
    alias = _get(env, f"{_base()}/file/assets/alias.png")
    assert alias.status_code == 403 and "symlink" in alias.json()["detail"]
    assert _get(env, f"{_base()}/file/assets/no_such_file.png").status_code == 404
    assert _get(env, f"{_base()}/file/assets").status_code == 404  # a directory is not a regular file


def test_array_npy_and_npz(env: dict[str, Any]) -> None:
    events = _events(env, EPISODE_ENDED)
    obs = next(e for e in events if e["event_type"] == "api_call_end" and e["data"].get("return", {}).get("type") == "dict")
    rgb = obs["data"]["return"]["items"]["robot0_robotview"]["items"]["images"]["items"]["rgb"]
    body = _strict_json(_get(env, f"{_base()}/array/{rgb['path']}").content)
    info = body["arrays"]["array"]
    assert info["shape"] == [128, 128, 3] and info["dtype"] == "uint8" and not info["is_mask"]
    assert info["preview_shape"] == [4, 16, 3] and info["truncated"]
    stored = np.load(env["root"] / RUN_ID / EPISODE_ENDED / rgb["path"])
    assert np.array_equal(np.asarray(info["preview"], dtype=np.uint8), stored[:4, :16, :3])
    # statistics were recorded in the event at capture time; the endpoint does not recompute them
    assert info["stats"] is None and rgb["stats"]["min"] is not None
    assert info["image"] == {"kind": "rgb", "count": 1, "height": 128, "width": 128}
    sam3 = next(e for e in events if e["event_type"] == "service_call_end" and e["artifacts"])
    npz = sam3["artifacts"][0]["path"]
    arrays = _strict_json(_get(env, f"{_base()}/array/{npz}").content)["arrays"]
    assert set(arrays) == {"a0", "a1"}  # the two SAM3 masks, bit-packed in the npz
    assert arrays["a0"]["is_mask"] and arrays["a0"]["packed"] and arrays["a0"]["shape"] == [128, 128]
    node = sam3["data"]["response_summary"]["items"][0]["items"]["mask"]
    assert node["npz"] == npz and node["npz_key"] == "a0__packbits" and node["is_mask"]
    by_key = _get(env, f"{_base()}/array/{npz}", params={"name": node["npz_key"]}).json()["arrays"]
    assert set(by_key) == {"a0__packbits"} and by_key["a0__packbits"]["shape"] == [128, 128]
    assert by_key["a0__packbits"]["stats"] is None and node["stats"] is not None  # the recorded ones
    assert _get(env, f"{_base()}/array/{npz}", params={"name": "nope"}).status_code == 404


def test_array_image(env: dict[str, Any]) -> None:
    events = _events(env, EPISODE_ENDED)
    npz = next(e for e in events if e["event_type"] == "service_call_end" and e["artifacts"])["artifacts"][0]["path"]
    resp = _get(env, f"{_base()}/array-image/{npz}", params={"name": "a1__packbits"})
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (128, 128) and img.mode == "L" and set(np.unique(np.asarray(img))) <= {0, 255}
    assert _get(env, f"{_base()}/array-image/{npz}").status_code == 400  # several arrays need a name
    stack = "/api/traces/runs/trap_run/episodes/arrays_ep/array-image/arrays/stack.npz"
    assert Image.open(io.BytesIO(_get(env, stack, params={"name": "m", "index": 2}).content)).size == (7, 5)
    assert _get(env, stack, params={"name": "m", "index": 5}).status_code == 400
    assert _get(env, stack, params={"name": "v"}).status_code == 400  # 1-D
    obs = next(e for e in events if e["event_type"] == "api_call_end" and e["data"].get("return", {}).get("type") == "dict")
    depth = obs["data"]["return"]["items"]["robot0_robotview"]["items"]["images"]["items"]["depth"]["path"]
    depth_png = Image.open(io.BytesIO(_get(env, f"{_base()}/array-image/{depth}").content))
    assert depth_png.size == (128, 128)
    arrays = f"/api/traces/runs/trap_run/episodes/arrays_ep/array-image/arrays"
    wide = Image.open(io.BytesIO(_get(env, f"{arrays}/huge_side.npy").content))
    assert max(wide.size) <= trace_arrays.MAX_IMAGE_SIDE


def test_npz_with_many_members(env: dict[str, Any]) -> None:
    url = "/api/traces/runs/trap_run/episodes/arrays_ep"
    listing = _get(env, f"{url}/array/arrays/sam3_masks.npz")
    assert listing.status_code == 200, listing.text
    assert len(listing.json()["arrays"]) == 50 and listing.json()["arrays"]["a49"]["packed"]
    one = _get(env, f"{url}/array/arrays/sam3_masks.npz", params={"name": "a49__packbits"}).json()["arrays"]
    assert set(one) == {"a49__packbits"} and one["a49__packbits"]["shape"] == [32, 40]
    img = Image.open(io.BytesIO(_get(env, f"{url}/array-image/arrays/sam3_masks.npz", params={"name": "a7"}).content))
    assert img.size == (40, 32) and np.asarray(img)[7].min() == 255 and np.asarray(img)[8].max() == 0


UNSCOPED = "unscoped.412300"


def test_unscoped_log_is_served_like_an_episode(env: dict[str, Any]) -> None:
    run_dir = env["root"] / RUN_ID
    detail = _get(env, f"/api/traces/runs/{RUN_ID}").json()
    assert detail["unscoped_files"][0]["log_id"] == UNSCOPED
    summary = _get(env, _base(UNSCOPED)).json()
    assert summary["kind"] == "unscoped" and summary["status"] == "unscoped" and summary["episode_id"] == UNSCOPED
    assert summary["files_dir"] == f"{UNSCOPED}.files" and summary["pid"] == 412300
    page = _get(env, f"{_base(UNSCOPED)}/events").json()
    assert [e["event_type"] for e in page["events"]] == ["model_request", "model_response"] and page["eof"]
    request = page["events"][0]
    payload_path = request["data"]["payload_path"]
    assert payload_path.startswith(f"{UNSCOPED}.files/payloads/")
    payload = _get(env, f"{_base(UNSCOPED)}/file/{payload_path}")
    assert payload.status_code == 200 and payload.headers["content-type"] == "application/json"
    assert payload.json()["messages"][0]["content"][0]["text"] == "Reply with the single word OK."
    response = page["events"][1]["data"]["payload_path"]
    assert _get(env, f"{_base(UNSCOPED)}/file/{response}").json()["choices"][0]["message"]["content"] == "OK"
    # only unscoped.<pid>.files/ is served: not the run's other files, not an episode's files
    for rel in ("run.json", "runner_records.jsonl", f"{UNSCOPED}.jsonl", f"{UNSCOPED}.files",
                f"{EPISODE_ENDED}/episode.json", "unscoped.9.files/payloads/x.json"):
        assert _get(env, f"{_base(UNSCOPED)}/file/{rel}").status_code == 400, rel
    assert _get(env, f"{_base(UNSCOPED)}/file/{UNSCOPED}.files/payloads/missing.json").status_code == 404
    leak = _get(env, f"{_base(UNSCOPED)}/file/{UNSCOPED}.files/leak.txt")
    assert leak.status_code in (403, 404) and SECRET not in leak.text
    # a log that does not exist, and ids that only look like one
    assert _get(env, _base("unscoped.999")).status_code == 404
    assert _get(env, f"{_base('unscoped.999')}/events").status_code == 404
    assert _get(env, _base("unscoped.x")).status_code == 404
    assert (run_dir / f"{UNSCOPED}.jsonl").is_file()


@pytest.mark.parametrize(
    "name",
    ["object.npy", "structured.npy", "complex.npy", "pickle.npy", "corrupt.npz", "odd_member.npz"],
)
def test_arrays_that_are_refused(env: dict[str, Any], name: str) -> None:
    resp = _get(env, f"/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays/{name}")
    assert resp.status_code == 400, resp.text


@pytest.mark.parametrize(
    ("name", "check"),
    [
        ("empty.npy", lambda a: a["array"]["stats"] is None and a["array"]["preview"] == []),
        ("nonfinite.npy", lambda a: a["array"]["preview"] == [["nan", "inf"], ["-inf", 1e308]]),
        ("all_nan.npy", lambda a: a["array"]["preview"][0] == ["nan"] * 5),
        ("fortran.npy", lambda a: a["array"]["preview"] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        ("big_endian.npy", lambda a: a["array"]["preview"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("scalar.npy", lambda a: a["array"]["preview"] == 2.5 and a["array"]["shape"] == []),
        ("two.npz", lambda a: set(a) == {"a", "b"} and a["b"]["preview"] == [[1.0] * 3] * 3),
    ],
)
def test_array_edge_cases_never_warn(env: dict[str, Any], name: str, check: Any, capfd: pytest.CaptureFixture[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resp = _get(env, f"/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays/{name}")
    assert resp.status_code == 200, resp.text
    assert check(_strict_json(resp.content)["arrays"])
    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""


def test_array_size_limits(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    url = "/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays"
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_FILE_BYTES", 100)
    assert _get(env, f"{url}/depth.npy").status_code == 400
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_FILE_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_DECOMPRESSED_BYTES", 50)
    assert _get(env, f"{url}/two.npz").status_code == 400
    assert _get(env, f"{url}/depth.npy").status_code == 400
    monkeypatch.setattr(trace_arrays, "MAX_ARRAY_DECOMPRESSED_BYTES", 256 * 1024 * 1024)
    monkeypatch.setattr(trace_arrays, "MAX_NPZ_MEMBERS", 1)
    assert _get(env, f"{url}/two.npz").status_code == 400
    monkeypatch.setattr(trace_arrays, "MAX_NPZ_MEMBERS", 64)
    body = _get(env, f"{url}/depth.npy").json()["arrays"]["array"]
    assert np.asarray(body["preview"]).size <= trace_arrays.MAX_PREVIEW_VALUES


def test_viser_in_viewer_only_mode(env: dict[str, Any]) -> None:
    assert _get(env, "/api/traces/viser").json() == {"available": False, "port": None, "reason": "viewer_only"}


def test_viser_in_full_mode_uses_the_probe(tmp_path: Path) -> None:
    """The probe is given this server's own port, so a web UI started on
    a Viser port (8080/8081) does not report itself as the Viser server."""
    seen: list[int | None] = []
    app = FastAPI()
    app.include_router(trace_api.create_trace_router(
        tmp_path, viewer_only=False, expose=False, viser_probe=lambda own_port: seen.append(own_port) or 8081))
    assert TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL).get("/api/traces/viser").json()["port"] == 8081
    assert seen == [8765]  # the port of LOOPBACK_BASE_URL
    on_8081 = TestClient(app, client=LOCAL, base_url="http://localhost:8081").get("/api/traces/viser").json()
    assert on_8081 == {"available": False, "port": None, "reason": "not_running"}  # never itself

    def broken(own_port: int | None) -> int:
        raise OSError("probe failed")

    app2 = FastAPI()
    app2.include_router(trace_api.create_trace_router(tmp_path, viewer_only=False, expose=False, viser_probe=broken))
    assert TestClient(app2, client=LOCAL, base_url=LOOPBACK_BASE_URL).get("/api/traces/viser").json()["available"] is False


def test_missing_trace_root(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(trace_api.create_trace_router(tmp_path / "nope", viewer_only=True, expose=False))
    client = TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL)
    assert client.get("/api/traces/config").json()["exists"] is False
    assert client.get("/api/traces/runs").json() == {"runs": []}
    assert client.get("/api/traces/runs/r1").status_code == 404


@pytest.mark.parametrize("path", ["/api/traces", "/api/traces/", "/api/traces/nope", "/api/traces/runs/a/b/c/d", "/api/traces/legacy"])
def test_unknown_trace_paths_are_json_404(env: dict[str, Any], path: str) -> None:
    resp = _get(env, path)
    assert resp.status_code == 404 and resp.headers["content-type"] == "application/json"


def test_episode_without_events_file_gives_an_empty_page(env: dict[str, Any]) -> None:
    body = _get(env, "/api/traces/runs/trap_run/episodes/arrays_ep/events").json()
    assert body["events"] == [] and body["eof"] and body["next_offset"] == 0
    assert _get(env, "/api/traces/runs/trap_run/episodes/arrays_ep/events", params={"after_offset": 5}).status_code == 409


# ---------------------------------------------------------------------------
# loopback gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1", True),
        ("127.3.4.5", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("testclient", False),  # the TestClient default
        ("10.0.0.5", False),
        ("192.168.1.20", False),
        ("::ffff:10.0.0.5", False),
        ("0.0.0.0", False),
        ("", False),
    ],
)
def test_loopback_gating(env: dict[str, Any], host: str, allowed: bool) -> None:
    client = TestClient(env["app"], client=(host, 40000), base_url=LOOPBACK_BASE_URL)
    for url in ("/api/traces/config", "/api/traces/runs", f"{_base()}/events", "/api/traces/legacy/runs", "/api/traces/nope"):
        resp = client.get(url)
        if allowed:
            assert resp.status_code in (200, 404), (url, resp.status_code)
        else:
            assert resp.status_code == 403, (url, resp.status_code)
            assert "loopback" in resp.json()["detail"]


def test_expose_answers_remote_clients(env: dict[str, Any]) -> None:
    app = create_app(viewer_only=True, trace_root=env["root"], expose_traces=True, runner_roots=[env["legacy"]], frontend_dist=env["base"] / "dist")
    client = TestClient(app, client=("10.0.0.5", 40000), base_url="http://192.168.1.10:8765")
    assert client.get("/api/traces/runs").status_code == 200
    assert client.get("/api/traces/config").json()["expose"] is True


def test_client_is_none_is_refused(env: dict[str, Any]) -> None:
    status, _, _ = raw_get(env["app"], "/api/traces/config", client=None)  # type: ignore[arg-type]
    assert status == 403


# ---------------------------------------------------------------------------
# path traversal on every endpoint
# ---------------------------------------------------------------------------

TRACE_TEMPLATES = [
    "/api/traces/runs/{run}",
    "/api/traces/runs/{run}/episodes/{ep}",
    "/api/traces/runs/{run}/episodes/{ep}/events",
    "/api/traces/runs/{run}/episodes/{ep}/file/{rel}",
    "/api/traces/runs/{run}/episodes/{ep}/array/{rel}",
    "/api/traces/runs/{run}/episodes/{ep}/array-image/{rel}",
]
LEGACY_TEMPLATES = [
    "/api/traces/legacy/runs/{run}",
    "/api/traces/legacy/trials/{run}/{task}/{cond}/{trial}",
    "/api/traces/legacy/file/{run}/{task}/{cond}/{trial}/{rel}",
]
TRACE_VALID = {"run": RUN_ID, "ep": EPISODE_ENDED, "rel": "episode.json"}
# the run's unscoped log, opened under the episode id unscoped.<pid> (file paths below unscoped.<pid>.files/)
UNSCOPED_TEMPLATES = TRACE_TEMPLATES[3:]
UNSCOPED_VALID = {"rel": "unscoped.412300.files/payloads/ev_6f851dc660025ec06e88.request.json"}
UNSCOPED_REL_ATTACKS = [
    "unscoped.412300.files/%2E%2E/run.json", "unscoped.412300.files/..%2F..%2F..%2F.env",
    "unscoped.412300.files%2F..%2Frun.json", "unscoped.412300.files/leak.txt", "unscoped.412300.files/../../secret.txt",
    "unscoped.412300.files/./payloads", "unscoped.412300.files//payloads", "../unscoped.412300.files/leak.txt",
]
LEGACY_VALID = {"run": "smoke_run", "task": "cube_lift", "cond": "S2", "trial": TRIAL_DIR, "rel": "code.py"}

# URL-encoded forms: the server decodes %2E and %2F before routing, as uvicorn does.
ENCODED_ID_ATTACKS = [
    "%2E", "%2E%2E", "%2e%2e", "..%2F..", "%2E%2E%2F%2E%2E%2F%2E%2E", "a%2F..%2F..%2F..", "%2F..%2F..",
    "%00", "a%00b", "%0A", "a%0Ab", "..%5C..", "%2E%2E%5C", "~", "a%20b",
]
ENCODED_REL_ATTACKS = [
    "..%2F..%2F..%2F..%2F.env", "%2E%2E/%2E%2E/%2E%2E/%2E%2E/.env", "assets/%2E%2E/%2E%2E/%2E%2E/%2E%2E/%2E%2E/.env",
    "%2Fetc%2Fpasswd", "assets%2F..%2F..%2F..%2F..%2Fsecret.txt", "assets/%2E/alias.png", "assets//alias.png",
    "assets/leak.txt", "assets/leak.npy", "leak.txt", "%00", "assets/alias.png%00.txt",
]
# Decoded forms sent as-is (uvicorn passes these through without removing dot segments).
LITERAL_ID_ATTACKS = [".", "..", "...", "a/../..", "../..", "..\\..", "\x00", "a\nb"]
LITERAL_REL_ATTACKS = [
    "../../../../.env", "../../../../../.env", "assets/../../../../../.env", "/etc/passwd", "assets/./alias.png",
    "./assets/alias.png", "assets//alias.png", "assets/../payloads", "..", ".", "",
]


def _cases(templates: list[str], valid: dict[str, str], id_attacks: list[str], rel_attacks: list[str]) -> list[tuple[str, str]]:
    cases = []
    for template in templates:
        for key in valid:
            if "{" + key + "}" not in template:
                continue
            for attack in rel_attacks if key == "rel" else id_attacks:
                values = dict(valid, **{key: attack})
                cases.append((template, template.format(**values)))
    return cases


def _assert_refused(status: int, body: bytes, url: str) -> None:
    assert status in (400, 403, 404), (url, status, body[:200])
    assert SECRET.encode() not in body and str(SECRET_NUMBER).encode() not in body, url
    assert b"<html>" not in body, url  # never the SPA page for an API path


@pytest.mark.parametrize(
    ("template", "url"),
    _cases(TRACE_TEMPLATES, TRACE_VALID, ENCODED_ID_ATTACKS, ENCODED_REL_ATTACKS)
    + _cases(LEGACY_TEMPLATES, LEGACY_VALID, ENCODED_ID_ATTACKS, ENCODED_REL_ATTACKS),
)
def test_encoded_traversal_is_refused(env: dict[str, Any], template: str, url: str) -> None:
    resp = _get(env, url)
    _assert_refused(resp.status_code, resp.content, url)


@pytest.mark.parametrize(
    ("template", "url"),
    _cases(TRACE_TEMPLATES, TRACE_VALID, LITERAL_ID_ATTACKS, LITERAL_REL_ATTACKS)
    + _cases(LEGACY_TEMPLATES, LEGACY_VALID, LITERAL_ID_ATTACKS, LITERAL_REL_ATTACKS),
)
def test_literal_traversal_is_refused(env: dict[str, Any], template: str, url: str) -> None:
    status, body, _ = raw_get(env["app"], url)
    _assert_refused(status, body, url)


@pytest.mark.parametrize(
    "url",
    [
        template.format(run=RUN_ID, ep="unscoped.412300", rel=attack)
        for template in UNSCOPED_TEMPLATES
        for attack in UNSCOPED_REL_ATTACKS + ENCODED_REL_ATTACKS
    ],
)
def test_traversal_through_the_unscoped_log_is_refused(env: dict[str, Any], url: str) -> None:
    resp = _get(env, url)
    _assert_refused(resp.status_code, resp.content, url)
    status, body, _ = raw_get(env["app"], url.replace("%2E", ".").replace("%2F", "/"))
    _assert_refused(status, body, url)


def test_the_valid_unscoped_file_url_works(env: dict[str, Any]) -> None:
    url = UNSCOPED_TEMPLATES[0].format(run=RUN_ID, ep="unscoped.412300", **UNSCOPED_VALID)
    status, body, headers = raw_get(env["app"], url)
    assert status == 200 and headers["content-type"] == "application/json", (url, status, body[:200])


def test_the_valid_forms_of_the_traversal_templates_work(env: dict[str, Any]) -> None:
    """The attack cases above differ from these working URLs in one segment only."""
    for template in TRACE_TEMPLATES:
        url = template.format(**TRACE_VALID)
        status, body, _ = raw_get(env["app"], url)
        expected = 400 if "/array" in url else 200  # episode.json is not an array file
        assert status == expected, (url, status, body[:200])
    for template in LEGACY_TEMPLATES:
        status, body, _ = raw_get(env["app"], template.format(**LEGACY_VALID))
        assert status == 200, (template, status, body[:200])


@pytest.mark.parametrize(
    "url",
    [
        "/api/traces/runs/escape_run",
        "/api/traces/runs/escape_run/episodes/ep1",
        "/api/traces/runs/escape_run/episodes/ep1/events",
        "/api/traces/runs/escape_run/episodes/ep1/file/secret.txt",
        "/api/traces/runs/escape_run/episodes/ep1/array/secret.npy",
        "/api/traces/runs/escape_run/episodes/ep1/array-image/secret.npy",
        f"/api/traces/runs/{RUN_ID}/episodes/escape_ep",
        f"/api/traces/runs/{RUN_ID}/episodes/escape_ep/events",
        f"/api/traces/runs/{RUN_ID}/episodes/escape_ep/file/secret.txt",
        f"/api/traces/runs/{RUN_ID}/episodes/escape_ep/array/secret.npy",
        f"/api/traces/runs/{RUN_ID}/episodes/escape_ep/array-image/secret.npy",
        f"{_base()}/file/assets/leak.txt",
        f"{_base()}/array/assets/leak.npy",
        f"{_base()}/array-image/assets/leak.npy",
        "/api/traces/legacy/runs/escape_run",
        f"/api/traces/legacy/trials/escape_run/cube_lift/S2/{TRIAL_DIR}",
        f"/api/traces/legacy/file/escape_run/cube_lift/S2/{TRIAL_DIR}/code.py",
        f"/api/traces/legacy/trials/smoke_run/cube_lift/escape_cond/{TRIAL_DIR}",
        f"/api/traces/legacy/file/smoke_run/cube_lift/escape_cond/{TRIAL_DIR}/code.py",
        f"/api/traces/legacy/file/smoke_run/cube_lift/S2/{TRIAL_DIR}/leak.txt",
    ],
)
def test_symlinks_leading_outside_are_refused(env: dict[str, Any], url: str) -> None:
    resp = _get(env, url)
    _assert_refused(resp.status_code, resp.content, url)
    assert resp.status_code in (403, 404)


def test_symlinked_fixed_files_are_ignored(env: dict[str, Any]) -> None:
    summary = _get(env, "/api/traces/runs/trap_run/episodes/trap_episode_json")
    assert summary.status_code == 200 and SECRET not in summary.text and summary.json()["episode"] is None
    assert summary.json()["episode_json_error"] == "symlink_not_followed"
    # A symlinked events.jsonl is an explicit error, not "not written yet"
    events = _get(env, "/api/traces/runs/trap_run/episodes/trap_events/events")
    assert events.status_code == 403 and "symlink" in events.json()["detail"] and SECRET not in events.text
    detail = _get(env, "/api/traces/runs/trap_run")
    assert SECRET not in detail.text
    for resp in (_get(env, "/api/traces/runs"), _get(env, f"/api/traces/runs/{RUN_ID}"), _get(env, "/api/traces/legacy/runs")):
        assert SECRET not in resp.text


# ---------------------------------------------------------------------------
# legacy runner outputs (section 9.4)
# ---------------------------------------------------------------------------


def test_legacy_runs(env: dict[str, Any]) -> None:
    body = _get(env, "/api/traces/legacy/runs").json()
    assert body["roots"] == [{"index": 0, "path": str(env["legacy"]), "exists": True}]
    assert [r["run_id"] for r in body["runs"]] == ["smoke_run"]
    run = body["runs"][0]
    assert run["description"] == "legacy test run" and run["n_trials"] == 1 and run["root_index"] == 0


def test_legacy_run_detail(env: dict[str, Any]) -> None:
    body = _get(env, "/api/traces/legacy/runs/smoke_run").json()
    assert body["run"]["run_id"] == "smoke_run" and body["records_info"]["corrupt_lines"] == 1
    (trial,) = body["trials"]
    assert trial["task"] == "cube_lift" and trial["condition"] == "S2" and trial["trial_dir"] == TRIAL_DIR
    assert trial["trial"] == 1 and trial["sandbox_rc"] == 0 and trial["reward"] == 1.0 and trial["task_completed"] is True
    assert trial["record"]["outcome"]["task_success"] is True and trial["record"]["attempt_statuses"] == ["completed"]
    assert _get(env, "/api/traces/legacy/runs/smoke_run", params={"root": 0}).status_code == 200
    assert _get(env, "/api/traces/legacy/runs/smoke_run", params={"root": 3}).status_code == 400
    assert _get(env, "/api/traces/legacy/runs/no_such_run").status_code == 404


def test_legacy_trial_view(env: dict[str, Any]) -> None:
    body = _get(env, f"/api/traces/legacy/trials/smoke_run/cube_lift/S2/{TRIAL_DIR}").json()
    assert body["model_input_recorded"] is False
    assert body["code"]["text"] == "print('hello')\n"
    summary = body["summary"]
    assert summary["parsed"] and summary["stdout"] == "hello\nworld\n"
    assert summary["stderr"].endswith("ValueError: boom\n") and summary["reward"] == 0.536
    assert summary["task_completed"] is False and summary["terminated"] is True and summary["num_code_blocks"] == 3
    assert body["trial_record"]["sandbox_rc"] == 0
    assert [v["path"] for v in body["videos"]] == ["video_combined.mp4"]
    (image,) = body["harness_images"]
    assert image["path"] == "visual_feedback_00.png" and image["label_ja"] == "ハーネス保存画像（送信を検証していない）"
    (prompt,) = body["harness_prompts"]
    assert prompt["path"] == "prompts_and_responses/initial_prompt.txt"
    assert prompt["label_ja"] == "ハーネスが保存したプロンプト文字列（送信payloadではない）"
    assert [f["path"] for f in body["other_files"]] == ["all_responses.json"]
    assert "leak.txt" not in json.dumps(body)


def test_legacy_file(env: dict[str, Any]) -> None:
    base = f"/api/traces/legacy/file/smoke_run/cube_lift/S2/{TRIAL_DIR}"
    assert _get(env, f"{base}/code.py").text == "print('hello')\n"
    video = _get(env, f"{base}/video_combined.mp4")
    assert video.headers["content-type"] == "video/mp4" and video.headers["x-content-type-options"] == "nosniff"
    assert _get(env, f"{base}/prompts_and_responses/initial_prompt.txt").text == "harness prompt text"
    assert _get(env, f"{base}/missing.txt").status_code == 404


def test_parse_summary_txt_on_unexpected_text() -> None:
    assert trace_api.parse_summary_txt("nothing here")["parsed"] is False
    tricky = SUMMARY_TXT.replace("  Stdout: hello\nworld\n", "  Stdout: a\n  Stderr: fake\n")
    parsed = trace_api.parse_summary_txt(tricky)
    assert parsed["parsed"] and parsed["ambiguous_split"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (TRIAL_DIR, {"trial": 1, "sandbox_rc": 0, "reward": 1.0, "task_completed": True}),
        ("trial_203_sandboxrc_1_reward_0.536_taskcompleted_0", {"trial": 203, "sandbox_rc": 1, "reward": 0.536, "task_completed": False}),
        ("trial_02_sandboxrc_0_reward_nan_taskcompleted_0", {"trial": 2, "sandbox_rc": 0, "reward": "nan", "task_completed": False}),
        ("trial_x", {"trial": None, "sandbox_rc": None, "reward": None, "task_completed": None}),
    ],
)
def test_parse_trial_dir_name(name: str, expected: dict[str, Any]) -> None:
    assert trace_api.parse_trial_dir_name(name) == expected


# ---------------------------------------------------------------------------
# section 2: handlers print nothing and emit no warning
# ---------------------------------------------------------------------------


def test_handlers_emit_nothing_under_warnings_as_errors(env: dict[str, Any], capfd: pytest.CaptureFixture[str]) -> None:
    events = _events(env, EPISODE_ENDED)
    npz = next(e for e in events if e["event_type"] == "service_call_end" and e["artifacts"])["artifacts"][0]["path"]
    urls = [
        "/api/traces/config",
        "/api/traces/runs",
        f"/api/traces/runs/{RUN_ID}",
        _base(),
        f"{_base()}/events",
        f"{_base(EPISODE_INCOMPLETE)}/events",
        f"{_base()}/array/{npz}",
        f"{_base()}/array-image/{npz}?name=a0",
        "/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays/nonfinite.npy",
        "/api/traces/runs/trap_run/episodes/arrays_ep/array-image/arrays/all_nan.npy",
        "/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays/pickle.npy",
        "/api/traces/viser",
        "/api/traces/legacy/runs",
        "/api/traces/legacy/runs/smoke_run",
        f"/api/traces/legacy/trials/smoke_run/cube_lift/S2/{TRIAL_DIR}",
        _base(UNSCOPED),
        f"{_base(UNSCOPED)}/events",
        f"{_base(UNSCOPED)}/file/{UNSCOPED_VALID['rel']}",
        "/api/traces/runs/trap_run/episodes/arrays_ep/array/arrays/sam3_masks.npz",
        "/api/traces/runs/trap_run/episodes/arrays_ep/array-image/arrays/sam3_masks.npz?name=a3__packbits",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        statuses = [_get(env, url).status_code for url in urls]
    assert statuses == [200] * 10 + [400] + [200] * 9
    captured = capfd.readouterr()
    assert captured.out == "" and captured.err == ""


def test_json_responses_are_strict(env: dict[str, Any]) -> None:
    for url in ("/api/traces/runs", f"/api/traces/runs/{RUN_ID}", _base(), f"{_base()}/events", "/api/traces/legacy/runs/smoke_run"):
        _strict_json(_get(env, url).content)


def test_gz_text_decodes_to_the_stored_bytes(env: dict[str, Any]) -> None:
    events = _events(env, EPISODE_ENDED)
    stdout = next(e for e in events if e["event_type"] == "stdout" and isinstance(e["data"]["text"], dict))
    path = env["root"] / RUN_ID / EPISODE_ENDED / stdout["data"]["text"]["asset"]
    resp = _get(env, f"{_base()}/file/{stdout['data']['text']['asset']}")
    assert resp.content == gzip.decompress(path.read_bytes())


# ---------------------------------------------------------------------------
# small trace roots for the tests below (each test gets its own)
# ---------------------------------------------------------------------------


def _event_line(seq: int, prefix: str = "ev", **data: Any) -> bytes:
    """One event line; lines with the same ``seq`` and prefix length have the same length."""
    event = {"schema_version": "capx-trace/1", "event_id": f"{prefix}_{seq:08d}", "seq": seq, "event_type": "frame", "data": data}
    return json.dumps(event, separators=(",", ":")).encode() + b"\n"


def _small_app(tmp_path: Path, **kwargs: Any) -> tuple[Path, TestClient]:
    root = tmp_path / "traces"
    root.mkdir()
    app = create_app(viewer_only=True, trace_root=root, runner_roots=kwargs.pop("runner_roots", []), frontend_dist=tmp_path / "no_dist", **kwargs)
    return root, TestClient(app, client=LOCAL, base_url=LOOPBACK_BASE_URL)


# ---------------------------------------------------------------------------
# Arrays that used to hang or draw wrongly, through the endpoints
# ---------------------------------------------------------------------------


def test_six_d_array_preview_ends(tmp_path: Path) -> None:
    """uint8 (1, 4, 4, 4, 16, 16): the preview loop used to halve the axis of size 1 forever."""
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    arr = (np.arange(16384) % 251).astype(np.uint8).reshape(1, 4, 4, 4, 16, 16)
    np.save(ep / "assets" / "six.npy", arr)
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(client.get("/api/traces/runs/r1/episodes/ep1/array/assets/six.npy")), daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive(), "the /array request did not end"
    info = result[0].json()["arrays"]["array"]
    assert info["shape"] == [1, 4, 4, 4, 16, 16] and info["preview_shape"] == [1, 1, 4, 4, 16, 16]
    assert np.array_equal(np.asarray(info["preview"], dtype=np.uint8), arr[:, :1])


def test_bool_masks_of_shape_h_w_1_through_the_endpoint(tmp_path: Path) -> None:
    """ (H, W, 1) bool is one H x W mask; (N, H, W) bool stays a stack."""
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    seg = np.zeros((48, 64, 1), dtype=np.int32)
    seg[10:20, 5:15] = 3
    np.save(ep / "assets" / "mask.npy", seg == 3)
    stack = np.zeros((2, 3, 2), dtype=bool)
    stack[1, 0, 1] = True
    np.save(ep / "assets" / "stack.npy", stack)
    base = "/api/traces/runs/r1/episodes/ep1"
    info = client.get(f"{base}/array/assets/mask.npy").json()["arrays"]["array"]
    assert info["image"] == {"kind": "mask", "count": 1, "height": 48, "width": 64} and info["is_mask"]
    png = Image.open(io.BytesIO(client.get(f"{base}/array-image/assets/mask.npy").content))
    assert png.size == (64, 48) and np.array_equal(np.asarray(png) == 255, seg[:, :, 0] == 3)
    info = client.get(f"{base}/array/assets/stack.npy").json()["arrays"]["array"]
    assert info["image"] == {"kind": "mask_stack", "count": 2, "height": 3, "width": 2}
    png = Image.open(io.BytesIO(client.get(f"{base}/array-image/assets/stack.npy", params={"index": 1}).content))
    assert png.size == (2, 3) and np.asarray(png).tolist() == [[0, 255], [0, 0], [0, 0]]
    assert client.get(f"{base}/array-image/assets/stack.npy", params={"index": 2}).status_code == 400


def test_empty_array_with_many_axes_through_the_endpoint(tmp_path: Path) -> None:
    """Regression case: a 128-byte file np.empty((4,) * 8 + (0,)) made
    /array build 87,381 lists, even with the read budgets at one byte."""
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    np.save(ep / "assets" / "empty.npy", np.empty((4,) * 8 + (0,), dtype=np.uint8))

    def lists(value: Any) -> int:
        return 1 + sum(lists(v) for v in value) if isinstance(value, list) else 0

    resp = client.get("/api/traces/runs/r1/episodes/ep1/array/assets/empty.npy")
    info = resp.json()["arrays"]["array"]
    assert resp.status_code == 200 and info["shape"] == [4] * 8 + [0] and info["truncated"]
    assert lists(info["preview"]) <= trace_arrays.MAX_PREVIEW_LISTS and len(resp.content) < 64 * 1024


# ---------------------------------------------------------------------------
# status without a heartbeat, runs with a damaged run.json
# ---------------------------------------------------------------------------


def test_episode_without_a_heartbeat_file_is_incomplete(tmp_path: Path) -> None:
    """A copied crashed episode has fresh file mtimes; without a heartbeat it is not running."""
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    ep.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "running", "task": "t"}))
    (ep / "events.jsonl").write_bytes(_event_line(0))
    summary = client.get("/api/traces/runs/r1/episodes/ep1").json()
    assert summary["status"] == "incomplete" and summary["liveness_source"] == "none"
    assert summary["status_detail"] == "no episode_end and no heartbeat file (liveness unknown)"
    assert client.get("/api/traces/runs").json()["runs"][0]["status_counts"]["incomplete"] == 1
    (ep / "heartbeat").touch()
    assert client.get("/api/traces/runs/r1/episodes/ep1").json()["status"] == "running"


@pytest.mark.parametrize("content", [b"", b"\n", b'{"run_id": "r1", "created_at": "2026-09', b"[1, 2]"])
def test_run_with_an_empty_or_damaged_run_json_is_listed_and_detailed(tmp_path: Path, content: bytes) -> None:
    root, client = _small_app(tmp_path)
    (root / "r1").mkdir()
    (root / "r1" / "run.json").write_bytes(content)
    assert [r["run_id"] for r in client.get("/api/traces/runs").json()["runs"]] == ["r1"]
    detail = client.get("/api/traces/runs/r1")
    assert detail.status_code == 200, detail.text
    body = detail.json()
    if content == b"[1, 2]":  # valid JSON, but not the run object: shown as it is
        assert body["run"] == [1, 2] and body["run_json_error"] is None
    else:
        assert body["run"] is None and body["run_json_error"] == "invalid_json"
    assert body["episodes"] == [] and body["runner_records"] == []
    assert client.get("/api/traces/runs/r2").status_code == 404


# ---------------------------------------------------------------------------
# the Host header must name a loopback host (DNS rebinding, same-host proxies)
# ---------------------------------------------------------------------------

GATED_URLS = [
    "/api/traces/config",
    "/api/traces/runs",
    f"/api/traces/runs/{RUN_ID}",
    f"/api/traces/runs/{RUN_ID}/episodes/{EPISODE_ENDED}",
    f"/api/traces/runs/{RUN_ID}/episodes/{EPISODE_ENDED}/events",
    f"/api/traces/runs/{RUN_ID}/episodes/{EPISODE_ENDED}/file/episode.json",
    "/api/traces/legacy/runs",
    f"/api/traces/legacy/file/smoke_run/cube_lift/S2/{TRIAL_DIR}/code.py",
    "/api/traces/nope",
]


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1:8765", True),
        ("localhost:8765", True),
        ("LocalHost:8765", True),
        ("localhost.:8765", True),
        ("localhost", True),
        ("127.3.4.5", True),
        ("[::1]:8765", True),
        ("[::1]", True),
        ("[::ffff:127.0.0.1]:8765", True),
        ("localhost:9000", True),  # an SSH port forward to another local port
        ("rebind.attacker.example:8765", False),
        ("attacker.example", False),
        ("localhost.attacker.example:8765", False),
        ("127.0.0.1.nip.io:8765", False),
        ("[::2]:8765", False),
        ("[::1]x:8765", False),
        ("192.168.1.10:8765", False),
        # The full web UI prints http://0.0.0.0:<port>; a browser reaches
        # the unspecified address only on this machine
        ("0.0.0.0:8765", True),
        ("0.0.0.0", True),
        ("[::]:8765", True),
        ("0.0.0.1:8765", False),
        ("localhost:http", False),
        ("localhost:", False),
        ("", False),
    ],
)
def test_host_header_must_name_a_loopback_host(env: dict[str, Any], host: str, allowed: bool) -> None:
    headers = [(b"host", host.encode())] if host else []
    for url in GATED_URLS:
        status, body, _ = raw_get(env["app"], url, headers=headers)  # the TCP peer is 127.0.0.1
        if allowed:
            assert status in (200, 404), (host, url, status, body[:200])
        else:
            assert status == 403, (host, url, status)
            assert b"loopback" in body and SECRET.encode() not in body and b"cube_stack" not in body


def test_host_check_through_the_test_client(env: dict[str, Any]) -> None:
    rebind = TestClient(env["app"], client=LOCAL, base_url="http://rebind.attacker.example:8765")
    resp = rebind.get(f"/api/traces/runs/{RUN_ID}")
    assert resp.status_code == 403 and "Host header" in resp.json()["detail"]
    forward = TestClient(env["app"], client=LOCAL, base_url="http://localhost:9000")
    assert forward.get(f"/api/traces/runs/{RUN_ID}").status_code == 200
    # --expose-traces answers any host and any client
    app = create_app(viewer_only=True, trace_root=env["root"], expose_traces=True, frontend_dist=env["base"] / "dist")
    assert TestClient(app, client=LOCAL, base_url="http://rebind.attacker.example:8765").get("/api/traces/runs").status_code == 200


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        ("http://localhost:5173", True),
        ("http://127.0.0.1:8765", True),
        ("http://[::1]:8765", True),
        ("http://0.0.0.0:8200", True),
        ("http://[::]:8200", True),
        ("null", True),  # an opaque origin: the CORS policy lets it read nothing
        ("http://rebind.attacker.example:8765", False),
        ("https://attacker.example", False),
        ("http://localhost.attacker.example", False),
        ("not a url", False),
    ],
)
def test_origin_header_if_present_must_be_loopback(env: dict[str, Any], origin: str, allowed: bool) -> None:
    resp = _get(env, "/api/traces/runs", headers={"Origin": origin})
    assert resp.status_code == (200 if allowed else 403), (origin, resp.status_code)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True), ("localhost:1", True), ("127.0.0.1", True), ("127.255.255.254:80", True),
        ("[::1]:65535", True), ("::1", True), ("[::ffff:127.0.0.2]", True),
        ("0.0.0.0", True), ("0.0.0.0:80", True), ("[::]", True), ("[::]:8200", True), ("[::ffff:0.0.0.0]", True),
        (None, False), ("", False), (" ", False), ("[::1", False), ("[::1]:", False), ("[::1]:a", False),
        ("localhost:1:2", False), ("127.0.0.1:-1", False), ("128.0.0.1", False), ("0x7f.1", False),
        ("2130706433", False), ("localhost..", False), ("xn--localhost", False),
        ("0.0.0.1", False), ("0", False), ("[::2]", False), ("0.0.0.0.nip.io", False),
    ],
)
def test_host_is_loopback(host: str | None, expected: bool) -> None:
    assert trace_api.host_is_loopback(host) is expected


def test_the_tcp_peer_must_still_be_loopback_for_the_unspecified_host(env: dict[str, Any]) -> None:
    """Accepting Host 0.0.0.0 does not open the API: the peer check comes first."""
    status, body, _ = raw_get(env["app"], "/api/traces/runs", client=("192.168.1.20", 40000), headers=[(b"host", b"0.0.0.0:8200")])
    assert status == 403 and b"loopback clients only" in body
    assert not trace_api._is_loopback_ip("0.0.0.0") and not trace_api._is_loopback_ip("::")


# ---------------------------------------------------------------------------
# an escaped lone surrogate in a recorded string does not break a response
# ---------------------------------------------------------------------------


def test_lone_surrogates_in_recorded_json_are_served(tmp_path: Path) -> None:
    """The recorder writes ASCII JSON, so a lone surrogate is stored as the escape \\udcff."""
    legacy = tmp_path / "runner"
    trial = legacy / "run1" / "task" / "cond" / TRIAL_DIR
    trial.mkdir(parents=True)
    root, client = _small_app(tmp_path, runner_roots=[legacy])
    message = "failed to load mesh /data/obj_\udcff.ply"
    ep = root / "r1" / "bad_ep"
    ep.mkdir(parents=True)
    doc = {"status": "ended", "trace_complete": True, "task": "t\ud83d", "exception": {"type": "OSError", "message": message}}
    (ep / "episode.json").write_text(json.dumps(doc, ensure_ascii=True, indent=1))
    (ep / "events.jsonl").write_text(json.dumps({"seq": 0, "event_type": "episode_start", "data": {"note": "\udcff"}}, ensure_ascii=True) + "\n")
    (root / "r1" / "run.json").write_text(json.dumps({"run_id": "r1", "note": "\udc80"}, ensure_ascii=True))
    (root / "r1" / "runner_records.jsonl").write_text(json.dumps({"task": "t", "note": "\ud800"}, ensure_ascii=True) + "\n")
    (legacy / "run1" / "run.json").write_text(json.dumps({"run_id": "run1", "manifest": {"description": "\udcff"}}, ensure_ascii=True))
    (trial / "trial_record.json").write_text(json.dumps({"sandbox_rc": 0, "note": "\udcff"}, ensure_ascii=True))
    for url in (
        "/api/traces/runs",
        "/api/traces/runs/r1",
        "/api/traces/runs/r1/episodes/bad_ep",
        "/api/traces/runs/r1/episodes/bad_ep/events",
        "/api/traces/legacy/runs",
        "/api/traces/legacy/runs/run1",
        f"/api/traces/legacy/trials/run1/task/cond/{TRIAL_DIR}",
    ):
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.status_code, resp.text[:200])
        resp.content.decode("ascii")  # pure ASCII: every non-ASCII character is escaped
        _strict_json(resp.content)
    detail = client.get("/api/traces/runs/r1")
    assert b"\\udcff" in detail.content and b"\\ud83d" in detail.content
    assert json.loads(detail.content)["episodes"][0]["episode"]["exception"]["message"] == message
    assert json.loads(client.get("/api/traces/runs/r1/episodes/bad_ep").content)["task"] == "t\ud83d"


def test_json_response_is_ascii_for_any_string() -> None:
    resp = trace_api.json_response({"a": "\udcff é \U0001f600", "b": float("nan")})
    body = resp.body.decode("ascii")
    assert json.loads(body) == {"a": "\udcff é \U0001f600", "b": "nan"}


# ---------------------------------------------------------------------------
# /events generation (a replaced or rewritten events.jsonl is detected)
# ---------------------------------------------------------------------------

EVENTS_URL = "/api/traces/runs/r1/episodes/ep1/events"


def _events_root(tmp_path: Path, lines: bytes) -> tuple[Path, TestClient]:
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    ep.mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "running"}))
    (ep / "events.jsonl").write_bytes(lines)
    return ep, client


def test_events_pages_carry_the_file_generation(tmp_path: Path) -> None:
    ep, client = _events_root(tmp_path, b"".join(_event_line(i) for i in range(3)))
    page = client.get(EVENTS_URL).json()
    gen = page["generation"]
    st = (ep / "events.jsonl").stat()
    assert isinstance(gen, str) and gen.startswith(f"{st.st_dev:x}-{st.st_ino:x}-") and not gen.endswith("pending")
    # the same generation keeps working, also after appends (the recorder only appends)
    with open(ep / "events.jsonl", "ab") as fh:
        fh.write(_event_line(3))
    same = client.get(EVENTS_URL, params={"after_offset": page["next_offset"], "generation": gen})
    assert same.status_code == 200 and [e["seq"] for e in same.json()["events"]] == [3] and same.json()["generation"] == gen
    # without the parameter nothing is checked (older clients)
    assert client.get(EVENTS_URL, params={"after_offset": page["next_offset"]}).status_code == 200


def test_replaced_events_file_with_a_valid_old_offset_answers_generation_changed(tmp_path: Path) -> None:
    """The old offset is a line boundary of the new file, so only the
    generation shows that the page would mix two files."""
    ep, client = _events_root(tmp_path, b"".join(_event_line(i, "ev_old") for i in range(3)))
    first = client.get(EVENTS_URL).json()
    new = tmp_path / "new.jsonl"
    new.write_bytes(b"".join(_event_line(i, "ev_new") for i in range(5)))
    os.replace(new, ep / "events.jsonl")
    stale = client.get(EVENTS_URL, params={"after_offset": first["next_offset"], "generation": first["generation"]})
    assert stale.status_code == 409
    body = stale.json()
    assert set(body) == {"error", "generation"} and body["error"] == "generation_changed"
    assert body["generation"] != first["generation"]
    restart = client.get(EVENTS_URL, params={"after_offset": 0, "generation": body["generation"]}).json()
    assert [e["event_id"] for e in restart["events"]] == [f"ev_new_{i:08d}" for i in range(5)]
    assert restart["generation"] == body["generation"]


def test_first_line_rewritten_in_place_changes_the_generation(tmp_path: Path) -> None:
    ep, client = _events_root(tmp_path, b"".join(_event_line(i, "ev_aaa") for i in range(2)))
    first = client.get(EVENTS_URL).json()
    inode = (ep / "events.jsonl").stat().st_ino
    with open(ep / "events.jsonl", "r+b") as fh:
        fh.write(_event_line(0, "ev_bbb"))
    assert (ep / "events.jsonl").stat().st_ino == inode
    stale = client.get(EVENTS_URL, params={"after_offset": first["next_offset"], "generation": first["generation"]})
    assert stale.status_code == 409 and stale.json()["error"] == "generation_changed"


def test_generation_of_a_missing_or_torn_events_file(tmp_path: Path) -> None:
    ep, client = _events_root(tmp_path, b"")
    (ep / "events.jsonl").unlink()
    missing = client.get(EVENTS_URL).json()
    assert missing["generation"] == "none" and missing["events"] == [] and missing["eof"]
    assert client.get(EVENTS_URL, params={"generation": "none"}).status_code == 200
    changed = client.get(EVENTS_URL, params={"generation": "somefile"})
    assert changed.status_code == 409 and changed.json() == {"error": "generation_changed", "generation": "none"}
    # a first line without its newline yet: "pending", which changes once the line is complete
    line = _event_line(0)
    (ep / "events.jsonl").write_bytes(line[:20])
    torn = client.get(EVENTS_URL, params={"generation": "none"})
    assert torn.status_code == 409 and torn.json()["generation"].endswith("-pending")
    pending = torn.json()["generation"]
    assert client.get(EVENTS_URL, params={"generation": pending}).json()["events"] == []
    with open(ep / "events.jsonl", "ab") as fh:
        fh.write(line[20:])
    done = client.get(EVENTS_URL, params={"generation": pending})
    assert done.status_code == 409 and not done.json()["generation"].endswith("-pending")
    assert len(client.get(EVENTS_URL, params={"generation": done.json()["generation"]}).json()["events"]) == 1


def _long_first_line(tag: str, pad: int) -> bytes:
    event = {"pad": "x" * pad, "event_id": f"{tag}_0", "seq": 0, "event_type": "episode_start", "data": {}}
    return json.dumps(event, separators=(",", ":")).encode() + b"\n"


@pytest.mark.parametrize("pad", [5000, 300_000])
def test_first_line_rewritten_beyond_its_first_4096_bytes_changes_the_generation(tmp_path: Path, pad: int) -> None:
    """Regression case: only the first 4096 bytes of the first line
    were hashed, so a rewrite that kept them gave 200 and a mix of two files."""
    ep, client = _events_root(tmp_path, _long_first_line("old", pad) + _event_line(1, "ev_old"))
    first = client.get(EVENTS_URL).json()
    path = ep / "events.jsonl"
    inode = path.stat().st_ino
    with open(path, "r+b") as fh:  # same inode, same first 4096 bytes, same length
        fh.write(_long_first_line("new", pad) + _event_line(1, "ev_new") + _event_line(2, "ev_new"))
    assert path.stat().st_ino == inode
    stale = client.get(EVENTS_URL, params={"after_offset": first["next_offset"], "generation": first["generation"]})
    assert stale.status_code == 409 and stale.json()["error"] == "generation_changed"
    assert stale.json()["generation"] != first["generation"]


def test_a_first_line_longer_than_the_hashed_limit_is_never_trusted_across_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A first line longer than GENERATION_LINE_BYTES is not hashed whole; its generation
    then changes with every change of the file, so an old offset is never reused."""
    monkeypatch.setattr(reader, "GENERATION_LINE_BYTES", 1024)
    ep, client = _events_root(tmp_path, _long_first_line("old", 5000) + _event_line(1))
    first = client.get(EVENTS_URL).json()
    assert client.get(EVENTS_URL).json()["generation"] == first["generation"]  # stable while unchanged
    assert first["generation"].count("-") == 4
    with open(ep / "events.jsonl", "ab") as fh:
        fh.write(_event_line(2))
    after = client.get(EVENTS_URL, params={"after_offset": first["next_offset"], "generation": first["generation"]})
    assert after.status_code == 409 and after.json()["error"] == "generation_changed"


def test_cached_pages_of_another_generation_are_not_served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A first line rewritten in place within one timestamp tick keeps (dev, inode, size,
    mtime, ctime); the cache must not serve the old page for the new generation."""
    ep, client = _events_root(tmp_path, b"".join(_event_line(i, "ev_aaa") for i in range(2)))
    monkeypatch.setattr(reader, "_stat_key", lambda st: (st.st_dev, st.st_ino, st.st_size))  # a coarse clock
    old = client.get(EVENTS_URL).json()
    with open(ep / "events.jsonl", "r+b") as fh:
        fh.write(b"".join(_event_line(i, "ev_bbb") for i in range(2)))
    new = client.get(EVENTS_URL).json()
    assert new["generation"] != old["generation"]
    assert [e["event_id"] for e in new["events"]] == [f"ev_bbb_{i:08d}" for i in range(2)]


# ---------------------------------------------------------------------------
# An events file that exists but cannot be read is an error
# ---------------------------------------------------------------------------


def _deny(monkeypatch: pytest.MonkeyPatch, name: str, exc: OSError) -> None:
    real = reader.Dir.open_file

    def open_file(self: reader.Dir, file_name: str) -> tuple[int, os.stat_result]:
        if file_name == name:
            raise exc
        return real(self, file_name)

    monkeypatch.setattr(reader.Dir, "open_file", open_file)


@pytest.mark.parametrize(
    ("exc", "status", "detail"),
    [
        (PermissionError(errno.EACCES, "Permission denied"), 403, "permission denied"),
        (OSError(errno.EIO, "Input/output error"), 500, "(EIO)"),
        (OSError(errno.ELOOP, "Too many levels of symbolic links"), 403, "symlink"),
        (reader.NotRegularFileError(errno.EINVAL, "not a regular file"), 500, "not a regular file"),
    ],
)
def test_events_that_exist_but_cannot_be_opened_are_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError, status: int, detail: str
) -> None:
    """They used to answer 200 with events: [], eof: true and generation "none", which a
    summary saying "ended" would have shown as a complete, empty episode."""
    _ep, client = _events_root(tmp_path, _event_line(0))
    _deny(monkeypatch, "events.jsonl", exc)
    resp = client.get(EVENTS_URL)
    assert resp.status_code == status and detail in resp.json()["detail"], resp.text
    assert "events" not in resp.json()


def test_events_read_errors_after_the_open_are_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ep, client = _events_root(tmp_path, _event_line(0))

    def failing_pread(fd: int, size: int, offset: int) -> bytes:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(reader.os, "pread", failing_pread)
    resp = client.get(EVENTS_URL)
    assert resp.status_code == 500 and "(EIO)" in resp.json()["detail"]


def test_unscoped_log_that_cannot_be_opened_is_an_error(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _deny(monkeypatch, f"{UNSCOPED}.jsonl", PermissionError(errno.EACCES, "Permission denied"))
    resp = _get(env, f"{_base(UNSCOPED)}/events")
    assert resp.status_code == 403 and "permission denied" in resp.json()["detail"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_unreadable_files_on_disk_are_reported_not_missing(tmp_path: Path) -> None:
    ep, client = _events_root(tmp_path, _event_line(0))
    run_json = ep.parent / "run.json"
    run_json.write_text(json.dumps({"run_id": "r1"}))
    for path in (ep / "events.jsonl", ep / "episode.json", run_json):
        path.chmod(0)
    try:
        events = client.get(EVENTS_URL)
        assert events.status_code == 403 and "permission denied" in events.json()["detail"]
        summary = client.get("/api/traces/runs/r1/episodes/ep1").json()
        assert summary["episode_json_error"] == "unreadable (EACCES)" and summary["status"] == "incomplete"
        detail = client.get("/api/traces/runs/r1").json()
        assert detail["run"] is None and detail["run_json_error"] == "unreadable (EACCES)"
    finally:
        for path in (ep / "events.jsonl", ep / "episode.json", run_json):
            path.chmod(0o644)
    readable = client.get("/api/traces/runs/r1/episodes/ep1").json()
    assert readable["episode_json_error"] is None


def test_symlinked_run_json_is_reported(tmp_path: Path) -> None:
    root, client = _small_app(tmp_path)
    (root / "r1").mkdir()
    (tmp_path / "outside.json").write_text(json.dumps({"run_id": "SECRET"}))
    (root / "r1" / "run.json").symlink_to(tmp_path / "outside.json")
    (root / "r1" / "runner_records.jsonl").write_text("")
    detail = client.get("/api/traces/runs/r1")
    assert detail.status_code == 200 and "SECRET" not in detail.text
    assert detail.json()["run"] is None and detail.json()["run_json_error"] == "symlink_not_followed"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_unreadable_run_files_are_reported_not_missing(tmp_path: Path) -> None:
    """The reader rule for the run files: an unreadable runner_records.jsonl used to look
    like a run without runner records, and a run whose only file is an unreadable
    run.json was listed by /runs but answered 404 by /runs/{run_id}."""
    root, client = _small_app(tmp_path)
    record = json.dumps({"task": "cube_lift", "condition": "S1", "seed": 1, "outcome": "success"}) + "\n"
    for name in ("r_records", "r_ok"):
        (root / name).mkdir()
        (root / name / "run.json").write_text(json.dumps({"run_id": name}))
        (root / name / "runner_records.jsonl").write_text(record)
    (root / "r_run_json").mkdir()
    (root / "r_run_json" / "run.json").write_text(json.dumps({"run_id": "r_run_json"}))
    (root / "r_link").mkdir()
    (root / "r_link" / "run.json").write_text(json.dumps({"run_id": "r_link"}))
    (tmp_path / "outside.jsonl").write_text(json.dumps({"task": "SECRET"}) + "\n")
    (root / "r_link" / "runner_records.jsonl").symlink_to(tmp_path / "outside.jsonl")
    unreadable = [root / "r_records" / "runner_records.jsonl", root / "r_run_json" / "run.json"]
    for path in unreadable:
        path.chmod(0)
    try:
        info = client.get("/api/traces/runs/r_records").json()
        assert info["runner_records"] == [] and info["runner_records_info"]["error"] == "unreadable (EACCES)"
        assert "r_run_json" in {r["run_id"] for r in client.get("/api/traces/runs").json()["runs"]}
        only_run_json = client.get("/api/traces/runs/r_run_json")
        assert only_run_json.status_code == 200, only_run_json.text
        assert only_run_json.json()["run"] is None and only_run_json.json()["run_json_error"] == "unreadable (EACCES)"
    finally:
        for path in unreadable:
            path.chmod(0o644)
    link = client.get("/api/traces/runs/r_link")
    assert link.json()["runner_records_info"]["error"] == "symlink_not_followed" and "SECRET" not in link.text
    ok = client.get("/api/traces/runs/r_ok").json()
    assert len(ok["runner_records"]) == 1 and "error" not in ok["runner_records_info"]
    (root / "r_ok" / "runner_records.jsonl").unlink()  # no runner records at all: not an error
    assert "error" not in client.get("/api/traces/runs/r_ok").json()["runner_records_info"]


def _file_env(tmp_path: Path) -> TestClient:
    root, client = _small_app(tmp_path)
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    (ep / "assets" / "x.txt").write_text("x")
    np.save(ep / "assets" / "a.npy", np.zeros((2, 2)))
    return client


@pytest.mark.parametrize(
    ("exc", "status", "detail"),
    [
        (OSError(errno.EIO, "Input/output error"), 500, "file cannot be read (EIO)"),
        (OSError(errno.EMFILE, "Too many open files"), 500, "file cannot be read (EMFILE)"),
        (PermissionError(errno.EACCES, "Permission denied"), 403, "permission denied"),
        (FileNotFoundError(errno.ENOENT, "No such file or directory"), 404, "file not found"),
        (OSError(errno.ENXIO, "No such device or address"), 404, "file not found"),  # a socket
        (reader.NotRegularFileError(errno.EINVAL, "not a regular file"), 404, "file not found"),
    ],
)
def test_files_that_cannot_be_opened_are_not_reported_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: OSError, status: int, detail: str
) -> None:
    """An I/O error or too many open files used to be answered 404 "file not found"."""
    client = _file_env(tmp_path)
    _deny(monkeypatch, "x.txt", exc)
    _deny(monkeypatch, "a.npy", exc)
    base = "/api/traces/runs/r1/episodes/ep1"
    for url in (f"{base}/file/assets/x.txt", f"{base}/array/assets/a.npy", f"{base}/array-image/assets/a.npy"):
        resp = client.get(url)
        assert resp.status_code == status and resp.json()["detail"] == detail, (url, resp.text)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_run_and_episode_dirs_that_cannot_be_opened_are_not_reported_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An episode directory without read permission was answered 404 "episode not found"."""
    client = _file_env(tmp_path)
    base = "/api/traces/runs/r1/episodes/ep1"
    ep = tmp_path / "traces" / "r1" / "ep1"
    ep.chmod(0)
    try:
        for url in (base, f"{base}/events", f"{base}/file/assets/x.txt"):
            resp = client.get(url)
            assert resp.status_code == 403 and resp.json()["detail"] == "permission denied", (url, resp.text)
    finally:
        ep.chmod(0o755)
    assert client.get(base).status_code == 200
    assert client.get("/api/traces/runs/r1/episodes/nope").status_code == 404
    real = reader.Dir.subdir
    failing: set[str] = set()

    def subdir(self: reader.Dir, name: str) -> reader.Dir:
        if name in failing:
            raise OSError(errno.EIO, "Input/output error")
        return real(self, name)

    monkeypatch.setattr(reader.Dir, "subdir", subdir)
    for name, what in (("ep1", "episode"), ("r1", "run")):
        failing.add(name)
        resp = client.get(base)
        assert resp.status_code == 500 and resp.json()["detail"] == f"{what} cannot be read (EIO)", resp.text


# ---------------------------------------------------------------------------
# One damaged start time does not break the run detail
# ---------------------------------------------------------------------------


def test_episodes_with_non_numeric_start_times_are_sorted(tmp_path: Path) -> None:
    root, client = _small_app(tmp_path)
    values = {
        "e_float": 1758900000.0, "e_int": 1758800000, "e_string": "2026-09-26T10:00:00Z", "e_list": [1, 2],
        "e_dict": {"t": 1}, "e_bool": True, "e_huge": 10**400, "e_null": None,
    }
    for name, value in values.items():
        ep = root / "r1" / name
        ep.mkdir(parents=True)
        (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True, "started_wall_time": value}))
    (root / "r1" / "e_nan").mkdir()
    (root / "r1" / "e_nan" / "episode.json").write_text('{"status": "ended", "trace_complete": true, "started_wall_time": NaN}')
    resp = client.get("/api/traces/runs/r1")
    assert resp.status_code == 200, resp.text
    order = [e["episode_id"] for e in resp.json()["episodes"]]
    assert order[:2] == ["e_int", "e_float"] and order[-1] == "e_huge"
    assert sorted(order) == sorted([*values, "e_nan"])
    by_id = {e["episode_id"]: e for e in resp.json()["episodes"]}
    assert by_id["e_string"]["start_wall_time"] == "2026-09-26T10:00:00Z"  # reported as read
    assert client.get("/api/traces/runs/r1").json()["episodes"] == resp.json()["episodes"]


def test_unscoped_log_pages_carry_a_generation(env: dict[str, Any]) -> None:
    page = _get(env, f"{_base(UNSCOPED)}/events").json()
    assert page["generation"] and page["generation"] != "none"
    stale = _get(env, f"{_base(UNSCOPED)}/events", params={"generation": "0-0-0"})
    assert stale.status_code == 409 and stale.json() == {"error": "generation_changed", "generation": page["generation"]}


def test_generation_is_checked_before_the_offset(tmp_path: Path) -> None:
    """A stale generation with an offset past the new file's end is generation_changed, so
    the client restarts instead of reporting a bad offset."""
    ep, client = _events_root(tmp_path, b"".join(_event_line(i) for i in range(4)))
    first = client.get(EVENTS_URL).json()
    new = tmp_path / "short.jsonl"
    new.write_bytes(_event_line(0, "ev_new"))
    os.replace(new, ep / "events.jsonl")
    resp = client.get(EVENTS_URL, params={"after_offset": first["next_offset"], "generation": first["generation"]})
    assert resp.status_code == 409 and resp.json()["error"] == "generation_changed"
    bad_offset = client.get(EVENTS_URL, params={"after_offset": first["next_offset"]})
    assert bad_offset.status_code == 409 and "detail" in bad_offset.json()


# ---------------------------------------------------------------------------
# npz members that are damaged or compressed with bzip2/LZMA: HTTP 400, never 500
# ---------------------------------------------------------------------------


def test_damaged_and_unsupported_npz_members_are_400_json(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    from trace_viewer_testlib import damaged_npz_bytes

    root, client = _small_app(tmp_path)
    capfd.readouterr()  # The standalone server reports a missing frontend at startup.
    ep = root / "r1" / "ep1"
    (ep / "assets").mkdir(parents=True)
    (ep / "episode.json").write_text(json.dumps({"status": "ended", "trace_complete": True}))
    (ep / "assets" / "damaged.npz").write_bytes(damaged_npz_bytes())
    buf = io.BytesIO()
    np.save(buf, np.zeros(16))
    for method, name in ((zipfile.ZIP_LZMA, "lzma.npz"), (zipfile.ZIP_BZIP2, "bzip2.npz")):
        with zipfile.ZipFile(ep / "assets" / name, "w", compression=method) as zf:
            zf.writestr("a.npy", buf.getvalue())
    base = "/api/traces/runs/r1/episodes/ep1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        damaged = client.get(f"{base}/array-image/assets/damaged.npz", params={"name": "a0"})
        assert damaged.status_code == 400 and damaged.headers["content-type"] == "application/json"
        assert damaged.json()["detail"] == "cannot read the .npz file (error)"
        for name in ("lzma.npz", "bzip2.npz"):
            for url in (f"{base}/array/assets/{name}", f"{base}/array/assets/{name}?name=a", f"{base}/array-image/assets/{name}?name=a"):
                resp = client.get(url)
                assert resp.status_code == 400 and "unsupported compression" in resp.json()["detail"], (url, resp.text)
        # the undamaged mask of the damaged file is still readable
        assert client.get(f"{base}/array/assets/damaged.npz", params={"name": "a1"}).status_code == 200
    assert capfd.readouterr() == ("", "")
