# CaP-X trace format and viewer

This specification describes the public CaP-X trace recorder and viewer. The recorder observes [capgym/cap-x](https://github.com/capgym/cap-x) and leaves its source unchanged. `scripts/make_demo_trace.py` creates a synthetic trace in the same format.

Section numbering starts at 2 so that section references in code and tests stay valid.

## Hook ids

Recorded traces identify recorder hooks by these ids. The short names below are used when this document refers to a hook.

| Id | Short name | What it records |
|---|---|---|
| H1 | HTTP hook | Model HTTP attempts, including service and perception requests. |
| H2 | trial role hook | Model roles, call kinds, turns and task text in a trial. |
| H3 | model call hook | Logical model calls, streamed responses and preflight requests. |
| H4 | trial lifecycle hook | Batch episode start and end, evaluator results, and `trial_videos` after completed upstream trials. |
| H5 | feedback hook | Feedback assembled for a later model turn. |
| H6 | output parser hook | The harness's reading of model output. |
| H7 | environment hook | Code blocks, streams, observations, video reads and resets. |
| H8 | API wrapper hook | API call starts, results, errors and nesting. |
| H9 | service wrapper hook | Perception service calls and their results or errors. |
| H10 | frame hook | Frame buffer positions, simulator steps and frame events. |
| H11 | traceback hook | Hides recorder frames from tracebacks and stack output. |
| H12 | video encoder hook | Frame hashes used to match video input to recorded frames. |
| H13 | web trial hook | Web episodes, model roles, feedback and end reasons. |
| H14 | perception parser hook | Parsed perception model replies and logical call completion. |
| H15 | runner identity hook | Runner id, output directory and trace fingerprint information. |
| H16 | runner record hook | Copies the runner's trial records into `runner_records.jsonl`. |

## 2. Rules the recorder keeps

The trace recorder observes a trial without changing its model inputs, results, simulator state, exceptions, or timing. API wrappers call the original method once and return the original object or raise the original exception. Hooks do not add simulator calls. The trial thread queues small summaries; a writer thread processes payloads and files.

The runner's trial alarm is a `TimeoutError` without an OS error number. Recorder handlers re-raise that alarm. A file-system `ETIMEDOUT` has an error number and follows the normal recorder error path, so a failed trace read does not end the trial.

The recorder avoids printing, logging, Python warnings, and changes to randomness. Only a startup-hook failure prints one short line to stderr. It hides recorder frames from tracebacks that the model may receive. It keeps HTTP headers, raw launch arguments, and environment dumps out of traces. It records configured model names and endpoint host/port from launch arguments. It redacts known secret fields and URL credentials before storage. Prompts, model replies, stdout, stderr, and images can still contain secrets, so protect the trace root.

When tracing is disabled, the recorder installs no hooks. Failed writes, dropped events, late events, and failed hook installs count as gaps. A trace is complete only when its episode ends without gaps.

## 3. Enable recording

Install this distribution in the same Python environment that runs CaP-X. The `.pth` hook (a `.pth` file that Python runs at startup) activates before CaP-X imports in every process, including spawned workers.

| Variable | Default | Meaning |
|---|---|---|
| `CAPX_TRACE_DIR` | unset (off) | Trace root; setting it turns tracing on. A relative path is resolved against the working directory, and the absolute path is written back to `os.environ`, so spawned workers use the same root. |
| `CAPX_TRACE_PROXY_PORTS` | `8110` | Comma-separated loopback ports treated as possible proxy endpoints. Requests to these ports get the viewer's proxy note. An empty value turns this guess off. Entries that are not a port number from 1 to 65535 are ignored and listed in `settings.problems`; if none remain, no port is treated as a proxy. |
| `CAPX_TRACE_RUN_ID` | derived | Run id (section 4.2). |
| `CAPX_TRACE_FRAME_FPS` | `4` | Maximum number of periodic frames saved per wall-clock second from the video frame buffer. `0` saves key frames only. |
| `CAPX_TRACE_JPEG_QUALITY` | `85` | JPEG quality of saved buffer frames. Images sent to a model are stored as sent and never re-encoded. |
| `CAPX_TRACE_STATE` | `1` | Turns on ground-truth state snapshots around code blocks and top-level API calls (section 8.4). An empty value, `0`, `false`, `no` or `off` turns them off. |
| `CAPX_TRACE_MIN_FREE_GB` | `2` | When free disk space falls below this value, the recorder skips binary files (frames, images, arrays, videos). Below half of it, the recorder writes one final `disk_low` marker and then stops writing. |
| `CAPX_TRACE_MAX_QUEUE_MB` | `512` | Maximum total size of the binary copies that wait in the writer queue (section 4.3). |
| `CAPX_TRACE_VIDEO_FILE_MB` | `256` | Maximum size of each upstream trial MP4 copied into an episode. The default limits one copy on the writer thread to 256 MiB and still fits short trial videos. The recorder skips larger files and writes a warning. `0` turns copying off and writes one warning per episode. |
| `CAPX_TRACE_ARRAY_CALL_MB` / `CAPX_TRACE_ARRAY_EPISODE_MB` | `64` / `2048` | Size limits for the array copies taken at call time, per call and per episode. Past the limit, the recorder keeps only the shape, dtype and a bounded hash (section 5.3). |

A value that is not a number, is NaN or infinite, or is below 0 (below 1 for the JPEG
quality) falls back to the default and is listed in
`episode_start.data.tracer.settings.problems`. A JPEG quality above 100 becomes 100 and is
listed as well. KB, MB and GB in these variable names and in the rest of this document
are binary units: 1 KB is 2^10 bytes, 1 MB is 2^20 bytes and 1 GB is 2^30 bytes.

The recorder also sets two variables itself for the processes it spawns:
`CAPX_TRACE_FALLBACK_RUN_ID` (the fallback run id of section 4.2) and
`CAPX_TRACE_RUNNER_RUN_DIR` (the runner's run directory, exported by H15).

If the imported CaP-X package has a `capx/tracing` directory (its own bundled recorder), the public recorder stays inactive and leaves the bundled recorder untouched. When CaP-X has no bundled recorder, the public recorder activates. The recorder makes this choice at the first CaP-X import, or at activation if CaP-X is already imported.

For a uv-managed CaP-X checkout, run:

```sh
uv pip install --python .venv/bin/python /path/to/capx-trace-viewer
.venv/bin/python -m capx_trace_recorder install-pth
export CAPX_TRACE_DIR=./traces
uv run --no-sync --active capx/envs/launch.py --config-path /path/to/config.yaml
```

For a pip-managed environment, run `python -m pip install /path/to/capx-trace-viewer` and `python -m capx_trace_recorder install-pth` with that environment's interpreter. `status` reports the interpreter, the hook path, and its state. It reports invalid UTF-8 hook content as `modified` and a read failure as `unreadable`. Install and uninstall refuse a modified hook and stop if they cannot read it. Run `uninstall-pth` before removing the package. After `uv sync`, or after changing the package layout of an editable install, reinstall this distribution and its hook. A leftover `.pth` file does nothing while tracing is off.

To trace a single process, use `python -m capx_trace_recorder run -m module [args...]` or `python -m capx_trace_recorder run script.py [args...]`. This launcher adds runpy frames to printed stack traces, so use the `.pth` hook for experiments that compare model inputs. Hooks patch functions in the modules that define them and leave globals in `__main__` unchanged. When `CAPX_TRACE_DIR` is exported, every other Python process in that environment also registers the CaP-X import check, except viewer-only servers. Only a startup-hook failure prints one short line to stderr.

## 4. Storage

### 4.1 Layout

```
<CAPX_TRACE_DIR>/                           # the trace root
  recorder_status.<pid>.json                # active public recorder after CaP-X import; absent when it stands down
  recorder_warnings.<pid>.jsonl             # H4 install failures before any episode opens
  <run_id>/
    run.json                                # created once with O_EXCL, never replaced
    runner_records.jsonl                    # redacted copies of the runner's trial records (runner parent process)
    unscoped.<pid>.jsonl                    # events with no episode (preflight, unattributed events, run-level warnings)
    unscoped.<pid>.files/                   # payloads and assets of those events
    unscoped.<pid>.status.json              # only when lines of that log could not be written
    <episode_id>/
      episode.json                          # summary; tmp file + rename on every update
      events.jsonl                          # append-only; one ASCII JSON event per line; file order == seq order
      heartbeat                             # empty file; mtime refreshed every 10 s while the episode is open
      payloads/<event_id>.request.json
      payloads/<event_id>.response.json
      payloads/<event_id>.feedback.json
      frames/<camera>/e<epoch>_<buffer_index>.jpg
      videos/video_turn_NN.mp4             # completed upstream trial videos copied by H4
      videos/video_combined.mp4
      videos/video_combined_<camera>.mp4
      assets/<sha256[:2]>/<sha256>.<ext>    # png, jpg, mp4, npy, npz, txt.gz, json.gz, ...
```

The first process that records something for a run writes the trace's `run.json`:
`schema_version`, `tracer_version`, `run_id`, `created_wall_time`, `created_at`,
`created_by` (`pid`, `host`, `python`, `git_commit`), `trace_dir` and `runner_run_dir`.
It is a different file from the runner's own `run.json` in the runner's output directory.

`episode.json` holds the episode's current state: `status` (`running` until
`episode_end` is written, then `ended`), `episode_end_written`, `trace_complete` (null
until the end), `gaps`, `dropped_events`, `late_events`, `n_warnings`, `n_events`,
`last_seq`, `events_bytes`, `heartbeat_interval_s`, `updated_wall_time`, `host`, and
`video_copy_pending`; the
start fields (`source`, `task`, `condition`, `seed`, `attempt_index`,
`runner_invocation`, `condition_label`, `models`, `config_flags`, `env`,
`started_wall_time`, `pid`); after the end, `ended_wall_time`, `end_reason`,
`duration_s`, `alarm_seen`, `n_code_blocks`, `n_model_calls`, `n_http_attempts`,
`tracer_overhead_s`, `harness_summary` and `exception`; and, where they apply,
`hooks_installed_during_episode`, `hooks_skipped_during_episode` and `apis_wrapped` (the hook metadata).

Events that belong to no episode go to the unscoped log of their run. Their files are in
`unscoped.<pid>.files/`, and their artifact paths are relative to the run directory.

### 4.2 Ids

- Every id matches `^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$`. Other characters become `-`, and
  a first character that is not an ASCII letter or digit becomes `x`.
- `run_id`, first match wins: `CAPX_TRACE_RUN_ID`; the experiment runner's run id, which
  the runner identity hook (H15) learns at the runner's `revision.fingerprint` call (after `--run-id` and
  `--output-root` are applied, before the preflight and the workers start) and exports as
  `CAPX_TRACE_RUN_ID`; for a runner episode, the name of the runner's run directory
  (`CAPX_TRACE_RUNNER_RUN_DIR`, else the first of `config["output_dir"]` and its 3 nearest
  ancestors that holds `run.json`); else the fallback `<UTC yyyymmddTHHMMSSZ>_<host>_<pid>`,
  derived once by the first process and exported as `CAPX_TRACE_FALLBACK_RUN_ID`, so the
  workers of one `launch.py` run share it.
- `episode_id` of a runner episode: `<task>__<condition>__s<seed>__i<invocation>__a<attempt>`.
  `task` and `condition` are the first two parts of `config["output_dir"]` below the
  runner's run directory. `invocation` is `len(run.json["invocations"])` of the runner's
  `run.json` when this process first read it; the runner appends one entry to that list
  each time its command runs for the run id. `attempt` is the 1-based count of
  `_run_single_trial` calls for (`config["output_dir"]`, seed) in this process.
- `episode_id` of a `launch.py` episode: `<UTC ts>__<task>__s<seed>__<6 hex>`, where `task`
  is the stem of the config path, or the env class name.
- `episode_id` of a web UI episode: `<UTC ts>__web__<first 8 characters of the session id>`.
- The recorder cuts an episode id longer than 200 characters before its tail, so the
  parts that tell trials apart (`__s<seed>__i<invocation>__a<attempt>`, or the random
  suffix) always survive. When the id is already used in this process or its directory
  exists, the recorder shortens the part before the tail and appends `__dup<n>`, starting
  at `__dup2` (for example `...__s1__i1__a1__dup2`). After 10000 numbered tries it appends
  a random suffix.
- `event_id` is `ev_` followed by 20 hex characters from `os.urandom`, and
  `logical_call_id` is `lc_` followed by 20 hex characters.

### 4.3 Writer

- Each process has one writer thread (`capx-trace-writer`, a daemon thread), created when
  the first item is queued. A process forked from a traced process starts with a fresh
  store.
- `emit()` runs on the calling thread. It sets `event_id` and the clocks, and inside a lock
  of the episode it assigns `seq` and puts the item on a `queue.SimpleQueue`, which never
  blocks. An item holds the event and, optionally, a builder function that the writer runs
  before it writes the line (payload normalisation, image decoding, hashing, JPEG, PNG and
  npz encoding).
- The calling thread takes everything a builder needs as data that nothing changes
  later: request bodies (`str` or `bytes`, which are immutable), array, frame and state
  copies taken at call time, a copy of the episode's segments taken when a model call
  starts (section 6.4), and references to the frame lists that `get_video_frames_range`
  returned (fresh copies that the harness only reads).
- The writer takes items in FIFO order. For each item it writes the files first (a tmp
  name, then `os.replace`), then appends the event line with one `write` call and checks
  the byte count. On a short write or an error it truncates `events.jsonl` back to its
  previous size and counts a `write_failed` gap. An event names a file only after the file
  is in place; otherwise the reference carries `written: false` and a reason.
- The writer checks that the `seq` numbers of each log have no holes. A hole becomes a
  `dropped_events` gap and a `trace_warning`.
- When the writer thread has died, the next `emit()` sees it (`is_alive()`) and stops
  queueing for that episode. The episode then has no `episode_end` and no fresh heartbeat,
  and the viewer shows it as incomplete.

Queue budget. `CAPX_TRACE_MAX_QUEUE_MB` limits the bytes of the binary copies that wait in
the queue: frames, harness renders, arrays, images, VDM video frames and ground-truth state
snapshots. When an item would go over the budget, the store releases its copies as it
queues the item (it swaps the builder for one that holds none of them). The writer then
counts one `queue_full` gap and records the files the item would have written with
`written: false, reason: "queue_full"`. Model request and response bodies are text, and
the writer always writes them, also over the budget; it then leaves out only the images and
videos inside the request, which counts one gap.

### 4.4 Episode end, late events, limits and failures

Episode end. The batch trial lifecycle hook (H4) or the web trial hook (H13) queues `episode_end` and
returns. When the writer reaches `episode_end`, it fills in `trace_complete`, `gaps`,
`dropped_events`, `late_events` and `n_warnings`, writes the line, rewrites
`episode.json` with `status: "ended"` and stops its regular heartbeat updates.
During a later video copy, the writer refreshes that episode's heartbeat and
the heartbeats of other open episodes between 1 MiB chunks, about every 10 seconds.

`trace_complete` is true only when `gaps` is empty. `gaps` lists `{code, count, first_seq}`
for each kind of counted problem: dropped or late events, failed writes, files skipped for
low disk space or for the queue budget, hooks that failed to install (at episode start or
while the episode was open), request bodies that could not be parsed or rebuilt, recorder
errors, events that could not be given to one episode, state snapshots that could not be
taken, arrays over the array budget, unsafe artifact paths (`unsafe_path`), and the recorder's path in a block's output. Every
`trace_warning` counts as a gap unless it carries `counts_as_gap: false`.

H4 queues a pending marker, used only inside the writer, before `episode_end`, and
queues `trial_videos` after it. `episode.json` has `video_copy_pending: true` while the
copy runs and false after `trial_videos` is written. If the writer is interrupted, the
marker stays true. The recorder itself emits `trial_videos`, and the writer treats it as
part of normal completion: it does not set `late` or count a gap. The reader accepts
`trial_videos` after the end and reads its artifacts. After the copy, `trace_complete`
in `episode.json` takes precedence over the value in the earlier `episode_end` event.
A failed copy updates `episode.json` with one gap for that file.

Late events. An episode stays reachable after `episode_end`, because the objects bound to
it (its env, its low-level env and the web UI's launch arguments) keep pointing to it
(the recorder lifecycle). The writer marks events that arrive later, for example from web UI threads
that were still running or from an alarm during an ensemble call, with `late: true`,
counts them as `late_event` gaps and rewrites `episode.json` with `trace_complete: false`.
Once `episode_end` is written, the writer keeps only a small record of the episode (seq,
counters, gaps and summary), and the recorder drops its references to the env, the frames,
the segments and the web session. A late event after that point carries `sim_step` and
`sim_time` null and has no key frames or state.

Process exit. An `atexit` handler queues a stop marker and waits at most 10 s. SIGKILL and
SIGTERM skip it: the events still in the queue (normally a few milliseconds' worth) are
lost, and the viewer later shows the episode as incomplete.

Low disk space. The writer checks the free space of the trace root every 5 s. Below
`CAPX_TRACE_MIN_FREE_GB` it skips binary files and counts `disk_low` gaps. Below half of
it, all writing stops for the rest of the process after one final `disk_low` warning line
in every open log and in the run's unscoped log. The episodes that were open then have no
`episode_end` and no fresh heartbeat, so the viewer shows them as incomplete; episodes
that had already ended keep their files untouched. If an episode's first queued item detects the stop, that episode cannot write its
start or end events and appears incomplete.

Write failures. A write that fails for an episode's files counts a `write_failed` gap in
that episode; one that fails for `run.json` or `runner_records.jsonl` leaves a
`write_failed` warning in the run's unscoped log. Lines that the unscoped log itself cannot
take are counted in `unscoped.<pid>.status.json` next to it, and the viewer lists that
count with the log.

A failure to read a source video counts as a `read_failed` gap in the episode. This
gap is separate from the `write_failed` gaps for failures to write the destination file.
The skipped artifact records `read_failed: <type>`.

### 4.5 Readers

Readers split `events.jsonl` on `b"\n"`. They skip a line that is not UTF-8 JSON of an
object, or that is longer than 8 MB, wherever it is, and count it in `corrupt_lines`. A
last line without a newline is a torn tail and stays unread. `NaN` and `Infinity` in a
line are served as the strings `"nan"`, `"inf"` and `"-inf"`. The backend and
local validation tools share the reader in `capx_trace_viewer/reader.py`, which
uses only the standard library and follows the rules of section 2.

## 5. Events

### 5.1 Envelope

| Field | Meaning |
|---|---|
| `schema_version` | `"capx-trace/1"` |
| `run_id`, `episode_id`, `event_id` | Section 4.2. `episode_id` is null in an unscoped log. |
| `parent_event_id` | The event this one belongs to: the code block of a top-level API call, the calling API of a nested one, the API call of a service call, the Molmo service call of a perception VLM request, the request of a response or error, the start event of an end or error event, the block of its stdout, stderr and evaluator result, and the response that a parse read. |
| `seq` | Position in the log: 1, 2, 3, ... in file order. |
| `wall_time` | `time.time()` when the hook ran. |
| `monotonic_time` | `time.monotonic()`, comparable only within one `pid`. |
| `pid`, `thread` | Process id and thread name. |
| `sim_step`, `sim_time`, `sim_epoch` | Simulator clock (section 8.3) or null. `sim_epoch` starts at 0 and grows by one with each further reset inside the episode; the web UI resets the env after a block timeout. |
| `event_type`, `status` | Section 5.2. `status` is `started`, `ok`, `error`, `degraded`, `warning` or `info`. |
| `late` | True for an event queued after `episode_end`. |
| `refs` | The ids that apply: `logical_call_id`, `http_attempt` (1-based HTTP attempt within a logical call), `code_block_id`, `api_call_id`, `model_event_id`, `turn_index`, `for_event_id`. |
| `artifacts` | The files this event wrote: `[{path, sha256, kind, mime, bytes, label, written, reason?}]`, plus `width`, `height`, `shape`, `dtype`, `camera`, `names` or `json_path` where they apply. Paths are relative to the episode directory (to the run directory in an unscoped log). `label` is `model_input` (an image or video sent to a model), `payload`, `harness_render`, `debug_frame`, `perception_input`, `perception_output`, `state`, `text` or `value` (an array or image that an API call took or returned). |
| `data` | Fields of the event type (section 5.2). |

Event lines are ASCII JSON (`ensure_ascii=True`) of at most 256 KB. Strings over 4000
characters (code, stdout, stderr, model text, error messages) go to `txt.gz` assets, and
state snapshots to `json.gz` assets. When a line would still be longer than 256 KB, the
writer saves the whole `data` as a `json.gz` asset (label `payload`) and keeps inline only
its scalar fields and its strings of at most 200 characters, plus `oversize: true`,
`data_asset` (the asset path) and `data_asset_written`. The viewer loads that asset and
merges it back when a pane needs the event (section 9.3).

### 5.2 Event types

`episode_start` (status `started`):

- `source`: `experiments.run`, `envs.runner` (`launch.py`) or `web_ui`;
- `task`, `condition`, `seed`, `attempt_index`, `runner_invocation` (section 4.2);
- `condition_label`: the condition name, for display only;
- `runner_condition`: fields copied from the matching condition entry of the runner's
  `run.json` (`task`, `condition`, `config_path`, `privileged`, `apis`, `multi_turn`,
  `uses_vdm`, `use_visual_feedback`, `use_oracle_code`, `note`);
- `config_flags`, read from the real config: `multi_turn` (a `multi_turn_prompt` is set),
  `use_visual_feedback`, `use_img_differencing`, `use_video_differencing`,
  `use_wrist_camera`, `use_parallel_ensemble`, `use_multimodel`, `record_video`,
  `use_oracle_code` and `use_legacy_multi_turn_decision_prompt`; with oracle code (a fixed
  program that runs instead of model output) also `oracle_code` (`source`, `sha256` and
  `length` of `env.oracle_code`), and in the web UI also `await_user_input_each_turn`;
- `models`: `{role: {model, endpoint}}`, copied field by field; `code_generator` carries an
  `oracle_code` flag, and `vdm` is present only when image or video differencing is on;
- `env`: `class`, `low_level_class`, `apis`, `privileged` and `viser_port` (when a Viser
  server exists). A web UI episode has `env: null` and `env_pending`, because the web UI
  creates the env inside the episode; these fields go to `episode.json` when the env is
  bound;
- `frames`: whether the frame hook (H10) was installed, its buffers and cameras, and how many buffer frames
  were already there;
- `tracer`: `version`, `settings`, `hooks_installed`, `hooks_skipped`, `hooks_failed`; required hook failures count as gaps;
- `output_dir`, `host`, `python`, `git_commit`, `process_start_time` (from
  `/proc/<pid>/stat`), and `web_session_id` for the web UI.

`git_commit` is the commit of the CaP-X checkout from which the loaded `capx`
package came. It is null when that package has no Git checkout.

`trial_videos` follows `episode_end` for completed upstream trials that have
`record_video: true`, no exception or timeout, and a `code_path`. Fork runner episodes
(`source: experiments.run`) use their runner artifacts instead. The event's `artifacts`
list holds the copied trial MP4 files. Each written video has `path` relative to the
episode (`videos/<original name>`), `bytes`, `sha256`, `mime: video/mp4`,
`written: true`, and `turn_index` parsed from `video_turn_NN` (null for combined videos).

The writer accepts a turn file only when two checks pass. Its number must be below the
episode's code block count. Its modification time must be no earlier than the final
`code.py` modification time minus 5 seconds; `code.py` and the videos share a filesystem
clock. If `code.py` cannot be stat'ed, the writer uses the episode start wall time minus
5 seconds instead. This rule excludes files left by a run minutes earlier and still
allows for timestamp rounding and short write delays. Combined files must pass the same
time rule.

For each accepted file, the writer opens the source without following symlinks, applies
the per-file size cap, and checks that free space after the copy stays above the
disk-low threshold. It copies in 1 MiB chunks and then renames the copy into place
atomically. The queue budget does not apply to these copies, which the writer makes
itself. Missing videos and stale files produce warnings but no gaps. Each attempted copy
that fails counts one gap.

Limits. The copy runs on the single writer thread. Very large videos on a slow disk can
delay later events. In a sequential run, they can also bring the next episode closer to
`queue_full`. If the copy is interrupted, `videos/.video_*.tmp.*` files may remain, the
pending marker stays true, and the viewer reports that the videos may not have been
saved. The viewer keeps polling while the copy is pending and the heartbeat is fresh. It
reads the summary every 20 seconds once the heartbeat is two minutes old and the
copy may have been interrupted. It reads events again when the pending marker clears.

`episode_end`:

- `end_reason`: `completed`, `exception`, `timeout_alarm`, `error`, `stopped_by_user` or
  `cancelled_replaced`, and `end_note` when the recorder ended the episode itself (a start
  that an alarm or a recorder error interrupted, or an episode left open on the trial
  thread that the next start closed). An `ETIMEDOUT` error inside a trial ends it with
  `exception`; the runner's alarm ends it with `timeout_alarm`;
- `harness_summary`: the fields of the returned `TrialSummary`, and `success_meaning`,
  which says what `success` means for the harness that returned it:
  - batch episodes: "sandbox_rc == 0 of the last code block that
    ran (0 also when its stderr says 'executing action in terminated episode'); earlier
    blocks may have failed; not task success". After the last block, the harness sets
    `sandbox_rc` to 0 when that block's stderr contains the quoted phrase, and `success`
    is `sandbox_rc == 0`. Only the last block counts.
  - web UI episodes: "task_completed or
    num_finishes > 0, as the web UI runner computes it (its task_completed also counts
    terminated with reward > 0); sandbox_rc is not used". `num_finishes` counts the
    decisions that the harness read as finish, which include every reply without
    REGENERATE and an empty reply.
  - older traces may store "code ran without exception (sandbox_rc == 0); not task success"
    in `success_meaning` for both kinds of episode. The viewer derives the correct
    explanation from the episode source;
- `exception` (`type`, `message`, `message_redacted`), `alarm_seen`, `duration_s`,
  `n_code_blocks`, `n_model_calls` (logical calls), `n_http_attempts`,
  `tracer_overhead_s`, `tracer_overhead_all_threads_s`, and `session_state` for the web
  UI;
- filled in by the writer: `trace_complete`, `gaps`, `dropped_events`, `late_events`,
  `n_warnings`.

`task_text`: the task text that the first code-generation request actually sent, that is
the text before `"\nAPIs:\n"` in its last user message (with the LIBERO goal already
filled in). The model call hook (H3) emits it once per episode, at the first code-generation request; an
episode that runs oracle code sends no such request and has no `task_text`. Fields:
`text`, `reason` (why no text was found), `logical_call_id`, `source`. The viewer shows it
at the top of the left pane.

`model_request` (one per HTTP attempt, status `started`): `role` (`code_generator`, `vdm`,
`perception_vlm`, `preflight` or `unknown`), `call_kind` (`initial_code`,
`multi_turn_decision`, `ensemble_member`, `ensemble_synthesis`,
`initial_scene_description`, `image_diff`, `video_diff` or `point`), `turn_index`,
`logical_call_id`, `http_attempt`, `method`, `endpoint` (`url` without userinfo, query and
fragment, `host`, `port`, `path`, `known_proxy`), `stream` and `body_type`. The writer adds
`payload_path`, `body_sha256`, `body_bytes`, `reconstruction`, `masked_fields`,
`assets_moved`, `model`, `params` (every top-level body key except `messages` and
`input`), `n_messages`, `images`, `n_images`, `n_videos` and `input_summary` (section 6).

`model_response`: `role`, `call_kind`, `logical_call_id`, `http_attempt`, `http_status`,
`latency_s` (the HTTP round trip, including any proxy retries within one attempt;
section 6.5), `streamed`, `n_chunks`, `ttft_s`, `content`
(the full text), `reasoning`, `usage`, `finish_reason`, `response_model`, `response_id`
and `payload_path`.
`reasoning` is `{text, field}` and is present only when the API returned it in
`choices[0].message.reasoning`, `choices[0].message.reasoning_content` or a streamed
reasoning delta; the recorder never infers it. The model call hook (H3) records a streamed response (web UI
code generation) from the stream. Its payload file holds the assembled
`{streamed, content, reasoning, n_chunks}`, and `done_chunk_seen` says whether the stream's
final chunk arrived.

`model_error`: `scope` is `attempt` for one failed HTTP attempt (an HTTP status of 400 or
more, with `body_excerpt`, or a transport exception) and `logical` for the failure of the
whole logical call (the exception that left `query_model`, or a stream that failed or that
its caller stopped reading, `GeneratorExit`, with `partial_content` and `n_chunks`). The
other fields are `role`, `call_kind`, `logical_call_id`, `http_attempt`, `http_status`,
`error_type`, `message` (redacted, at most 2000 characters, with `message_redacted`),
`latency_s` where it is known, and `body_excerpt` (the first 2000 characters of the error
body, redacted, with `body_excerpt_redacted`).

`model_output_parsed`: the harness's reading of a reply. With `parser: "_extract_code"` it
has `code_blocks` and `code_source` (`model`, `oracle` or `unknown`). With
`parser: "_parse_multi_turn_decision"` it has `decision`, `harness_reading` ("REGENERATE
found -> regenerate" or "no REGENERATE -> finish"), `model_said_finish` (the reply
contains the literal `FINISH`) and `regenerate_text`. With `parser: "molmo._parse_points"`
it has `logical_call_id`, `parsed_text`, `points` and `norm_scale`. Every parse has
`model_event_id`, the response it read (null when a Molmo request got no response), and
`content_empty`. The harness parser reads any reply without REGENERATE as finish, so
`decision` records the harness's reading, and `model_said_finish` records whether the
model wrote FINISH. The viewer shows the reading with the prefix `harness:` (for example
"harness: no REGENERATE -> finish"), and it shows 「モデルは完了と主張」 (the model claims it
finished) only when `model_said_finish` is true.

`feedback_built`: what `_build_multi_turn_decision_prompt` (or its `_legacy` form)
assembled for the next decision request: `builder`, `turn_index` (that of the next
decision request), `vdm_model_event_id`, `user_feedback`, `is_video_feedback`,
`has_visual_feedback`, `image_pixel_sha256`, `n_messages`, `segments` and `texts_path`.
`segments` describes the formatted multi-turn text (`kind: "multi_turn_prompt"`, whose
`contains` lists the registered texts found in it), the VDM text and the user feedback,
each with `text_sha256`, `length` and `source_event_ids`. The texts themselves are in
`payloads/<event_id>.feedback.json` (`complete_multi_turn_prompt`,
`visual_differencing_feedback`, `user_feedback`).

`code_block_start`: `code_block_id` (equal to `event_id`), `block_index`, `turn_index`,
`source_model_event_id` (the code-generation response the code came from), `code_source`
(`model`, `oracle` or `unknown`), `code` (inline, or a text asset when long),
`code_sha256`, `direct_access`, `frame_ref` and `state_ref`. `direct_access.names` says
which of `env`, `obs`, `APIS` and `INPUTS` the code names; calls through `APIS[...]` are
recorded, and calls through the other three are not.

`code_block_end`: `code_block_id`, `sandbox_rc`, `duration_s` (the whole `step`),
`exec_duration_s` (the `exec` of the code), `stdout_len`, `stderr_len`,
`stdout_event_id`, `stderr_event_id`, `reward`, `terminated`, `truncated`,
`task_completed`, `result_summary`, `frame_ref`, `state_ref`, `sim_step_range`,
`exception` (when `step` itself raised) and `tracer_frame_visible`. `tracer_frame_visible`
is true when the recorder's path appears in the block's stdout or stderr, which should
never happen; it also counts a gap.

`stdout`, `stderr`: one of each per code block, holding exactly what `step()` returned:
`code_block_id`, `text` (inline up to 4000 characters, else a text asset) and `length`.

`api_call_start`: `api_call_id`, `name`, `qualname`, `api_class`, `depth` (0 for a call
from the generated code, 1 and more for an API called by another API), `privileged`,
`args`, `units` (section 5.3), `call_site`, `stdout_offset`, `stderr_offset`, `frame_ref`
and `state_ref`. For depth 0, `call_site` is the line of the generated code that led to
the call, `{code_block_id, line, source_line}`, plus `function`, `function_line` and
`function_source_line` when the call came from a function the code defined
(`function_defined_in_earlier_block` when that function came from an earlier block).

`api_call_end`, `api_call_error`: `api_call_id`, `name`, `depth`, `duration_s`,
`stdout_slice` and `stderr_slice` (`{start, end, text}`: what this call printed),
`frame_ref`, `state_ref`, `sim_step_delta` and `buffer_frames` (`[first, last]` buffer
indices appended during the call). `api_call_end` adds `return`. `api_call_error` adds
`error_type` (module-qualified unless the class is built in), `message` (redacted, with
`message_redacted` and `message_cut`), `alarm_timeout`, `traceback` (recorder frames
removed, redacted, with `traceback_redacted` and `traceback_cut`) and `cause` (`type`,
`message`).

`service_call_start`, `service_call_end`, `service_call_error`: one call of a perception or
motion service closure: `service` (`sam3` or `sam2` for segmentation, `owlvit` for
detection, `molmo` for pointing, `graspnet` for grasp planning, `pyroki` or `curobo` for
inverse kinematics and trajectory planning), `function` (the closure attribute),
`request_summary`, `endpoint`, `response_summary`, `latency_s`, `http_attempts`
(`[{status, latency_s, error, endpoint}]`, with `model_event_id` for Molmo requests) and,
for Molmo, `model_logical_call_ids`. The status is `degraded` when an HTTP attempt failed
and the closure still returned. The
recorder stores the image each service received as PNG (`perception_input`) and the array
outputs (masks bit-packed with `np.packbits`, boxes, scores, grasp poses, IK solutions)
together in one `npz` (`perception_output`); their statistics are in `response_summary`.

`observation_snapshot`: a frame the harness rendered with `CodeExecutionEnvBase.render`
or `render_wrist`: `camera`, `purpose` (the calling function), `pixel_sha256` (sha256 of
the raw `uint8` pixels), `width` and `height`. It is stored as PNG with label
`harness_render`.

`video_frames_read`: a call of `get_video_frames_range` or
`get_wrist_video_frames_range` by the harness (for VDM videos): `start`, `end`,
`n_frames`, `camera` and `frame_pixel_sha256` (one hash per frame; null with
`reason: "queue_full"` when the queue budget dropped the frames).

`frame`: a frame copied from the video buffer: `camera`, `epoch`, `buffer_index`,
`sim_step`, `reason` (`periodic` or `keyframe`) and `for_event_id`. It is stored as JPEG
with label `debug_frame`.

`env_reset`: a reset inside an episode (the web UI resets after a block timeout):
`sim_epoch`, `resets_in_episode`, `env_class`.

`evaluator_result`: `scope` (`code_block` or `episode`), `code_block_id`, `reward`,
`task_completed`, `terminated`, `truncated` and `source` (`CodeExecutionEnvBase.step`, or
the `TrialSummary` returned by the harness). It holds only values the harness computed.

`trace_warning`: `code`, `message` and `related_event_id`. The codes are `write_failed`, `read_failed`,
`disk_low`, `queue_full`, `dropped_events`, `hook_install_failed`, `payload_unparsed`,
`array_budget`, `tracer_frame_visible`, `ambiguous_episode`, `internal_error`,
`stale_context` and `state_unavailable`. `stale_context` reports context entries that an
interrupted wrapper left on the trial thread, which trial lifecycle hook (H4) clears when an episode starts (not
a gap of the new episode) or ends (a gap). `state_unavailable` reports a supported
simulator whose state could not be read. `ambiguous_episode` means an event could match more than one open episode, so the recorder cannot assign it safely; the warning lists the candidate episode ids and the affected episode has a gap.
Warnings can also carry `writer_generated` (found by the writer), `counts_as_gap: false`,
`hook` and `target` with `during_episode` or `run_level` (hook failures), `candidates`
(`ambiguous_episode`), `outside_episode`, `episode_id`, `alarm` and `note` (a recorder
error or an interrupted start written to the unscoped log), `file` and `count` (write
failures), `when` and `removed` (`stale_context`), and `message_redacted`. `late_event`
appears only as a gap code: late events carry `late: true` instead of a warning.
`video_missing` means upstream produced no matching MP4 files or no trial directory.
`video_skipped` names a file and its reason (`stale`, `turn_index_out_of_range`,
`not_regular`, `unsupported_platform`, `disk_low`, or
`write_failed`, `read_failed: <type>`). `video_size_limit` names an oversized file, or
reports that `CAPX_TRACE_VIDEO_FILE_MB=0` disabled copying (one warning per episode).
The recorder writes `disk_low` once per episode even if several videos cannot be copied. These warnings carry
`counts_as_gap: false`. Each attempted copy that fails adds one gap on its own.
The `turn_index_out_of_range` and `unsupported_platform` reasons also produce warnings
without gaps.

`runner_records.jsonl`, written by the runner record hook (H16), holds a copy of every record that the experiment runner
appends to its `trials.jsonl`, so the viewer can show the runner's verdict (attempt
statuses, `task_success`, `code_execution_success`, `agent_finish`,
`termination_reason`) without reading files outside the trace root. The error texts of a
record (`attempts[].error`, `error`, `setup_error`) are redacted (section 6.3), and the
record lists the changed ones in `trace_redacted_fields` and `trace_cut_fields`.

### 5.3 Argument and return summaries

`summarize(value)` runs on the calling thread: before the call for the arguments, and
before the wrapper returns for the return value. It never evaluates or consumes anything
the program did not: it calls no `repr()`, iterates no generator and reads no attribute of
an arbitrary object, and it calls base-class methods directly, so a subclass's `__str__`
or `__float__` never runs.

- `None`, `bool`, `int`, `float`: inline; non-finite floats become the strings `"nan"`,
  `"inf"` and `"-inf"`.
- `str`: inline up to 4000 characters; a longer string becomes `{truncated, preview,
  length}`, with the whole text in a `txt.gz` asset.
- numpy arrays: `shape` and `dtype`, and the values inline when `size <= 64`.
  `summarize` copies a larger array now, within the array budget, and the writer saves the
  copy as `npy` with `stats` (`min`, `max`, `mean`, `n_finite`, `n_nonfinite` over the
  finite values; null values when there are none). The writer saves a `uint8` image of at
  least 8 x 8 pixels with 3 or 4 channels that a service receives as PNG instead. Object
  and structured dtypes keep shape and dtype only.
- An array over the array budget keeps `shape`, `dtype`, `not_saved: "array_budget"` and a
  bounded hash: `sha256` of the whole array up to 1 MB, else `sha256_prefix` of its first
  1 MB in C order with `sha256_prefix_bytes`. One summary hashes at most 4 MB, and after
  that the array gets `sha256_skipped: "hash_budget"` (`"hash_failed"` when hashing
  failed). The event then carries `array_budget_exceeded`, and an `array_budget` warning
  follows.
- `list`, `tuple`, `dict` and the dict types of `collections`: recursive, at most 4
  levels deep, 50 items per level and 2000 nodes per summary; the rest is marked
  `truncated`. Named tuples and dataclass instances: their fields.
- PIL images: size, mode and a PNG copy (only for a loaded image, within the budget).
  torch tensors: shape, dtype and device. `bytes`: type and length.
- Any other object (a generator, iterator, file, module, ...): its type name only.
- Parameters, keys and fields whose name is in the secret-key list of section 6.3 (exact,
  case-insensitive) have their value replaced by `"***masked***"`.
- Units and coordinate frames come from a table in the trace recorder for API
  functions whose docstrings state them, for example `get_object_pose` of the Franka API
  classes (metres, and a unit quaternion in `(w, x, y, z)` order). For everything else the
  viewer shows 「単位・座標系: 不明」 (units and coordinate frame unknown).

## 6. Model payloads

### 6.1 Capture point

Every model request leaves through `requests.Session.send`, including ordinary,
streamed and perception-model requests. There `PreparedRequest.body` is exactly what is
sent: a `str` for `data=json.dumps(...)`, or `bytes` for `json=`. The HTTP hook (H1) keeps a reference to
it and never reads the headers.

The HTTP hook (H1) classifies a request by the innermost open context of its thread:

- inside a logical model call opened by the model call hook (H3): a `model_request` for this HTTP attempt;
- inside a Molmo service call handled by the service wrapper hook (H9): a `perception_vlm` request; each object's retry loop
  is one logical call, which the HTTP hook (H1) opens and the perception parser hook (H14) closes;
- inside another service call: one HTTP attempt of that call;
- anywhere else: nothing is recorded.

For `stream=True`, the HTTP hook (H1) records the request and, when `send` returns, the status and the
timing. It never touches `r.content`, `r.text`, `r.json()`, `r.raw` or `iter_*`; the model call hook (H3)
generator wrapper records the streamed text. For a response that is not streamed, `send`
has already read the body, and the HTTP hook (H1) keeps the cached `bytes`.

### 6.2 Normalisation and images

`payloads/<event_id>.request.json` is the parsed body with each `data:` URL replaced by a
placeholder:

```json
{"$asset": "assets/3f/3f9a....png", "sha256": "...", "mime": "image/png", "bytes": 312004}
```

The asset holds the decoded bytes exactly as sent. The placeholder also records `prefix`
when the data URL did not start with the plain `data:<mime>;base64,`,
`noncanonical_base64: true` when the base64 text was not canonical, and `written: false`
with a `reason` when the asset was not written.

Each entry of `model_request.images` has `path` (the JSON path in the payload), `order`
(the position among the media parts, so the order of a VDM's "before" and "after" images
is explicit), `kind` (`image`, `video`, `external_url`, or `other` for another data URL),
`asset`, `written`, `sha256`, `mime`, `bytes`, `width`, `height`, `pixel_sha256` and
`source`.

`source` says where the image came from, by exact hash match only:

- The writer decodes a PNG image below 50 megapixels and hashes its raw pixels. Its
  `source` is the harness render (`observation_snapshot`) with the same `pixel_sha256`:
  that event's `kind`, `event_id`, `camera`, `sim_step` and `wall_time`. The writer
  prefers renders to images seen in API arguments or return values; the image of a Molmo
  request matches the `perception_input` of its service call.
- Two renders of an unchanged scene have the same pixels. The recorder assigns sources
  from the last media part of a request to the first, and each part takes the latest
  matching render that no later part of the same request took. Such a source carries
  `ambiguous: true` and lists every match in `candidates`.
- A VDM video is matched through video encoder hook (H12). The recorder hashes the frames passed to
  `_encode_video_base64`, finds the `video_frames_read` event with the same frame hashes,
  and indexes the MP4 bytes; the request's video then gets
  `{kind: "video_frames_read", event_id, frame_range, camera, n_frames}`.
- Otherwise `source` is `{kind: "unknown"}`, with a `reason` when a dropped item explains
  it. JPEG images are not decoded, so their source is always unknown.

The recorder keeps external `http(s)://` image URLs, with the query removed, as
`kind: "external_url"` and `fetched: false`, and never fetches them. The stored
asset is the image the model received, and harness renders and buffer frames are separate
files with their own labels.

### 6.3 Masking, redaction and reconstruction

Masking of payloads. The recorder masks the values of these key names, compared exactly and
case-insensitively: `api_key`, `apikey`, `x-api-key`, `authorization`, `access_token`,
`refresh_token`, `id_token`, `token`, `client_secret`, `secret`, `password`,
`api-key`, `x-goog-api-key`, `aws_secret_access_key`, `private_key`,
`auth_token`, `session_token`, `passwd`. Only whole
names match, so the recorder keeps `max_tokens`, `max_completion_tokens` and
`budget_tokens`. It replaces the whole value by `"***masked***"` whatever its type, and it
moves nothing below a masked key to an asset or lists it as an external URL.
`masked_fields` lists the masked JSON paths and the external URLs whose query was removed.

Reconstruction. Before masking, the writer puts back each placeholder as
`data:<mime>;base64,<base64 of the asset>` (with the original prefix), serialises the
result with `json.dumps(obj)` (default separators, `ensure_ascii=True`) and compares its
sha256 with `body_sha256`. For a `str` body it compares the ISO-8859-1 bytes that
`http.client` sends. The result is `{verified: true, method: "json_roundtrip"}`,
`{verified: true, method: "masked_compare", masked: [paths]}` when masking changed the
stored copy, or `{verified: false, reason}`. A failed check (`body_not_str_or_bytes`,
`not_json`, `roundtrip_mismatch`, `noncanonical_base64`, ...) counts a `payload_unparsed`
gap. The writer stores non-finite numbers as strings and then sets
`nonfinite_values_stored_as_strings`. The raw body is never stored.

The viewer labels the stored request 「保存用正規化済みpayload」 (payload normalised for
storage) and lists which parts were moved to assets and which fields were masked.

Redaction of stored error texts. `payload.redact()` removes secrets from every error text
the recorder stores: exception messages, causes and tracebacks, HTTP error bodies and the
errors of HTTP attempts, hook failure reasons, recorder errors, and the error texts of
runner records. It replaces only what the four rules below describe. Names, paths with
trial ids, digests, code and prose outside these rules are stored as they are.

- A URL loses its userinfo and its whole query, and keeps its scheme, host, path and
  fragment. Text counts as a URL when it begins with `scheme://`, or when it is a path
  that begins with `/` at the start of the text or after whitespace, a quote or one of
  `( [ { < = : ,`.
- The credential after `Bearer`, `Basic`, `Token` or `Digest` (the schemes of an HTTP
  `Authorization` header) becomes `***redacted***`.
- API keys of known formats become `***redacted***`: OpenAI (`sk-`), Google (`AIza`),
  GitHub (`ghp_`, `gho_`, `ghs_`, `github_pat_`), Hugging Face (`hf_`), Slack (`xoxa-`,
  `xoxb-`, `xoxp-`, `xoxr-`), AWS (`AKIA`, `ASIA`), Groq (`gsk_`),
  xAI (`xai-`), Replicate (`r8_`), and PEM and PGP private-key blocks.
- The value of each secret key name listed above becomes `***masked***`. The name counts
  when a separator (`:` or `=`) follows it, with any whitespace, line breaks included, on
  either side. The name may be quoted, the index of a subscript
  (`config["password"] = ...`) or a flag (`--password=...`). The whole value is masked: a
  string up to its closing quote (f-strings included), a value in `[...]`, `{...}` or
  `(...)` up to its matching bracket across lines, or any other value up to whitespace or
  one of `, ; & ) ] }`. In a Python annotation (`password: str = "..."`) the value after
  `=` is masked. Empty values, `None`, `null`, `true`, `false` and references such as
  `os.environ["API_KEY"]`, `get_key()` and `args.api_key` stay.

An error text can hold a JSON string or the `repr` of a Python string, and both write a
line break or tab as `\n`, `\r` or `\t`. Before a URL path, an `Authorization` scheme, an
API key or a key name, and around the separator after a key name, the rules read such an
escape as whitespace. A secret at the start of an escaped line is therefore found as in
plain text. The trace recorder defines the complete redaction rules. Its redaction has limits.

`redact()` redacts the first 65536 characters of a text as a whole and then cuts the
result to the stored length of the field, so a secret that crosses the cut is replaced
whole. A field gets `<field>_redacted: true` when a rule replaced something in the stored
part of the text. A text stored without a length limit and longer than 65536 characters
keeps its start, ends with a note, and gets `<field>_cut: true`.

### 6.4 Input summary and provenance

The recorder keeps, per episode, text segments whose source it knows, and it records where
they occur in every model request. The viewer groups them into the categories
タスク/API説明 (task and API description), 実行コード履歴 (history of executed code),
stdout/stderr, VDM説明 (VDM description) and その他 (other).

| Kind | Registered from | Viewer category |
|---|---|---|
| `task_prompt` | the task text of each code-generation request (the text before `"\nAPIs:\n"` in its last user message); the same text after `_patch_libero_goal`; `env._task_prompt`; the text before `"\nAPIs:\n"` in `env._full_prompt` | タスク/API説明 |
| `api_docs` | the text after `"\nAPIs:\n"` in `env._full_prompt` | タスク/API説明 |
| `harness_template` | `env._system_prompt`; the separator `"\nAPIs:\n"`; the literal parts of `multi_turn_prompt`; the prose string constants of the harness functions that build prompts, read from their code objects | その他 (harness template) |
| `code_history` | the code of each executed block (the action passed to `step`) | 実行コード履歴 |
| `stdout`, `stderr` | each block's stdout and stderr as `step` returned them | stdout/stderr |
| `vdm_description` | the content of each VDM response | VDM説明 |
| `user_feedback` | text injected through the web UI | その他 (user) |

The model call hook (H3) copies the segments for a request when its logical call starts, from the
env's state at that moment, because some envs rebuild their prompt at reset and LIBERO
fills in the goal after reset. Every HTTP attempt of the call uses that copy. A perception
VLM (Molmo) request gets no copy, so its prompt text shows as 「由来不明」 (unknown origin).

For each request the writer searches every segment of at least 8 characters, and the
7-character separator `"\nAPIs:\n"`, as an exact substring of every text part, at most 20
times per segment and part. It merges segments with the same kind and text and searches
them once, naming all their sources. It matches nothing by meaning, numbers or
coordinates. Offsets count the characters of the Python string (code points).
`input_summary`:

```json
{
  "n_images": 1, "n_videos": 0, "no_image_input": false,
  "categories": {
    "images": [{"path": "messages[1].content[1].image_url.url", "order": 0, "kind": "image", "asset": "assets/3f/3f9a....png", "source": {"kind": "observation_snapshot", "event_id": "ev_...", "camera": "...", "sim_step": 126, "wall_time": 1790000000.0}}],
    "task_api_description": [{"kind": "api_docs", "path": "messages[1].content[0].text", "start": 812, "end": 20417, "source_event_ids": []}],
    "code_history": [], "stdout_stderr": [], "vdm_description": [], "other": []
  },
  "unattributed": [{"path": "messages[1].content[0].text", "start": 0, "end": 11}],
  "registered_not_found": [{"kind": "vdm_description", "length": 1843, "source_event_ids": ["ev_..."]}]
}
```

`unattributed` lists the character ranges that no segment covers, at most 200
(`unattributed_truncated: true` when the list was cut). Ranges that hold only whitespace,
such as the newlines that join text parts, carry `whitespace_only: true`; the viewer shows
the other ranges as 「由来不明」 and counts the whitespace ranges on a separate line.
`registered_not_found` lists the dynamic segments (code history, stdout, stderr, VDM
descriptions, user feedback) produced earlier in the episode that this input does not
contain, for example the first VDM scene description, which the later decision prompts no
longer carry.

Videos are counted separately from images. The viewer marks a request with neither as
「この呼び出しは画像入力なし」 (no image input in this call), and only when zero images and
zero videos were measured; a request whose body was not parsed shows 画像: 未記録 (images:
not recorded) instead.

For an API call or a code block, the viewer reports whether the block's output reached a
later model input (the later-input report), using the recorded `input_summary` only. It
checks every model request after the block's `code_block_end`, each HTTP attempt
separately, up to and including the first code-generator request, followed by the other
HTTP attempts of that code-generator call, which it marks as resends of the same logical
call. For each request it reports:

- For each of the block's streams, stdout and stderr: 「含まれています」 (included) with the
  payload path and offsets; 「含まれていません（照合済み）」 (not included, searched) only when
  the recorder searched the stream for this request (`registered_not_found` names it);
  「空（出力なし）」 (empty); or 未確認 (unchecked) when the stream has 1 to 7 characters
  (too short to be registered), was not among the segments searched for this request, or
  is missing from the record although `code_block_end` gives its length. When the block's
  stdout or stderr event is missing and its length is unknown, the report says 未記録 (not
  recorded).
- Inside a found stream, the API's own printed range (`stdout_slice`, `stderr_slice`).
  The viewer maps offsets into the payload only when the segment is exactly this block's
  stream: it names this block's stream event (or the block, when that event is missing)
  and has the stream's length. Otherwise the report says that the position inside the
  segment was not computed. The API's printed range is 未記録 when `api_call_end` or its
  slice is missing.
- For `api_call_error`, the report checks whether the exception text `str(e)` (at least 8 characters) or this
  exception's final traceback line appears in the block's stdout or stderr, and therefore in the
  later request. The final traceback line must be `<type>: <message>` with the whole message;
  the type may carry its module; an empty `str(e)` matches the bare type; a message of
  several lines spans those lines; and the type alone never matches when `str(e)` is not
  empty. When the same line appears several times, the report says that the exception it
  belongs to cannot be told apart. A message stored with `message_redacted: true` or
  `message_cut: true` differs from what the program printed, so it is not searched, and
  the report says 未確認 with that reason.

For a value that the API returned without printing it, the viewer says that the API
printed nothing, whether the block's stdout (N characters, including prints by the
generated code) was in the next code-generator input and where, and that it did not check
whether this value is part of it. It never searches the model input for numbers or
coordinates. When no later code-generator request exists, it shows
「このエピソードでは以後のモデル呼び出しなし」 (no later model call in this episode).

### 6.5 What the recorded body shows

The body is what the CaP-X client sent to the endpoint. Upstream CaP-X includes an
OpenRouter proxy in `capx/serving/openrouter_server.py`. Its request model defaults to
`temperature=0.2` and `max_tokens=256`. The proxy strips an `openrouter/` model prefix and
forwards the request to OpenRouter. Its Pydantic request model drops unknown request
fields by default and rejects message content items other than `text` and `image_url`.
The OpenAI SDK client retries 429 and 5xx responses within one client attempt using its
default `max_retries`. For non-streaming replies, the proxy rebuilds the response without
`usage` and `reasoning`. The recorded body is the request before any proxy
changes. `endpoint.known_proxy` is a guess: it is true for a loopback host on a port
listed in `CAPX_TRACE_PROXY_PORTS` (default `8110`), and the viewer then shows a proxy
note. If a different service listens on that port, the note may not apply to it. Set
the variable to an empty string to turn the note off.

## 7. Hook installation and compatibility

The recorder installs hooks after each target module loads. A missing required symbol or an invalid callable goes in `hooks_failed` and counts as a trace gap. Only symbols explicitly listed as optional in `bootstrap.OPTIONAL_SYMBOLS` go in `hooks_skipped`; the upstream target modules currently have no such omissions. The recorder attempts an install only for modules that are imported. If H4 cannot install, `recorder_warnings.<pid>.jsonl` at the trace root explains why no episodes were recorded.

### 7.1 Table

`capx_trace_recorder/bootstrap.py` installs hooks when tracing is on. Each install
checks that the target exists and has the expected shape. A missing required symbol or
an incompatible callable goes in `hooks_failed`. If the optional H2 target
`_patch_libero_goal` is missing (upstream 53e9966 defines it), it goes in `hooks_skipped`. Wrappers carry `__capx_trace_wrapper__ = True` and are
never applied twice to the same callable.

| Id | Target | How | Records |
|---|---|---|---|
| H1 | `requests.sessions.Session.send` | class attribute wrapper | `model_request`, `model_response`, `model_error` per HTTP attempt; the HTTP attempts of service calls; Molmo requests and their logical calls |
| H2 | `capx.envs.trial`: `_describe_initial_scene`, `_get_visual_differencing_feedback`, `_get_video_differencing_feedback`, `_query_initial_code`, `_handle_multi_turn_step`, `_patch_libero_goal` | module attribute rebinding (the trial loop calls them as module globals); the module's aliases of H3, H5, H6 and H12 targets are rebound as well | role context (`role`, `call_kind`, `turn_index`); the task text after `_patch_libero_goal` |
| H3 | `capx.llm.client`: `query_model`, `query_model_streaming`, `query_model_ensemble`, `query_single_model_ensemble` and the module's own underscore aliases | rebinding in the defining module (callers bind their aliases at import, after the patch) | logical calls (the only place a `logical_call_id` is created for code-generation and VDM calls); ensemble member and synthesis roles; streamed responses; preflight requests; `task_text` |
| H4 | `capx.envs.trial._run_single_trial`, also rebound in `capx.envs.runner` | rebinding before `capx.envs.runner` and `capx.experiments.run` import it; the first four parameters must be `env, trial, args, config` | `episode_start`, `episode_end`, the episode's `evaluator_result`, and `trial_videos`; copies completed upstream trial MP4 files from the returned `code_path` directory; installs H10 on the env before the original resets it |
| H5 | `capx.utils.launch_utils`: `_build_multi_turn_decision_prompt`, `_build_multi_turn_decision_prompt_legacy` | rebinding | `feedback_built` |
| H6 | `capx.utils.launch_utils`: `_parse_multi_turn_decision`, `_extract_code` | rebinding | `model_output_parsed` |
| H7 | `CodeExecutionEnvBase`: `step`, `_exec_user_code`, `render`, `render_wrist`, `get_video_frames_range`, `get_wrist_video_frames_range`, `reset` | class attribute wrappers | `code_block_start`, `code_block_end`, `stdout`, `stderr`, the block's `evaluator_result`, `observation_snapshot`, `video_frames_read`, `env_reset`; binds a web UI env to its episode; installs H8 and H9 at the first `step` of an env in an episode |
| H8 | the public API methods that `api.functions()` returns | instance attributes `types.MethodType(functools.wraps(fn.__func__)(traced), fn.__self__)` | `api_call_start`, `api_call_end`, `api_call_error`, including nested `self.x()` calls |
| H9 | the service closures on API instances that exist among `sam3_seg_fn`, `sam3_point_prompt_fn`, `sam2_seg_fn`, `sam2_point_prompt_fn`, `owl_vit_det_fn`, `molmo_point_fn`, `grasp_net_plan_fn`, `grasp_net_plan_point_clouds_fn`, `ik_solve_fn`, `trajopt_plan_fn` | instance attribute replacement | `service_call_start`, `service_call_end`, `service_call_error` |
| H10 | the low-level env's `_record_frame` | instance attribute wrapper, installed at episode start (web UI: when the env is bound) | buffer index to `sim_step`; `frame` events |
| H11 | `traceback.walk_tb`, `traceback.walk_stack`, `traceback._walk_tb_with_full_positions` (Python 3.11+) | module attributes of the stdlib `traceback` module | recorder frames hidden from printed tracebacks and stacks (section 7.4) |
| H12 | `capx.utils.video_utils._encode_video_base64` | rebinding in the defining module | the frame hashes of each VDM video, matched to `video_frames_read` |
| H13 | `capx.web.async_trial_runner.run_trial_async` (an async wrapper); the web runner's aliases `_query_model` (VDM calls) and `_query_model_streaming` (code generation); `SessionManager.inject_prompt` and `stop_session` in `capx.web.session_manager` | rebinding in the defining modules | web UI episodes and their roles, user feedback, `end_reason` |
| H14 | `capx.integrations.vision.molmo._parse_points` | rebinding | Molmo's reply as parsed (`model_output_parsed`); closes the object's logical call |
| H15 | `capx.experiments.manifest.load_manifest`, `capx.experiments.revision.fingerprint` | rebinding in the defining modules (`run.py` binds `load_manifest` at import, after the patch, and calls `revision.fingerprint` as a module attribute) | adds the `tracing` fingerprint part; learns the run id and run directory and exports `CAPX_TRACE_RUN_ID` and `CAPX_TRACE_RUNNER_RUN_DIR` before the preflight and the workers start |
| H16 | `capx.experiments.records.append_jsonl`, for the runner's `trials.jsonl` only | rebinding | `runner_records.jsonl` |

Install order. H11 is installed first and runs a self-test (section 7.4). H1, H4 and H7
put recorder frames on the stack while generated code runs, and so do H8, H9 and H10,
which H4 and H7 install later. The bootstrap installs these hooks only when the H11
self-test passed. Otherwise it records each of them in `hooks_failed`.

Hooks that fail. A public function that H8 cannot wrap, or a service closure that H9
cannot replace, makes the install partial. The episode records a partial install as
`hook_install_failed`, which is a gap. In the web UI server the trial module and the env
classes are first imported inside an episode, so the bootstrap installs H2, H4 and H7
while that episode is open. It adds such installs to `episode.json` as
`hooks_installed_during_episode` and turns a failure into a `trace_warning` with
`during_episode: true`. In the experiment runner the recorder also writes each hook
failure once to the run's unscoped log (`run_level: true`), no later than the moment H15
learns the run id. When H4 has failed, the recorder labels the runner's requests `unknown`
instead of `preflight`, because they cannot be told apart from trial requests.

`episode.json` also records `apis_wrapped`: for each API instance of the env, the
wrappers present on it now (`wrapped`, `services`), whether they were installed in this
episode (`installed_in_this_episode`), and the functions skipped. An env reused across a
worker's episodes keeps the wrappers of its first episode.

Interrupted starts. H4 and H13 run these steps as one unit with its own cleanup: open the
episode, bind its objects, queue `episode_start`, and push the episode on the thread. If
an alarm or a recorder error interrupts that unit, the cleanup removes the episode from
the set of open episodes, from the thread's stack, from its env, low-level env and
launch-argument bindings, and from the web session table. The H10 hook also stops copying
frames for it. When its `episode_start` may already be queued, the recorder ends the episode (`end_reason` `timeout_alarm`, `error` or
`exception`) with an `end_note`. After a recorder error the trial runs untraced. After an
alarm, or another exception that is not a subclass of `Exception`, the recorder runs this cleanup
inside `guard()` and then re-raises the original exception before the trial starts. A
failure inside the cleanup becomes an `internal_error` warning, and the original exception
still propagates. A recorder error, or an alarm that came before anything was queued, also
writes an `internal_error` warning with `outside_episode: true`, `episode_id`, `alarm` and
`note` to the run's unscoped log.

### 7.2 API wrapping (H8)

`combined_doc()` loops over `self.functions()`, which returns `self.<name>` bound methods,
and writes `name + str(inspect.signature(fn))` and `inspect.getdoc(fn)` for each. Wrappers
installed as instance attributes of the `MethodType` form keep both values identical,
including inherited docstrings. A plain function wrapper would lose them. For each API
instance the recorder computes `combined_doc()`, installs the wrappers and computes it
again. If the two results differ in any way, it removes the wrappers from that instance and
records `hook_install_failed`. The CaP-X integration test checks each concrete `ApiBase` subclass available in that environment.

Each thread keeps a stack of open API calls. `depth` is the number of open API calls, and
`parent_event_id` is the innermost open call, or the code block for depth 0. H8 wraps only
the methods listed in `functions()`. A depth-0 call records the call site, key frames, a
ground-truth state snapshot and the offsets of the buffers that capture stdout and stderr
during `exec`. The wrapper summarises the arguments before the call and the return value
right after it. For an `api_call_error`, the trial thread keeps the traceback as
`(filename, lineno, name)` for each frame outside the recorder. The writer then looks up
the source lines and formats the text. This way `linecache` never runs on the trial
thread, where its `except OSError` could swallow the alarm.

### 7.3 Episode and role lookup

Only the H3 wrappers create logical model calls, except for Molmo, where H1 creates one per
object and H14 closes it. The H2 and H13 wrappers set the role context (`role`,
`call_kind`, `turn_index`) in per-thread state and clear it when the wrapped function
returns or raises. Ensemble members run on a `ThreadPoolExecutor` in
`query_model_ensemble` and `query_single_model_ensemble` in `capx/llm/client.py`. The web runner runs its env on an executor thread (`async_trial_runner.py:143`)
and streams code generation on others (`:364` and `:710`). None of these threads inherits
the context. The H3 wrapper therefore takes the role from the first source that applies:
the thread's own role stack; the episode's current role, which the caller set before the
pool started; `preflight` in the experiment runner before any episode (when H4 is
installed); and otherwise `unknown`.

An event with no enclosing context of its own finds its episode by these steps; the first
one that gives an answer wins:

1. the web UI episode of the current `contextvars` context, which H13 sets and which
   reaches `asyncio.to_thread` workers (VDM calls) but not `run_in_executor` threads;
2. the thread's own episode stack, which H4 pushes on the trial thread;
3. the innermost open context of the thread (code block, API call, logical model call,
   service call), each of which carries its episode;
4. the binding of a known object: the env or its low-level env (bound by H4 and H7), or the
   web runner's launch arguments (bound by H13). A binding outlives its episode, so events
   from the env of an ended episode go to that episode as late events. When an object was
   bound to a newer episode after a thread last used it, a thread started for one job keeps
   its events on the older episode. On a pool worker, the main thread or a thread of
   unknown kind, the event is ambiguous (`_bound_episode` in `capx_trace_recorder/context.py`);
5. an open web UI episode whose session holds the env (`session.env`), so that the env of
   each web trial is bound to its own episode even while two web episodes are open;
6. the single open episode of the process.

With more than one candidate, a model call (H3) and its HTTP attempts go to the unscoped log
of the first candidate's run, and every other event is dropped. Each candidate episode
counts one `ambiguous_episode` gap per such event or model call, and one explaining warning
per (hook, run, candidates) goes to the run's unscoped log and to each candidate. The
warning's text says "were not recorded" for every hook, although for H3 the calls are in
the unscoped log. A model call with no candidate episode is recorded only in the experiment
runner's parent process, in the run's unscoped log (the preflight, for example).

H13 wraps the web runner's `_query_model_streaming` as a generator (`yield from`). It
sets the context in the iterating thread for the whole iteration, and the H3 wrapper
inside it records `model_response` from the `done` chunk (content, reasoning, `n_chunks`,
`ttft_s`), or `model_error` with the partial text on an exception or `GeneratorExit`.

The wrappers push each context as the first statement inside their own `try` and pop it
in its `finally`, so an alarm cannot leave one behind between the push and the `try`, or
after the original returned. H4 clears whatever is still on the trial thread when an
episode starts or ends, records it as `stale_context`, and ends an episode that an
interrupted exit path left open (`end_reason: error`, with an `end_note`). An alarm that
lands after the original `_run_single_trial` returned still ends the episode, as
`timeout_alarm`, before the `TimeoutError` leaves H4.

### 7.4 Traceback protection (H11)

Sandbox tracebacks and stacks printed through the `traceback` module must be the same as
without the recorder, whoever prints them: `base.py:178`, generated code with its own
`import traceback`, `two_arm_lift.py` (a function-local import), or `logging`. H11 replaces
`walk_tb`, `walk_stack` and, on Python 3.11+, `_walk_tb_with_full_positions` in the stdlib
`traceback` module with generators that skip the frames of recorder code (module name `capx_trace_recorder` or below, or a file under `capx_trace_recorder/`). All formatting functions
(`print_exc`, `format_exc`, `print_exception`, `TracebackException`, `extract_tb`,
`print_stack`, `extract_stack`) take their frames from these generators, so `limit` counts
the same frames as in an untraced run. For `walk_stack(None)` the start frame counts only
visible frames, with the number of hops measured from the running interpreter's own
`walk_stack` (2 on Python 3.10, 4 on 3.12).

On any internal error a replacement yields the original sequence, and a `TimeoutError`
that lands inside it is re-raised, as it would be from the stdlib code in an untraced run.
The self-test runs a toy exception through a wrapper and checks that the traceback text
is the same with and without it, that `print_stack` inside a wrapped call matches too, and
that `walk_stack(None)` starts at the same frame. When anything differs, the recorder
restores the originals and skips the hooks that add frames.

H11 does not cover code that walks `exc.__traceback__` or frames by hand, and the C-level
printing of an uncaught exception at the top level (which is not model input). When an
alarm or `KeyboardInterrupt` lands inside recorder code, the traceback differs from an
untraced run in any case, because the interrupted location differs. As a last check, the
writer scans each block's stdout and stderr for the recorder's path
(`tracer_frame_visible`).

### Upstream compatibility

The recorder was tested with upstream CaP-X 53e9966 through the `.pth` hook. Two
traced Cube Stack trials ran in two spawned workers, and the SHA-256 of each copied
video matched the original file.

| Target | Upstream CaP-X status | Recorded behavior | Skipped behavior |
|---|---|---|---|
| `requests.sessions` | Present | H1 HTTP requests and responses | None |
| `capx.llm.client` | Present; model reply is a dictionary despite its `str` annotation | H3 logical and streamed calls | None |
| `capx.utils.launch_utils` | Present | H5 feedback prompt; H6 output parsing | None |
| `capx.utils.video_utils` | Present | H12 frame hashes | None |
| `capx.envs.trial` | Present; `_save_trial_artifacts` creates the final named directory and `_save_turn_and_combined_videos` writes into it before `_run_single_trial` returns | H2 roles; H4 batch episodes and completed MP4 copies | None |
| `capx.envs.runner` | Present | Rebinds H4 trial alias | None |
| `capx.envs.tasks.base` | Present | H7 code, streams, observations; H8 API and H9 service setup; H10 frames | None |
| `capx.web.async_trial_runner` | Present | H13 web trial lifecycle | None |
| `capx.web.session_manager` | Present | H13 session feedback and end reasons | None |
| `capx.integrations.vision.molmo` | Present | H14 parsed perception replies | None |
| `capx.experiments.revision` | Absent | Applies H15 only when a CaP-X variant provides and imports this module | H15 runner identity on upstream |
| `capx.experiments.manifest` | Absent | Applies H15 only when a CaP-X variant provides and imports this module | H15 manifest metadata on upstream |
| `capx.experiments.records` | Absent | Applies H16 only when a CaP-X variant provides and imports this module | H16 runner record copy on upstream |


## 8. Frames, clocks and state

### 8.1 Frame sources

- The existing video frame buffer (`_frame_buffer`, and `_wrist_frame_buffer` when
  present), read by index and copied. While recording is on, `_record_frame` appends to it
  every 1 to 10 control steps, depending on the simulator: every `_subsample_rate` steps
  for robosuite and LIBERO (1 in `robosuite_handover.py`, 2 for Cube Lift, 10 for Spill
  Wipe) and at every `step` for R1Pro. Some manifests turn recording off
  (`record_video: false`); such an episode has no buffer frames, and the viewer shows
  「未取得」 (not captured).
- Frames the harness renders itself through `render` and `render_wrist` (environment hook H7): visual feedback,
  wrist images and web UI frames. Only these can match a model-input image.
- The buffer entries of R1Pro are dicts of camera arrays, so frames are keyed by
  (`epoch`, `buffer_index`, `camera`).

The main camera's name is the low-level env's `save_camera_name`, else `agentview` for
LIBERO and `main` otherwise; the wrist camera's name is its `_wrist_camera_name`, else
`wrist`.

### 8.2 Key frames and periodic frames

- Key frames: at each `code_block_start` and `code_block_end` and at the start and end of
  each depth-0 API call, the recorder copies the latest frame of each buffer. The event
  stores `frame_ref = {epoch, buffer_index, camera, frame_sim_step, path, written}`
  (`frame_refs` when there are several buffers) next to its own `sim_step`, and the viewer
  shows the difference as 「このフレームは N step 前」 (this frame is N steps earlier).
- Periodic frames: when a new frame arrives in the main buffer, at most
  `CAPX_TRACE_FRAME_FPS` frames per wall-clock second.
- `epoch` increments when a buffer is cleared or replaced (its length drops, or it is a new
  list object).
- Frames already in the buffer when the frame hook (H10) is attached are back-filled with
  `sim_step: null`.

### 8.3 Simulator clock

The recorder reads the clock without side effects. `sim_step` is the first of
`low_level_env._sim_step_count`, `robosuite_env.timestep` and `env.episode_steps`
(R1Pro). `sim_time` is `sim.data.time` of the MuJoCo simulation (robosuite, or LIBERO's
`handle.env.sim`), else LIBERO's `get_current_time_s()`. Only the thread that drives the
simulator reads the clock: the trial thread, or the web UI's env thread. Other threads use
the last value read.

### 8.4 Ground-truth state

For MuJoCo-based envs (robosuite and LIBERO), the recorder copies `sim.data.time`, `qpos`,
and `xpos` and `xquat` of all named bodies at the start and end of code blocks and
depth-0 API calls when `CAPX_TRACE_STATE=1`. The writer stores the snapshot as a `json.gz`
asset (`state_ref`) with `quaternion_order: "wxyz (MuJoCo xquat)"` and
`frame: "MuJoCo world"`. The recorder caches body names per MuJoCo model object and checks
them against `model.nbody` on every use, because robosuite frees the model at each reset
and a new model can get the freed address. Other simulators get
`{path: null, reason: "unsupported_simulator"}`, which is not a gap. A supported simulator
whose state cannot be read gets a `state_unavailable` gap, and a snapshot dropped for the
queue budget gets `{path: null, written: false, reason: "queue_full"}`. The viewer shows
the state only in the debug tab.

## 9. Viewer

### 9.1 Launch and access

The standalone server serves the viewer at `/traces` and redirects `/` there. Run `python -m capx_trace_viewer.server --trace-root /path/to/traces --port 8290`. The default trace root is `CAPX_TRACE_DIR`, then `outputs/traces` under the working directory. The default runner root is `outputs/runner`. Use `--runner-root` to read other runner outputs. A missing frontend build returns 503 and a build hint.

The API serves `/api/traces`. Unless `--expose-traces` is set, it accepts requests only from loopback peers with a local `Host` and a local `Origin` when present. A reverse proxy must pass the real client address through `X-Forwarded-For`. Vite forwards `/api` to `127.0.0.1:8290` during development. `--expose-traces` adds no authentication. Anyone who can reach the API can read files under the trace root.

### 9.2 Backend (`capx_trace_viewer/trace_api.py`)

In both modes the server includes the router `/api/traces` before the catch-all route
that serves the page for any other path. The handlers are plain `def` functions, which
FastAPI runs in its thread pool, and they follow the rules of section 2. An unknown path
under `/api/traces` gets a JSON 404.

Ids and paths. Every run id, episode id and path segment must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$`, so `.`, `..`, empty segments and a decoded `%2F`
never reach the filesystem. The backend opens files through `capx_trace_viewer.reader.Dir`,
which opens the trace root (or runner root) once, then each path segment relative to its
parent with `O_NOFOLLOW`, and serves a file from the descriptor it checked. A symlink
below the root gets 403, even one that points inside it.

Errors. 404 means that a run, episode or file is not there or is not a regular file (a
directory, socket or device); 403 means a symlink or a permission error; 500 means
anything else (an I/O error, too many open files), with `cannot be read (<ERRNO>)` in
`detail`.

Files. The backend serves files with `X-Content-Type-Options: nosniff`,
`Content-Security-Policy: sandbox`, `Cache-Control: no-cache` and a MIME type from an
allowlist; any other file downloads as an attachment. It sends `*.txt.gz`, `*.json.gz`
and `*.jsonl.gz` with `Content-Encoding: gzip`, so the browser decodes them, and it
honours a single byte range, so videos can seek. `OpenFileResponse` in
`capx_trace_viewer/trace_api.py` describes how it treats malformed ranges and files that change
while they are sent. JSON answers are ASCII (`ensure_ascii=True`) and carry
`Cache-Control: no-store`.

Arrays. The backend reads `.npy` and `.npz` files without `np.load` (numeric and bool
dtypes only) and refuses oversized or damaged files with HTTP 400; the limits are in
`capx_trace_viewer/trace_arrays.py`. At most 2 array requests run at once, and another one waits up
to 20 s before it gets 503. `/array` returns shape, dtype, a preview of the leading values
and `image`, which says how `/array-image` draws the array: as a mask, a gray image, an RGB
image or a stack of them, at most 1024 pixels per side (`index` picks one image of a
stack). `stats` is always null, because the recorder stored the statistics in the event. A
bit-packed mask is presented as one bool array with `packed: true`.

Endpoints:

| Method and path | Response |
|---|---|
| `GET /api/traces/config` | `{trace_root, exists, viewer_only, runner_roots, expose, schema_version, heartbeat_stale_s}` |
| `GET /api/traces/runs` | `{runs: [{run_id, n_episodes, last_modified, status_counts}]}`, newest first |
| `GET /api/traces/runs/{run_id}` | `{run_id, run, run_json_error, runner_records, runner_records_info, episodes: [summary], unscoped_files}` |
| `GET /api/traces/runs/{run_id}/episodes/{episode_id}` | episode summary |
| `GET /api/traces/runs/{run_id}/episodes/{episode_id}/events?after_offset=0&max_bytes=2000000&max_events=5000&generation=` | `{events, next_offset, eof, torn_tail, corrupt_lines, corrupt_offsets, file_size, generation}` |
| `GET /api/traces/runs/{run_id}/episodes/{episode_id}/file/{relpath}` | a file below the episode directory |
| `GET /api/traces/runs/{run_id}/episodes/{episode_id}/array/{relpath}?name=` | `{arrays: {name: {shape, dtype, stats, preview, preview_shape, truncated, is_mask, packed, image}}}` |
| `GET /api/traces/runs/{run_id}/episodes/{episode_id}/array-image/{relpath}?name=&index=` | PNG of a mask, a depth or other numeric image, an RGB image, or one image of a stack |
| `GET /api/traces/viser` | `{available, port, reason}` |
| `GET /api/traces/legacy/runs`, `/legacy/runs/{run_id}`, `/legacy/trials/{run_id}/{task}/{condition}/{trial_dir}`, `/legacy/file/{run_id}/{task}/{condition}/{trial_dir}/{relpath}` | read-only views of runner outputs (section 9.4) |

Runs. A directory under the trace root is a run when it holds a regular `run.json` or
`runner_records.jsonl` (readable or not), an episode directory, an unscoped log or an
unscoped status file; the list and the detail use the same rule. A run whose `run.json` is
empty, damaged or unreadable is still answered, with `run: null` and `run_json_error`
(`symlink_not_followed`, `not_a_regular_file`, `unreadable (<ERRNO>)`, `invalid_json`,
`too_large (...)` or `not_an_object`); a missing `run.json` is not an error.
`runner_records_info.error` is set when `runner_records.jsonl` exists but cannot be read.
Episodes are ordered by `start_wall_time` when it is a finite number, else by the newest
file time, then by `episode_id`. `unscoped_files` lists `{file, bytes, log_id,
write_status}` per unscoped log; `write_status` is the recorder's
`unscoped.<pid>.status.json` or null, a log that exists only as that status file is listed
with `bytes` and `log_id` null, and a status file that cannot be read gives
`write_status_error`.

Episode summary. The summary has the fields `source`, `task`, `condition`,
`condition_label`, `seed`, `attempt_index`, `runner_invocation`, `start_wall_time`,
`end_wall_time`, `end_reason`, `trace_complete`, `gaps`, `dropped_events`,
`late_events`, `n_warnings`, `viser_port`, `pid` and `host`, plus `status`,
`status_detail`, `recorded_status`, `heartbeat_age_s`, `liveness_source`,
`last_modified`, `events_bytes`, `episode` (the raw `episode.json`) and
`episode_json_error`. It comes from `episode.json` and the heartbeat. The backend reads the
first line of `events.jsonl` (at most 1 MB) only when `episode.json` does not name one of
the start fields (a field present with a null value counts as named), and the run list
reads no `events.jsonl` at all. Episode status:

- `ended`: `episode.json` says `ended` and `trace_complete` is true;
- `ended_with_gaps`: `episode.json` says `ended` and `trace_complete` is not true;
- `running`: no `episode_end`, and the heartbeat file is younger than 30 s;
- `incomplete`: no `episode_end`, and the heartbeat file is older than 30 s or missing; the
  viewer shows 「記録が途中で止まりました（episode_end なし）」 (recording stopped before
  episode_end). File modification times other than the heartbeat's are never used, because
  a copied or unpacked episode has fresh times although nothing writes to it.

Events. `/events` pages `events.jsonl` by byte offset; a single line longer than
`max_bytes` is still read whole, so paging always advances. `corrupt_lines` counts the
lines skipped in this page. Each page carries the file's `generation`, an id built from
the file's device, its inode and a hash of its first line (`reader.events_generation`), so
a client can tell when the file was replaced or rewritten. A request that names another
generation gets HTTP 409 `{"error": "generation_changed", "generation": "<new>"}`. An
`after_offset` past the end of the file or not at a line start also gets 409.

Unscoped logs. The episode endpoints also serve a run's unscoped log, under the episode id
`unscoped.<pid>`: its summary has `kind: "unscoped"` and `status: "unscoped"`,
`/events` pages the log, and `/file`, `/array` and `/array-image` serve only paths below
`unscoped.<pid>.files/`. A real episode directory of that name would take precedence.

Viser. `/viser` probes ports 8080 and 8081 on 127.0.0.1 for a running Viser server,
never the server's own port. With `--viewer-only` it always answers
`{available: false, port: null, reason: "viewer_only"}`, because no Viser proxy route
exists there.

### 9.3 Layout

#### Default simple view

The default view groups records by one code execution. Select `実行回`,
`前の実行`, or `次の実行` to switch the video interval, generating prompt,
model response, stdout/stderr and result together. With no event selected it
starts at the first execution. The header explicitly labels the time-verified
runner outcome as `試行全体の最終結果`; each execution has its own result.
`試行を選ぶ` opens the experiment selector and trial cards.

For runner videos, the left panel plays only the selected execution's simulator-step
interval, mapped to the combined video using recorded frames (interpolated
between frames). A zero-length interval shows `動作なし` and a recorded starting
image. When a mapping is missing or ambiguous, the panel plays no unverified
interval. For upstream copied videos, a matching turn video plays directly. If only a
combined video is available and its interval cannot be mapped, the panel plays the
whole trial and labels it as such. Native video timestamps refer to the whole movie.

The right panel offers `入力プロンプト`, `モデル出力`, and `実行ログ`.
The code block's source response ID determines its generating HTTP request;
requests sharing a video timestamp remain separate. Missing source links show
unrecorded data rather than borrowing another input. The prompt panel shows
the latest user message verbatim, with earlier messages collapsed. The output
panel includes the stored response and expandable executed code; the log panel
shows that block's stdout and stderr. Recorded input images remain labelled.
External media URLs are never fetched. Missing payloads and loading errors
remain visible. Other perception calls remain accessible in the detailed view.

`詳細を見る` switches to the diagnostic layout below without changing the
selected run, episode or event; `シンプル表示に戻る` switches back. `view=details`
in the URL opens the diagnostic view directly. Metadata, API trees, hashes and
provenance tabs remain available there. Narrow screens stack the two simple panels.

#### Diagnostic view: in-page video playback

The media pane now opens `動画＋プロンプト` when a combined MP4 is available.
Native video controls support playback and seeking; the default speed is 0.5×.
`再生位置に追従` selects the model call at the current video position, scrolls
the event list, and reveals the actual request messages in the detail pane.
`コード・API` follows execution events instead. Clicking a cue pauses and seeks
the video and disables automatic following so the selected input stays visible.

Video lookup uses the time-verified final attempt's recorded artifact paths and
an exact match to a registered runner root. Earlier retry attempts never inherit
the final attempt's video. Existing endpoints retain their path and symlink checks;
the viewer neither opens arbitrary recorded absolute paths nor changes trace files.

Synchronization uses recorded simulator-step/buffer-index pairs and the existing
encoder's 30 fps, with linear interpolation between frame anchors. It is not a
wall-clock replay: model waits are absent from the MP4, and multiple requests can
share a video position. All such cues remain individually selectable; automatic
following selects the last cue at that position in event order. Automatic mapping
requires one recorded camera, epoch zero, an initial buffer index of zero, monotonic
step anchors, and a last-frame index consistent with video duration within about
one frame. Resets, other camera files and incomplete mappings disable following
with an explanation while retaining video playback. No new recording is required.
If no eligible MP4 is available, the recorded-frame view remains the default.

For upstream episodes, the simple view matches `video_turn_NN` to the recorded
`code_block_start.data.block_index`. It never guesses a missing index. Turn videos
play at 0.5× speed. If a turn video is absent, it uses a combined video. When the frame
records cannot identify the turn's clip, it labels the playback as the whole-trial video.
When several cameras have combined videos, it chooses `video_combined.mp4` first, then the
camera whose name comes first alphabetically. The detailed camera list shows combined
videos only. For `source: envs.runner` episodes, once run details have loaded with zero
runner records, the simple header uses the episode's `task_completed` value. Web UI and
fork runner episodes keep an unknown verdict until a matching runner record is available.

#### Diagnostic view: event and detail panes

The page uses the dark theme tokens of `web-ui/tailwind.config.js`, is laid out for a
desktop screen, and has Japanese UI text with technical identifiers kept as they are. It
adds no runtime npm dependency. Font ligatures are off, so recorded text such as `->` or `!=`
shows as it was sent.

Top bar. It shows the run, episode, task, seed, attempt and runner invocation; the
condition label marked 「ラベル（表示のみ）」 (label, display only) next to the flags of the
real config; the models and roles, with configured roles that sent nothing marked
「未使用」 (unused) and roles that are not configured marked 「（設定外）」; the episode status;
the polling status; recording gaps; three separate outcome fields; and the Live / Browse
switch with a 更新 (refresh) button.

- Recording gaps: the `episode_end` gaps and the `trace_warning` codes in plain words,
  hooks that failed (also during the episode), corrupt lines, a torn tail, a replaced
  `events.jsonl`, and `episode_json_error`. When the backend says `ended` but the reader
  found corrupt lines or failed hooks, the status reads
  「終了（記録完了と報告。ただし欠落を検出）」 (ended, reported complete, gaps detected).
- Outcomes: the evaluator (`task_completed` and reward, with their source); the model
  (「モデルは完了と主張」 only when a code-generator parse has `model_said_finish`); and the
  runner's verdict from `runner_records.jsonl`.
- Runner verdict. Runner records carry no invocation number, so the viewer joins a record
  to the episode by task, condition, seed and attempt, then by time: the record's attempt
  with this episode's attempt number must have started before the episode and ended after
  its start (runner times are whole seconds, hence a 2 s margin). Zero or several matches
  show 「対応する runner 記録を特定できません」 (no single matching runner record) with the
  reason; without an episode start time a single record is shown marked as unchecked. A
  record's `outcome` describes the trial's last attempt only, so the bar shows it as the episode's verdict (a green or red
  `task_success`) only when the episode is that attempt. For an earlier attempt it first
  shows that attempt's own status, for example
  「この試行（attempt 1）: timeout（生成コードの実行中）」 (this attempt: timeout, while the
  generated code ran), and then the `outcome` in plain text labelled
  「trial の最終結果（attempt 2。このエピソードの試行ではありません）」 (the trial's final result,
  attempt 2, not this episode's attempt). An attempt that starts and ends in the same second
  and does not contain the episode start (a record the runner's parent writes when a worker
  fails, always attempt 1) cannot be checked by time, and the bar says so. While the
  records are being read, when the read failed, and when `runner_records_info` reports
  that the file could not be read, the bar says so instead of 「runner の記録なし」 (no
  runner record). It also names lines of `runner_records.jsonl` that could not be parsed,
  a torn last line and a file cut at the reader's size limit.
- A run's unscoped log gets a one-line explanation instead of the episode fields.

Picker. It lists runs newest first, then the episodes of a run with status badges
(attempts of one seed next to each other), the `run_json_error` of a run and the
`episode_json_error` of an episode. The run's unscoped logs are listed under the episodes
and open like an episode, with the lines the recorder could not write (`write_status`).
Runner directories without a trace are listed separately (section 9.4). The URL holds
`run`, `episode` and `event` (and `legacy_run`, `legacy_trial`, `legacy_root` for a
runner trial without a trace), so a view can be shared.

Left pane (resizable). The task text comes first (「タスク文」, from `task_text`), then the
event timeline, grouped by cause from the ids of section 5: model call, parsed code, code
block, API calls, stdout/stderr, evaluator, VDM call, feedback, next model call. Events are
never linked by time proximity; an event whose parent id does not resolve is shown as an
orphan with the reason. The timeline groups HTTP retries under their logical call and folds
nested API calls; frames and snapshots never become cards. Cards:

- model call: role, model, image count, latency, retries and status. A `model_error` with
  scope `logical` is the call's result even after an HTTP 2xx answer, and attempt errors of
  earlier retries do not fail a call that was answered. A logical error of type
  `GeneratorExit` (the caller stopped reading a stream) shows as 中断 (stopped) instead of
  失敗 (failed), with its partial text;
- code block: highlighted code folded to a few lines, `sandbox_rc`, duration, and an
  excerpt of stdout and stderr;
- API call: name, argument and return summaries, duration, status, before and after key
  frames (top-level calls only), perception input images and at most two mask thumbnails
  of its service calls;
- VDM call: an excerpt of the description; evaluator: reward and `task_completed`.

Filters: event type, role (selecting roles shows the model calls of those roles, with their
callers as dimmed context rows), and failures only (failed, degraded, or open after the
episode stopped). Free-text search covers the text stored inline in the events (up to 4000
characters): code, stdout and stderr, response content and reasoning, API and service
names, and error messages; texts stored as assets are not searched. An open start event
shows 「終了イベントなし（実行中）」 (no end event, running) while the episode runs and
「終了イベントなし（プロセス終了）」 (no end event, process ended) otherwise. Late events are
marked 「エピソード終了後に届いたイベント」 (event that arrived after the episode ended). A
warning card names its code in Japanese and marks `run_level` (the whole process),
`outside_episode` and `counts_as_gap: false` (not a recording gap). Each
card, the top bar, the media pane and each detail tab has an error boundary, so an event
of an unexpected shape replaces only that part with a notice; the top bar's fallback keeps
the button that opens the picker.

Right pane, media (resizable against the details):

- for model calls, the images and videos that were sent, in payload order, labelled
  「モデルに送信」 (sent to the model), with hash, size and source. The recorder's `source`
  is shown as recorded; only when it is absent does the viewer look for an
  `observation_snapshot` taken before the request with the same `pixel_sha256`, and it
  labels that as a match found by the viewer;
- for code blocks and top-level API calls, the before and after key frames side by side,
  with their `sim_step` and 「このフレームは N step 前」;
- for every event, a flipbook over the recorded frames of the chosen camera: the frames
  within the event's `seq` span plus 8 before and after, or, for an open event, every frame
  since its start (at most 400), labelled 「デバッグ用フレーム（モデル入力ではない）」 (debug
  frame, not model input), with play, pause, step and speeds of 0.25 to 4 times a base of
  4 fps; `seq` only orders the frames;
- a camera switch and click to enlarge;
- the 「ライブ3D」 (live 3D) tab only when the episode is running, its recorded Viser port is
  the one the server's proxy serves, and the server is in full mode. It shows the
  `/viser-proxy/` iframe labelled 「現在のシミュレータ状態（選択中の過去イベントではありません）」
  (the current simulator state, not the selected past event). Otherwise the pane shows
  「過去3D未対応：記録済みフレームを表示しています」 (past 3D not supported; showing recorded
  frames).

Right pane, details, five tabs:

1. 実際のモデル入力 (actual model input): for model calls, the stored request in message
   order (role, full text with fold and unfold, search, copy), images inline with their
   metadata, the six-category summary (画像枚数, タスク/API説明, 実行コード履歴,
   stdout/stderr, VDM説明, その他) with positions marked in the text, 「保存用正規化済みpayload」
   with the masked and moved fields, the proxy note, and 「この呼び出しは画像入力なし」. For API
   and code-block events: 「このAPI実行時に新しいLLM呼び出しはありません」 (no new LLM call
   during this API call) when no model call is nested under the event, otherwise the nested
   perception VLM calls with links; the statement that no code-generator or VDM call
   happened during the event, which the viewer checks (a code-generator or VDM request
   inside the event's `seq` span is listed in a red notice instead); links to the call that
   produced the code and to the next one; and the later-input report of section 6.4. When
   no payload was recorded (the hook was not installed, a write failed, the disk was low,
   or the episode has no trace): 「未記録」 (not recorded).
2. モデル出力/コード (model output and code): the raw response, the reasoning the API
   provided (labelled with its field), the parsed code and the harness decision, usage,
   latency and every HTTP attempt; for a failed or stopped stream, the text received until
   then.
3. API/実行ログ (API and execution log): arguments, the return value (arrays with shape,
   dtype, the recorded statistics and a preview; units, or 「単位・座標系: 不明」), stdout and
   stderr with the API's own slice highlighted, the exception and traceback (with a note
   when they were redacted or cut), duration, simulator steps, and links to the code block
   (with the call line marked) and to the model call.
4. デバッグ状態 (debug state), always under the fixed banner 「モデル入力とは別（デバッグ用）」
   (separate from the model input, for debugging): ground truth before and after,
   evaluator results, the harness summary, perception inputs and outputs (mask previews
   and grasp arrays through `/array` and `/array-image`, with 「記録時の統計」, the
   statistics recorded at capture time), and the raw events; for the episode start and
   end, the recorder's own state (env, settings, hooks installed at the start and during
   the episode, failed hooks, `apis_wrapped`, `end_note`). Next to `success`, the viewer
   states what it means, chosen by the episode's `source` (section 5.2). For a batch
   episode it shows 「最終ブロックの sandbox_rc == 0（ハーネスの補正後。途中のブロックの失敗は含まない）」
   (the last block's `sandbox_rc` is 0 after the harness's override; failures of earlier
   blocks do not count), and for a web UI episode
   「task_completed または num_finishes > 0（Web UI の判定。sandbox_rc は見ない）」 (the web
   UI's rule, which ignores `sandbox_rc`). Both are followed by
   「タスク成功の判定とは別の値です」 (a different value from the task-success verdict). An
   older `success_meaning` text that says "code ran without exception" is marked
   「以前のレコーダーが保存した誤った説明。正しい意味は success の行を参照」 (a wrong description
   stored by an earlier recorder; see the success row for the meaning).
5. 入力差分 (input diff): the selected request against the previous request of the same
   family in the same episode. The families are: code-generator calls (`initial_code` and
   `multi_turn_decision` together), ensemble members, ensemble syntheses, all VDM calls,
   and role plus `call_kind` for anything else. The previous call is represented by its
   attempt with an ok response, else its last attempt. Messages are aligned by role and
   text, and changed messages get a line diff computed in the browser; images are
   compared by `sha256` (added, removed, unchanged). For a retry, the tab also says whether
   its body sha256 equals that of attempt 1. A request whose images were not recorded
   shows 「未記録」.

Polling. On a running episode in Live mode, the viewer reads new events and the episode
summary about every second and keeps the newest event selected. Any manual selection, or a
URL that names an event, switches to Browse, where only the summary is re-read (every
5 s). After the episode stops running, the viewer keeps polling every second for a grace
period of 15 to 60 s, so the tail of `events.jsonl` and late events are read; events after
that need 更新 (refresh). While the open episode has no matching runner record and the run
had a running episode in the last 60 s, the viewer re-reads the runner records every 30 s.
`web-ui/src/trace/live.ts` states the exact rules.

File generations. When the `generation` of `events.jsonl` changes, the viewer drops what it
has read, reads the file again from offset 0, and says so in the top bar
(「表示中に events.jsonl が置き換えられ、最初から読み直しました」). File and `/array-image` URLs
carry the generation, so a trace replaced in place never shows a cached file of the old
trace.

Data assets. When the writer moved an event's `data` to a `data_asset` file (section 5.1),
the viewer loads that file for the selected event and for the events its panes read. A
pane that needs such a file shows 読み込み中 (loading), 読み込めません (cannot be loaded)
with 再読み込み (reload), or 未記録 (not recorded) when the file was never written. Beyond 64
nested service calls, it shows 未読み込み (not loaded) with a 読み込む (load) button.

Arrays. The statistics shown are the ones recorded in the event. Previews follow the
endpoint's `image` layout, and the viewer applies the same rule itself when an older
server sends no `image`. A preview cut by the listing budget shows a note.

The browser's `localStorage` keeps the pane sizes.

### 9.4 Episodes without a trace

The picker lists runner run directories written before the recorder existed separately,
under 「トレースなし（既存ログ）」 (no trace, existing logs). The backend reads them from the
runner roots (default `outputs/runner`, changed with
`--runner-root`) with the same path rules. A runner run is a directory that holds
`run.json` or `trials.jsonl`, and its trials are `<task>/<condition>/trial_*`. The same run
id can exist under two runner roots, so every request carries the root index (`?root=`).

A trial directory shows `code.py`, stdout and stderr parsed from `summary.txt`, the
videos, `visual_feedback_*.png` labelled 「ハーネス保存画像（送信を検証していない）」 (image
the harness saved; not verified as sent), `trial_record.json`, and the values encoded in
the directory name (`sandbox_rc`, reward, `task_completed`). When the trial has no
`codegen_turn_<n>_messages.json` file, the model-input tab shows 「未記録」. When one exists,
the tab lists the saved messages and state logs. Files in `prompts_and_responses/*.txt`
appear separately under the heading
「ハーネスが保存したプロンプト文字列（送信payloadではない）」 (prompt strings the harness saved;
not the sent payload).
