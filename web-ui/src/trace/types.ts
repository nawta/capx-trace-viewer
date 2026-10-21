/**
 * Types of the experiment trace viewer.
 *
 * Mirrors docs/trace-format.md section 5 (event envelope 5.1, per-type data
 * 5.2, argument summaries 5.3, payload metadata 6.2-6.4) and the responses of
 * section 9.2. The recorder is written separately and can omit fields (a hook
 * that failed, an older schema), so every data field is optional and nullable:
 * the viewer never assumes a field exists.
 *
 * Related modules: api.ts (HTTP client for 9.2), model.ts (causal tree built
 * from these events), laterInput.ts (the 6.4 "later input" report).
 */

// ---------------------------------------------------------------------------
// 5.1 Envelope
// ---------------------------------------------------------------------------

export type EventStatus = 'started' | 'ok' | 'error' | 'degraded' | 'warning' | 'info';

export type ArtifactLabel =
  | 'model_input'
  | 'harness_render'
  | 'debug_frame'
  | 'perception_input'
  | 'perception_output'
  | 'payload'
  | 'state'
  | 'text';

/** One entry of `artifacts` (paths are relative to the episode dir). */
export interface Artifact {
  path?: string | null;
  sha256?: string | null;
  kind?: string | null;
  mime?: string | null;
  bytes?: number | null;
  width?: number | null;
  height?: number | null;
  shape?: number[] | null;
  dtype?: string | null;
  camera?: string | null;
  label?: ArtifactLabel | string | null;
  written?: boolean | null;
  reason?: string | null;
  name?: string | null;
}

export interface EventRefs {
  logical_call_id?: string | null;
  http_attempt?: number | null;
  code_block_id?: string | null;
  api_call_id?: string | null;
  model_event_id?: string | null;
  turn_index?: number | null;
  for_event_id?: string | null;
  [key: string]: unknown;
}

/** One line of events.jsonl. `data` is typed per event type through `dataOf`. */
export interface TraceEvent {
  schema_version?: string | null;
  run_id?: string | null;
  episode_id?: string | null;
  event_id: string;
  parent_event_id?: string | null;
  seq: number;
  wall_time?: number | null;
  monotonic_time?: number | null;
  pid?: number | null;
  thread?: string | null;
  sim_step?: number | null;
  sim_time?: number | null;
  sim_epoch?: number | null;
  event_type: string;
  status?: EventStatus | string | null;
  late?: boolean | null;
  refs?: EventRefs | null;
  artifacts?: Artifact[] | null;
  data?: unknown;
}

// ---------------------------------------------------------------------------
// Shared value shapes
// ---------------------------------------------------------------------------

/**
 * Text that is inline up to 4000 characters and otherwise an asset (5.1).
 * The recorder may store a plain string, or an object that carries a preview
 * and the asset path of the full text.
 */
export type TextValue =
  | string
  | {
      text?: string | null;
      preview?: string | null;
      head?: string | null;
      asset?: string | null;
      path?: string | null;
      $asset?: string | null;
      length?: number | null;
      chars?: number | null;
      truncated?: boolean | null;
      [key: string]: unknown;
    };

/** Key frame reference stored next to an event's own sim_step (8.2). */
export interface FrameRef {
  epoch?: number | null;
  buffer_index?: number | null;
  camera?: string | null;
  frame_sim_step?: number | null;
  path?: string | null;
  written?: boolean | null;
  reason?: string | null;
}

/** Ground-truth state reference (8.4); a path string, an object, or null. */
export type StateRef =
  | string
  | {
      path?: string | null;
      asset?: string | null;
      sha256?: string | null;
      written?: boolean | null;
      reason?: string | null;
    }
  | null;

export interface EndpointInfo {
  url?: string | null;
  host?: string | null;
  port?: number | null;
  path?: string | null;
  known_proxy?: boolean | string | null;
}

/** One snapshot whose pixels match a sent image (6.2, `source.candidates`). */
export interface ImageSourceCandidate {
  event_id?: string | null;
  kind?: string | null;
  sim_step?: number | null;
  wall_time?: number | null;
}

/**
 * `source` of a model-input image; filled only from exact hash matches (6.2).
 * When several snapshots have the same pixels (an unchanged scene rendered
 * twice), the recorder picks one, sets `ambiguous` and lists every match in
 * `candidates`. `reason` explains a `{kind: "unknown"}` video source.
 */
export interface ImageSource {
  kind?: string | null;
  camera?: string | null;
  sim_step?: number | null;
  wall_time?: number | null;
  event_id?: string | null;
  frame_range?: number[] | null;
  ambiguous?: boolean | null;
  candidates?: ImageSourceCandidate[] | null;
  reason?: string | null;
  n_frames?: number | null;
  [key: string]: unknown;
}

/** One entry of `model_request.images` (6.2). */
export interface ImageInfo {
  path?: string | null;
  order?: number | null;
  /** `image`, `video`, `external_url`, or another media kind (the trace recorder). */
  kind?: string | null;
  asset?: string | null;
  sha256?: string | null;
  mime?: string | null;
  bytes?: number | null;
  width?: number | null;
  height?: number | null;
  pixel_sha256?: string | null;
  source?: ImageSource | null;
  fetched?: boolean | null;
  url?: string | null;
  written?: boolean | null;
  reason?: string | null;
}

export interface Reconstruction {
  verified?: boolean | null;
  method?: string | null;
  masked?: string[] | null;
  reason?: string | null;
}

/** A registered segment found in a request text part (6.4). */
export interface SegmentMatch {
  kind?: string | null;
  path?: string | null;
  start?: number | null;
  end?: number | null;
  source_event_ids?: string[] | null;
}

export interface TextRange {
  path?: string | null;
  start?: number | null;
  end?: number | null;
  /** An unattributed range that holds only whitespace (for example the newline between two segments). */
  whitespace_only?: boolean | null;
}

export interface InputSummaryCategories {
  images?: ImageInfo[] | null;
  task_api_description?: SegmentMatch[] | null;
  code_history?: SegmentMatch[] | null;
  stdout_stderr?: SegmentMatch[] | null;
  vdm_description?: SegmentMatch[] | null;
  other?: SegmentMatch[] | null;
}

export type SegmentCategory = Exclude<keyof InputSummaryCategories, 'images'>;

export interface InputSummary {
  n_images?: number | null;
  n_videos?: number | null;
  no_image_input?: boolean | null;
  categories?: InputSummaryCategories | null;
  unattributed?: TextRange[] | null;
  /** The recorder keeps at most 200 unattributed ranges and sets this when it cut the list. */
  unattributed_truncated?: boolean | null;
  /** Registered segments (at least 8 characters) that were searched in this request and not found. */
  registered_not_found?: { kind?: string | null; length?: number | null; source_event_ids?: string[] | null }[] | null;
}

/** `{start, end, text}`: what one API call printed (5.2). */
export interface OutputSlice {
  start?: number | null;
  end?: number | null;
  text?: TextValue | null;
}

export interface Gap {
  code?: string | null;
  count?: number | null;
  first_seq?: number | null;
}

// ---------------------------------------------------------------------------
// 5.2 Data per event type
// ---------------------------------------------------------------------------

export interface ModelInfo {
  model?: string | null;
  endpoint?: string | EndpointInfo | null;
}

export interface EnvInfo {
  class?: string | null;
  low_level_class?: string | null;
  apis?: string[] | null;
  viser_port?: number | null;
  privileged?: boolean | null;
  [key: string]: unknown;
}

/** One entry of episode.json `apis_wrapped` (API and service wrappers installed on an instance). */
export interface ApiWrappedInfo {
  api?: string | null;
  class?: string | null;
  wrapped?: string[] | null;
  services?: string[] | null;
  /** False when the env (and its API instances) was reused from an earlier episode of the worker. */
  installed_in_this_episode?: boolean | null;
  [key: string]: unknown;
}

export interface EpisodeStartData {
  source?: string | null;
  task?: string | null;
  condition?: string | null;
  seed?: number | null;
  attempt_index?: number | null;
  runner_invocation?: number | null;
  condition_label?: string | null;
  runner_condition?: Record<string, unknown> | null;
  config_flags?: Record<string, unknown> | null;
  models?: Record<string, ModelInfo | null> | null;
  env?: EnvInfo | null;
  /** Web UI: the env is created inside the episode; episode.json then holds `env`. */
  env_pending?: string | null;
  tracer?: {
    version?: string | null;
    settings?: Record<string, unknown> | null;
    hooks_installed?: string[] | null;
    hooks_failed?: unknown[] | null;
  } | null;
  host?: string | null;
  python?: string | null;
  git_commit?: string | null;
  process_start_time?: number | null;
}

export interface EpisodeEndData {
  end_reason?: string | null;
  /**
   * Set when the recorder closed an episode that was left open on the trial thread, at the
   * start of the next one (end_reason "error").
   */
  end_note?: string | null;
  harness_summary?: Record<string, unknown> | null;
  exception?: { type?: string | null; message?: TextValue | null; message_redacted?: boolean | null; message_cut?: boolean | null } | null;
  alarm_seen?: boolean | null;
  duration_s?: number | null;
  n_code_blocks?: number | null;
  n_model_calls?: number | null;
  tracer_overhead_s?: number | null;
  trace_complete?: boolean | null;
  gaps?: Gap[] | null;
  dropped_events?: number | null;
  late_events?: number | null;
  n_warnings?: number | null;
}

export interface TaskTextData {
  text?: TextValue | null;
  /** Why `text` is null (for example no "\nAPIs:\n" in the request). */
  reason?: string | null;
  logical_call_id?: string | null;
  source?: string | null;
  model_event_id?: string | null;
}

export interface ModelRequestData {
  role?: string | null;
  call_kind?: string | null;
  turn_index?: number | null;
  logical_call_id?: string | null;
  http_attempt?: number | null;
  endpoint?: EndpointInfo | null;
  model?: string | null;
  params?: Record<string, unknown> | null;
  stream?: boolean | null;
  payload_path?: string | null;
  body_sha256?: string | null;
  body_bytes?: number | null;
  reconstruction?: Reconstruction | null;
  masked_fields?: string[] | null;
  images?: ImageInfo[] | null;
  n_messages?: number | null;
  n_images?: number | null;
  n_videos?: number | null;
  input_summary?: InputSummary | null;
}

export interface ModelResponseData {
  role?: string | null;
  call_kind?: string | null;
  logical_call_id?: string | null;
  http_attempt?: number | null;
  http_status?: number | null;
  latency_s?: number | null;
  streamed?: boolean | null;
  n_chunks?: number | null;
  ttft_s?: number | null;
  content?: TextValue | null;
  reasoning?: { text?: TextValue | null; field?: string | null } | null;
  usage?: Record<string, unknown> | null;
  finish_reason?: string | null;
  response_model?: string | null;
  response_id?: string | null;
  payload_path?: string | null;
}

export interface ModelErrorData {
  scope?: 'attempt' | 'logical' | string | null;
  role?: string | null;
  call_kind?: string | null;
  logical_call_id?: string | null;
  http_attempt?: number | null;
  http_status?: number | null;
  error_type?: string | null;
  message?: TextValue | null;
  /** The recorder removed secrets from the stored message / body excerpt (payload.redact). */
  message_redacted?: boolean | null;
  body_excerpt?: TextValue | null;
  body_excerpt_redacted?: boolean | null;
  latency_s?: number | null;
  /** Streamed calls (web trial hook): the text received before the error or the stop (GeneratorExit). */
  streamed?: boolean | null;
  n_chunks?: number | null;
  partial_content?: TextValue | null;
}

export interface ModelOutputParsedData {
  model_event_id?: string | null;
  /**
   * The logical call the parse belongs to. Molmo's parse (Molmo parser hook, parser "molmo._parse_points")
   * carries it, with model_event_id null when the request got no response (for example the
   * Molmo server refused the connection and the API parsed an empty text).
   */
  logical_call_id?: string | null;
  parser?: string | null;
  code_blocks?: (TextValue | null)[] | null;
  decision?: string | null;
  /** The recorder's wording of the decision, e.g. "no REGENERATE -> finish". */
  harness_reading?: string | null;
  regenerate_text?: TextValue | null;
  model_said_finish?: boolean | null;
  content_empty?: boolean | null;
  code_source?: string | null;
}

export interface FeedbackSegment {
  kind?: string | null;
  text_sha256?: string | null;
  length?: number | null;
  source_event_ids?: (string | null)[] | null;
  /** For the `multi_turn_prompt` segment: the registered texts found inside it. */
  contains?: FeedbackSegment[] | null;
}

export interface FeedbackBuiltData {
  builder?: string | null;
  turn_index?: number | null;
  segments?: FeedbackSegment[] | null;
  image_pixel_sha256?: string | string[] | null;
  vdm_model_event_id?: string | null;
  user_feedback?: TextValue | null;
  is_video_feedback?: boolean | null;
  has_visual_feedback?: boolean | null;
  n_messages?: number | null;
  /**
   * JSON file with the builder's text arguments ({complete_multi_turn_prompt,
   * visual_differencing_feedback, user_feedback}); written by the writer thread.
   */
  texts_path?: string | null;
  payload_path?: string | null;
}

export interface CodeBlockStartData {
  code_block_id?: string | null;
  block_index?: number | null;
  turn_index?: number | null;
  source_model_event_id?: string | null;
  code_source?: string | null;
  code?: TextValue | null;
  code_sha256?: string | null;
  direct_access?: string[] | Record<string, unknown> | null;
  frame_ref?: FrameRef | FrameRef[] | null;
  state_ref?: StateRef;
}

export interface CodeBlockEndData {
  code_block_id?: string | null;
  /** Event ids of this block's stdout / stderr events (written before code_block_end). */
  stdout_event_id?: string | null;
  stderr_event_id?: string | null;
  exec_duration_s?: number | null;
  /** The exception step() raised (the block then has no reward). */
  exception?: { type?: string | null; message?: TextValue | null } | null;
  /** Arrays of the block result that were over the array budget (only shape, dtype and sha256 kept). */
  array_budget_exceeded?: number | null;
  sandbox_rc?: number | null;
  duration_s?: number | null;
  stdout_len?: number | null;
  stderr_len?: number | null;
  reward?: number | null;
  terminated?: boolean | null;
  truncated?: boolean | null;
  task_completed?: boolean | null;
  result_summary?: unknown;
  frame_ref?: FrameRef | FrameRef[] | null;
  state_ref?: StateRef;
  sim_step_range?: (number | null)[] | null;
  tracer_frame_visible?: boolean | null;
}

export interface StreamTextData {
  code_block_id?: string | null;
  text?: TextValue | null;
  /** Length of the full text in characters (code points). */
  length?: number | null;
}

export interface CallSite {
  code_block_id?: string | null;
  line?: number | null;
  source_line?: string | null;
}

export interface ApiCallStartData {
  api_call_id?: string | null;
  name?: string | null;
  qualname?: string | null;
  api_class?: string | null;
  depth?: number | null;
  privileged?: boolean | null;
  args?: unknown;
  call_site?: CallSite | null;
  stdout_offset?: number | null;
  stderr_offset?: number | null;
  frame_ref?: FrameRef | FrameRef[] | null;
  state_ref?: StateRef;
  units?: unknown;
  array_budget_exceeded?: number | null;
}

export interface ApiCallEndData {
  api_call_id?: string | null;
  array_budget_exceeded?: number | null;
  duration_s?: number | null;
  return?: unknown;
  error_type?: string | null;
  /** str(e): inline up to 4000 characters, else a text asset (the trace recorder). */
  message?: TextValue | null;
  /**
   * True when the recorder removed secrets from the stored message (payload.redact_text):
   * the stored text then differs from the str(e) the program printed.
   */
  message_redacted?: boolean | null;
  /**
   * The message was longer than the recorder reads (payload.REDACT_WINDOW, 64 KB): only its
   * start is stored, followed by a note line, so the stored text is not what the program printed.
   */
  message_cut?: boolean | null;
  traceback?: TextValue | null;
  /** The recorder removed secrets from the stored traceback (or its last line, the message). */
  traceback_redacted?: boolean | null;
  /** The traceback was longer than the recorder reads; only its start is stored. */
  traceback_cut?: boolean | null;
  /** The chained exception (__cause__ or __context__), its message redacted like `message`. */
  cause?: { type?: string | null; message?: string | null; message_redacted?: boolean | null } | null;
  alarm_timeout?: boolean | null;
  stdout_slice?: OutputSlice | null;
  stderr_slice?: OutputSlice | null;
  frame_ref?: FrameRef | FrameRef[] | null;
  state_ref?: StateRef;
  sim_step_delta?: number | null;
  buffer_frames?: (number | null)[] | null;
  units?: unknown;
}

export interface HttpAttemptInfo {
  status?: number | null;
  latency_s?: number | null;
  error?: string | null;
  /** The recorder removed secrets from the stored error text. */
  error_redacted?: boolean | null;
}

export interface ServiceCallData {
  service_call_id?: string | null;
  service?: string | null;
  function?: string | null;
  endpoint?: string | EndpointInfo | null;
  request_summary?: unknown;
  response_summary?: unknown;
  latency_s?: number | null;
  http_attempts?: HttpAttemptInfo[] | null;
  error_type?: string | null;
  /** str(e): inline up to 4000 characters, else a text asset (the trace recorder). */
  message?: TextValue | null;
  /** The recorder removed secrets from the stored message (see ApiCallEndData). */
  message_redacted?: boolean | null;
  /** Only the start of a very long message is stored (see ApiCallEndData). */
  message_cut?: boolean | null;
  degraded_reason?: string | null;
  /** Arrays over CAPX_TRACE_ARRAY_CALL_MB / _EPISODE_MB: only shape, dtype and sha256 kept. */
  array_budget_exceeded?: number | null;
}

export interface ObservationSnapshotData {
  camera?: string | null;
  purpose?: string | null;
  pixel_sha256?: string | null;
  width?: number | null;
  height?: number | null;
}

/** A get_video_frames_range call of the harness (the trace recorder). */
export interface VideoFramesReadData {
  start?: number | null;
  end?: number | null;
  n_frames?: number | null;
  camera?: string | null;
  /** sha256 of each frame's raw uint8 pixels; null when the queue budget dropped the copies. */
  frame_pixel_sha256?: string[] | null;
  /** Why the hashes are missing (queue_full). */
  reason?: string | null;
}

export interface FrameData {
  camera?: string | null;
  epoch?: number | null;
  buffer_index?: number | null;
  sim_step?: number | null;
  reason?: 'periodic' | 'keyframe' | string | null;
  for_event_id?: string | null;
}

export interface EnvResetData {
  sim_epoch?: number | null;
  reason?: string | null;
}

export interface EvaluatorResultData {
  scope?: 'code_block' | 'episode' | string | null;
  code_block_id?: string | null;
  reward?: number | null;
  task_completed?: boolean | null;
  terminated?: boolean | null;
  truncated?: boolean | null;
  source?: string | null;
}

export interface TraceWarningData {
  code?: string | null;
  message?: string | null;
  related_event_id?: string | null;
  count?: number | null;
  /** hook_install_failed: the hook that failed, and whether it failed while this episode was open. */
  hook?: string | null;
  target?: string | null;
  during_episode?: boolean | null;
  /** ambiguous_episode: the open episodes an event could have belonged to. */
  candidates?: string[] | null;
  /** False for warnings that are not a recording gap (ambiguous_episode, stale_context at episode start). */
  counts_as_gap?: boolean | null;
  /** hook_install_failed in a run's unscoped log: a failure of the whole process, not of one episode. */
  run_level?: boolean | null;
  /** internal_error of recorder code that ran outside any episode (unscoped log). */
  outside_episode?: boolean | null;
  /**
   * An episode start the recorder undid (the trace recorder abandoned episode cleanup): the
   * id of that episode, whether the trial alarm caused it, and what happened to the trial
   * ("...; the trial ran untraced").
   */
  episode_id?: string | null;
  alarm?: boolean | null;
  note?: string | null;
  /** The recorder removed secrets from the stored message. */
  message_redacted?: boolean | null;
}

/**
 * Markers the writer leaves in `data` when an event line would exceed 256 KB
 * (when the trace recorder limits line size): the full `data` is saved as a gzip JSON asset
 * (label `payload`, also listed in `artifacts`) and only scalar fields and strings of at
 * most 200 characters stay inline. The viewer loads the asset on demand (dataAsset.ts).
 */
export interface DataAssetMarkers {
  oversize?: boolean | null;
  data_asset?: string | null;
  data_asset_written?: boolean | null;
}

/** Event type -> data shape. Unknown event types are kept and shown as "other". */
export interface EventDataMap {
  episode_start: EpisodeStartData;
  episode_end: EpisodeEndData;
  task_text: TaskTextData;
  model_request: ModelRequestData;
  model_response: ModelResponseData;
  model_error: ModelErrorData;
  model_output_parsed: ModelOutputParsedData;
  feedback_built: FeedbackBuiltData;
  code_block_start: CodeBlockStartData;
  code_block_end: CodeBlockEndData;
  stdout: StreamTextData;
  stderr: StreamTextData;
  api_call_start: ApiCallStartData;
  api_call_end: ApiCallEndData;
  api_call_error: ApiCallEndData;
  service_call_start: ServiceCallData;
  service_call_end: ServiceCallData;
  service_call_error: ServiceCallData;
  observation_snapshot: ObservationSnapshotData;
  video_frames_read: VideoFramesReadData;
  frame: FrameData;
  env_reset: EnvResetData;
  evaluator_result: EvaluatorResultData;
  trace_warning: TraceWarningData;
}

export type KnownEventType = keyof EventDataMap;

/**
 * Typed view of `ev.data`. Returns an empty object when data is missing or
 * not an object, so callers can read optional fields without further checks.
 */
export function dataOf<T extends KnownEventType>(ev: TraceEvent | null | undefined, _type: T): EventDataMap[T] {
  const d = ev?.data;
  if (d && typeof d === 'object' && !Array.isArray(d)) return d as EventDataMap[T];
  return {} as EventDataMap[T];
}

// ---------------------------------------------------------------------------
// 9.2 Responses
// ---------------------------------------------------------------------------

export type EpisodeStatus = 'ended' | 'ended_with_gaps' | 'running' | 'incomplete';

export interface TraceConfigResponse {
  trace_root?: string | null;
  exists?: boolean | null;
  viewer_only?: boolean | null;
  runner_roots?: string[] | null;
  expose?: boolean | null;
}

export interface RunListItem {
  run_id: string;
  n_episodes?: number | null;
  last_modified?: number | string | null;
  status_counts?: Record<string, number> | null;
}

export interface RunListResponse {
  runs: RunListItem[];
}

/** Summary of one episode (episode.json plus the derived status, 9.2). */
export interface EpisodeSummary {
  run_id?: string | null;
  episode_id: string;
  status?: EpisodeStatus | string | null;
  status_detail?: string | null;
  source?: string | null;
  task?: string | null;
  condition?: string | null;
  condition_label?: string | null;
  seed?: number | null;
  attempt_index?: number | null;
  runner_invocation?: number | null;
  start_wall_time?: number | null;
  end_wall_time?: number | null;
  last_modified?: number | string | null;
  heartbeat_age_s?: number | null;
  end_reason?: string | null;
  trace_complete?: boolean | null;
  gaps?: Gap[] | null;
  dropped_events?: number | null;
  late_events?: number | null;
  n_warnings?: number | null;
  viser_port?: number | null;
  /** "unscoped" for a run's log of events with no episode (served under the id unscoped.<pid>). */
  kind?: string | null;
  /** episode.json as the recorder wrote it (null when missing). */
  episode?: EpisodeJson | null;
  /**
   * Why episode.json could not be used: "missing", "symlink_not_followed",
   * "not_a_regular_file", "unreadable (<ERRNO>)", "invalid_json", "too_large (...)" or
   * "not_an_object" (capx_trace_viewer/reader.py read_json_at); null when it was read.
   */
  episode_json_error?: string | null;
  [key: string]: unknown;
}

/** Fields of episode.json the viewer reads beyond the summary (the trace recorder). */
export interface EpisodeJson {
  env?: EnvInfo | null;
  apis_wrapped?: ApiWrappedInfo[] | null;
  /** Hooks installed while the episode was open (web UI: trial role hook, trial lifecycle hook, environment hook are first imported inside it). */
  hooks_installed_during_episode?: string[] | null;
  tracer_overhead_s?: number | null;
  [key: string]: unknown;
}

/** One record of the runner's trials.jsonl copied into runner_records.jsonl (5.2). */
export interface RunnerRecord {
  run_id?: string | null;
  task?: string | null;
  condition?: string | null;
  trial?: number | null;
  seed?: number | null;
  attempts?: {
    attempt_id?: number | null;
    status?: string | null;
    started_at?: string | null;
    ended_at?: string | null;
    duration_s?: number | null;
    error?: string | null;
    /** A timeout: the alarm fired inside the generated code (the sandbox caught it). */
    timed_out_in_sandbox?: boolean | null;
    task_success?: boolean | null;
  }[] | null;
  outcome?: {
    task_success?: boolean | null;
    code_execution_success?: boolean | null;
    agent_finish?: boolean | null;
    termination_reason?: string | null;
    [key: string]: unknown;
  } | null;
  reward?: number | null;
  [key: string]: unknown;
}

/**
 * How runner_records.jsonl was read (capx_trace_viewer/reader.py run_detail_at): `error` when the
 * file exists but could not be opened or read (e.g. "unreadable (EACCES)"), lines that did not
 * parse, a last line without a newline, and records left out beyond the read limit.
 */
export interface RunnerRecordsInfo {
  error?: string | null;
  corrupt_lines?: number | null;
  torn_tail?: boolean | null;
  truncated?: boolean | null;
}

/**
 * The recorder's unscoped.<pid>.status.json (the trace recorder unscoped write failure handler):
 * lines of that unscoped log that could not be written.
 */
export interface UnscopedWriteStatus {
  file?: string | null;
  pid?: number | null;
  lines_not_written?: number | null;
  last_reason?: string | null;
  updated_wall_time?: number | null;
  [key: string]: unknown;
}

export interface UnscopedFileInfo {
  file?: string | null;
  /** null when the log exists only as its status file (the .jsonl is missing). */
  bytes?: number | null;
  log_id?: string | null;
  write_status?: UnscopedWriteStatus | null;
  /** Why the status file could not be read. */
  write_status_error?: string | null;
}

export interface RunDetailResponse {
  run_id: string;
  run?: Record<string, unknown> | null;
  /** Why run.json could not be read (it is then null). */
  run_json_error?: string | null;
  runner_records?: RunnerRecord[] | null;
  runner_records_info?: RunnerRecordsInfo | null;
  episodes: EpisodeSummary[];
  /** unscoped.<pid>.jsonl files of the run (events with no episode); `log_id` opens one in the viewer. */
  unscoped_files?: UnscopedFileInfo[] | null;
}

export interface EventsPage {
  events: TraceEvent[];
  next_offset: number;
  eof: boolean;
  torn_tail?: boolean | null;
  corrupt_lines?: number | null;
  /**
   * Identity of the events.jsonl file the page was read from (device, inode and a hash of
   * its first line). When it changes, the file was replaced: the client drops what it read
   * and reads again from offset 0. Sent back as the `generation` query parameter; a
   * mismatch answers HTTP 409 {"error": "generation_changed", "generation": "<new>"}.
   */
  generation?: string | null;
}

export interface ArrayStats {
  min?: number | string | null;
  max?: number | string | null;
  mean?: number | string | null;
  [key: string]: unknown;
}

/**
 * How `/array-image` draws an array (capx_trace_viewer/trace_arrays.py image_layout): `mask` or
 * `gray` (one image: a 2-D array, or 3-D with a last axis of size 1), `rgb`, `mask_stack`
 * (3-D bool with a last axis larger than 1: masks along axis 0) or `stack` (other 3-D:
 * images along axis 0); `index` of /array-image picks one of `count`.
 */
export interface ArrayImageLayout {
  kind: string;
  count: number;
  height: number;
  width: number;
}

export interface ArrayInfo {
  shape?: number[] | null;
  dtype?: string | null;
  /** Always null from current servers: statistics are recorded in the event (9.2). */
  stats?: ArrayStats | null;
  preview?: unknown;
  is_mask?: boolean | null;
  truncated?: boolean | null;
  packed?: boolean | null;
  preview_shape?: number[] | null;
  /** How the array is drawn; null when it cannot be drawn. Absent from older servers. */
  image?: ArrayImageLayout | null;
  /** Why no preview was sent (`listing_budget`: a listing of every array of a file hit its byte limit). */
  preview_omitted?: string | null;
  [key: string]: unknown;
}

export interface ArrayResponse {
  arrays: Record<string, ArrayInfo>;
}

export interface ViserResponse {
  available?: boolean | null;
  port?: number | null;
  reason?: string | null;
}
