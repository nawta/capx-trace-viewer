/**
 * "Did an API result reach a later model input?" (docs/trace-format.md 6.4).
 *
 * For an API call or code block, the later inputs are every model request
 * (each HTTP attempt, so retries are listed too) after the block's
 * `code_block_end`, up to and including the next `code_generator` request;
 * the other HTTP attempts of that code-generator call follow as resends. For
 * each of them this module reports, from the recorded `input_summary` only:
 *   - whether the block's stdout/stderr segment was found (path and offsets),
 *     searched and not found (`registered_not_found`), or never searched
 *     (the recorder registers only texts of at least 8 characters), and
 *     inside a found segment the API's own `stdout_slice`/`stderr_slice`;
 *   - for `api_call_error`, where the exception text (str(e), at least 8
 *     characters) or this exception's final traceback line
 *     (`<type>: <message>`, type and message both matching) sits in the
 *     block's streams and so in the request.
 * It never searches the model input for numbers or other values. Offsets are
 * mapped into a segment only when the segment is exactly the block's stream
 * (it names this block's stream event, and it has the stream's length; the
 * recorder merges identical texts of several blocks into one segment that names
 * them all); otherwise the report says the position inside the segment is not
 * computed. Whatever was not measured is reported as unchecked, never
 * as "not included".
 *
 * Related: model.ts (tree and call order), components/tabs/ModelInputTab.tsx
 * (renders the report).
 */

import { dataOf, type SegmentMatch, type TraceEvent } from './types';
import { codePointLength, utf16ToCodePoint } from './offsets';
import {
  attemptNumber,
  enclosingBlock,
  numOrNull,
  resolveText,
  type Attempt,
  type EpisodeModel,
  type LogicalCall,
  type TNode,
} from './model';

/** Texts shorter than this are neither registered nor searched by the recorder. */
export const MIN_SEGMENT_CHARS = 8;

export interface SegmentHit {
  path: string;
  start: number;
  end: number;
  sourceEventIds: string[];
  /** The segment is exactly this block's stream, so stream offsets map 1:1. */
  exact: boolean;
}

export interface MappedRange {
  path: string;
  start: number;
  end: number;
}

/**
 * How one of the block's streams was checked in one request:
 * - included: a segment of this stream is in the request (`hits`);
 * - not_included: the stream was searched in this request and not found;
 * - empty: the stream is empty, there is nothing to find;
 * - short: 1-7 characters, never registered or searched (unchecked);
 * - unsearched: not among the texts searched for this request (unchecked);
 * - not_recorded: the block's stream event is missing and its length is unknown.
 */
export type StreamCheckState = 'included' | 'not_included' | 'empty' | 'short' | 'unsearched' | 'not_recorded';

export interface StreamCheck {
  state: StreamCheckState;
  hits: SegmentHit[];
}

export interface ErrorTextFinding {
  what: 'message' | 'traceback_line';
  text: string;
  stream: 'stdout' | 'stderr';
  /** Offset of the text in the block's stream (code points). */
  streamOffset: number;
  /** Where it sits in this request, when the stream segment maps exactly. */
  mapped: MappedRange | null;
  /** The stream segment is in the request but its offsets do not map exactly. */
  inUnmappedSegment: boolean;
  /** How the stream itself was checked in this request. */
  streamState: StreamCheckState;
}

export interface LaterRequestReport {
  call: LogicalCall;
  attempt: Attempt;
  /** 1-based HTTP attempt number within the logical call. */
  httpAttempt: number;
  /** A resend of the same logical call (another HTTP attempt), not a new plan. */
  isRetry: boolean;
  /** input_summary was recorded for this request. */
  recorded: boolean;
  /** An attempt of the next code-generator call (the end of the range). */
  isNextCodegen: boolean;
  stdout: StreamCheck;
  stderr: StreamCheck;
  /** The API's own printed slices mapped into this request. */
  apiSlices: { stream: 'stdout' | 'stderr'; range: MappedRange }[];
  /** Why an API slice that exists could not be mapped (null when not needed). */
  sliceNote: string | null;
  errorFindings: ErrorTextFinding[];
  /**
   * For an API error with no finding: `unchecked` when nothing could be searched (texts not
   * loaded, str(e) shorter than 8 characters and no traceback line), `not_in_streams` when
   * the searched texts are not in the block's stdout/stderr at all.
   */
  errorNote: 'unchecked' | 'not_in_streams' | null;
}

export interface ApiFacts {
  /** The api_call_end / api_call_error event is in the record. */
  endRecorded: boolean;
  /** null: not recorded (no end event, or no slice in it). */
  printedStdout: boolean | null;
  printedStderr: boolean | null;
  stdoutSlice: { start: number; end: number } | null;
  stderrSlice: { start: number; end: number } | null;
  isError: boolean;
  errorType: string | null;
  /**
   * The full str(e); null when it is not known: not recorded, or a long message whose asset
   * is not loaded (`messageIncomplete`).
   */
  errorMessage: string | null;
  /** str(e) is stored as an asset that has not been loaded, so it was not searched. */
  messageIncomplete: boolean;
  /**
   * The recorder removed secrets from the stored str(e) (`message_redacted`), so the stored
   * text is not what the program printed and is not searched: the result is unchecked,
   * never "not printed".
   */
  messageRedacted: boolean;
  /**
   * The stored str(e) holds only the first 64 KB of the message plus a note line
   * (`message_cut`), so it is not searched either: the result is unchecked.
   */
  messageCut: boolean;
  /** str(e) is too short to search (fewer than 8 characters). */
  messageTooShort: boolean;
  /** This exception's final traceback line in the block's stderr (type and message match). */
  tracebackLine: string | null;
  /** Where that line starts in the stderr (UTF-16 index of the loaded text). */
  tracebackOffset: number | null;
  /** How many places of the stderr hold that line; above 1 the exception it belongs to is ambiguous. */
  tracebackMatches: number;
  /** The texts needed for the exception search were not available. */
  missingStreamText: boolean;
}

export interface LaterInputReport {
  applicable: boolean;
  reason: string | null;
  block: TNode | null;
  blockEnd: TraceEvent | null;
  stdoutEvent: TraceEvent | null;
  stderrEvent: TraceEvent | null;
  stdoutLen: number | null;
  stderrLen: number | null;
  requests: LaterRequestReport[];
  nextCodegen: LogicalCall | null;
  api: ApiFacts | null;
}

export interface StreamTexts {
  /** Full stdout of the block ('' when there is no stdout event), or null when not loaded. */
  stdout: string | null;
  stderr: string | null;
  /**
   * Full str(e) of the selected API error when the caller resolved it (a long message is a
   * text asset); undefined lets this module use the inline value.
   */
  errorMessage?: string | null;
}

/**
 * The full text of one of the block's streams for the exception search, or null when it is
 * not known. The recorder writes a stdout and a stderr event for every block whose step()
 * returned text, empty or not (the trace recorder), and code_block_end names their
 * lengths and event ids. So:
 * - the stream event is in the record: its text, or null while only a preview is loaded;
 * - no stream event, but the block end names a non-empty length or an event id: the event
 *   was lost (for example a corrupt line), so the text is unknown (null), never '';
 * - no stream event and a recorded length of 0, or neither length nor event id (step()
 *   returned no text): '' (nothing was printed).
 */
export function blockStreamText(
  event: TraceEvent | null,
  loaded: { text: string | null; partial: boolean },
  recordedLen: number | null | undefined,
  recordedEventId: string | null | undefined,
): string | null {
  if (event) return loaded.partial ? null : loaded.text;
  if (typeof recordedLen === 'number') return recordedLen === 0 ? '' : null;
  return typeof recordedEventId === 'string' && recordedEventId ? null : '';
}

function streamLen(ev: TraceEvent | null, recorded: number | null | undefined, text: string | null): number | null {
  if (typeof recorded === 'number') return recorded;
  if (text !== null && ev) return codePointLength(text);
  if (!ev) return null;
  const d = dataOf(ev, 'stdout');
  const n = numOrNull(d.length);
  if (n !== null) return n;
  const r = resolveText(d.text);
  if (r.length !== null) return r.length;
  if (r.text !== null && !r.truncated) return codePointLength(r.text);
  return null;
}

function hitsFor(
  segs: SegmentMatch[],
  kind: 'stdout' | 'stderr',
  ids: Set<string>,
  streamId: string | null,
  len: number | null,
): SegmentHit[] {
  const out: SegmentHit[] = [];
  for (const s of segs) {
    const src = (s.source_event_ids ?? []).filter((x): x is string => typeof x === 'string');
    if (!src.some((x) => ids.has(x))) continue;
    if (s.kind && s.kind !== kind) continue;
    if (!s.kind && streamId && !src.includes(streamId)) continue;
    const start = numOrNull(s.start);
    const end = numOrNull(s.end);
    if (start === null || end === null || !s.path) continue;
    // The recorder registers each stream with source_event_ids = [stream event,
    // code_block_start] and merges segments with the same kind and text
    // (the trace recorder compute_input_summary). So a segment of this kind that names
    // this block's stream (or, when that event is missing, this block) holds exactly this
    // block's stream text, even when other blocks that printed the same text are named too;
    // the length check only guards against a malformed record.
    const exact = len !== null && end - start === len && (streamId === null || src.includes(streamId));
    out.push({ path: s.path, start, end, sourceEventIds: src, exact });
  }
  return out;
}

/** Checks one stream of the block against one request's input_summary. */
export function checkStream(
  summary: { categories?: { stdout_stderr?: SegmentMatch[] | null } | null; registered_not_found?: { kind?: string | null; source_event_ids?: string[] | null }[] | null } | null | undefined,
  kind: 'stdout' | 'stderr',
  ids: Set<string>,
  streamEvent: TraceEvent | null,
  len: number | null,
): StreamCheck {
  const hits = hitsFor(summary?.categories?.stdout_stderr ?? [], kind, ids, streamEvent?.event_id ?? null, len);
  if (hits.length > 0) return { state: 'included', hits };
  if (len === 0) return { state: 'empty', hits };
  if (len !== null && len < MIN_SEGMENT_CHARS) return { state: 'short', hits };
  const searched = (summary?.registered_not_found ?? []).some(
    (r) => r && r.kind === kind && (r.source_event_ids ?? []).some((x) => typeof x === 'string' && ids.has(x)),
  );
  if (searched) return { state: 'not_included', hits };
  if (!streamEvent && len === null) return { state: 'not_recorded', hits };
  return { state: 'unsearched', hits };
}

function sliceOf(v: unknown): { start: number; end: number } | null {
  if (!v || typeof v !== 'object') return null;
  const o = v as { start?: unknown; end?: unknown };
  const start = numOrNull(o.start);
  const end = numOrNull(o.end);
  if (start === null || end === null) return null;
  return { start, end };
}

const DOTTED_PREFIX = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\.$/;

/**
 * This exception's final traceback line in `stderr`, as Python prints it: `<type>: <message>`
 * (the type may carry its module, `pkg.mod.<type>: <message>`), or only `<type>` when str(e)
 * is empty. Both the type and the whole message must match, so the line of another
 * exception of the same type is never taken. A message with newlines spans several lines;
 * the returned text then spans them too. Returns the last such place, its offset (UTF-16)
 * and how many places match; null when there is none, or when the type or the message is
 * unknown.
 */
export function findTracebackLine(
  stderr: string,
  errorType: string | null,
  message: string | null,
): { line: string; offset: number; count: number } | null {
  if (!errorType || message === null) return null;
  const found: { line: string; offset: number }[] = [];
  const needle = message === '' ? errorType : `${errorType}: ${message}`;
  let i = stderr.indexOf(needle);
  while (i >= 0) {
    const lineStart = stderr.lastIndexOf('\n', i - 1) + 1;
    const prefix = stderr.slice(lineStart, i);
    const end = i + needle.length;
    const atLineEnd = end === stderr.length || stderr[end] === '\n' || stderr[end] === '\r';
    if ((prefix === '' || DOTTED_PREFIX.test(prefix)) && atLineEnd) {
      found.push({ line: stderr.slice(lineStart, end), offset: lineStart });
    }
    i = stderr.indexOf(needle, i + 1);
  }
  if (found.length === 0) return null;
  const last = found[found.length - 1];
  return { line: last.line, offset: last.offset, count: found.length };
}

/**
 * The later requests of 6.4: every HTTP attempt whose request comes after `afterSeq`, in
 * seq order, up to and including the first code-generator request; the other attempts of
 * that code-generator call are added after it (resends of the same input).
 */
export function laterAttempts(
  model: EpisodeModel,
  afterSeq: number,
): { items: { call: LogicalCall; attempt: Attempt }[]; nextCodegen: LogicalCall | null } {
  const all: { call: LogicalCall; attempt: Attempt }[] = [];
  for (const call of model.calls) {
    for (const attempt of call.attempts) {
      if (attempt.request.seq > afterSeq) all.push({ call, attempt });
    }
  }
  all.sort((a, b) => a.attempt.request.seq - b.attempt.request.seq);
  const b = all.findIndex((x) => x.call.role === 'code_generator');
  if (b < 0) return { items: all, nextCodegen: null };
  const boundary = all[b].call;
  const items = all.slice(0, b + 1);
  for (const x of all.slice(b + 1)) if (x.call === boundary) items.push(x);
  return { items, nextCodegen: boundary };
}

export function computeLaterInput(model: EpisodeModel, node: TNode, texts: StreamTexts): LaterInputReport {
  const empty: LaterInputReport = {
    applicable: false,
    reason: null,
    block: null,
    blockEnd: null,
    stdoutEvent: null,
    stderrEvent: null,
    stdoutLen: null,
    stderrLen: null,
    requests: [],
    nextCodegen: null,
    api: null,
  };
  const block = enclosingBlock(model, node);
  if (!block) return { ...empty, reason: 'コードブロックに属さないイベントです' };
  const blockEnd = block.end;
  if (!blockEnd) {
    return { ...empty, block, reason: 'このブロックの code_block_end がまだ無いため、以後の入力は判定できません' };
  }
  const stdoutEvent = block.children.find((c) => c.kind === 'stdout')?.primary ?? null;
  const stderrEvent = block.children.find((c) => c.kind === 'stderr')?.primary ?? null;
  const endData = dataOf(blockEnd, 'code_block_end');
  const stdoutLen = streamLen(stdoutEvent, endData.stdout_len, texts.stdout);
  const stderrLen = streamLen(stderrEvent, endData.stderr_len, texts.stderr);

  const blockIds = new Set<string>([block.primary.event_id, blockEnd.event_id]);
  const blockId = dataOf(block.primary, 'code_block_start').code_block_id;
  if (blockId) blockIds.add(blockId);
  const stdoutIds = new Set(blockIds);
  const stderrIds = new Set(blockIds);
  if (stdoutEvent) stdoutIds.add(stdoutEvent.event_id);
  if (stderrEvent) stderrIds.add(stderrEvent.event_id);

  // API facts.
  let api: ApiFacts | null = null;
  if (node.kind === 'api_call') {
    const endEv = node.end;
    const ed = endEv ? dataOf(endEv, 'api_call_end') : null;
    const so = sliceOf(ed?.stdout_slice);
    const se = sliceOf(ed?.stderr_slice);
    const isError = endEv?.event_type === 'api_call_error';
    let message: string | null = null;
    let messageIncomplete = false;
    const messageRedacted = isError && ed?.message_redacted === true;
    const messageCut = isError && ed?.message_cut === true;
    if (isError && !messageRedacted && !messageCut && ed?.message !== undefined && ed?.message !== null) {
      const r = resolveText(ed.message);
      if (texts.errorMessage !== undefined) {
        message = texts.errorMessage;
        messageIncomplete = message === null && r.truncated;
      } else {
        message = r.truncated ? null : r.text;
        messageIncomplete = r.truncated;
      }
    }
    const errorType = isError ? ed?.error_type ?? null : null;
    const tb = isError && texts.stderr !== null ? findTracebackLine(texts.stderr, errorType, message) : null;
    api = {
      endRecorded: !!endEv,
      printedStdout: so ? so.end > so.start : null,
      printedStderr: se ? se.end > se.start : null,
      stdoutSlice: so,
      stderrSlice: se,
      isError,
      errorType,
      errorMessage: message,
      messageIncomplete,
      messageRedacted,
      messageCut,
      messageTooShort: isError && message !== null && codePointLength(message) < MIN_SEGMENT_CHARS,
      tracebackLine: tb?.line ?? null,
      tracebackOffset: tb?.offset ?? null,
      tracebackMatches: tb?.count ?? 0,
      missingStreamText: isError && (texts.stdout === null || texts.stderr === null),
    };
  }

  const { items, nextCodegen } = laterAttempts(model, blockEnd.seq);

  const requests: LaterRequestReport[] = items.map(({ call, attempt }) => {
    const summary = dataOf(attempt.request, 'model_request').input_summary ?? null;
    const stdout = checkStream(summary, 'stdout', stdoutIds, stdoutEvent, stdoutLen);
    const stderr = checkStream(summary, 'stderr', stderrIds, stderrEvent, stderrLen);
    const apiSlices: LaterRequestReport['apiSlices'] = [];
    const notes: string[] = [];
    const errorFindings: ErrorTextFinding[] = [];
    let errorNote: LaterRequestReport['errorNote'] = null;

    if (api) {
      const mapSlice = (stream: 'stdout' | 'stderr', slice: { start: number; end: number } | null, check: StreamCheck) => {
        if (!slice || slice.end <= slice.start) return;
        for (const h of check.hits) {
          if (h.exact) apiSlices.push({ stream, range: { path: h.path, start: h.start + slice.start, end: h.start + slice.end } });
          else notes.push(`${stream} のセグメントはこのブロックの ${stream} 全体と一致しないため、APIの出力位置は算出していません`);
        }
      };
      mapSlice('stdout', api.stdoutSlice, stdout);
      mapSlice('stderr', api.stderrSlice, stderr);

      if (api.isError) {
        const search = (what: ErrorTextFinding['what'], needle: string | null, stream: 'stdout' | 'stderr', offset16?: number) => {
          const hay = stream === 'stdout' ? texts.stdout : texts.stderr;
          if (!needle || hay === null) return;
          const idx = offset16 ?? hay.indexOf(needle);
          if (idx < 0) return;
          const off = utf16ToCodePoint(hay, idx);
          const len = codePointLength(needle);
          const check = stream === 'stdout' ? stdout : stderr;
          if (check.hits.length === 0) {
            errorFindings.push({ what, text: needle, stream, streamOffset: off, mapped: null, inUnmappedSegment: false, streamState: check.state });
            return;
          }
          for (const h of check.hits) {
            errorFindings.push({
              what,
              text: needle,
              stream,
              streamOffset: off,
              mapped: h.exact ? { path: h.path, start: h.start + off, end: h.start + off + len } : null,
              inUnmappedSegment: !h.exact,
              streamState: check.state,
            });
          }
        };
        const messageSearchable = api.errorMessage !== null && !api.messageTooShort;
        if (messageSearchable) {
          search('message', api.errorMessage, 'stdout');
          search('message', api.errorMessage, 'stderr');
        }
        if (api.tracebackLine !== null && api.tracebackOffset !== null) {
          search('traceback_line', api.tracebackLine, 'stderr', api.tracebackOffset);
        }
        if (errorFindings.length === 0) {
          errorNote = api.missingStreamText || (!messageSearchable && api.tracebackLine === null) ? 'unchecked' : 'not_in_streams';
        }
      }
    }

    return {
      call,
      attempt,
      httpAttempt: attemptNumber(call, attempt),
      isRetry: call.attempts.indexOf(attempt) > 0,
      recorded: !!summary,
      isNextCodegen: call === nextCodegen,
      stdout,
      stderr,
      apiSlices,
      sliceNote: notes.length ? Array.from(new Set(notes)).join(' / ') : null,
      errorFindings,
      errorNote,
    };
  });

  return {
    applicable: true,
    reason: null,
    block,
    blockEnd,
    stdoutEvent,
    stderrEvent,
    stdoutLen,
    stderrLen,
    requests,
    nextCodegen,
    api,
  };
}
