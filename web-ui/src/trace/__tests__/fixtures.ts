/**
 * Builders of recorder-shaped events for the trace viewer's tests. Field
 * names and nesting follow the trace recorder's HTTP, model, and environment hooks.
 * Ids are predictable (ev_0001, …) so tests can name them.
 */

import type { InputSummary, SegmentMatch, TraceEvent } from '../types';

export interface EvOpts {
  id?: string;
  parent?: string | null;
  refs?: Record<string, unknown>;
  status?: string;
  late?: boolean;
  artifacts?: TraceEvent['artifacts'];
  wall?: number;
}

export class Rec {
  seq = 0;
  events: TraceEvent[] = [];

  add(type: string, data: Record<string, unknown>, o: EvOpts = {}): TraceEvent {
    this.seq += 1;
    const ev: TraceEvent = {
      schema_version: 'capx-trace/1',
      run_id: 'run',
      episode_id: 'ep',
      event_id: o.id ?? `ev_${String(this.seq).padStart(4, '0')}`,
      parent_event_id: o.parent ?? null,
      seq: this.seq,
      wall_time: o.wall ?? 1_790_000_000 + this.seq,
      event_type: type,
      status: o.status ?? 'ok',
      refs: o.refs ?? {},
      artifacts: o.artifacts ?? [],
      data,
    };
    if (o.late) ev.late = true;
    this.events.push(ev);
    return ev;
  }

  episodeStart(data: Record<string, unknown> = {}, o: EvOpts = {}): TraceEvent {
    return this.add('episode_start', { source: 'experiments.run', task: 'cube_lift', condition: 'C', seed: 1, attempt_index: 1, runner_invocation: 1, ...data }, { status: 'started', ...o });
  }

  episodeEnd(data: Record<string, unknown> = {}, o: EvOpts = {}): TraceEvent {
    return this.add('episode_end', { end_reason: 'completed', trace_complete: true, ...data }, o);
  }

  /** One HTTP attempt of a logical call (HTTP hook). */
  request(role: string, lcid: string, attempt = 1, data: Record<string, unknown> = {}, o: EvOpts = {}): TraceEvent {
    return this.add(
      'model_request',
      {
        role,
        call_kind: role === 'vdm' ? 'image_diff' : role === 'code_generator' ? 'multi_turn_decision' : null,
        turn_index: 0,
        logical_call_id: lcid,
        http_attempt: attempt,
        model: 'm',
        stream: false,
        payload_path: null,
        n_images: 0,
        n_videos: 0,
        images: [],
        ...data,
      },
      { status: 'started', refs: { logical_call_id: lcid, http_attempt: attempt }, ...o },
    );
  }

  response(req: TraceEvent, httpStatus = 200, content: unknown = 'ok', data: Record<string, unknown> = {}): TraceEvent {
    const d = req.data as { logical_call_id: string; http_attempt: number; role: string };
    return this.add(
      'model_response',
      { role: d.role, logical_call_id: d.logical_call_id, http_attempt: d.http_attempt, http_status: httpStatus, latency_s: 0.5, content, ...data },
      { parent: req.event_id, refs: { logical_call_id: d.logical_call_id, http_attempt: d.http_attempt, model_event_id: req.event_id } },
    );
  }

  /** An attempt-scope error (HTTP >= 400 or a transport exception, HTTP hook). */
  attemptError(req: TraceEvent, httpStatus: number | null, message = 'HTTP 503'): TraceEvent {
    const d = req.data as { logical_call_id: string; http_attempt: number; role: string };
    return this.add(
      'model_error',
      { scope: 'attempt', role: d.role, logical_call_id: d.logical_call_id, http_attempt: d.http_attempt, http_status: httpStatus, error_type: 'HTTPError', message },
      { status: 'error', parent: req.event_id, refs: { logical_call_id: d.logical_call_id, http_attempt: d.http_attempt } },
    );
  }

  /** The call's final error (model call hook end_logical_error, or web trial hook StreamState.failed with partial_content). */
  logicalError(lastReq: TraceEvent, errorType: string, message: unknown, data: Record<string, unknown> = {}): TraceEvent {
    const d = lastReq.data as { logical_call_id: string; http_attempt: number; role: string };
    return this.add(
      'model_error',
      { scope: 'logical', role: d.role, logical_call_id: d.logical_call_id, http_attempt: d.http_attempt, http_status: 200, error_type: errorType, message, ...data },
      { status: 'error', parent: lastReq.event_id, refs: { logical_call_id: d.logical_call_id } },
    );
  }

  parsed(resp: TraceEvent, data: Record<string, unknown> = {}): TraceEvent {
    return this.add('model_output_parsed', { parser: '_extract_code', model_event_id: resp.event_id, code_blocks: ['x = 1'], ...data }, { parent: resp.event_id });
  }

  blockStart(source: TraceEvent | null, code = 'x = 1', data: Record<string, unknown> = {}): TraceEvent {
    const ev = this.add('code_block_start', { block_index: 0, turn_index: 0, source_model_event_id: source?.event_id ?? null, code_source: 'model', code, ...data }, { status: 'started' });
    (ev.data as Record<string, unknown>).code_block_id = ev.event_id;
    ev.refs = { code_block_id: ev.event_id };
    return ev;
  }

  apiStart(block: TraceEvent, name: string, args: unknown = {}, parentApi: TraceEvent | null = null, data: Record<string, unknown> = {}): TraceEvent {
    const ev = this.add(
      'api_call_start',
      { name, qualname: `Api.${name}`, api_class: 'Api', depth: parentApi ? 1 : 0, args, call_site: { code_block_id: block.event_id, line: 3, source_line: `${name}()` }, ...data },
      { status: 'started', parent: parentApi?.event_id ?? block.event_id },
    );
    (ev.data as Record<string, unknown>).api_call_id = ev.event_id;
    return ev;
  }

  apiEnd(start: TraceEvent, ret: unknown = null, slices: { stdout?: [number, number] | null; stderr?: [number, number] | null } = {}): TraceEvent {
    const sl = (v: [number, number] | null | undefined) => (v === null ? null : { start: v?.[0] ?? 0, end: v?.[1] ?? 0, text: '' });
    return this.add(
      'api_call_end',
      { api_call_id: start.event_id, name: (start.data as { name: string }).name, duration_s: 0.01, return: ret, stdout_slice: sl(slices.stdout), stderr_slice: sl(slices.stderr) },
      { parent: start.event_id },
    );
  }

  apiError(start: TraceEvent, errorType: string, message: unknown, slices: { stdout?: [number, number] | null; stderr?: [number, number] | null } = {}): TraceEvent {
    const sl = (v: [number, number] | null | undefined) => (v === null ? null : { start: v?.[0] ?? 0, end: v?.[1] ?? 0, text: '' });
    return this.add(
      'api_call_error',
      { api_call_id: start.event_id, name: (start.data as { name: string }).name, duration_s: 0.01, error_type: errorType, message, stdout_slice: sl(slices.stdout), stderr_slice: sl(slices.stderr) },
      { status: 'error', parent: start.event_id },
    );
  }

  serviceStart(api: TraceEvent, service: string, fn: string): TraceEvent {
    return this.add('service_call_start', { service, function: fn, request_summary: {} }, { status: 'started', parent: api.event_id });
  }

  serviceEnd(start: TraceEvent, data: Record<string, unknown> = {}, type: 'service_call_end' | 'service_call_error' = 'service_call_end'): TraceEvent {
    const s = start.data as { service: string; function: string };
    return this.add(type, { service: s.service, function: s.function, latency_s: 0.1, ...data }, { status: type === 'service_call_error' ? 'error' : 'ok', parent: start.event_id });
  }

  stream(block: TraceEvent, kind: 'stdout' | 'stderr', text: unknown, length?: number): TraceEvent {
    return this.add(kind, { code_block_id: block.event_id, text, length: length ?? (typeof text === 'string' ? [...text].length : null) }, { parent: block.event_id, refs: { code_block_id: block.event_id } });
  }

  evaluator(block: TraceEvent, reward = 0.5): TraceEvent {
    return this.add('evaluator_result', { scope: 'code_block', code_block_id: block.event_id, reward, task_completed: false }, { parent: block.event_id, refs: { code_block_id: block.event_id } });
  }

  blockEnd(block: TraceEvent, stdoutLen: number | null, stderrLen: number | null, data: Record<string, unknown> = {}): TraceEvent {
    return this.add('code_block_end', { code_block_id: block.event_id, sandbox_rc: 0, duration_s: 0.1, stdout_len: stdoutLen, stderr_len: stderrLen, ...data }, { parent: block.event_id, refs: { code_block_id: block.event_id } });
  }
}

/** An input_summary for a later request: `found` segments of the block's streams, `notFound` registered ones. */
export function summaryWith(found: SegmentMatch[], notFound: { kind: string; source_event_ids: string[]; length?: number }[] = []): InputSummary {
  return {
    n_images: 0,
    n_videos: 0,
    no_image_input: true,
    categories: { images: [], task_api_description: [], code_history: [], stdout_stderr: found, vdm_description: [], other: [] },
    unattributed: [],
    registered_not_found: notFound,
  };
}
