/**
 * A fake /api/traces server for the page tests: one run with episodes whose
 * summaries, events and files a test changes while the page is mounted (an
 * episode ends, a runner record is appended, events.jsonl is replaced).
 * Answers follow capx_trace_viewer/trace_api.py (9.2): /events pages by offset (one
 * event = 100 bytes here) and answers 409 generation_changed when the client
 * names another generation.
 *
 * Related: harness/browser.ts (the fetch it plugs into).
 */

import { net, type FakeAnswer, type FakeRequest } from './harness/browser';
import type { EpisodeSummary, RunnerRecord, TraceEvent } from '../types';

export interface FakeEpisode {
  summary: EpisodeSummary;
  events: TraceEvent[];
  generation: string;
  /** Files below the episode dir by relpath. */
  files: Map<string, FakeAnswer>;
}

export interface FakeRun {
  runId: string;
  records: RunnerRecord[];
  episodes: Map<string, FakeEpisode>;
  latency: { config: number; run: number; summary: number; events: number; file: number };
  viewerOnly: boolean;
}

export function makeRun(runId = 'r1'): FakeRun {
  return { runId, records: [], episodes: new Map(), latency: { config: 5, run: 20, summary: 20, events: 20, file: 5 }, viewerOnly: true };
}

export function addEpisode(run: FakeRun, id: string, summary: Partial<EpisodeSummary>, events: TraceEvent[]): FakeEpisode {
  const ep: FakeEpisode = { summary: { episode_id: id, ...summary }, events, generation: 'g1', files: new Map() };
  run.episodes.set(id, ep);
  return ep;
}

export function ev(epId: string, seq: number, type: string, data: Record<string, unknown>, o: Partial<TraceEvent> = {}): TraceEvent {
  return {
    schema_version: 'capx-trace/1',
    run_id: 'r1',
    episode_id: epId,
    event_id: `${epId}_${seq}`,
    parent_event_id: null,
    seq,
    wall_time: 1_790_000_000 + seq,
    event_type: type,
    status: type.endsWith('_start') || type === 'model_request' ? 'started' : 'ok',
    refs: {},
    artifacts: [],
    data,
    ...o,
  };
}

/** Serves `run` through the harness fetch. */
export function serve(run: FakeRun): void {
  net.handler = (req: FakeRequest): FakeAnswer | null => {
    const p = req.path;
    if (p === '/api/traces/config') {
      return { json: { trace_root: '/x', exists: true, viewer_only: run.viewerOnly, runner_roots: [], expose: false }, latency: run.latency.config };
    }
    const base = `/api/traces/runs/${run.runId}`;
    if (p === base) {
      return {
        json: { run_id: run.runId, run: {}, runner_records: run.records, episodes: Array.from(run.episodes.values()).map((e) => e.summary) },
        latency: run.latency.run,
      };
    }
    const m = p.slice(base.length).match(/^\/episodes\/([^/]+)(\/events|\/file\/(.+))?$/);
    if (!p.startsWith(`${base}/`) || !m) return null;
    const ep = run.episodes.get(decodeURIComponent(m[1]));
    if (!ep) return { status: 404, json: { detail: 'episode not found' } };
    if (!m[2]) return { json: ep.summary, latency: run.latency.summary };
    if (m[2] === '/events') {
      const gen = req.query.get('generation');
      if (gen && gen !== ep.generation) return { status: 409, json: { error: 'generation_changed', generation: ep.generation }, latency: run.latency.events };
      const after = Number(req.query.get('after_offset') ?? '0');
      return {
        json: { events: ep.events.slice(Math.floor(after / 100)), next_offset: ep.events.length * 100, eof: true, torn_tail: false, corrupt_lines: 0, generation: ep.generation },
        latency: run.latency.events,
      };
    }
    const file = ep.files.get(decodeURIComponent(m[3] ?? ''));
    return file ? { latency: run.latency.file, ...file } : { status: 404, json: { detail: 'file not found' }, latency: run.latency.file };
  };
}
