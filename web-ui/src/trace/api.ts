/**
 * HTTP client for the trace API (docs/trace-format.md section 9.2).
 *
 * Every endpoint of 9.2 has a function here. `/events` is paged by byte offset:
 * callers pass the `next_offset` of the previous page as `after_offset`.
 * Files under the episode dir are fetched lazily (payloads, text assets,
 * state snapshots) and cached per URL. The panes build those URLs with
 * episodeFileUrl, which adds the events.jsonl generation (`g`, ignored by the
 * server): when a trace is replaced its files get new URLs, so neither this
 * cache nor the browser's shows a file of the old trace. A file that is not
 * valid JSON is dropped from the cache, and an explicit retry fetches again.
 *
 * Related: types.ts (response shapes), hooks.ts (paging and live polling),
 * capx_trace_viewer/trace_api.py (the server side).
 */

import type {
  ArrayResponse,
  EpisodeSummary,
  EventsPage,
  RunDetailResponse,
  RunListResponse,
  TraceConfigResponse,
  ViserResponse,
} from './types';

const BASE = '/api/traces';

export class ApiError extends Error {
  readonly status: number;
  readonly url: string;
  /** The parsed JSON body of the error answer, when it had one (e.g. 409 generation_changed). */
  readonly body: unknown;

  constructor(status: number, message: string, url: string, body: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.url = url;
    this.body = body;
  }
}

function seg(id: string): string {
  return encodeURIComponent(id);
}

/** Encodes a relative path segment by segment, keeping the slashes. */
export function encodeRelPath(relpath: string): string {
  return relpath
    .split('/')
    .filter((p) => p.length > 0)
    .map((p) => encodeURIComponent(p))
    .join('/');
}

function rootQuery(root: number | null | undefined): string {
  return typeof root === 'number' && root >= 0 ? `?root=${root}` : '';
}

function episodeBase(runId: string, episodeId: string): string {
  return `${BASE}/runs/${seg(runId)}/episodes/${seg(episodeId)}`;
}

/**
 * An open episode, with the generation of its events.jsonl as the pager last read it
 * (device, inode and first-line hash; null before the first page).
 */
export interface EpisodeRef {
  runId: string;
  episodeId: string;
  generation?: string | null;
}

function withGeneration(url: string, generation: string | null | undefined): string {
  if (!generation) return url;
  return `${url}${url.includes('?') ? '&' : '?'}g=${encodeURIComponent(generation)}`;
}

async function readError(resp: Response): Promise<{ message: string; body: unknown }> {
  try {
    const ct = resp.headers.get('content-type') ?? '';
    if (ct.includes('application/json')) {
      const body = (await resp.json()) as unknown;
      const o = body && typeof body === 'object' ? (body as { detail?: unknown; error?: unknown }) : {};
      if (typeof o.detail === 'string') return { message: o.detail, body };
      if (o.detail !== undefined) return { message: JSON.stringify(o.detail), body };
      if (typeof o.error === 'string') return { message: o.error, body };
      return { message: JSON.stringify(body).slice(0, 300), body };
    }
    const text = await resp.text();
    return { message: text.slice(0, 300) || resp.statusText, body: null };
  } catch {
    return { message: resp.statusText, body: null };
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const resp = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  if (!resp.ok) {
    const e = await readError(resp);
    throw new ApiError(resp.status, e.message, url, e.body);
  }
  const ct = resp.headers.get('content-type') ?? '';
  if (!ct.includes('json')) {
    // The chat server's SPA fallback answers unknown GETs with index.html.
    throw new ApiError(
      resp.status,
      'JSON ではない応答です。このサーバーに /api/traces が無い可能性があります。',
      url,
    );
  }
  return (await resp.json()) as T;
}

export interface EventsQuery {
  maxBytes?: number;
  maxEvents?: number;
  /**
   * The file generation of the previous page. The server answers 409
   * {"error": "generation_changed", "generation": "<new>"} when events.jsonl was replaced.
   */
  generation?: string | null;
}

export const traceApi = {
  config(signal?: AbortSignal): Promise<TraceConfigResponse> {
    return getJson(`${BASE}/config`, signal);
  },

  runs(signal?: AbortSignal): Promise<RunListResponse> {
    return getJson(`${BASE}/runs`, signal);
  },

  run(runId: string, signal?: AbortSignal): Promise<RunDetailResponse> {
    return getJson(`${BASE}/runs/${seg(runId)}`, signal);
  },

  episode(runId: string, episodeId: string, signal?: AbortSignal): Promise<EpisodeSummary> {
    return getJson(episodeBase(runId, episodeId), signal);
  },

  events(
    runId: string,
    episodeId: string,
    afterOffset: number,
    query: EventsQuery = {},
    signal?: AbortSignal,
  ): Promise<EventsPage> {
    const params = new URLSearchParams({
      after_offset: String(afterOffset),
      max_bytes: String(query.maxBytes ?? 2_000_000),
      max_events: String(query.maxEvents ?? 5000),
    });
    if (query.generation) params.set('generation', query.generation);
    return getJson(`${episodeBase(runId, episodeId)}/events?${params.toString()}`, signal);
  },

  fileUrl(runId: string, episodeId: string, relpath: string): string {
    return `${episodeBase(runId, episodeId)}/file/${encodeRelPath(relpath)}`;
  },

  array(runId: string, episodeId: string, relpath: string, name?: string, signal?: AbortSignal): Promise<ArrayResponse> {
    const q = name ? `?name=${encodeURIComponent(name)}` : '';
    return getJson(`${episodeBase(runId, episodeId)}/array/${encodeRelPath(relpath)}${q}`, signal);
  },

  /** PNG of a 2-D mask or depth array; `index` picks a slice of a 3-D stack. */
  arrayImageUrl(runId: string, episodeId: string, relpath: string, name?: string, index?: number): string {
    const params = new URLSearchParams();
    if (name) params.set('name', name);
    if (index !== undefined && index > 0) params.set('index', String(index));
    const q = params.toString();
    return `${episodeBase(runId, episodeId)}/array-image/${encodeRelPath(relpath)}${q ? `?${q}` : ''}`;
  },

  viser(signal?: AbortSignal): Promise<ViserResponse> {
    return getJson(`${BASE}/viser`, signal);
  },

  legacyRuns(signal?: AbortSignal): Promise<unknown> {
    return getJson(`${BASE}/legacy/runs`, signal);
  },

  /** `root` is the runner-root index from /legacy/runs (the same run id can exist under two roots). */
  legacyRun(runId: string, root: number | null, signal?: AbortSignal): Promise<unknown> {
    return getJson(`${BASE}/legacy/runs/${seg(runId)}${rootQuery(root)}`, signal);
  },

  legacyTrial(runId: string, task: string, condition: string, trialDir: string, root: number | null, signal?: AbortSignal): Promise<unknown> {
    return getJson(`${BASE}/legacy/trials/${seg(runId)}/${seg(task)}/${seg(condition)}/${seg(trialDir)}${rootQuery(root)}`, signal);
  },

  /** URL of a file below a legacy trial dir (9.4). */
  legacyFileUrl(runId: string, task: string, condition: string, trialDir: string, relpath: string, root: number | null): string {
    return `${BASE}/legacy/file/${seg(runId)}/${seg(task)}/${seg(condition)}/${seg(trialDir)}/${encodeRelPath(relpath)}${rootQuery(root)}`;
  },
};

// ---------------------------------------------------------------------------
// Lazy file loading (payloads, text assets, gzip JSON state snapshots)
// ---------------------------------------------------------------------------

const textCache = new Map<string, Promise<string>>();
const MAX_CACHE_ENTRIES = 200;

function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

/**
 * URL of a file under the episode dir, tagged with the events.jsonl generation the episode
 * was read under. Payload paths are named after event ids, which a replaced trace can
 * reuse; with the generation in the URL a replaced trace never gets a cached file of the
 * old one.
 */
export function episodeFileUrl(ep: EpisodeRef, relpath: string): string {
  return withGeneration(traceApi.fileUrl(ep.runId, ep.episodeId, relpath), ep.generation);
}

/** traceApi.arrayImageUrl for an open episode, tagged with its generation (see episodeFileUrl). */
export function episodeArrayImageUrl(ep: EpisodeRef, relpath: string, name?: string, index?: number): string {
  return withGeneration(traceApi.arrayImageUrl(ep.runId, ep.episodeId, relpath, name, index), ep.generation);
}

export interface FetchFileOptions {
  /** Drop a cached copy and fetch again (an explicit retry). */
  fresh?: boolean;
}

/**
 * Fetches a file as UTF-8 text. `.gz` files are decompressed in the browser
 * (checked by magic bytes, so a server that already sent Content-Encoding:
 * gzip also works). Results are cached per URL; a failed fetch is not kept.
 */
export function fetchText(url: string, opts: FetchFileOptions = {}): Promise<string> {
  if (opts.fresh) textCache.delete(url);
  const cached = textCache.get(url);
  if (cached) return cached;
  const p = (async () => {
    const resp = await fetch(url);
    if (!resp.ok) {
      const e = await readError(resp);
      throw new ApiError(resp.status, e.message, url, e.body);
    }
    let bytes: Uint8Array = new Uint8Array(await resp.arrayBuffer());
    if (isGzip(bytes)) bytes = await gunzip(bytes);
    return new TextDecoder('utf-8').decode(bytes);
  })();
  textCache.set(url, p);
  p.catch(() => textCache.delete(url));
  if (textCache.size > MAX_CACHE_ENTRIES) {
    const first = textCache.keys().next().value;
    if (first !== undefined) textCache.delete(first);
  }
  return p;
}

/**
 * Fetches and parses a JSON file (fetchText). A body that is not valid JSON (a file still
 * being written, or broken) is dropped from the cache, so the next attempt reads the file
 * again instead of the same broken text.
 */
export async function fetchJsonFile<T = unknown>(url: string, opts: FetchFileOptions = {}): Promise<T> {
  const p = fetchText(url, opts);
  const text = await p;
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    if (textCache.get(url) === p) textCache.delete(url);
    throw new Error(`JSON として読めません（${e instanceof Error ? e.message : String(e)}）`);
  }
}

/** Human-readable text of an error thrown by this module or by fetch. */
export function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return `見つかりません (404): ${err.message}`;
    if (err.status === 403) return `アクセスできません (403): ${err.message}`;
    return `HTTP ${err.status}: ${err.message}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}
