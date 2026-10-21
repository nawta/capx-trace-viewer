/**
 * React hooks of the trace viewer (docs/trace-format.md 9.2 paging, 9.3
 * Live/Browse and shareable URLs).
 *
 * - useUrlParams: the `run`, `episode` and `event` query parameters (9.3), so
 *   a view can be shared as a link.
 * - useEpisodeEvents: loads events.jsonl through `/events` by byte offset
 *   (9.2, pager.ts) and, while `poll` is true, fetches new lines about every
 *   second. The page decides when to poll (live.ts).
 * - useDataAssets: loads the data of events that the writer moved to an asset
 *   because the line was too long (dataAsset.ts), for the events a pane needs.
 *   Files are fetched under the events.jsonl generation (api.ts
 *   episodeFileUrl), so a replaced trace never shows the old trace's files; a
 *   retry fetches again past the cache.
 * - useAsync, useStoredState, useFullText: small helpers; full texts that were
 *   stored as assets are fetched only when a pane asks for them.
 *
 * Related: api.ts, pager.ts, TraceViewerPage.tsx.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { episodeFileUrl, errorText, fetchJsonFile, fetchText, traceApi, type EpisodeRef } from './api';
import { dataAssetRef, parseDataAsset, type DataAssetState } from './dataAsset';
import { resolveText } from './model';
import { EMPTY_SNAPSHOT, EventsPager, type PagerSnapshot } from './pager';
import { LIVE_POLL_MS } from './live';
import type { TextValue, TraceEvent } from './types';

// ---------------------------------------------------------------------------
// URL query parameters
// ---------------------------------------------------------------------------

export type UrlParams = Record<string, string | null>;

function readParams(): URLSearchParams {
  return new URLSearchParams(window.location.search);
}

/**
 * Query-string state. `set` merges keys (null removes a key); `push` adds a
 * history entry (run/episode changes), otherwise the entry is replaced
 * (event selection), so the back button goes back to the previous episode.
 */
export function useUrlParams(): [URLSearchParams, (next: UrlParams, push?: boolean) => void] {
  const [params, setParams] = useState<URLSearchParams>(readParams);
  useEffect(() => {
    const onPop = () => setParams(readParams());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);
  const set = useCallback((next: UrlParams, push = false) => {
    const p = readParams();
    for (const [k, v] of Object.entries(next)) {
      if (v === null || v === '') p.delete(k);
      else p.set(k, v);
    }
    const qs = p.toString();
    const url = `${window.location.pathname}${qs ? `?${qs}` : ''}${window.location.hash}`;
    if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      if (push) window.history.pushState(null, '', url);
      else window.history.replaceState(null, '', url);
    }
    setParams(new URLSearchParams(qs));
  }, []);
  return [params, set];
}

// ---------------------------------------------------------------------------
// Generic async loader
// ---------------------------------------------------------------------------

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  /**
   * Reads again. When a read for the same deps is in flight it is not aborted: one more
   * read starts when it settles, so a change made before the call is always seen
   * (the 更新 button, the end of an episode). A read in flight for STALLED_READ_MS or
   * more is given up and a new one starts at once.
   */
  reload: () => void;
  /**
   * A poll tick: reads again unless a read for the same deps is still in flight (that read
   * is kept, and the tick is skipped; after STALLED_READ_MS it is given up and read
   * again). For useInterval.
   */
  poll: () => void;
}

/**
 * A read still in flight after this long is given up by the next poll() or reload() and
 * started again: one request that never settles (a dropped SSH forward, a hung
 * connection) must not stop the polling for good. Reads slower than a poll interval but
 * faster than this are kept. Twice the longest poll interval (the
 * runner records, 30 s), so every poll skips at least one tick before giving a read up.
 */
export const STALLED_READ_MS = 60_000;

/** The read in flight for `key`, unless it has been in flight for STALLED_READ_MS or more. */
function unstalledFlight<F extends { key: string; startedAt: number }>(f: F | null, key: string): F | null {
  return f && f.key === key && Date.now() - f.startedAt < STALLED_READ_MS ? f : null;
}

/**
 * Runs `fn` whenever `deps` change and on `reload()` and `poll()`. When `deps` change the
 * read in flight is aborted and the previous data is dropped at once (data and errors are
 * tagged with the deps they were fetched for, so a render with new deps never sees the
 * old result). A reload or poll keeps a read in flight for the same deps: aborting it on
 * every tick would discard a read slower than the poll interval. After STALLED_READ_MS,
 * the read is abandoned; the previous data is
 * kept until the new result arrives, and also when the read fails (so one failed poll
 * does not blank the page).
 */
export function useAsync<T>(fn: ((signal: AbortSignal) => Promise<T>) | null, deps: unknown[]): AsyncState<T> {
  const [result, setResult] = useState<{ key: string; data: T } | null>(null);
  const [failure, setFailure] = useState<{ key: string; error: string } | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const depsKey = JSON.stringify(deps.map((d) => (typeof d === 'function' ? 'fn' : d ?? null)));
  const active = fn !== null;
  // The deps reads are made for now (null while inactive), and the read in flight.
  const keyRef = useRef<string | null>(null);
  const flight = useRef<{ key: string; ctrl: AbortController; again: boolean; startedAt: number } | null>(null);

  const start = useCallback((key: string) => {
    const f = fnRef.current;
    if (!f) return;
    // A read still in flight here is one that was given up (stalled): drop it.
    const old = flight.current;
    if (old) old.ctrl.abort();
    const ctrl = new AbortController();
    const entry = { key, ctrl, again: false, startedAt: Date.now() };
    flight.current = entry;
    setLoadingKey(key);
    f(ctrl.signal)
      .then((d) => {
        if (ctrl.signal.aborted) return;
        setResult({ key, data: d });
        setFailure(null);
      })
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) setFailure({ key, error: errorText(e) });
      })
      .finally(() => {
        if (ctrl.signal.aborted || flight.current !== entry) return;
        flight.current = null;
        // A reload came while this read ran: read once more.
        if (entry.again && keyRef.current === key) start(key);
        else setLoadingKey(null);
      });
  }, []);

  useEffect(() => {
    if (!active) {
      keyRef.current = null;
      setLoadingKey(null);
      return;
    }
    keyRef.current = depsKey;
    start(depsKey);
    return () => {
      // New deps (another episode, another file) or unmount: the old read is of no use.
      keyRef.current = null;
      const f = flight.current;
      if (f && f.key === depsKey) {
        f.ctrl.abort();
        flight.current = null;
      }
    };
    // `depsKey` stands for `deps`: callers pass the values `fn` depends on.
  }, [depsKey, active, start]);

  const reload = useCallback(() => {
    const key = keyRef.current;
    if (key === null) return;
    const f = unstalledFlight(flight.current, key);
    if (f) f.again = true;
    else start(key);
  }, [start]);
  const poll = useCallback(() => {
    const key = keyRef.current;
    if (key === null) return;
    if (!unstalledFlight(flight.current, key)) start(key);
  }, [start]);

  if (!active) return { data: null, error: null, loading: false, reload, poll };
  const hasResult = result !== null && result.key === depsKey;
  const data = hasResult ? result.data : null;
  const error = failure && failure.key === depsKey ? failure.error : null;
  // Loading until a result or an error for these deps exists (covers the render before the effect runs).
  const loading = loadingKey === depsKey || (!hasResult && error === null);
  return { data, error, loading, reload, poll };
}

// ---------------------------------------------------------------------------
// Events: byte-offset paging and live polling
// ---------------------------------------------------------------------------

export type EventsState = PagerSnapshot;

export interface EpisodeEvents extends EventsState {
  /**
   * Fetches everything new once (used when an episode stops running, and by the refresh
   * button). When a fetch is already running, it reads once more after it, so nothing
   * written before the call is missed.
   */
  refresh: () => void;
}

export function useEpisodeEvents(runId: string | null, episodeId: string | null, poll: boolean): EpisodeEvents {
  // Which episode `state` belongs to. The pager for a new episode is created in an effect,
  // so for one render the ids are new while `state` still holds the previous episode's
  // events; those must never be shown (or used to build file URLs) under the new ids.
  const [state, setState] = useState<{ key: string | null; snap: PagerSnapshot }>({ key: null, snap: EMPTY_SNAPSHOT });
  const currentKey = runId && episodeId ? `${runId}\u0000${episodeId}` : null;
  const pagerRef = useRef<EventsPager | null>(null);

  useEffect(() => {
    if (!runId || !episodeId) {
      pagerRef.current = null;
      setState({ key: null, snap: EMPTY_SNAPSHOT });
      return;
    }
    const key = `${runId}\u0000${episodeId}`;
    const ctrl = new AbortController();
    const pager = new EventsPager(
      (after, generation) => traceApi.events(runId, episodeId, after, { generation }, ctrl.signal),
      (snap) => setState({ key, snap }),
    );
    pagerRef.current = pager;
    setState({ key, snap: pager.snapshot });
    void pager.pump();
    return () => {
      ctrl.abort();
      pager.dispose();
      if (pagerRef.current === pager) pagerRef.current = null;
    };
  }, [runId, episodeId]);

  // Polling about every second while `poll` is true.
  useEffect(() => {
    if (!poll || !runId || !episodeId) return;
    let stopped = false;
    let timer: number | undefined;
    const tick = async () => {
      if (stopped) return;
      const pager = pagerRef.current;
      if (pager) await pager.pump();
      if (!stopped) timer = window.setTimeout(tick, LIVE_POLL_MS);
    };
    timer = window.setTimeout(tick, LIVE_POLL_MS);
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [poll, runId, episodeId]);

  const refresh = useCallback(() => {
    void pagerRef.current?.pump();
  }, []);

  if (state.key !== currentKey) return { ...EMPTY_SNAPSHOT, loading: currentKey !== null, refresh };
  return { ...state.snap, refresh };
}

// ---------------------------------------------------------------------------
// Data moved to assets (oversize events)
// ---------------------------------------------------------------------------

export interface DataAssets {
  /** Loading state per event id (only events that were asked for). */
  states: ReadonlyMap<string, DataAssetState>;
  /** Starts loading the data assets of these events (each at most once). */
  ensure: (events: TraceEvent[]) => void;
  /** Loads one event's data asset again (after a failure). */
  retry: (ev: TraceEvent) => void;
}

const NO_ASSET_STATES: ReadonlyMap<string, DataAssetState> = new Map();

/**
 * Loading state of the data assets of one episode. The state is dropped when the episode's
 * generation changes or `resetKey` changes (the events were read again from the start
 * because events.jsonl was replaced); assets are then fetched under the new generation.
 */
export function useDataAssets(ep: EpisodeRef | null, resetKey: string): DataAssets {
  const runId = ep?.runId ?? null;
  const episodeId = ep?.episodeId ?? null;
  const generation = ep?.generation ?? null;
  const key = runId && episodeId ? `${runId}\u0000${episodeId}\u0000${generation ?? ''}\u0000${resetKey}` : '';
  const [store, setStore] = useState<{ key: string; map: ReadonlyMap<string, DataAssetState> }>({ key: '', map: NO_ASSET_STATES });
  const keyRef = useRef(key);
  keyRef.current = key;
  const requested = useRef<{ key: string; ids: Set<string> }>({ key, ids: new Set() });

  const put = useCallback((forKey: string, id: string, st: DataAssetState) => {
    setStore((prev) => {
      if (keyRef.current !== forKey) return prev;
      const m = new Map(prev.key === forKey ? prev.map : NO_ASSET_STATES);
      m.set(id, st);
      return { key: forKey, map: m };
    });
  }, []);

  const load = useCallback(
    (ev: TraceEvent, force: boolean) => {
      const k = keyRef.current;
      if (!k || !runId || !episodeId) return;
      if (requested.current.key !== k) requested.current = { key: k, ids: new Set() };
      if (!force && requested.current.ids.has(ev.event_id)) return;
      const ref = dataAssetRef(ev);
      if (!ref) return;
      requested.current.ids.add(ev.event_id);
      if (!ref.path || !ref.written) {
        put(k, ev.event_id, { status: 'unwritten', reason: ref.reason });
        return;
      }
      put(k, ev.event_id, { status: 'loading' });
      // A retry reads the file again past the cache (it may have been repaired or rewritten).
      fetchJsonFile<unknown>(episodeFileUrl({ runId, episodeId, generation }, ref.path), { fresh: force })
        .then((v) => {
          const obj = parseDataAsset(v);
          put(k, ev.event_id, obj ? { status: 'loaded', data: obj } : { status: 'failed', error: '保存された data が JSON オブジェクトではありません' });
        })
        .catch((e: unknown) => put(k, ev.event_id, { status: 'failed', error: errorText(e) }));
    },
    [runId, episodeId, generation, put],
  );

  const ensure = useCallback((events: TraceEvent[]) => events.forEach((ev) => load(ev, false)), [load]);
  const retry = useCallback((ev: TraceEvent) => load(ev, true), [load]);
  return { states: store.key === key ? store.map : NO_ASSET_STATES, ensure, retry };
}

/**
 * Calls `fn` every `ms` while `active` (the episode summary in Live and Browse mode, the
 * Viser check, the runner records). Pass an AsyncState's `poll`, which skips a tick while
 * the previous read is still in flight.
 */
export function useInterval(fn: () => void, ms: number, active: boolean): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => ref.current(), ms);
    return () => window.clearInterval(id);
  }, [ms, active]);
}

// ---------------------------------------------------------------------------
// Per-viewer conveniences (pane sizes)
// ---------------------------------------------------------------------------

export function useStoredState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw !== null) return JSON.parse(raw) as T;
    } catch {
      /* storage unavailable: use the default */
    }
    return initial;
  });
  const set = useCallback(
    (v: T) => {
      setValue(v);
      try {
        window.localStorage.setItem(key, JSON.stringify(v));
      } catch {
        /* storage unavailable: keep the value for this page only */
      }
    },
    [key],
  );
  return [value, set];
}

// ---------------------------------------------------------------------------
// Full text of a TextValue (inline, or an asset loaded on demand)
// ---------------------------------------------------------------------------

export interface FullText {
  text: string | null;
  /** Only a preview is available until the asset is loaded. */
  partial: boolean;
  loading: boolean;
  error: string | null;
  asset: string | null;
  length: number | null;
}

/**
 * The full text of a TextValue: inline, or its asset fetched under the episode's generation
 * (a replaced trace gets a new URL, so the old text is never shown for it).
 */
export function useFullText(ep: EpisodeRef | null, value: TextValue | null | undefined, enabled = true): FullText {
  const r = resolveText(value);
  const needFetch = enabled && r.asset !== null && r.truncated && !!ep;
  const url = needFetch && ep && r.asset ? episodeFileUrl(ep, r.asset) : null;
  const [loaded, setLoaded] = useState<{ url: string; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchText(url)
      .then((t) => {
        if (!cancelled) setLoaded({ url, text: t });
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(errorText(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [url]);
  if (url && loaded && loaded.url === url) {
    return { text: loaded.text, partial: false, loading: false, error: null, asset: r.asset, length: r.length };
  }
  return {
    text: r.text,
    partial: r.truncated,
    loading: url ? loading : false,
    error: url ? error : null,
    asset: r.asset,
    length: r.length,
  };
}
