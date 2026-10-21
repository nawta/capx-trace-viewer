/**
 * Events whose `data` did not fit in one event line (docs/trace-format.md 5.1:
 * lines are at most 256 KB). The writer then saves the whole `data` as a gzip
 * JSON asset and keeps only scalar fields and short strings inline, with the
 * markers `oversize`, `data_asset` and `data_asset_written`
 * (when the trace recorder limits line size). The viewer loads the asset when an
 * event that needs it is shown and merges it back, so the panes see the full
 * arguments, return values, input summaries and so on. Until then the event
 * carries only the inline part, and the panes say so.
 *
 * Every event a pane reads for the selection is loaded, not only the selected
 * one: the later requests of 6.4, the previous request of the input diff, and
 * the service calls nested below an API call or block (the debug tab's
 * recorded statistics; at most MAX_NESTED_ASSET_TARGETS of them). A pane that
 * reads such an event says whether its data is loading, failed (with a retry),
 * was never written, or was not loaded (beyond that bound; with a button that
 * loads it), instead of 未記録.
 *
 * Pure functions here; the loading state lives in hooks.ts (useDataAssets).
 *
 * Related: model.ts, laterInput.ts (which events a selection needs),
 * components/DetailTabs.tsx (loading and failed and not recorded notice).
 */

import { laterAttempts } from './laterInput';
import { descendants, enclosingBlock, mainAttempt, previousSameFamily, strOrNull, type EpisodeModel, type Selection } from './model';
import type { DataAssetMarkers, TraceEvent } from './types';

export interface DataAssetRef {
  /** Asset path relative to the episode dir, or null when none was recorded. */
  path: string | null;
  /** False when the writer could not save the asset. */
  written: boolean;
  /** Why the asset is missing (from the matching artifact entry). */
  reason: string | null;
}

export type DataAssetState =
  | { status: 'loading' }
  | { status: 'loaded'; data: Record<string, unknown> }
  | { status: 'failed'; error: string }
  | { status: 'unwritten'; reason: string | null };

function markers(ev: TraceEvent): DataAssetMarkers | null {
  const d = ev.data;
  return d && typeof d === 'object' && !Array.isArray(d) ? (d as DataAssetMarkers) : null;
}

/** True when the writer moved this event's data to an asset. */
export function isOversize(ev: TraceEvent): boolean {
  const m = markers(ev);
  return !!m && (m.oversize === true || typeof m.data_asset === 'string');
}

/** True when the asset of an oversize event was loaded and merged back (mergeDataAsset). */
export function isAssetLoaded(ev: TraceEvent): boolean {
  return (markers(ev) as { data_asset_loaded?: unknown } | null)?.data_asset_loaded === true;
}

/** How far the data of an oversize event is, for a pane that reads it. */
export type PendingAsset =
  | { kind: 'loading'; path: string }
  | { kind: 'failed'; path: string; error: string }
  | { kind: 'unwritten'; path: string | null; reason: string | null }
  /** Never requested: not among the events loaded for this selection (see withTargetsLoading). */
  | { kind: 'idle'; path: string };

/**
 * Why an event still lacks its data: null for a normal event and for one whose asset was
 * merged; otherwise loading, failed, never written, or idle (never requested; the page
 * marks the events it is about to load as loading, withTargetsLoading).
 */
export function pendingAsset(ev: TraceEvent, states: ReadonlyMap<string, DataAssetState>): PendingAsset | null {
  const ref = dataAssetRef(ev);
  if (!ref || isAssetLoaded(ev)) return null;
  const st = states.get(ev.event_id);
  if (st?.status === 'loaded') return null;
  if (!ref.path || !ref.written) return { kind: 'unwritten', path: ref.path, reason: ref.reason };
  if (st?.status === 'unwritten') return { kind: 'unwritten', path: ref.path, reason: st.reason ?? ref.reason };
  if (st?.status === 'failed') return { kind: 'failed', path: ref.path, error: st.error };
  if (st?.status === 'loading') return { kind: 'loading', path: ref.path };
  return { kind: 'idle', path: ref.path };
}

/**
 * `states` with every oversize target that has no state yet marked as loading: the page
 * requests the targets in an effect after the render, so for that one render they would
 * otherwise read as never requested. Returns `states` itself when nothing is added.
 */
export function withTargetsLoading(states: ReadonlyMap<string, DataAssetState>, targets: TraceEvent[]): ReadonlyMap<string, DataAssetState> {
  let out: Map<string, DataAssetState> | null = null;
  for (const ev of targets) {
    if (!isOversize(ev) || states.has(ev.event_id)) continue;
    if (!out) out = new Map(states);
    out.set(ev.event_id, { status: 'loading' });
  }
  return out ?? states;
}

/** Where the full data of an oversize event is, or null for a normal event. */
export function dataAssetRef(ev: TraceEvent): DataAssetRef | null {
  if (!isOversize(ev)) return null;
  const m = markers(ev) as DataAssetMarkers;
  const path = strOrNull(m.data_asset);
  const art = path ? (ev.artifacts ?? []).find((a) => a && a.path === path) : undefined;
  const written = !!path && m.data_asset_written !== false && art?.written !== false;
  return { path, written, reason: art?.reason ?? (path ? null : 'data_asset のパスがありません') };
}

/**
 * The event with its full data merged back: the asset's fields, then the markers (so the
 * panes can still say the data came from an asset). `full` must be the JSON object the
 * writer saved; anything else leaves the event unchanged.
 */
export function mergeDataAsset(ev: TraceEvent, full: unknown): TraceEvent {
  if (!full || typeof full !== 'object' || Array.isArray(full)) return ev;
  const inline = markers(ev) ?? {};
  const data: Record<string, unknown> = {
    ...(full as Record<string, unknown>),
    oversize: true,
    data_asset: inline.data_asset ?? null,
    data_asset_written: inline.data_asset_written ?? null,
    data_asset_loaded: true,
  };
  return { ...ev, data };
}

/**
 * `events` with every loaded data asset merged back. Returns the same array when nothing
 * was loaded, so memoised consumers do not rebuild.
 */
export function applyDataAssets(events: TraceEvent[], states: ReadonlyMap<string, DataAssetState>): TraceEvent[] {
  let changed = false;
  const out = events.map((ev) => {
    const st = states.get(ev.event_id);
    if (st && st.status === 'loaded' && isOversize(ev)) {
      changed = true;
      return mergeDataAsset(ev, st.data);
    }
    return ev;
  });
  return changed ? out : events;
}

/** Parses a loaded asset: the saved data must be a JSON object. */
export function parseDataAsset(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/**
 * Nested service events loaded for one selection at most (a block can hold many calls).
 * A pane that reads one beyond it says it was not loaded and offers to load it.
 */
export const MAX_NESTED_ASSET_TARGETS = 64;

/**
 * Oversize events the panes need for a selection: the selected node's own events, the
 * events of its code block (API calls and streams), the requests of the 6.4 later-input
 * range (their input summaries), the previous request of the same family (input diff), the
 * service calls nested below a selected API call or block (their recorded response
 * summaries give the debug tab's statistics; at most MAX_NESTED_ASSET_TARGETS events), and
 * always episode_start, episode_end and task_text (top bar, left pane).
 */
export function dataAssetTargets(model: EpisodeModel, selection: Selection | null): TraceEvent[] {
  const out = new Map<string, TraceEvent>();
  const add = (ev: TraceEvent | null | undefined) => {
    if (ev && isOversize(ev) && !out.has(ev.event_id)) out.set(ev.event_id, ev);
  };
  add(model.episodeStart);
  add(model.episodeEnd);
  add(model.taskText);
  if (!selection) return Array.from(out.values());
  const node = selection.node;
  for (const ev of node.events) add(ev);
  if (node.kind === 'model_call' && node.call) {
    const prev = previousSameFamily(model, node.call);
    add(prev ? mainAttempt(prev)?.request : null);
    // The execution viewer also shows the first block produced by this call.
    const produced = node.call.producedBlocks[0];
    if (produced) {
      for (const ev of produced.events) add(ev);
      for (const child of produced.children) {
        if (child.kind === 'stdout' || child.kind === 'stderr' || child.kind === 'evaluator') {
          for (const ev of child.events) add(ev);
        }
      }
    }
  }
  const block = node.kind === 'model_call' ? null : enclosingBlock(model, node);
  if (block) {
    for (const ev of block.sourceCall?.node.events ?? []) add(ev);
    for (const ev of block.events) add(ev);
    for (const c of block.children) if (c.kind === 'stdout' || c.kind === 'stderr' || c.kind === 'evaluator') for (const ev of c.events) add(ev);
    if (block.end) for (const { attempt } of laterAttempts(model, block.end.seq).items) add(attempt.request);
  }
  if (node.kind === 'api_call' || node.kind === 'code_block') {
    let nested = 0;
    for (const d of descendants(node)) {
      if (d.kind !== 'service_call') continue;
      for (const ev of d.events) {
        if (nested >= MAX_NESTED_ASSET_TARGETS) break;
        if (isOversize(ev) && !out.has(ev.event_id)) {
          add(ev);
          nested += 1;
        }
      }
    }
  }
  return Array.from(out.values());
}
