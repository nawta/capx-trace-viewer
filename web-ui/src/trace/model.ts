/**
 * Causal tree of one episode, built from its events.
 *
 * Implements the grouping of docs/trace-format.md section 9.3 (left pane)
 * from the ids of section 5: HTTP attempts (`model_request`) are grouped under
 * their `logical_call_id`, `model_response`/`model_error` hang under their
 * request (`parent_event_id`), code blocks point back to the call that
 * produced them (`source_model_event_id`), API calls nest under code blocks
 * and under each other (`parent_event_id`, `api_call_id`), service calls and
 * perception VLM requests nest under API calls, and stdout/stderr/evaluator
 * results hang under their block (`code_block_id`). Events are never linked
 * by time proximity: an event whose parent id does not resolve is shown as an
 * orphan with the reason. `seq` (file order) is used only for ordering.
 *
 * Outcome of a logical model call (callOutcome): a `model_error` with scope
 * `logical` is the call's final result and wins over an HTTP 2xx answer (the
 * client can fail after HTTP 200, for example on a malformed body, or a
 * stream can be cut); `model_error`s with scope `attempt` belong to one HTTP
 * attempt and do not fail the call when a later attempt was answered.
 *
 * Related: types.ts (event shapes), laterInput.ts (6.4 report),
 * components/Timeline.tsx (renders the tree), components/MediaPane.tsx (frames),
 * dataAsset.ts (events whose data was moved to an asset).
 */

import { LABELS } from './labels';
import {
  dataOf,
  type Artifact,
  type FrameRef,
  type ImageInfo,
  type ModelRequestData,
  type StateRef,
  type TextValue,
  type TraceEvent,
} from './types';

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

export type NodeKind =
  | 'model_call'
  | 'code_block'
  | 'api_call'
  | 'service_call'
  | 'stdout'
  | 'stderr'
  | 'evaluator'
  | 'feedback'
  | 'episode_start'
  | 'episode_end'
  | 'env_reset'
  | 'warning'
  | 'other';

/** One HTTP attempt of a logical model call. */
export interface Attempt {
  request: TraceEvent;
  responses: TraceEvent[];
  errors: TraceEvent[];
}

/** A logical model call: every HTTP attempt that shares a logical_call_id. */
export interface LogicalCall {
  id: string;
  /** True when the requests carried no logical_call_id (one attempt, keyed by its event id). */
  synthetic: boolean;
  role: string;
  callKind: string;
  turnIndex: number | null;
  model: string | null;
  attempts: Attempt[];
  /** `model_error` with scope `logical`: the call's final failure (after all attempts). */
  logicalErrors: TraceEvent[];
  /** Responses that name this call but no known attempt. */
  unmatched: TraceEvent[];
  /** Attempt-scope `model_error`s whose attempt (request) is not in the record. */
  strayErrors: TraceEvent[];
  parsed: TraceEvent[];
  producedBlocks: TNode[];
  node: TNode;
}

export interface TNode {
  key: string;
  kind: NodeKind;
  /** Anchor event: the start event, the first request, or the event itself. */
  primary: TraceEvent;
  /** End or error event of a start/end pair. */
  end: TraceEvent | null;
  events: TraceEvent[];
  children: TNode[];
  parent: TNode | null;
  seq: number;
  lastSeq: number;
  /** A start event without its end event. */
  open: boolean;
  failed: boolean;
  degraded: boolean;
  late: boolean;
  /** trace_warning events whose related_event_id points into this node. */
  warnings: TraceEvent[];
  /** Why the parent id did not resolve (shown on the card); null when linked. */
  orphan: string | null;
  call: LogicalCall | null;
  /** For code blocks: the call named by source_model_event_id. */
  sourceCall: LogicalCall | null;
  /** Lower-cased text for the free-text search. */
  searchText: string;
}

export interface FrameRec {
  event: TraceEvent;
  camera: string;
  epoch: number;
  bufferIndex: number;
  simStep: number | null;
  reason: string;
  forEventId: string | null;
  path: string | null;
  written: boolean;
  writeReason: string | null;
}

export interface EpisodeModel {
  events: TraceEvent[];
  byId: Map<string, TraceEvent>;
  roots: TNode[];
  allNodes: TNode[];
  nodeOfEvent: Map<string, TNode>;
  calls: LogicalCall[];
  callOfEvent: Map<string, LogicalCall>;
  blocks: TNode[];
  blockById: Map<string, TNode>;
  apiById: Map<string, TNode>;
  frames: FrameRec[];
  framesByCamera: Map<string, FrameRec[]>;
  frameByKey: Map<string, FrameRec>;
  keyframesFor: Map<string, FrameRec[]>;
  snapshots: TraceEvent[];
  snapshotByPixelSha: Map<string, TraceEvent>;
  videoReads: TraceEvent[];
  episodeStart: TraceEvent | null;
  episodeEnd: TraceEvent | null;
  taskText: TraceEvent | null;
  warnings: TraceEvent[];
  evaluators: TraceEvent[];
  feedbacks: TraceEvent[];
  rolesSeen: Set<string>;
  lastSeq: number;
  countsByType: Map<string, number>;
}

// ---------------------------------------------------------------------------
// Small readers
// ---------------------------------------------------------------------------

export interface ResolvedText {
  /** Inline text (possibly a truncated preview). */
  text: string | null;
  /** Asset path of the full text, relative to the episode dir. */
  asset: string | null;
  truncated: boolean;
  length: number | null;
}

export function resolveText(v: TextValue | null | undefined): ResolvedText {
  if (v === null || v === undefined) return { text: null, asset: null, truncated: false, length: null };
  if (typeof v === 'string') return { text: v, asset: null, truncated: false, length: v.length };
  if (typeof v !== 'object') return { text: String(v), asset: null, truncated: false, length: null };
  const text = strOrNull(v.text) ?? strOrNull(v.preview) ?? strOrNull(v.head);
  const asset = strOrNull(v.asset) ?? strOrNull(v.$asset) ?? strOrNull(v.path);
  const length = numOrNull(v.length) ?? numOrNull(v.chars);
  const truncated = v.truncated === true || (asset !== null && (length === null || text === null || text.length < length));
  return { text, asset, truncated, length };
}

export function strOrNull(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

/**
 * Inline text of a value that may be a plain string or a text-asset reference
 * ({truncated, preview, length, asset}): the string, or the preview; '' otherwise.
 * Never throws on an unexpected shape (a long exception message is an object).
 */
export function textOf(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v && typeof v === 'object' && !Array.isArray(v)) return resolveText(v as TextValue).text ?? '';
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return '';
}

export function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function frameRefsOf(v: FrameRef | FrameRef[] | null | undefined): FrameRef[] {
  if (!v) return [];
  if (Array.isArray(v)) return v.filter((r): r is FrameRef => !!r && typeof r === 'object');
  if (typeof v === 'object') return [v];
  return [];
}

export interface ResolvedStateRef {
  path: string | null;
  reason: string | null;
  written: boolean;
}

export function resolveStateRef(v: StateRef | undefined): ResolvedStateRef | null {
  if (v === undefined || v === null) return null;
  if (typeof v === 'string') return { path: v, reason: null, written: true };
  const path = strOrNull(v.path) ?? strOrNull(v.asset);
  return { path, reason: strOrNull(v.reason), written: v.written !== false && path !== null };
}

function lower(s: unknown): string {
  return typeof s === 'string' ? s.toLowerCase() : '';
}

function isImageArtifact(a: Artifact): boolean {
  const mime = a.mime ?? '';
  const p = a.path ?? '';
  return mime.startsWith('image/') || /\.(png|jpe?g|webp|gif)$/i.test(p);
}

export function frameKey(camera: string, epoch: number, bufferIndex: number): string {
  return `${camera}|${epoch}|${bufferIndex}`;
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

function mkNode(kind: NodeKind, ev: TraceEvent, key?: string): TNode {
  return {
    key: key ?? ev.event_id,
    kind,
    primary: ev,
    end: null,
    events: [ev],
    children: [],
    parent: null,
    seq: ev.seq,
    lastSeq: ev.seq,
    open: false,
    failed: false,
    degraded: false,
    late: false,
    warnings: [],
    orphan: null,
    call: null,
    sourceCall: null,
    searchText: '',
  };
}

function isAncestor(candidate: TNode, node: TNode): boolean {
  let cur: TNode | null = candidate;
  let guard = 0;
  while (cur && guard < 10_000) {
    if (cur === node) return true;
    cur = cur.parent;
    guard += 1;
  }
  return false;
}

function attach(parent: TNode, child: TNode): boolean {
  if (parent === child || isAncestor(parent, child)) return false;
  child.parent = parent;
  parent.children.push(child);
  return true;
}

/** Builds the episode model. `events` may be in any order; seq decides order. */
export function buildEpisodeModel(input: TraceEvent[]): EpisodeModel {
  const events = input.filter((e) => e && typeof e.event_id === 'string').slice();
  events.sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));

  const byId = new Map<string, TraceEvent>();
  const countsByType = new Map<string, number>();
  for (const ev of events) {
    byId.set(ev.event_id, ev);
    countsByType.set(ev.event_type, (countsByType.get(ev.event_type) ?? 0) + 1);
  }

  const allNodes: TNode[] = [];
  const nodeOfEvent = new Map<string, TNode>();
  const blockById = new Map<string, TNode>();
  const apiById = new Map<string, TNode>();
  const serviceById = new Map<string, TNode>();
  const callsById = new Map<string, LogicalCall>();
  const callOfEvent = new Map<string, LogicalCall>();
  const roots: TNode[] = [];
  const pendingParent: { node: TNode; parentId: string | null; fallbackBlockId: string | null; expect: string }[] = [];

  const frames: FrameRec[] = [];
  const snapshots: TraceEvent[] = [];
  const videoReads: TraceEvent[] = [];
  const warnings: TraceEvent[] = [];
  const evaluators: TraceEvent[] = [];
  const feedbacks: TraceEvent[] = [];
  let episodeStart: TraceEvent | null = null;
  let episodeEnd: TraceEvent | null = null;
  let taskText: TraceEvent | null = null;

  const addNode = (n: TNode): TNode => {
    allNodes.push(n);
    for (const e of n.events) nodeOfEvent.set(e.event_id, n);
    return n;
  };
  const addEventToNode = (n: TNode, ev: TraceEvent) => {
    n.events.push(ev);
    nodeOfEvent.set(ev.event_id, n);
  };
  const orphanNode = (ev: TraceEvent, reason: string, kind: NodeKind = 'other'): TNode => {
    const n = addNode(mkNode(kind, ev));
    n.orphan = reason;
    roots.push(n);
    return n;
  };

  // Pass 1: start events and requests.
  for (const ev of events) {
    const refs = ev.refs ?? {};
    switch (ev.event_type) {
      case 'code_block_start': {
        const d = dataOf(ev, 'code_block_start');
        const id = d.code_block_id ?? strOrNull(refs.code_block_id) ?? ev.event_id;
        const n = addNode(mkNode('code_block', ev));
        blockById.set(id, n);
        blockById.set(ev.event_id, n);
        break;
      }
      case 'api_call_start': {
        const d = dataOf(ev, 'api_call_start');
        const id = d.api_call_id ?? strOrNull(refs.api_call_id) ?? ev.event_id;
        const n = addNode(mkNode('api_call', ev));
        apiById.set(id, n);
        apiById.set(ev.event_id, n);
        break;
      }
      case 'service_call_start': {
        const d = dataOf(ev, 'service_call_start');
        const n = addNode(mkNode('service_call', ev));
        serviceById.set(ev.event_id, n);
        if (d.service_call_id) serviceById.set(d.service_call_id, n);
        break;
      }
      case 'model_request': {
        const d = dataOf(ev, 'model_request');
        const lid = d.logical_call_id ?? strOrNull(refs.logical_call_id);
        const id = lid ?? `req:${ev.event_id}`;
        let call = callsById.get(id);
        if (!call) {
          call = {
            id,
            synthetic: lid === null,
            role: d.role ?? 'unknown',
            callKind: d.call_kind ?? '',
            turnIndex: d.turn_index ?? numOrNull(refs.turn_index),
            model: d.model ?? null,
            attempts: [],
            logicalErrors: [],
            unmatched: [],
            strayErrors: [],
            parsed: [],
            producedBlocks: [],
            node: null as unknown as TNode,
          };
          callsById.set(id, call);
        }
        call.attempts.push({ request: ev, responses: [], errors: [] });
        callOfEvent.set(ev.event_id, call);
        break;
      }
      default:
        break;
    }
  }

  const findAttempt = (call: LogicalCall, ev: TraceEvent, httpAttempt: number | null): Attempt | null => {
    if (ev.parent_event_id) {
      const a = call.attempts.find((x) => x.request.event_id === ev.parent_event_id);
      if (a) return a;
    }
    if (httpAttempt !== null) {
      const a = call.attempts.find((x) => {
        const rd = dataOf(x.request, 'model_request');
        return (rd.http_attempt ?? numOrNull(x.request.refs?.http_attempt)) === httpAttempt;
      });
      if (a) return a;
    }
    return null;
  };

  const resolveAnyNode = (id: string | null | undefined): TNode | null => {
    if (!id) return null;
    return nodeOfEvent.get(id) ?? apiById.get(id) ?? blockById.get(id) ?? serviceById.get(id) ?? null;
  };

  // Pass 2: everything else.
  for (const ev of events) {
    const refs = ev.refs ?? {};
    switch (ev.event_type) {
      case 'code_block_start':
      case 'api_call_start':
      case 'service_call_start':
      case 'model_request':
        break;
      case 'code_block_end': {
        const d = dataOf(ev, 'code_block_end');
        const id = d.code_block_id ?? strOrNull(refs.code_block_id) ?? ev.parent_event_id ?? null;
        const n = id ? blockById.get(id) : undefined;
        if (n && !n.end) {
          n.end = ev;
          addEventToNode(n, ev);
        } else if (n) {
          addEventToNode(n, ev);
        } else {
          orphanNode(ev, 'code_block_id に対応する code_block_start がありません');
        }
        break;
      }
      case 'api_call_end':
      case 'api_call_error': {
        const d = dataOf(ev, 'api_call_end');
        const id = d.api_call_id ?? strOrNull(refs.api_call_id) ?? null;
        const n = id ? apiById.get(id) : undefined;
        if (n && !n.end) {
          n.end = ev;
          addEventToNode(n, ev);
        } else if (n) {
          addEventToNode(n, ev);
        } else {
          orphanNode(ev, 'api_call_id に対応する api_call_start がありません');
        }
        break;
      }
      case 'service_call_end':
      case 'service_call_error': {
        const d = dataOf(ev, 'service_call_end');
        const candidates = [d.service_call_id, strOrNull(refs.for_event_id), ev.parent_event_id];
        let n: TNode | undefined;
        for (const c of candidates) {
          if (c && serviceById.has(c)) {
            n = serviceById.get(c);
            break;
          }
        }
        if (n && !n.end) {
          n.end = ev;
          addEventToNode(n, ev);
        } else if (n) {
          addEventToNode(n, ev);
        } else {
          // No id pairs this end with a start: shown on its own under its parent (never paired by time).
          const orphan = addNode(mkNode('service_call', ev));
          orphan.end = ev;
          pendingParent.push({ node: orphan, parentId: ev.parent_event_id ?? null, fallbackBlockId: null, expect: 'api' });
          orphan.orphan = '対応する service_call_start を id で特定できません';
        }
        break;
      }
      case 'model_response':
      case 'model_error': {
        const d = dataOf(ev, 'model_response');
        const ed = dataOf(ev, 'model_error');
        const lid = d.logical_call_id ?? strOrNull(refs.logical_call_id);
        const httpAttempt = d.http_attempt ?? numOrNull(refs.http_attempt);
        let call: LogicalCall | undefined = lid ? callsById.get(lid) : undefined;
        if (!call && ev.parent_event_id) call = callOfEvent.get(ev.parent_event_id);
        if (!call) {
          if (ev.event_type === 'model_error' && lid) {
            // An error of a call whose requests were never recorded.
            const logical = ed.scope === 'logical';
            call = {
              id: lid,
              synthetic: false,
              role: ed.role ?? 'unknown',
              callKind: ed.call_kind ?? '',
              turnIndex: numOrNull(refs.turn_index),
              model: null,
              attempts: [],
              logicalErrors: logical ? [ev] : [],
              unmatched: [],
              strayErrors: logical ? [] : [ev],
              parsed: [],
              producedBlocks: [],
              node: null as unknown as TNode,
            };
            callsById.set(lid, call);
            callOfEvent.set(ev.event_id, call);
          } else {
            orphanNode(ev, '対応する model_request がありません');
          }
          break;
        }
        callOfEvent.set(ev.event_id, call);
        if (ev.event_type === 'model_error' && ed.scope === 'logical') {
          call.logicalErrors.push(ev);
          break;
        }
        const attempt = findAttempt(call, ev, httpAttempt);
        if (attempt) {
          if (ev.event_type === 'model_response') attempt.responses.push(ev);
          else attempt.errors.push(ev);
        } else if (ev.event_type === 'model_error') {
          // An attempt error whose request is not in the record; it is not the call's final result.
          call.strayErrors.push(ev);
        } else {
          call.unmatched.push(ev);
        }
        break;
      }
      case 'model_output_parsed': {
        const d = dataOf(ev, 'model_output_parsed');
        const mid = d.model_event_id ?? strOrNull(refs.model_event_id);
        // A parse of a call that got no response (Molmo after a refused connection) names only
        // its logical call; it belongs to that call, not to "no parent".
        const lid = d.logical_call_id ?? strOrNull(refs.logical_call_id);
        const call = (mid ? callOfEvent.get(mid) : undefined) ?? (lid ? callsById.get(lid) : undefined);
        if (call) {
          call.parsed.push(ev);
          callOfEvent.set(ev.event_id, call);
        } else {
          orphanNode(ev, 'model_event_id に対応するモデル呼び出しがありません');
        }
        break;
      }
      case 'stdout':
      case 'stderr': {
        const d = dataOf(ev, 'stdout');
        const n = addNode(mkNode(ev.event_type, ev));
        pendingParent.push({
          node: n,
          parentId: d.code_block_id ?? strOrNull(refs.code_block_id) ?? ev.parent_event_id ?? null,
          fallbackBlockId: null,
          expect: 'block',
        });
        break;
      }
      case 'evaluator_result': {
        evaluators.push(ev);
        const d = dataOf(ev, 'evaluator_result');
        const n = addNode(mkNode('evaluator', ev));
        const blockId = d.code_block_id ?? strOrNull(refs.code_block_id);
        if (d.scope === 'code_block' || blockId) {
          pendingParent.push({ node: n, parentId: blockId ?? ev.parent_event_id ?? null, fallbackBlockId: null, expect: 'block' });
        } else {
          roots.push(n);
        }
        break;
      }
      case 'feedback_built': {
        feedbacks.push(ev);
        roots.push(addNode(mkNode('feedback', ev)));
        break;
      }
      case 'episode_start': {
        if (!episodeStart) episodeStart = ev;
        roots.push(addNode(mkNode('episode_start', ev)));
        break;
      }
      case 'episode_end': {
        if (!episodeEnd) episodeEnd = ev;
        roots.push(addNode(mkNode('episode_end', ev)));
        break;
      }
      case 'task_text': {
        if (!taskText) taskText = ev;
        break;
      }
      case 'env_reset': {
        roots.push(addNode(mkNode('env_reset', ev)));
        break;
      }
      case 'trace_warning': {
        warnings.push(ev);
        roots.push(addNode(mkNode('warning', ev)));
        break;
      }
      case 'frame': {
        const d = dataOf(ev, 'frame');
        const camera = d.camera ?? ev.artifacts?.find((a) => a.camera)?.camera ?? 'camera';
        const epoch = d.epoch ?? ev.sim_epoch ?? 0;
        const bufferIndex = d.buffer_index ?? -1;
        const art = (ev.artifacts ?? []).find((a) => isImageArtifact(a)) ?? (ev.artifacts ?? [])[0];
        const written = art ? art.written !== false : false;
        frames.push({
          event: ev,
          camera,
          epoch,
          bufferIndex,
          simStep: d.sim_step ?? ev.sim_step ?? null,
          reason: d.reason ?? 'periodic',
          forEventId: d.for_event_id ?? strOrNull(refs.for_event_id),
          path: art?.path ?? null,
          written: written && !!art?.path,
          writeReason: art?.reason ?? (art ? null : 'artifact なし'),
        });
        break;
      }
      case 'observation_snapshot':
        snapshots.push(ev);
        break;
      case 'video_frames_read':
        videoReads.push(ev);
        break;
      default:
        roots.push(addNode(mkNode('other', ev)));
        break;
    }
  }

  // Pass 3: call nodes.
  const calls = Array.from(callsById.values());
  for (const call of calls) {
    call.attempts.sort((a, b) => a.request.seq - b.request.seq);
    const anchor = call.attempts[0]?.request ?? call.logicalErrors[0] ?? call.strayErrors[0];
    const node = mkNode('model_call', anchor, `call:${call.id}`);
    node.events = [];
    for (const a of call.attempts) {
      node.events.push(a.request, ...a.responses, ...a.errors);
    }
    node.events.push(...call.logicalErrors, ...call.strayErrors, ...call.unmatched, ...call.parsed);
    node.events.sort((a, b) => a.seq - b.seq);
    node.call = call;
    call.node = node;
    addNode(node);
    if (!call.role || call.role === 'unknown') {
      const r = call.attempts.map((a) => dataOf(a.request, 'model_request').role).find((x) => !!x);
      if (r) call.role = r;
    }
  }
  calls.sort((a, b) => a.node.seq - b.node.seq);

  // Pass 4: parents.
  for (const n of allNodes) {
    if (n.kind === 'api_call') {
      const d = dataOf(n.primary, 'api_call_start');
      pendingParent.push({
        node: n,
        parentId: n.primary.parent_event_id ?? null,
        fallbackBlockId: d.call_site?.code_block_id ?? strOrNull(n.primary.refs?.code_block_id),
        expect: 'block-or-api',
      });
    } else if (n.kind === 'service_call' && n.primary.event_type === 'service_call_start') {
      pendingParent.push({ node: n, parentId: n.primary.parent_event_id ?? null, fallbackBlockId: null, expect: 'api' });
    } else if (n.kind === 'model_call') {
      const pid = n.primary.parent_event_id ?? null;
      if (pid) pendingParent.push({ node: n, parentId: pid, fallbackBlockId: null, expect: 'api-optional' });
      else roots.push(n);
    } else if (n.kind === 'code_block') {
      roots.push(n);
    }
  }
  const allowedParents: Record<string, NodeKind[]> = {
    block: ['code_block'],
    'block-or-api': ['code_block', 'api_call'],
    api: ['api_call'],
    'api-optional': ['api_call', 'service_call', 'code_block'],
  };
  for (const p of pendingParent) {
    const allowed = allowedParents[p.expect] ?? [];
    let parent = resolveAnyNode(p.parentId);
    const resolvedSomething = parent !== null;
    if (parent && (parent === p.node || !allowed.includes(parent.kind))) parent = null;
    if (!parent && p.fallbackBlockId) parent = blockById.get(p.fallbackBlockId) ?? null;
    if (parent && attach(parent, p.node)) continue;
    if (p.expect === 'api-optional' && (resolvedSomething || !p.parentId)) {
      // A model call whose parent is not an API call stays at the top level.
      roots.push(p.node);
      continue;
    }
    if (!p.node.orphan) {
      p.node.orphan = !p.parentId
        ? '親イベントの id が記録されていません'
        : resolvedSomething
          ? `親イベント ${p.parentId} は想定外の種類です`
          : `親イベント ${p.parentId} が記録にありません`;
    }
    roots.push(p.node);
  }

  // Pass 5: code block -> producing call.
  const blocks = allNodes.filter((n) => n.kind === 'code_block');
  for (const b of blocks) {
    const d = dataOf(b.primary, 'code_block_start');
    const sid = d.source_model_event_id;
    if (!sid) continue;
    const call = callOfEvent.get(sid);
    if (call) {
      b.sourceCall = call;
      call.producedBlocks.push(b);
    }
  }
  blocks.sort((a, b) => a.seq - b.seq);

  // Warnings attach to the node they name.
  for (const w of warnings) {
    const rid = dataOf(w, 'trace_warning').related_event_id;
    const target = rid ? resolveAnyNode(rid) : null;
    if (target && target.kind !== 'warning') target.warnings.push(w);
  }

  // Pass 6: flags, order, search text.
  for (const n of allNodes) {
    n.events.sort((a, b) => a.seq - b.seq);
    n.seq = Math.min(...n.events.map((e) => e.seq));
    n.lastSeq = Math.max(...n.events.map((e) => e.seq));
    n.late = n.events.some((e) => e.late === true);
    n.degraded = n.events.some((e) => e.status === 'degraded');
    computeFlags(n);
    n.searchText = buildSearchText(n);
  }
  const sortRec = (list: TNode[]) => {
    list.sort((a, b) => a.seq - b.seq);
    for (const c of list) sortRec(c.children);
  };
  sortRec(roots);

  // Frames.
  frames.sort((a, b) => a.event.seq - b.event.seq);
  const framesByCamera = new Map<string, FrameRec[]>();
  const frameByKey = new Map<string, FrameRec>();
  const keyframesFor = new Map<string, FrameRec[]>();
  for (const f of frames) {
    const list = framesByCamera.get(f.camera) ?? [];
    list.push(f);
    framesByCamera.set(f.camera, list);
    frameByKey.set(frameKey(f.camera, f.epoch, f.bufferIndex), f);
    if (f.forEventId) {
      const k = keyframesFor.get(f.forEventId) ?? [];
      k.push(f);
      keyframesFor.set(f.forEventId, k);
    }
  }
  const snapshotByPixelSha = new Map<string, TraceEvent>();
  for (const s of snapshots) {
    const sha = dataOf(s, 'observation_snapshot').pixel_sha256;
    if (sha && !snapshotByPixelSha.has(sha)) snapshotByPixelSha.set(sha, s);
  }

  const rolesSeen = new Set<string>();
  for (const c of calls) if (c.attempts.length > 0) rolesSeen.add(c.role);

  return {
    events,
    byId,
    roots,
    allNodes,
    nodeOfEvent,
    calls,
    callOfEvent,
    blocks,
    blockById,
    apiById,
    frames,
    framesByCamera,
    frameByKey,
    keyframesFor,
    snapshots,
    snapshotByPixelSha,
    videoReads,
    episodeStart,
    episodeEnd,
    taskText,
    warnings,
    evaluators,
    feedbacks,
    rolesSeen,
    lastSeq: events.length ? events[events.length - 1].seq : -1,
    countsByType,
  };
}

function computeFlags(n: TNode): void {
  const anyError = n.events.some((e) => e.status === 'error');
  switch (n.kind) {
    case 'code_block': {
      n.open = n.end === null;
      const rc = n.end ? dataOf(n.end, 'code_block_end').sandbox_rc : null;
      n.failed = anyError || (typeof rc === 'number' && rc !== 0);
      break;
    }
    case 'api_call':
      n.open = n.end === null;
      n.failed = anyError || n.end?.event_type === 'api_call_error';
      break;
    case 'service_call':
      n.open = n.end === null;
      n.failed = anyError || n.end?.event_type === 'service_call_error';
      break;
    case 'model_call': {
      const call = n.call;
      if (!call) break;
      const state = callOutcome(call).state;
      n.open = state === 'open';
      n.failed = state === 'failed';
      break;
    }
    case 'episode_end': {
      const reason = dataOf(n.primary, 'episode_end').end_reason;
      n.failed = anyError || reason === 'exception' || reason === 'timeout_alarm' || reason === 'error';
      break;
    }
    default:
      n.failed = anyError;
      break;
  }
}

export function isOkResponse(ev: TraceEvent): boolean {
  if (ev.event_type !== 'model_response') return false;
  const st = dataOf(ev, 'model_response').http_status;
  return ev.status !== 'error' && (typeof st !== 'number' || (st >= 200 && st < 300));
}

export type CallState = 'ok' | 'failed' | 'open';

export interface CallOutcome {
  /**
   * `failed` when a logical error was recorded (it wins over an HTTP 2xx answer), else `ok`
   * when an attempt was answered, else `open` when the last attempt has no result yet, else
   * `failed` (every attempt ended with an error).
   */
  state: CallState;
  /** The final `model_error` with scope `logical`, when the call failed. */
  finalError: TraceEvent | null;
  /** The first HTTP 2xx `model_response` of any attempt (also when the call failed afterwards). */
  httpAnswer: TraceEvent | null;
  /**
   * The final error is a GeneratorExit: the caller stopped reading the stream (for example
   * the web UI's stop button). The call has no result, so `state` is still `failed`, but the
   * panes name it a stop, not a failure of the model or the endpoint.
   */
  stopped: boolean;
}

export function callOutcome(call: LogicalCall): CallOutcome {
  const finalError = call.logicalErrors.length ? call.logicalErrors[call.logicalErrors.length - 1] : null;
  const httpAnswer = httpOkResponse(call);
  if (finalError) {
    const stopped = dataOf(finalError, 'model_error').error_type === 'GeneratorExit';
    return { state: 'failed', finalError, httpAnswer, stopped };
  }
  if (httpAnswer) return { state: 'ok', finalError: null, httpAnswer, stopped: false };
  const last = call.attempts[call.attempts.length - 1];
  if (last && last.responses.length === 0 && last.errors.length === 0) return { state: 'open', finalError: null, httpAnswer: null, stopped: false };
  return { state: 'failed', finalError: null, httpAnswer: null, stopped: false };
}

/**
 * Logical errors that report the outcome of this attempt: the ones whose parent is the
 * attempt's request or that name its http_attempt; one naming neither belongs to the last
 * attempt (the recorder writes the logical error after the last attempt).
 */
export function logicalErrorsOfAttempt(call: LogicalCall, attempt: Attempt): TraceEvent[] {
  const last = call.attempts[call.attempts.length - 1] ?? null;
  const no = attemptNumber(call, attempt);
  return call.logicalErrors.filter((e) => {
    if (e.parent_event_id) return e.parent_event_id === attempt.request.event_id;
    const ha = dataOf(e, 'model_error').http_attempt ?? numOrNull(e.refs?.http_attempt);
    if (ha !== null && ha !== undefined) return ha === no;
    return attempt === last;
  });
}

/** 1-based HTTP attempt number (the recorded one, else the position in the call). */
export function attemptNumber(call: LogicalCall, attempt: Attempt): number {
  const d = dataOf(attempt.request, 'model_request');
  return d.http_attempt ?? numOrNull(attempt.request.refs?.http_attempt) ?? call.attempts.indexOf(attempt) + 1;
}

export interface ImageCounts {
  /** Recorded number of images (null: not recorded, for example the body was not parsed). */
  images: number | null;
  videos: number | null;
  /**
   * true only when the recorder measured zero images and zero videos; null when the
   * counts were not recorded (never guessed from a missing `images` list).
   */
  noImage: boolean | null;
}

/** Image and video counts of a model request, from recorded fields only (6.4). */
export function requestImageCounts(d: ModelRequestData | null | undefined): ImageCounts {
  if (!d) return { images: null, videos: null, noImage: null };
  const list = Array.isArray(d.images) ? d.images : null;
  const isVideo = (im: ImageInfo) => im.kind === 'video' || (!im.kind && (im.mime ?? '').startsWith('video/'));
  const isImage = (im: ImageInfo) => (im.kind ? im.kind === 'image' || im.kind === 'external_url' : !isVideo(im));
  const images = numOrNull(d.n_images) ?? numOrNull(d.input_summary?.n_images) ?? (list ? list.filter(isImage).length : null);
  const videos = numOrNull(d.n_videos) ?? numOrNull(d.input_summary?.n_videos) ?? (list ? list.filter(isVideo).length : null);
  const flag = d.input_summary?.no_image_input;
  const noImage = typeof flag === 'boolean' ? flag : images !== null && videos !== null ? images === 0 && videos === 0 : null;
  return { images, videos, noImage };
}

function buildSearchText(n: TNode): string {
  const parts: string[] = [];
  for (const ev of n.events) {
    switch (ev.event_type) {
      case 'model_request': {
        const d = dataOf(ev, 'model_request');
        parts.push(lower(d.role), lower(d.call_kind), lower(d.model));
        break;
      }
      case 'model_response': {
        const d = dataOf(ev, 'model_response');
        parts.push(lower(resolveText(d.content).text), lower(resolveText(d.reasoning?.text).text));
        break;
      }
      case 'model_error': {
        const d = dataOf(ev, 'model_error');
        parts.push(lower(d.error_type), lower(textOf(d.message)), lower(textOf(d.partial_content)));
        break;
      }
      case 'code_block_start':
        parts.push(lower(resolveText(dataOf(ev, 'code_block_start').code).text));
        break;
      case 'api_call_start': {
        const d = dataOf(ev, 'api_call_start');
        parts.push(lower(d.name), lower(d.qualname), lower(d.api_class));
        break;
      }
      case 'api_call_error': {
        const d = dataOf(ev, 'api_call_error');
        parts.push(lower(d.error_type), lower(textOf(d.message)));
        break;
      }
      case 'service_call_start':
      case 'service_call_end':
      case 'service_call_error': {
        const d = dataOf(ev, 'service_call_start');
        parts.push(lower(d.service), lower(d.function), lower(textOf(d.message)));
        break;
      }
      case 'stdout':
      case 'stderr':
        parts.push(lower(resolveText(dataOf(ev, 'stdout').text).text));
        break;
      case 'feedback_built':
        parts.push(lower(resolveText(dataOf(ev, 'feedback_built').user_feedback).text));
        break;
      case 'trace_warning': {
        const d = dataOf(ev, 'trace_warning');
        parts.push(lower(d.code), lower(textOf(d.message)));
        break;
      }
      default:
        break;
    }
  }
  parts.push(lower(n.primary.event_type));
  return parts.filter((p) => p.length > 0).join('\n');
}

// ---------------------------------------------------------------------------
// Queries used by the panes
// ---------------------------------------------------------------------------

export function apiDepth(n: TNode): number {
  if (n.kind !== 'api_call') return 0;
  const d = dataOf(n.primary, 'api_call_start').depth;
  if (typeof d === 'number') return d;
  let depth = 0;
  let p = n.parent;
  while (p && p.kind === 'api_call') {
    depth += 1;
    p = p.parent;
  }
  return depth;
}

/** The code block a node belongs to (itself for a block). */
export function enclosingBlock(model: EpisodeModel, n: TNode): TNode | null {
  let cur: TNode | null = n;
  while (cur) {
    if (cur.kind === 'code_block') return cur;
    cur = cur.parent;
  }
  if (n.kind === 'api_call') {
    const id = dataOf(n.primary, 'api_call_start').call_site?.code_block_id;
    if (id) return model.blockById.get(id) ?? null;
  }
  if (n.kind === 'stdout' || n.kind === 'stderr') {
    const id = dataOf(n.primary, 'stdout').code_block_id;
    if (id) return model.blockById.get(id) ?? null;
  }
  return null;
}

/** Model calls nested anywhere below a node (perception VLM calls under an API call). */
export function nestedModelCalls(n: TNode): LogicalCall[] {
  const out: LogicalCall[] = [];
  const walk = (x: TNode) => {
    for (const c of x.children) {
      if (c.kind === 'model_call' && c.call) out.push(c.call);
      walk(c);
    }
  };
  walk(n);
  return out;
}

/** Code-generator and VDM requests whose seq lies inside the node's span. */
export function codegenOrVdmDuring(model: EpisodeModel, n: TNode): LogicalCall[] {
  const out: LogicalCall[] = [];
  for (const c of model.calls) {
    if (c.role !== 'code_generator' && c.role !== 'vdm') continue;
    if (c.attempts.some((a) => a.request.seq > n.seq && a.request.seq < n.lastSeq)) out.push(c);
  }
  return out;
}

/** First logical call of a role whose first request comes after `seq`. */
export function nextCallAfter(model: EpisodeModel, seq: number, role: string): LogicalCall | null {
  for (const c of model.calls) {
    const first = c.attempts[0]?.request;
    if (first && first.seq > seq && c.role === role) return c;
  }
  return null;
}

/** Call-kind family used to pick the previous request for the input diff (9.3 tab 5). */
export function callFamily(call: LogicalCall): string {
  if (call.role === 'vdm') return 'vdm';
  if (call.role === 'code_generator') {
    if (call.callKind === 'ensemble_member') return 'code_generator:ensemble_member';
    if (call.callKind === 'ensemble_synthesis') return 'code_generator:ensemble_synthesis';
    return 'code_generator';
  }
  return `${call.role}:${call.callKind}`;
}

export function previousSameFamily(model: EpisodeModel, call: LogicalCall): LogicalCall | null {
  const fam = callFamily(call);
  let prev: LogicalCall | null = null;
  for (const c of model.calls) {
    if (c === call) break;
    if (c.attempts.length > 0 && callFamily(c) === fam) prev = c;
  }
  return prev;
}

/** The attempt the viewer shows by default: the one that got an ok response, else the last. */
export function mainAttempt(call: LogicalCall): Attempt | null {
  const ok = call.attempts.find((a) => a.responses.some((r) => isOkResponse(r)));
  return ok ?? call.attempts[call.attempts.length - 1] ?? null;
}

/** The first HTTP 2xx response of any attempt, whatever the call's final outcome. */
export function httpOkResponse(call: LogicalCall): TraceEvent | null {
  for (const a of call.attempts) {
    const r = a.responses.find((x) => isOkResponse(x));
    if (r) return r;
  }
  return null;
}

/**
 * The answer the harness received: the HTTP 2xx response of a call that did not fail.
 * null when the call failed logically, even after an HTTP 200 (callOutcome).
 */
export function okResponse(call: LogicalCall): TraceEvent | null {
  return callOutcome(call).state === 'ok' ? httpOkResponse(call) : null;
}

export function lastParsed(call: LogicalCall): TraceEvent | null {
  return call.parsed.length ? call.parsed[call.parsed.length - 1] : null;
}

/** Newest node for Live mode; nested API calls resolve to their depth-0 caller. */
export function liveTarget(model: EpisodeModel): { node: TNode; eventId: string } | null {
  for (let i = model.events.length - 1; i >= 0; i -= 1) {
    const ev = model.events[i];
    let node = model.nodeOfEvent.get(ev.event_id);
    if (!node) continue;
    while (node.kind === 'api_call' && apiDepth(node) > 0 && node.parent && node.parent.kind === 'api_call') {
      node = node.parent;
    }
    return { node, eventId: selectionEventId(node) };
  }
  return null;
}

/** Event id written to the URL when a node is selected. */
export function selectionEventId(node: TNode): string {
  if (node.kind === 'model_call' && node.call) {
    const a = mainAttempt(node.call);
    if (a) return a.request.event_id;
  }
  return node.primary.event_id;
}

export interface Selection {
  node: TNode;
  eventId: string;
  /** For model calls: the attempt named by the selected event. */
  attempt: Attempt | null;
}

export function resolveSelection(model: EpisodeModel, eventId: string | null): Selection | null {
  if (!eventId) return null;
  const node = model.nodeOfEvent.get(eventId);
  if (!node) return null;
  let attempt: Attempt | null = null;
  if (node.call) {
    attempt =
      node.call.attempts.find(
        (a) =>
          a.request.event_id === eventId ||
          a.responses.some((r) => r.event_id === eventId) ||
          a.errors.some((r) => r.event_id === eventId),
      ) ?? mainAttempt(node.call);
  }
  return { node, eventId, attempt };
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

export interface KeyFrame {
  /** null when the ref names no camera (for example {written: false, reason: "no_buffer_frames"}). */
  camera: string | null;
  ref: FrameRef | null;
  frame: FrameRec | null;
  path: string | null;
  eventSimStep: number | null;
  frameSimStep: number | null;
  /** event sim_step minus frame sim_step; null when either is unknown. */
  offset: number | null;
  epoch: number | null;
  bufferIndex: number | null;
  /** Why there is no image (ref.reason or the frame artifact's reason). */
  reason: string | null;
}

function keyFramesOfEvent(model: EpisodeModel, ev: TraceEvent | null, refs: FrameRef[]): KeyFrame[] {
  if (!ev) return [];
  const out: KeyFrame[] = [];
  const seen = new Set<string>();
  const eventSimStep = ev.sim_step ?? null;
  for (const ref of refs) {
    const camera = ref.camera ?? null;
    const epoch = ref.epoch ?? null;
    const bi = ref.buffer_index ?? null;
    const frame = camera !== null && epoch !== null && bi !== null ? model.frameByKey.get(frameKey(camera, epoch, bi)) ?? null : null;
    const path = ref.path ?? frame?.path ?? (camera !== null && epoch !== null && bi !== null ? `frames/${camera}/e${epoch}_${bi}.jpg` : null);
    const frameSimStep = ref.frame_sim_step ?? frame?.simStep ?? null;
    const written = ref.written !== false && (frame ? frame.written : true);
    out.push({
      camera,
      ref,
      frame,
      path: written ? path : null,
      eventSimStep,
      frameSimStep,
      offset: eventSimStep !== null && frameSimStep !== null ? eventSimStep - frameSimStep : null,
      epoch,
      bufferIndex: bi,
      reason: ref.reason ?? frame?.writeReason ?? (path === null ? 'frame_ref に画像の位置がありません' : null),
    });
    if (camera !== null) seen.add(camera);
  }
  for (const f of model.keyframesFor.get(ev.event_id) ?? []) {
    if (seen.has(f.camera)) continue;
    seen.add(f.camera);
    out.push({
      camera: f.camera,
      ref: null,
      frame: f,
      path: f.written ? f.path : null,
      eventSimStep,
      frameSimStep: f.simStep,
      offset: eventSimStep !== null && f.simStep !== null ? eventSimStep - f.simStep : null,
      epoch: f.epoch,
      bufferIndex: f.bufferIndex,
      reason: f.written ? null : f.writeReason,
    });
  }
  return out;
}

/** Key frames at the start and end of a code block or API call (8.2). */
export function keyFramesOf(model: EpisodeModel, n: TNode): { before: KeyFrame[]; after: KeyFrame[] } {
  if (n.kind === 'code_block') {
    return {
      before: keyFramesOfEvent(model, n.primary, frameRefsOf(dataOf(n.primary, 'code_block_start').frame_ref)),
      after: keyFramesOfEvent(model, n.end, frameRefsOf(n.end ? dataOf(n.end, 'code_block_end').frame_ref : null)),
    };
  }
  if (n.kind === 'api_call') {
    return {
      before: keyFramesOfEvent(model, n.primary, frameRefsOf(dataOf(n.primary, 'api_call_start').frame_ref)),
      after: keyFramesOfEvent(model, n.end, frameRefsOf(n.end ? dataOf(n.end, 'api_call_end').frame_ref : null)),
    };
  }
  return { before: [], after: [] };
}

/**
 * Recorded frames of one camera around a node, for the flipbook: the frames
 * captured while the node ran (by seq), plus a few before and after. For an
 * open node (no end event yet) every frame since its start is included, the
 * newest last. At most `max` frames are returned (the newest are kept).
 */
export function framesAround(model: EpisodeModel, n: TNode, camera: string, pad = 8, max = 400): { frames: FrameRec[]; start: number } {
  const list = model.framesByCamera.get(camera) ?? [];
  if (list.length === 0) return { frames: [], start: 0 };
  let lo = list.findIndex((f) => f.event.seq >= n.seq);
  if (lo < 0) lo = list.length;
  let hi = lo;
  while (hi < list.length && list[hi].event.seq <= n.lastSeq) hi += 1;
  let from = Math.max(0, lo - pad);
  const to = n.open ? list.length : Math.min(list.length, hi + pad);
  if (to - from > max) from = to - max;
  const frames = list.slice(from, to);
  return { frames, start: Math.min(Math.max(0, lo - from), Math.max(0, frames.length - 1)) };
}

export function camerasOf(model: EpisodeModel): string[] {
  return Array.from(model.framesByCamera.keys()).sort();
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

export function nodeTitle(n: TNode): string {
  switch (n.kind) {
    case 'model_call': {
      const c = n.call;
      return c ? `${c.role}${c.callKind ? ` · ${c.callKind}` : ''}` : 'model call';
    }
    case 'code_block': {
      const d = dataOf(n.primary, 'code_block_start');
      return `コードブロック #${d.block_index ?? '?'}`;
    }
    case 'api_call': {
      const d = dataOf(n.primary, 'api_call_start');
      return d.name ?? d.qualname ?? 'API';
    }
    case 'service_call': {
      const d = dataOf(n.primary, 'service_call_start');
      return `${d.service ?? 'service'}${d.function ? `.${d.function}` : ''}`;
    }
    case 'stdout':
      return 'stdout';
    case 'stderr':
      return 'stderr';
    case 'evaluator':
      return '評価器';
    case 'feedback':
      return '次の入力のフィードバック';
    case 'episode_start':
      return 'エピソード開始';
    case 'episode_end':
      return 'エピソード終了';
    case 'env_reset':
      return '環境リセット';
    case 'warning':
      return `記録の警告: ${dataOf(n.primary, 'trace_warning').code ?? ''}`;
    default:
      return n.primary.event_type;
  }
}

/** Duration in seconds for cards and details, when recorded. */
export function nodeDuration(n: TNode): number | null {
  switch (n.kind) {
    case 'code_block':
      return n.end ? numOrNull(dataOf(n.end, 'code_block_end').duration_s) : null;
    case 'api_call':
      return n.end ? numOrNull(dataOf(n.end, 'api_call_end').duration_s) : null;
    case 'service_call':
      return n.end ? numOrNull(dataOf(n.end, 'service_call_end').latency_s) : null;
    case 'model_call': {
      const r = n.call ? httpOkResponse(n.call) : null;
      return r ? numOrNull(dataOf(r, 'model_response').latency_s) : null;
    }
    default:
      return null;
  }
}

/**
 * Arrays whose values were not saved because they were over the array budget
 * (`array_budget_exceeded` on api_call_*, service_call_* and code_block_end, 5.3).
 */
export function arrayBudgetOf(n: TNode): number {
  let total = 0;
  for (const ev of n.events) {
    const d = ev.data;
    if (d && typeof d === 'object' && !Array.isArray(d)) {
      const v = (d as { array_budget_exceeded?: unknown }).array_budget_exceeded;
      if (typeof v === 'number' && v > 0) total += v;
    }
  }
  return total;
}

/** Service calls directly under an API call that failed or were degraded. */
export function troubledServices(n: TNode): TNode[] {
  return n.children.filter((c) => c.kind === 'service_call' && (c.failed || c.degraded || c.open));
}

/** Artifacts of a given label across the events of a node. */
export function artifactsOf(n: TNode, label: string): Artifact[] {
  const out: Artifact[] = [];
  for (const ev of n.events) for (const a of ev.artifacts ?? []) if (a.label === label) out.push(a);
  return out;
}

export function descendants(n: TNode): TNode[] {
  const out: TNode[] = [];
  const walk = (x: TNode) => {
    for (const c of x.children) {
      out.push(c);
      walk(c);
    }
  };
  walk(n);
  return out;
}

/**
 * Where a model-input image came from (6.2). The recorder's `source` is used
 * as is; `{kind: "unknown"}` stays 由来不明. Only when no `source` was
 * recorded at all does the viewer look for an observation_snapshot taken
 * before the request with the same pixel_sha256, and says that it did so.
 * An ambiguous source (several snapshots with the same pixels) carries the
 * note of `ambiguousSource` unless `withAmbiguity` is false (callers that show
 * the note on its own line).
 */
export function imageSourceText(model: EpisodeModel, im: ImageInfo, requestSeq: number | null, withAmbiguity = true): string {
  const src = im.source;
  if (src && src.kind && src.kind !== 'unknown') {
    const parts = [src.kind];
    const ev = src.event_id ? model.byId.get(src.event_id) : undefined;
    if (ev) parts.push(`#${ev.seq}`);
    if (src.camera) parts.push(src.camera);
    if (src.sim_step !== undefined && src.sim_step !== null) parts.push(`sim_step ${src.sim_step}`);
    if (Array.isArray(src.frame_range)) parts.push(`frames ${src.frame_range.join('–')}`);
    const amb = withAmbiguity ? ambiguousSource(model, im) : null;
    return amb ? `${parts.join(' / ')}（${amb}）` : parts.join(' / ');
  }
  if (src) {
    const why = typeof src.reason === 'string' && src.reason ? `: ${src.reason}` : '';
    return `${LABELS.unattributed}（完全一致する記録フレームなし${why}）`;
  }
  if (im.pixel_sha256) {
    const snap = model.snapshotByPixelSha.get(im.pixel_sha256);
    if (snap && (requestSeq === null || snap.seq < requestSeq)) {
      const camera = dataOf(snap, 'observation_snapshot').camera ?? '?';
      return `observation_snapshot / ${camera} / sim_step ${snap.sim_step ?? '?'}（source の記録なし。ビューワーで pixel_sha256 が一致）`;
    }
  }
  return `${LABELS.unattributed}（source の記録なし）`;
}

/**
 * The note for an image whose pixels match several snapshots (6.2: the recorder sets
 * `source.ambiguous` and lists `candidates`), or null. An unchanged scene rendered twice
 * gives identical pixels, so which render was sent cannot be told from the pixels.
 */
export function ambiguousSource(model: EpisodeModel, im: ImageInfo): string | null {
  const src = im.source;
  if (!src || src.ambiguous !== true) return null;
  const cands = Array.isArray(src.candidates) ? src.candidates : [];
  const seqs = cands
    .map((c) => (c && c.event_id ? model.byId.get(c.event_id)?.seq : undefined))
    .filter((s): s is number => typeof s === 'number');
  const list = seqs.length ? `: ${seqs.map((s) => `#${s}`).join(', ')}` : '';
  return `同一画素のスナップショットが ${cands.length || '複数'} 件${list}。どれを送ったかは画素からは区別できません`;
}
