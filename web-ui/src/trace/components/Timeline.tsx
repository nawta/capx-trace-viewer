/**
 * Left pane of the trace viewer (docs/trace-format.md 9.3): the task text
 * first, then the event timeline grouped by cause (model call -> parsed code
 * -> code block -> API calls -> stdout/stderr -> evaluator -> VDM call ->
 * feedback -> next model call), with the card contents of 9.3, filters by
 * event type, role and failures, and a free-text search over code,
 * stdout/stderr, model text and API names. Nested API calls are folded by
 * default; frames and snapshots are never cards. Open start events and late
 * events carry the fixed markers of 9.3. A model call whose logical call failed
 * (also after an HTTP 200) is shown as failed with that final error. Each card
 * is wrapped in an error boundary, so one event with an unexpected shape
 * cannot blank the page.
 *
 * Related: model.ts (tree), VirtualList.tsx (windowing), TraceViewerPage.tsx.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { episodeArrayImageUrl, episodeFileUrl } from '../api';
import { dataAssetRef, isOversize } from '../dataAsset';
import { firstLines, fmtDuration, fmtNum, truncate } from '../format';
import { useFullText } from '../hooks';
import { LABELS, NOT_MODEL_INPUT, REDACTED_SHORT, WARNING_CODE_LABELS, arrayBudgetLabel, harnessReadingLabel, modelErrorLabel } from '../labels';
import {
  apiDepth,
  arrayBudgetOf,
  callOutcome,
  keyFramesOf,
  lastParsed,
  mainAttempt,
  nodeDuration,
  nodeTitle,
  requestImageCounts,
  resolveText,
  textOf,
  troubledServices,
  type EpisodeModel,
  type LogicalCall,
  type TNode,
} from '../model';
import { dataOf, type ModelOutputParsedData } from '../types';
import { CodeView } from './CodeView';
import { Badge, EventLink, statusTone, type Tone } from './common';
import { ErrorBoundary } from './ErrorBoundary';
import { TraceImage } from './TraceImage';
import { argsSummary, valueSummary, type EpisodeRef } from './ValueView';
import { VirtualList } from './VirtualList';

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

type TypeKey =
  | 'model_call'
  | 'code_block'
  | 'api_call'
  | 'service_call'
  | 'output'
  | 'evaluator'
  | 'feedback'
  | 'episode'
  | 'warning'
  | 'other';

const TYPE_FILTERS: { key: TypeKey; label: string }[] = [
  { key: 'model_call', label: 'モデル呼び出し' },
  { key: 'code_block', label: 'コードブロック' },
  { key: 'api_call', label: 'API呼び出し' },
  { key: 'service_call', label: 'サービス呼び出し' },
  { key: 'output', label: 'stdout/stderr' },
  { key: 'evaluator', label: '評価器' },
  { key: 'feedback', label: 'フィードバック' },
  { key: 'episode', label: 'エピソード' },
  { key: 'warning', label: '記録の警告' },
  { key: 'other', label: 'その他' },
];

function typeKeyOf(n: TNode): TypeKey {
  switch (n.kind) {
    case 'stdout':
    case 'stderr':
      return 'output';
    case 'episode_start':
    case 'episode_end':
    case 'env_reset':
      return 'episode';
    default:
      return n.kind;
  }
}

interface Filters {
  types: Set<TypeKey>;
  roles: Set<string>;
  failuresOnly: boolean;
  query: string;
}

interface Row {
  node: TNode;
  depth: number;
  /** Shown only because a descendant matches the filters. */
  context: boolean;
}

function isFailure(n: TNode, running: boolean): boolean {
  return n.failed || n.degraded || (n.open && !running);
}

function flatten(
  roots: TNode[],
  f: Filters,
  collapsed: Set<string>,
  nestedOpen: Set<string>,
  running: boolean,
): { rows: Row[]; matched: number } {
  const q = f.query.trim().toLowerCase();
  const active = f.types.size > 0 || f.roles.size > 0 || f.failuresOnly || q.length > 0;
  let matched = 0;
  const selfMemo = new Map<TNode, boolean>();
  const selfMatch = (n: TNode): boolean => {
    let v = selfMemo.get(n);
    if (v !== undefined) return v;
    v = true;
    if (f.types.size > 0 && !f.types.has(typeKeyOf(n))) v = false;
    // A role filter selects model calls of those roles; their callers stay visible as context rows.
    if (v && f.roles.size > 0 && !(n.kind === 'model_call' && n.call && f.roles.has(n.call.role))) v = false;
    if (v && f.failuresOnly && !isFailure(n, running)) v = false;
    if (v && q && !n.searchText.includes(q)) v = false;
    selfMemo.set(n, v);
    return v;
  };
  const visMemo = new Map<TNode, boolean>();
  const vis = (n: TNode): boolean => {
    let v = visMemo.get(n);
    if (v !== undefined) return v;
    v = !active || selfMatch(n) || n.children.some(vis);
    visMemo.set(n, v);
    return v;
  };
  const rows: Row[] = [];
  const walk = (list: TNode[], depth: number) => {
    for (const n of list) {
      if (!vis(n)) continue;
      const self = !active || selfMatch(n);
      if (active && self) matched += 1;
      rows.push({ node: n, depth, context: active && !self });
      if (n.children.length === 0) continue;
      if (!active && collapsed.has(n.key)) continue;
      let kids = n.children;
      if (n.kind === 'api_call' && !nestedOpen.has(n.key)) {
        kids = kids.filter((c) => c.kind !== 'api_call' || (active && vis(c)));
      }
      walk(kids, depth + 1);
    }
  };
  walk(roots, 0);
  return { rows, matched: active ? matched : rows.length };
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

interface TimelineProps {
  model: EpisodeModel;
  ep: EpisodeRef;
  running: boolean;
  selectedKey: string | null;
  selectedEventId: string | null;
  onSelect: (node: TNode, eventId?: string) => void;
  scrollToken: number;
  /** False for a run's unscoped log, which is no episode and has no task text. */
  showTaskText?: boolean;
}

export function Timeline({ model, ep, running, selectedKey, selectedEventId, onSelect, scrollToken, showTaskText = true }: TimelineProps) {
  const [types, setTypes] = useState<Set<TypeKey>>(new Set());
  const [roles, setRoles] = useState<Set<string>>(new Set());
  const [failuresOnly, setFailuresOnly] = useState(false);
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [nestedOpen, setNestedOpen] = useState<Set<string>>(new Set());
  // Bumped when the filters are cleared, so the selected card is brought back into view.
  const [filterToken, setFilterToken] = useState(0);
  // Playback can select an API hidden inside a collapsed parent.
  useEffect(() => {
    const node = model.allNodes.find(n => n.key === selectedKey);
    if (!node) return;
    const parents: string[] = [];
    for (let p = node.parent; p; p = p.parent) parents.push(p.key);
    setCollapsed(old => old.size && parents.some(k => old.has(k)) ? new Set([...old].filter(k => !parents.includes(k))) : old);
    setNestedOpen(old => parents.some(k => !old.has(k)) ? new Set([...old, ...parents]) : old);
  }, [selectedKey, model]);

  const filters = useMemo<Filters>(() => ({ types, roles, failuresOnly, query }), [types, roles, failuresOnly, query]);
  const { rows, matched } = useMemo(
    () => flatten(model.roots, filters, collapsed, nestedOpen, running),
    [model, filters, collapsed, nestedOpen, running],
  );
  const roleList = useMemo(() => Array.from(new Set(model.calls.map((c) => c.role))).sort(), [model]);

  const toggle = (set: Set<string>, key: string, setter: (s: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setter(next);
  };

  const itemKey = useCallback((r: Row) => r.node.key, []);
  const estimate = useCallback((r: Row) => {
    switch (r.node.kind) {
      case 'code_block':
        return 210;
      case 'api_call':
        return 118;
      case 'model_call':
        return 96;
      case 'stdout':
      case 'stderr':
        return 78;
      default:
        return 60;
    }
  }, []);

  const filtersActive = types.size > 0 || roles.size > 0 || failuresOnly || query.trim().length > 0;
  const selectionHidden = selectedKey !== null && !rows.some((r) => r.node.key === selectedKey);
  const clearFilters = () => {
    setTypes(new Set());
    setRoles(new Set());
    setFailuresOnly(false);
    setQuery('');
    setFilterToken((t) => t + 1);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    e.preventDefault();
    const idx = rows.findIndex((r) => r.node.key === selectedKey);
    const next = e.key === 'ArrowDown' ? Math.min(rows.length - 1, idx + 1) : Math.max(0, idx - 1);
    if (rows[next]) onSelect(rows[next].node);
  };

  const renderItem = useCallback(
    (r: Row) => (
      <ErrorBoundary label={`カード #${r.node.seq}（${r.node.primary.event_type}）`} resetKey={`${r.node.lastSeq}|${r.node.events.length}`}>
        <NodeCard
          row={r}
          model={model}
          ep={ep}
          running={running}
          selected={r.node.key === selectedKey}
          selectedEventId={selectedEventId}
          collapsed={collapsed.has(r.node.key)}
          nestedOpen={nestedOpen.has(r.node.key)}
          onSelect={onSelect}
          onToggle={() => toggle(collapsed, r.node.key, setCollapsed)}
          onToggleNested={() => toggle(nestedOpen, r.node.key, setNestedOpen)}
        />
      </ErrorBoundary>
    ),
    // Every input of the card is listed, so a card re-renders when any of them changes.
    [model, ep, running, selectedKey, selectedEventId, collapsed, nestedOpen, onSelect],
  );

  return (
    <div className="flex flex-col h-full min-h-0" onKeyDown={onKeyDown}>
      {showTaskText && <TaskText model={model} ep={ep} />}
      <div className="flex-shrink-0 px-3 py-2 border-b border-surface-border space-y-2 bg-surface-raised">
        <input
          type="search"
          name="trace-timeline-search"
          aria-label="タイムラインを検索"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="検索: コード / stdout・stderr / モデル出力 / API名"
          title="記録にインラインで入っているテキストを検索します。4000文字を超えて asset に保存された全文は対象外です。"
          className="w-full px-2.5 py-1.5 bg-surface-sunken border border-surface-border rounded-md text-xs text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent/40"
        />
        <div className="flex flex-wrap gap-1">
          {TYPE_FILTERS.map((t) => (
            <FilterChip key={t.key} active={types.has(t.key)} onClick={() => toggle(types as Set<string>, t.key, (s) => setTypes(s as Set<TypeKey>))}>
              {t.label}
            </FilterChip>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[11px] text-text-tertiary mr-1">role</span>
          {roleList.length === 0 && <span className="text-[11px] text-text-muted">モデル呼び出しなし</span>}
          {roleList.map((r) => (
            <FilterChip key={r} active={roles.has(r)} onClick={() => toggle(roles, r, setRoles)}>
              {r}
            </FilterChip>
          ))}
          <span className="mx-1 h-4 w-px bg-surface-border" />
          <FilterChip active={failuresOnly} onClick={() => setFailuresOnly(!failuresOnly)} tone="red">
            失敗のみ
          </FilterChip>
          <span className="ml-auto text-[11px] text-text-tertiary">
            {matched} 件表示
          </span>
        </div>
        {filtersActive && (
          <div className="flex items-center gap-2 text-[11px]">
            {selectionHidden && <span className="text-amber-300">選択中のイベントは絞り込みで表示されていません</span>}
            <button type="button" onClick={clearFilters} className="ml-auto text-accent-light hover:text-accent">
              絞り込みを解除
            </button>
          </div>
        )}
      </div>
      <VirtualList
        items={rows}
        itemKey={itemKey}
        estimateHeight={estimate}
        renderItem={renderItem}
        scrollToKey={selectedKey}
        scrollToken={scrollToken + filterToken}
        className="flex-1 min-h-0 py-1"
        empty={<div className="p-4 text-xs text-text-tertiary">表示するイベントがありません</div>}
      />
    </div>
  );
}

/** Last non-empty lines (a traceback ends with the exception line). */
function lastLines(s: string, n: number): string {
  return s
    .split('\n')
    .filter((l) => l.trim().length > 0)
    .slice(-n)
    .join('\n');
}

function FilterChip({ active, onClick, children, tone = 'gold' }: { active: boolean; onClick: () => void; children: ReactNode; tone?: 'gold' | 'red' }) {
  const on = tone === 'red' ? 'border-red-500/60 bg-red-500/15 text-red-300' : 'border-accent/60 bg-accent/15 text-accent';
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`px-1.5 py-0.5 rounded border text-[11px] transition-colors ${
        active ? on : 'border-surface-border text-text-tertiary hover:text-text-secondary hover:border-surface-border-light'
      }`}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Task text (top of the left pane)
// ---------------------------------------------------------------------------

function TaskText({ model, ep }: { model: EpisodeModel; ep: EpisodeRef }) {
  const [open, setOpen] = useState(false);
  const ev = model.taskText;
  const td = ev ? dataOf(ev, 'task_text') : null;
  const full = useFullText(ep, td ? td.text : null, open);
  const text = full.text ?? '';
  const { text: head, more } = firstLines(text, 4);
  return (
    <div className="flex-shrink-0 px-3 py-2 border-b border-surface-border bg-surface-raised">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">タスク文</span>
        {ev && full.text !== null ? (
          <span className="text-[11px] text-text-tertiary">最初のコード生成要求で実際に送られた文</span>
        ) : (
          <Badge tone="neutral">{LABELS.notRecorded}</Badge>
        )}
      </div>
      {ev && full.text === null ? (
        <div className="text-[11px] text-text-tertiary">
          最初のコード生成要求からタスク文を切り出せませんでした{td?.reason ? `（${td.reason}）` : ''}。送信内容はその要求の「実際のモデル入力」タブで確認できます。
        </div>
      ) : ev ? (
        <div className="text-xs text-text-primary whitespace-pre-wrap break-words max-h-64 overflow-y-auto">
          {open ? text : head}
          {full.partial && open && full.loading && <span className="text-text-tertiary"> …読み込み中</span>}
          {(more > 0 || full.partial) && (
            <button type="button" onClick={() => setOpen(!open)} className="block mt-1 text-[11px] text-accent-light hover:text-accent">
              {open ? '折りたたむ' : '全文を表示'}
            </button>
          )}
        </div>
      ) : (
        <div className="text-[11px] text-text-tertiary">
          task_text イベントがありません（最初のコード生成要求の前、oracle コード、または記録の欠落）。
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

const KIND_TAG: Record<TNode['kind'], { label: string; tone: Tone }> = {
  model_call: { label: 'モデル', tone: 'gold' },
  code_block: { label: 'コード', tone: 'green' },
  api_call: { label: 'API', tone: 'sky' },
  service_call: { label: 'サービス', tone: 'violet' },
  stdout: { label: 'stdout', tone: 'neutral' },
  stderr: { label: 'stderr', tone: 'red' },
  evaluator: { label: '評価器', tone: 'amber' },
  feedback: { label: 'フィードバック', tone: 'violet' },
  episode_start: { label: '開始', tone: 'neutral' },
  episode_end: { label: '終了', tone: 'neutral' },
  env_reset: { label: 'リセット', tone: 'amber' },
  warning: { label: '警告', tone: 'amber' },
  other: { label: 'その他', tone: 'neutral' },
};

export function roleTone(role: string | null | undefined): Tone {
  switch (role) {
    case 'code_generator':
      return 'gold';
    case 'vdm':
      return 'violet';
    case 'perception_vlm':
      return 'sky';
    default:
      return 'neutral';
  }
}

interface NodeCardProps {
  row: Row;
  model: EpisodeModel;
  ep: EpisodeRef;
  running: boolean;
  selected: boolean;
  selectedEventId: string | null;
  collapsed: boolean;
  nestedOpen: boolean;
  onSelect: (node: TNode, eventId?: string) => void;
  onToggle: () => void;
  onToggleNested: () => void;
}

function NodeCard({ row, model, ep, running, selected, selectedEventId, collapsed, nestedOpen, onSelect, onToggle, onToggleNested }: NodeCardProps) {
  const { node, depth, context } = row;
  const tag = KIND_TAG[node.kind];
  const dur = nodeDuration(node);
  const hasKids = node.children.length > 0;
  // Keyboard navigation: when focus was on another card, it follows the selection.
  const cardRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const active = document.activeElement;
    if (selected && cardRef.current && active !== cardRef.current && active instanceof HTMLElement && active.dataset.timelineCard !== undefined) {
      cardRef.current.focus({ preventScroll: true });
    }
  }, [selected]);
  return (
    <div className="relative pr-2 py-0.5" style={{ paddingLeft: 8 + depth * 14 }}>
      {depth > 0 && <div className="absolute top-0 bottom-0 border-l border-surface-border" style={{ left: 8 + depth * 14 - 8 }} />}
      <div
        ref={cardRef}
        data-timeline-card=""
        role="button"
        tabIndex={selected ? 0 : -1}
        aria-pressed={selected}
        onClick={() => onSelect(node)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect(node);
          }
        }}
        className={`rounded-md border transition-colors cursor-pointer ${
          selected ? 'border-accent/70 bg-accent/[0.06] border-gold-glow' : 'border-surface-border bg-surface-raised hover:border-surface-border-light'
        } ${context ? 'opacity-45' : ''} ${node.late ? 'border-dashed' : ''}`}
      >
        <div className="flex items-center gap-1.5 px-2 pt-1.5">
          {hasKids ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onToggle();
              }}
              aria-label={collapsed ? '展開' : '折りたたむ'}
              className="w-4 h-4 flex items-center justify-center text-text-tertiary hover:text-accent"
            >
              <svg className={`w-3 h-3 transition-transform ${collapsed ? '' : 'rotate-90'}`} fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            </button>
          ) : (
            <span className="w-4" />
          )}
          <Badge tone={tag.tone}>{tag.label}</Badge>
          <span className="font-display text-[13px] text-text-primary truncate min-w-0" title={nodeTitle(node)}>
            {nodeTitle(node)}
          </span>
          <span className="ml-auto flex items-center gap-1 flex-shrink-0">
            {dur !== null && <span className="text-[11px] text-text-tertiary font-mono">{fmtDuration(dur)}</span>}
            <StatusBadge node={node} running={running} />
            <span className="text-[10px] text-text-muted font-mono">#{node.seq}</span>
          </span>
        </div>
        <div className="px-2 pb-1.5 pt-1 pl-7 space-y-1 min-w-0">
          <CardBody node={node} model={model} ep={ep} selectedEventId={selectedEventId} onSelect={onSelect} nestedOpen={nestedOpen} onToggleNested={onToggleNested} />
          <Markers node={node} running={running} />
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ node, running }: { node: TNode; running: boolean }) {
  // episode_start carries status "started" by definition; the episode's state is in the top bar.
  if (node.kind === 'episode_start') return null;
  if (node.open) return <Badge tone={running ? 'gold' : 'red'}>{running ? '実行中' : '未完了'}</Badge>;
  if (node.failed) {
    // A stream the caller stopped reading (GeneratorExit) has no result, but it is a stop, not a failure.
    if (node.kind === 'model_call' && node.call && callOutcome(node.call).stopped) return <Badge tone="amber">中断</Badge>;
    return <Badge tone="red">失敗</Badge>;
  }
  if (node.degraded) return <Badge tone="amber">degraded</Badge>;
  const st = node.end?.status ?? node.primary.status;
  if (node.kind === 'warning') return <Badge tone="amber">warning</Badge>;
  if (st === 'ok' || node.kind === 'code_block' || node.kind === 'api_call' || node.kind === 'model_call' || node.kind === 'service_call') {
    return <Badge tone="green">ok</Badge>;
  }
  return st ? <Badge tone={statusTone(st)}>{st}</Badge> : null;
}

function Markers({ node, running }: { node: TNode; running: boolean }) {
  const items: ReactNode[] = [];
  if (node.open) items.push(<Badge key="open" tone={running ? 'gold' : 'red'}>{running ? LABELS.openWhileRunning : LABELS.openProcessEnded}</Badge>);
  if (node.late) items.push(<Badge key="late" tone="violet">{LABELS.lateEvent}</Badge>);
  if (node.orphan) items.push(<Badge key="orphan" tone="red" title={node.orphan}>親イベント不明: {truncate(node.orphan, 40)}</Badge>);
  if (node.warnings.length) items.push(<Badge key="warn" tone="amber">記録の警告 {node.warnings.length}</Badge>);
  const pendingData = node.events.filter((e) => isOversize(e) && (e.data as { data_asset_loaded?: unknown }).data_asset_loaded !== true);
  const unwritten = pendingData.filter((e) => dataAssetRef(e)?.written !== true).length;
  if (pendingData.length - unwritten > 0) {
    items.push(
      <Badge key="oversize" tone="amber" title="イベント行が 256 KB を超えたため、data の本体は別ファイルに保存されています。選択すると読み込みます。">
        data は別ファイル（選択で読み込み）
      </Badge>,
    );
  }
  if (unwritten > 0) {
    items.push(
      <Badge key="oversize-unwritten" tone="red" title="イベント行が 256 KB を超えたため data の本体を別ファイルに移しましたが、そのファイルは保存されていません。行に残った短い項目だけがあります。">
        data の本体は{LABELS.notRecorded}（256 KB 超）
      </Badge>,
    );
  }
  const budget = arrayBudgetOf(node);
  if (budget > 0) {
    items.push(
      <Badge key="budget" tone="amber" title={arrayBudgetLabel(budget)}>
        配列 {budget} 件は値を保存せず（上限超過）
      </Badge>,
    );
  }
  if (node.kind === 'api_call') {
    const bad = troubledServices(node);
    if (bad.length) {
      const failed = bad.filter((c) => c.failed).length;
      items.push(
        <Badge key="svc" tone={failed ? 'red' : 'amber'} title="この API の中の知覚・計画サービス呼び出し">
          内部のサービス呼び出し: {failed ? `失敗 ${failed}` : ''}
          {bad.length - failed ? `${failed ? ' / ' : ''}degraded・未完了 ${bad.length - failed}` : ''}
        </Badge>,
      );
    }
  }
  if (items.length === 0) return null;
  return <div className="flex flex-wrap gap-1">{items}</div>;
}

interface BodyProps {
  node: TNode;
  model: EpisodeModel;
  ep: EpisodeRef;
  selectedEventId: string | null;
  onSelect: (node: TNode, eventId?: string) => void;
  nestedOpen: boolean;
  onToggleNested: () => void;
}

function CardBody(props: BodyProps) {
  const { node } = props;
  switch (node.kind) {
    case 'model_call':
      return node.call ? <ModelCallBody {...props} call={node.call} /> : null;
    case 'code_block':
      return <CodeBlockBody {...props} />;
    case 'api_call':
      return <ApiCallBody {...props} />;
    case 'service_call':
      return <ServiceBody node={node} />;
    case 'stdout':
    case 'stderr':
      return <StreamBody node={node} />;
    case 'evaluator':
      return <EvaluatorBody node={node} />;
    case 'feedback':
      return <FeedbackBody {...props} />;
    case 'episode_start':
      return <EpisodeStartBody node={node} />;
    case 'episode_end':
      return <EpisodeEndBody node={node} />;
    case 'env_reset': {
      const d = dataOf(node.primary, 'env_reset');
      return <div className="text-[11px] text-text-secondary">sim_epoch {d.sim_epoch ?? node.primary.sim_epoch ?? '?'}{d.reason ? ` · ${d.reason}` : ''}</div>;
    }
    case 'warning': {
      const d = dataOf(node.primary, 'trace_warning');
      const label = d.code ? WARNING_CODE_LABELS[d.code] : undefined;
      return (
        <div className="text-[11px] text-amber-200 break-words space-y-0.5">
          {(label || d.run_level === true || d.outside_episode === true || d.counts_as_gap === false) && (
            <div className="flex flex-wrap gap-1">
              {label && <Badge tone="amber">{label}</Badge>}
              {d.run_level === true && <Badge tone="neutral">プロセス全体（run 単位）</Badge>}
              {d.outside_episode === true && <Badge tone="neutral">エピソード外</Badge>}
              {d.counts_as_gap === false && <Badge tone="neutral">記録の欠落には数えない</Badge>}
            </div>
          )}
          <div>
            {textOf(d.message)}
            {d.message_redacted === true ? `（${REDACTED_SHORT}）` : ''}
          </div>
          {typeof d.episode_id === 'string' && d.episode_id && d.outside_episode === true && (
            <div className="text-text-secondary">
              対象のエピソード: <span className="font-mono">{d.episode_id}</span>
              {d.alarm === true ? '（trial の alarm による中断）' : ''}
            </div>
          )}
          {typeof d.note === 'string' && d.note && <div className="text-text-secondary">{d.note}</div>}
        </div>
      );
    }
    default:
      return <div className="text-[11px] text-text-tertiary">event_type: {node.primary.event_type}</div>;
  }
}

function ModelCallBody({ call, selectedEventId, onSelect, node }: BodyProps & { call: LogicalCall }) {
  const attempt = mainAttempt(call);
  const rd = attempt ? dataOf(attempt.request, 'model_request') : null;
  const outcome = callOutcome(call);
  // The HTTP answer (latency, status) is shown even when the call failed afterwards.
  const resp = outcome.httpAnswer;
  const rsp = resp ? dataOf(resp, 'model_response') : null;
  const counts = requestImageCounts(rd);
  // A REGENERATE reply is parsed twice (_parse_multi_turn_decision, then _extract_code):
  // the decision comes from the parse that has one, the code blocks from the last parse.
  const parsed = lastParsed(call);
  const decisionEv = [...call.parsed].reverse().find((p) => !!dataOf(p, 'model_output_parsed').decision) ?? null;
  const pd = parsed ? { ...dataOf(parsed, 'model_output_parsed'), ...(decisionEv ? pickDecision(dataOf(decisionEv, 'model_output_parsed')) : {}) } : null;
  // Content is shown only for an answer the harness used (not after a logical failure).
  const content = outcome.state === 'ok' ? resolveText(rsp?.content).text ?? '' : '';
  const finalError = outcome.finalError ? dataOf(outcome.finalError, 'model_error') : null;
  const attemptError = !finalError && !resp ? [...call.attempts.flatMap((a) => a.errors), ...call.strayErrors].pop() ?? null : null;
  const ed = attemptError ? dataOf(attemptError, 'model_error') : null;
  return (
    <>
      <div className="flex flex-wrap items-center gap-1">
        <Badge tone={roleTone(call.role)}>{call.role}</Badge>
        {call.callKind && <Badge tone="neutral">{call.callKind}</Badge>}
        {call.turnIndex !== null && <span className="text-[11px] text-text-tertiary">ターン {call.turnIndex}</span>}
        {counts.noImage === true ? (
          <Badge tone="neutral">画像なし</Badge>
        ) : counts.noImage === null && rd ? (
          <Badge tone="neutral" title="画像と動画の数が記録されていません（本文を解析できなかった、または記録の欠落）">
            画像数 {LABELS.notRecorded}
          </Badge>
        ) : (
          <>
            {counts.images !== null && counts.images > 0 && <Badge tone="gold">画像 {counts.images}</Badge>}
            {counts.videos !== null && counts.videos > 0 && <Badge tone="gold">動画 {counts.videos}</Badge>}
          </>
        )}
        {call.attempts.length > 1 && <Badge tone="amber">再試行 {call.attempts.length - 1}</Badge>}
      </div>
      <div className="text-[11px] text-text-secondary font-mono truncate" title={call.model ?? undefined}>
        {call.model ?? rd?.model ?? 'model ?'}
        {rsp?.latency_s !== undefined && rsp?.latency_s !== null && <span className="text-text-tertiary"> · {fmtDuration(rsp.latency_s)}</span>}
        {rsp?.http_status ? <span className="text-text-tertiary"> · HTTP {rsp.http_status}</span> : null}
      </div>
      {call.role === 'vdm' && content && <div className="text-[11px] text-text-secondary line-clamp-3 break-words">{truncate(content, 240)}</div>}
      {call.role !== 'vdm' && pd && (
        <div className="flex flex-wrap items-center gap-1 text-[11px]">
          {Array.isArray(pd.code_blocks) && <span className="text-text-secondary">コード {pd.code_blocks.length} ブロック</span>}
          {pd.decision && (
            <Badge tone="neutral" title="ハーネスの読み取り（モデルの主張ではない）">
              {harnessReadingLabel(pd.decision, pd.harness_reading)}
            </Badge>
          )}
          {pd.model_said_finish === true && <Badge tone="green">{LABELS.modelClaimsFinish}</Badge>}
          {pd.content_empty === true && <Badge tone="red">応答が空</Badge>}
          {pd.code_source === 'oracle' && <Badge tone="amber">oracle</Badge>}
        </div>
      )}
      {finalError && (
        <div className="text-[11px] text-red-300 break-words">
          {resp ? `HTTP ${rsp?.http_status ?? '2xx'} の応答の後に失敗 · ` : ''}
          {modelErrorLabel(finalError)}
          {textOf(finalError.message) ? `: ${truncate(textOf(finalError.message), 160)}` : ''}
          {finalError.streamed ? `（受信した chunk ${finalError.n_chunks ?? '?'}）` : ''}
        </div>
      )}
      {ed && (
        <div className="text-[11px] text-red-300 break-words">
          {ed.http_status ? `HTTP ${ed.http_status} ` : ''}
          {ed.error_type ?? 'error'}: {truncate(textOf(ed.message), 160)}
        </div>
      )}
      {call.attempts.length > 1 && (
        <div className="flex flex-wrap gap-1">
          {call.attempts.map((a, i) => {
            const r = a.responses[0];
            const e = a.errors[0];
            const st = r ? dataOf(r, 'model_response').http_status : e ? dataOf(e, 'model_error').http_status : null;
            const isSel = selectedEventId === a.request.event_id;
            return (
              <button
                key={a.request.event_id}
                type="button"
                onClick={(ev) => {
                  ev.stopPropagation();
                  onSelect(node, a.request.event_id);
                }}
                className={`px-1.5 py-px rounded border text-[10px] font-mono ${
                  isSel ? 'border-accent text-accent' : r ? 'border-nv-green/40 text-nv-green' : e ? 'border-red-500/40 text-red-300' : 'border-surface-border text-text-tertiary'
                }`}
              >
                attempt {dataOf(a.request, 'model_request').http_attempt ?? i + 1}: {st ?? (r ? 'ok' : e ? 'error' : '…')}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}

function pickDecision(d: ModelOutputParsedData): Partial<ModelOutputParsedData> {
  return { decision: d.decision, harness_reading: d.harness_reading, model_said_finish: d.model_said_finish };
}

function CodeBlockBody({ node, onSelect }: BodyProps) {
  const d = dataOf(node.primary, 'code_block_start');
  const ed = node.end ? dataOf(node.end, 'code_block_end') : null;
  const code = resolveText(d.code);
  const out = node.children.find((c) => c.kind === 'stdout');
  const err = node.children.find((c) => c.kind === 'stderr');
  const outText = out ? resolveText(dataOf(out.primary, 'stdout').text).text ?? '' : '';
  const errText = err ? resolveText(dataOf(err.primary, 'stderr').text).text ?? '' : '';
  return (
    <>
      <div className="flex flex-wrap items-center gap-1 text-[11px]">
        {d.turn_index !== undefined && d.turn_index !== null && <span className="text-text-tertiary">ターン {d.turn_index}</span>}
        {d.code_source && <Badge tone={d.code_source === 'oracle' ? 'amber' : 'neutral'}>{d.code_source}</Badge>}
        {ed && typeof ed.sandbox_rc === 'number' && <Badge tone={ed.sandbox_rc === 0 ? 'green' : 'red'}>sandbox_rc={ed.sandbox_rc}</Badge>}
        {node.sourceCall && (
          <span className="text-text-tertiary">
            生成元:{' '}
            <EventLink onClick={() => onSelect(node.sourceCall!.node)}>
              {node.sourceCall.role}
              {node.sourceCall.turnIndex !== null ? ` ターン${node.sourceCall.turnIndex}` : ''}
            </EventLink>
          </span>
        )}
      </div>
      {code.text !== null ? (
        <CodeView code={code.text} maxLines={6} compact showCopy={false} />
      ) : (
        <div className="text-[11px] text-text-tertiary">コード: {LABELS.notRecorded}</div>
      )}
      {code.truncated && <div className="text-[10px] text-text-tertiary">（先頭のみ。全文は詳細タブ）</div>}
      {outText && <div className="text-[11px] text-text-secondary font-mono whitespace-pre-wrap break-all line-clamp-2">{firstLines(outText, 2).text}</div>}
      {errText && <div className="text-[11px] text-red-300 font-mono whitespace-pre-wrap break-all line-clamp-2">{lastLines(errText, 2)}</div>}
    </>
  );
}

function ApiCallBody({ node, model, ep, nestedOpen, onToggleNested }: BodyProps) {
  const d = dataOf(node.primary, 'api_call_start');
  const endEv = node.end;
  const ed = endEv ? dataOf(endEv, 'api_call_end') : null;
  const isErr = endEv?.event_type === 'api_call_error';
  const depth = apiDepth(node);
  const nested = node.children.filter((c) => c.kind === 'api_call').length;
  const kf = keyFramesOf(model, node);
  const before = kf.before[0];
  const after = kf.after.find((k) => k.camera === before?.camera) ?? kf.after[0];
  const serviceArts = node.children.filter((c) => c.kind === 'service_call').flatMap((c) => c.events.flatMap((e) => e.artifacts ?? []));
  const perception = serviceArts
    .filter((a) => (a.label === 'perception_output' || a.label === 'perception_input') && a.path && a.written !== false && /\.(png|jpe?g|webp)$/i.test(a.path))
    .slice(0, 2);
  // Decoded perception outputs: mask previews through /array-image for the mask nodes of
  // the service responses (npz + npz_key). Other arrays (boxes, grasp poses, IK
  // solutions) are not images; they are shown in the debug tab.
  const maskThumbs = maskNodesOf(node);
  const thumb = (path: string | null) => (path ? episodeFileUrl(ep, path) : null);
  return (
    <>
      <div className="flex flex-wrap items-center gap-1 text-[11px]">
        {depth > 0 && <Badge tone="neutral">depth {depth}</Badge>}
        {d.privileged === true && <Badge tone="amber">privileged</Badge>}
        {d.call_site?.line ? <span className="text-text-tertiary">行 {d.call_site.line}</span> : null}
      </div>
      <div className="text-[11px] font-mono text-text-secondary break-all">({argsSummary(d.args, 140)})</div>
      {endEv && !isErr && <div className="text-[11px] font-mono text-emerald-300/90 break-all">→ {valueSummary(ed?.return, 140)}</div>}
      {isErr && (
        <div className="text-[11px] font-mono text-red-300 break-all">
          ✕ {ed?.error_type ?? 'Error'}: {truncate(textOf(ed?.message), 160)}
        </div>
      )}
      {((before && before.path) || (after && after.path) || perception.length > 0 || maskThumbs.length > 0) && (
        <div className="flex flex-wrap items-end gap-x-3 gap-y-1 pt-0.5">
          {((before && before.path) || (after && after.path)) && (
            <div className="flex items-end gap-1.5">
              {before && (
                <div className="text-center">
                  <TraceImage src={thumb(before.path)} alt="開始時" provenance="debug_frame" compact className="h-12 w-16" />
                  <div className="text-[9px] text-text-muted">開始</div>
                </div>
              )}
              {after && (
                <div className="text-center">
                  <TraceImage src={thumb(after.path)} alt="終了時" provenance="debug_frame" compact className="h-12 w-16" />
                  <div className="text-[9px] text-text-muted">終了</div>
                </div>
              )}
              <span className="text-[9px] text-sky-300/80 self-center max-w-[7rem] leading-3">{LABELS.debugFrame}</span>
            </div>
          )}
          {(perception.length > 0 || maskThumbs.length > 0) && (
            <div className="flex items-end gap-1.5">
              {perception.map((a) => (
                <div key={a.path} className="text-center">
                  <TraceImage src={thumb(a.path ?? null)} alt={a.path ?? ''} provenance="perception" compact className="h-12 w-16" />
                  <div className="text-[9px] text-text-muted">{a.label === 'perception_output' ? '知覚出力' : '知覚入力'}</div>
                </div>
              ))}
              {maskThumbs.map((m) => (
                <MaskThumb key={`${m.npz}|${m.key}`} src={episodeArrayImageUrl(ep, m.npz, m.key)} alt={`${m.npz}:${m.key}`} />
              ))}
              <span className="text-[9px] text-violet-300/80 self-center max-w-[7rem] leading-3">知覚サービスの入出力（デバッグ用）</span>
            </div>
          )}
        </div>
      )}
      {nested > 0 && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggleNested();
          }}
          className="text-[11px] text-accent-light hover:text-accent"
        >
          {nestedOpen ? '内部呼び出しを隠す' : `内部呼び出し ${nested} 件を表示`}
        </button>
      )}
    </>
  );
}

/**
 * Mask arrays among the decoded outputs of the service calls directly under an API call.
 * The recorder stores them bit-packed in one npz per call and marks their summary nodes
 * with `is_mask`, `npz` and `npz_key` (the trace recorder). At most two.
 */
function maskNodesOf(node: TNode): { npz: string; key: string }[] {
  const out: { npz: string; key: string }[] = [];
  const seen = new Set<string>();
  const walk = (v: unknown, depth: number) => {
    if (out.length >= 2 || depth > 8 || !v || typeof v !== 'object') return;
    if (Array.isArray(v)) {
      for (const x of v) walk(x, depth + 1);
      return;
    }
    const o = v as Record<string, unknown>;
    const shape = Array.isArray(o.shape) ? o.shape : null;
    const isMask = o.is_mask === true || (o.dtype === 'bool' && shape !== null && shape.length === 2);
    if (isMask && typeof o.npz === 'string' && typeof o.npz_key === 'string' && o.written !== false) {
      const k = `${o.npz}|${o.npz_key}`;
      if (!seen.has(k)) {
        seen.add(k);
        out.push({ npz: o.npz, key: o.npz_key });
      }
      return;
    }
    for (const x of Object.values(o)) walk(x, depth + 1);
  };
  for (const c of node.children) {
    if (c.kind === 'service_call' && c.end) walk(dataOf(c.end, 'service_call_end').response_summary, 0);
  }
  return out;
}

/** Card thumbnail of a decoded perception output; left out entirely when no image can be made. */
function MaskThumb({ src, alt }: { src: string; alt: string }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <div className="text-center">
      <img src={src} alt={alt} loading="lazy" onError={() => setOk(false)} className="h-12 w-16 object-contain rounded border border-surface-border bg-black" />
      <div className="text-[9px] text-text-muted">知覚出力</div>
    </div>
  );
}

function ServiceBody({ node }: { node: TNode }) {
  const d = dataOf(node.end ?? node.primary, 'service_call_end');
  const sd = dataOf(node.primary, 'service_call_start');
  const attempts = d.http_attempts ?? sd.http_attempts ?? [];
  // The recorder learns the endpoint from the HTTP attempt, so it is on the end event.
  const epInfo = (node.end ? d.endpoint : null) ?? sd.endpoint ?? null;
  const endpoint = typeof epInfo === 'string' ? epInfo : epInfo?.url ?? epInfo?.host ?? null;
  return (
    <>
      <div className="text-[11px] text-text-secondary font-mono truncate">{endpoint ?? ''}</div>
      <div className="flex flex-wrap gap-1 text-[11px]">
        {typeof d.latency_s === 'number' && <span className="text-text-tertiary">{fmtDuration(d.latency_s)}</span>}
        {attempts.length > 0 && <span className="text-text-tertiary">HTTP 試行 {attempts.length}</span>}
        {node.end?.event_type === 'service_call_error' && <span className="text-red-300">{d.error_type}: {truncate(textOf(d.message), 100)}</span>}
      </div>
      {d.response_summary !== undefined && <div className="text-[11px] font-mono text-emerald-300/90 break-all">→ {valueSummary(d.response_summary, 120)}</div>}
    </>
  );
}

function StreamBody({ node }: { node: TNode }) {
  const d = dataOf(node.primary, 'stdout');
  const r = resolveText(d.text);
  const t = r.text ?? '';
  const { text, more } = firstLines(t, 3);
  const len = r.length ?? t.length;
  return (
    <>
      <pre className={`text-[11px] font-mono whitespace-pre-wrap break-all max-h-16 overflow-hidden ${node.kind === 'stderr' ? 'text-red-300' : 'text-text-secondary'}`}>
        {t ? text : '（空）'}
      </pre>
      <div className="text-[10px] text-text-muted">
        {len} 文字{more > 0 ? ` · 残り ${more} 行` : ''}
        {r.truncated ? ' · 全文は asset' : ''}
      </div>
    </>
  );
}

function EvaluatorBody({ node }: { node: TNode }) {
  const d = dataOf(node.primary, 'evaluator_result');
  return (
    <div className="flex flex-wrap items-center gap-1 text-[11px]">
      <span className="text-text-secondary">reward={fmtNum(d.reward)}</span>
      <Badge tone={d.task_completed === true ? 'green' : d.task_completed === false ? 'red' : 'neutral'}>task_completed={String(d.task_completed ?? '—')}</Badge>
      {d.source && <span className="text-text-tertiary">source: {d.source}</span>}
      <Badge tone="sky">{NOT_MODEL_INPUT}</Badge>
    </div>
  );
}

function FeedbackBody({ node, model, onSelect }: BodyProps) {
  const d = dataOf(node.primary, 'feedback_built');
  const segs = d.segments ?? [];
  const byKind = new Map<string, number>();
  for (const s of segs) byKind.set(s.kind ?? '?', (byKind.get(s.kind ?? '?') ?? 0) + (s.length ?? 0));
  const contained = Array.from(new Set(segs.flatMap((s) => (s.contains ?? []).map((c) => c.kind ?? '?'))));
  const next =
    typeof d.turn_index === 'number'
      ? model.calls.find((c) => c.role === 'code_generator' && c.turnIndex === d.turn_index && c.node.seq > node.seq) ?? null
      : null;
  const vdm = d.vdm_model_event_id ? model.callOfEvent.get(d.vdm_model_event_id) ?? null : null;
  const user = resolveText(d.user_feedback).text;
  return (
    <>
      <div className="flex flex-wrap gap-1 text-[11px]">
        {d.builder && <span className="text-text-tertiary font-mono">{d.builder}</span>}
        {typeof d.turn_index === 'number' && <span className="text-text-tertiary">ターン {d.turn_index}</span>}
        {Array.from(byKind.entries()).map(([k, n]) => (
          <Badge key={k} tone="neutral">
            {k} {n}字
          </Badge>
        ))}
      </div>
      {contained.length > 0 && <div className="text-[11px] text-text-tertiary">プロンプトに含まれる記録済みテキスト: {contained.join(', ')}</div>}
      {user && <div className="text-[11px] text-text-secondary">user: {truncate(user, 160)}</div>}
      <div className="flex flex-wrap gap-3 text-[11px]">
        {vdm && <EventLink onClick={() => onSelect(vdm.node)}>VDM 呼び出し</EventLink>}
        {next && <EventLink onClick={() => onSelect(next.node)}>この入力を使った次のコード生成</EventLink>}
      </div>
    </>
  );
}

function EpisodeStartBody({ node }: { node: TNode }) {
  const d = dataOf(node.primary, 'episode_start');
  return (
    <div className="text-[11px] text-text-secondary">
      {[d.task, d.condition, d.seed !== undefined && d.seed !== null ? `seed ${d.seed}` : null, d.attempt_index ? `attempt ${d.attempt_index}` : null, d.source]
        .filter(Boolean)
        .join(' · ')}
    </div>
  );
}

function EpisodeEndBody({ node }: { node: TNode }) {
  const d = dataOf(node.primary, 'episode_end');
  return (
    <div className="flex flex-wrap gap-1 text-[11px] text-text-secondary">
      {d.end_reason && <Badge tone={node.failed ? 'red' : 'neutral'}>end_reason={d.end_reason}</Badge>}
      {typeof d.duration_s === 'number' && <span>{fmtDuration(d.duration_s)}</span>}
      {d.trace_complete === false && <Badge tone="amber">記録に欠落あり</Badge>}
      {typeof d.n_code_blocks === 'number' && <span>ブロック {d.n_code_blocks}</span>}
      {typeof d.n_model_calls === 'number' && <span>モデル呼び出し {d.n_model_calls}</span>}
    </div>
  );
}
