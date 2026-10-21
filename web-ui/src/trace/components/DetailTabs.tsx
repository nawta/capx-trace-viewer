/**
 * Details part of the right pane: the five tabs of docs/trace-format.md 9.3
 * (実際のモデル入力, モデル出力/コード, API/実行ログ, デバッグ状態, 入力差分).
 * Tab contents fetch payload files lazily, only for the selected event. The
 * tab body is keyed by the selected event, so nothing a tab loaded for one
 * event (an array preview, a search) is shown for another; a render error in
 * one tab shows a notice instead of blanking the page.
 *
 * Events whose data the writer moved to an asset (line over 256 KB) get a
 * notice while that data is loading, when it failed, or when it was never
 * written (dataAsset.ts): the selected event's own, and in one line the other
 * events the panes read for this selection (later requests, the previous
 * request, nested service calls). The tabs get the loading states too, so a
 * pane that reads such an event says why its data is missing.
 *
 * Related: tabs/*.tsx, TraceViewerPage.tsx.
 */

import { useLayoutEffect, useRef, type ReactNode } from 'react';
import { dataAssetRef, pendingAsset, type DataAssetState } from '../dataAsset';
import { LABELS } from '../labels';
import type { EpisodeModel, Selection, TNode } from '../model';
import type { EpisodeSummary, TraceEvent } from '../types';
import { nodeTitle } from '../model';
import { Notice } from './common';
import { DataAssetInline, NO_ASSET_STATES } from './DataAssetState';
import { ErrorBoundary } from './ErrorBoundary';
import { ApiLogTab } from './tabs/ApiLogTab';
import { DebugTab } from './tabs/DebugTab';
import { InputDiffTab } from './tabs/InputDiffTab';
import { ModelInputTab } from './tabs/ModelInputTab';
import { ModelOutputTab } from './tabs/ModelOutputTab';
import type { EpisodeRef } from './ValueView';

export interface RangeFocus {
  eventId: string;
  path: string;
  start: number;
  end: number;
  token: number;
}

export interface TabProps {
  model: EpisodeModel;
  ep: EpisodeRef;
  /** The episode summary (with episode.json as `episode`); null until loaded. */
  summary: EpisodeSummary | null;
  selection: Selection;
  running: boolean;
  onSelect: (node: TNode, eventId?: string) => void;
  /** Opens the model-input tab of a request with a payload range marked. */
  onFocusRange: (eventId: string, path: string, start: number, end: number) => void;
  focus: RangeFocus | null;
  /** Loading state of data moved to assets, per event id (dataAsset.ts). */
  assetStates?: ReadonlyMap<string, DataAssetState>;
  /** Loads one event's data asset again, bypassing the file cache. */
  onRetryAsset?: (ev: TraceEvent) => void;
}

interface DetailTabsProps extends Omit<TabProps, 'selection'> {
  revealPrompt?: boolean;
  selection: Selection | null;
  tab: number;
  onTab: (i: number) => void;
  /** An event id from the URL that the loaded events do not contain. */
  missingEventId?: string | null;
  /** Every event whose data asset the panes need for the selection (dataAssetTargets). */
  assetEvents?: TraceEvent[];
}

/**
 * Notice for the events of the selection whose data is in an asset and not (yet) merged:
 * loading, failed (with a retry), or never written. Nothing for loaded ones. `related` are
 * the other events the panes read for this selection; they get one line each in a
 * compact notice.
 */
export function DataAssetNotice({
  events,
  related = [],
  states,
  onRetry,
}: {
  events: TraceEvent[];
  related?: TraceEvent[];
  states: ReadonlyMap<string, DataAssetState>;
  onRetry?: (ev: TraceEvent) => void;
}) {
  const rows: ReactNode[] = [];
  for (const ev of events) {
    const ref = dataAssetRef(ev);
    if (!ref) continue;
    const st = states.get(ev.event_id);
    if (st?.status === 'loaded') continue;
    const where = `#${ev.seq} ${ev.event_type}`;
    if (!ref.path || !ref.written || st?.status === 'unwritten') {
      rows.push(
        <Notice key={ev.event_id} tone="amber">
          {where} の data は 256 KB を超えたため別ファイルに移されましたが、そのファイルは{LABELS.notRecorded}です
          {ref.reason ? `（${ref.reason}）` : ''}。行に残った短い項目だけを表示しています。
        </Notice>,
      );
    } else if (st?.status === 'failed') {
      rows.push(
        <Notice key={ev.event_id} tone="red">
          {where} の data（{ref.path}）を読み込めません: {st.error}。行に残った短い項目だけを表示しています。{' '}
          {onRetry && (
            <button type="button" onClick={() => onRetry(ev)} className="underline underline-offset-2">
              再読み込み
            </button>
          )}
        </Notice>,
      );
    } else {
      rows.push(
        <Notice key={ev.event_id} tone="gold">
          {where} の data は 256 KB を超えたため {ref.path} に保存されています。読み込み中…（読み込むまでは行に残った短い項目だけを表示します）
        </Notice>,
      );
    }
  }
  const own = new Set(events.map((e) => e.event_id));
  const others = related.filter((ev) => !own.has(ev.event_id) && pendingAsset(ev, states) !== null);
  if (others.length > 0) {
    const bad = others.some((ev) => pendingAsset(ev, states)?.kind !== 'loading');
    rows.push(
      <Notice key="related" tone={bad ? 'amber' : 'gold'}>
        <div>この選択で参照する別のイベントの data（256 KB を超えたため別ファイル）:</div>
        {others.map((ev) => (
          <div key={ev.event_id} className="pl-2">
            #{ev.seq} {ev.event_type}: <DataAssetInline ev={ev} states={states} onRetry={onRetry} />
          </div>
        ))}
      </Notice>,
    );
  }
  return rows.length ? <div className="space-y-1 mb-2">{rows}</div> : null;
}

export function DetailTabs({ tab, onTab, selection, revealPrompt = false, missingEventId = null, assetStates = NO_ASSET_STATES, onRetryAsset, assetEvents = [], ...rest }: DetailTabsProps) {
  // A new tab or a new event starts at the top. (Runs before the children's passive
  // effects, so a range focus or a marked line can still scroll itself into view.)
  const panelRef = useRef<HTMLDivElement>(null);
  const eventId = selection?.eventId ?? null;
  useLayoutEffect(() => {
    if (panelRef.current) panelRef.current.scrollTop = 0;
  }, [tab, eventId]);
  useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!panel || !revealPrompt || tab !== 0) return;
    const reveal = () => {
      const target = panel.querySelector<HTMLElement>('[data-playback-input]');
      if (!target) return false;
      panel.scrollTop += target.getBoundingClientRect().top - panel.getBoundingClientRect().top;
      return true;
    };
    if (reveal()) return;
    const observer = new MutationObserver(() => { if (reveal()) observer.disconnect(); });
    observer.observe(panel, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [tab, eventId, revealPrompt]);
  let body: ReactNode;
  if (!selection) {
    body = (
      <div className="p-4 text-xs text-text-tertiary space-y-1">
        {missingEventId && <div className="text-amber-300">URL で指定されたイベント {missingEventId} は、このエピソードの記録にありません。</div>}
        <div>左のタイムラインでイベントを選択してください。</div>
      </div>
    );
  } else {
    const props: TabProps = { ...rest, selection, assetStates, onRetryAsset };
    switch (tab) {
      case 0:
        body = <ModelInputTab {...props} />;
        break;
      case 1:
        body = <ModelOutputTab {...props} />;
        break;
      case 2:
        body = <ApiLogTab {...props} />;
        break;
      case 3:
        body = <DebugTab {...props} />;
        break;
      default:
        body = <InputDiffTab {...props} />;
        break;
    }
  }
  return (
    <div className="flex flex-col h-full min-h-0 bg-surface">
      <div className="flex-shrink-0 flex items-end gap-0.5 px-2 pt-1.5 border-b border-surface-border bg-surface-raised overflow-x-auto">
        {LABELS.tabs.map((label, i) => (
          <button
            key={label}
            type="button"
            role="tab"
            aria-selected={tab === i}
            onClick={() => onTab(i)}
            className={`px-3 py-1.5 text-xs font-display whitespace-nowrap rounded-t-md border-b-2 transition-colors ${
              tab === i ? 'text-accent border-accent bg-surface' : 'text-text-tertiary border-transparent hover:text-text-primary'
            }`}
          >
            {label}
          </button>
        ))}
        {selection && (
          <span className="ml-auto mb-1.5 pl-3 text-[11px] text-text-tertiary truncate font-mono" title={selection.eventId}>
            {nodeTitle(selection.node)} · {selection.eventId}
          </span>
        )}
      </div>
      <div ref={panelRef} role="tabpanel" className="flex-1 min-h-0 overflow-y-auto p-3">
        {selection && <DataAssetNotice events={selection.node.events} related={assetEvents} states={assetStates} onRetry={onRetryAsset} />}
        <ErrorBoundary key={`${tab}|${eventId ?? ''}`} label={`「${LABELS.tabs[tab] ?? ''}」タブ`}>
          {body}
        </ErrorBoundary>
      </div>
    </div>
  );
}
