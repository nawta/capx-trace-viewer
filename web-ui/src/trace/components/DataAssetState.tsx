/**
 * Where a pane reads an event whose data the writer moved to an asset (line
 * over 256 KB, dataAsset.ts) and that asset is not merged yet: a short
 * statement of its state (loading, failed with a retry, never written, or not
 * loaded with a button that loads it), so the pane never says 「未記録」 for
 * data that was recorded but is still loading or failed to load.
 * "Not loaded" is an event the page does not load by itself for the
 * selection (beyond dataAsset.ts MAX_NESTED_ASSET_TARGETS).
 *
 * Related: DetailTabs.tsx (the notice above the tabs), tabs/ModelInputTab.tsx
 * (later-input report), tabs/DebugTab.tsx (recorded statistics),
 * tabs/InputDiffTab.tsx, TopBar.tsx.
 */

import { pendingAsset, type DataAssetState } from '../dataAsset';
import { LABELS } from '../labels';
import type { TraceEvent } from '../types';

export const NO_ASSET_STATES: ReadonlyMap<string, DataAssetState> = new Map();

export function DataAssetInline({
  ev,
  states,
  onRetry,
}: {
  ev: TraceEvent;
  states: ReadonlyMap<string, DataAssetState>;
  onRetry?: (ev: TraceEvent) => void;
}) {
  const p = pendingAsset(ev, states);
  if (!p) return null;
  if (p.kind === 'unwritten') {
    return (
      <span className="text-amber-300">
        {LABELS.notRecorded}（data は 256 KB を超えたため別ファイルに移されましたが、そのファイルは書かれていません{p.reason ? `: ${p.reason}` : ''}）
      </span>
    );
  }
  if (p.kind === 'failed') {
    return (
      <span className="text-red-300">
        読み込めません（data は 256 KB を超えたため別ファイル {p.path} にあります: {p.error}）
        {onRetry && (
          <button type="button" onClick={() => onRetry(ev)} className="ml-1 underline underline-offset-2">
            再読み込み
          </button>
        )}
      </span>
    );
  }
  if (p.kind === 'idle') {
    return (
      <span className="text-amber-300">
        未読み込み（data は 256 KB を超えたため別ファイル {p.path} にあります。この選択では自動で読み込みません）
        {onRetry && (
          <button type="button" onClick={() => onRetry(ev)} className="ml-1 underline underline-offset-2">
            読み込む
          </button>
        )}
      </span>
    );
  }
  return <span className="text-text-tertiary">読み込み中…（data は 256 KB を超えたため別ファイル {p.path} にあります）</span>;
}
