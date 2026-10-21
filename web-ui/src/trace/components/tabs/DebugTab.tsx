/**
 * Tab 4 「デバッグ状態」 (docs/trace-format.md 9.3, 8.4, 5.2), always under the
 * fixed banner 「モデル入力とは別（デバッグ用）」: ground truth before/after
 * (state_ref json.gz assets, with the bodies that moved), evaluator results
 * (only values the harness computed), the harness summary (its `success` is the
 * last block's sandbox_rc == 0 after the harness's override in batch episodes, and
 * task_completed or num_finishes > 0 in web UI episodes; never the task verdict),
 * perception inputs and outputs (images, mask previews, grasp arrays via
 *  /array and /array-image), and the raw events of the selection. For the
 * episode start and end, the recorder's own state: env, settings, hooks
 * installed at the start and during the episode (episode.json
 * hooks_installed_during_episode), failed hooks, and the API wrappers
 * (apis_wrapped, installed_in_this_episode). The recorded statistics of
 * perception outputs come from the service events; when those events' data was
 * moved to an asset (line over 256 KB), the section says whether it is
 * loading, failed or never written.
 *
 * Related: ValueView.tsx (arrays), api.ts (gzip JSON), model.ts.
 */

import { useMemo, useState } from 'react';
import { episodeFileUrl, fetchJsonFile, traceApi } from '../../api';
import { pendingAsset, type DataAssetState } from '../../dataAsset';
import { fmtDuration, fmtNum } from '../../format';
import { useAsync, useFullText } from '../../hooks';
import { LABELS, NOT_MODEL_INPUT, REDACTED_MESSAGE_NOTE, REDACTED_SHORT } from '../../labels';
import { descendants, enclosingBlock, imageSourceText, mainAttempt, resolveStateRef, textOf, type EpisodeModel, type TNode } from '../../model';
import { dataOf, type ArrayStats, type Artifact, type EpisodeSummary, type StateRef, type TextValue, type TraceEvent } from '../../types';
import { Badge, DebugBanner, EventLink, KV, Notice, SectionTitle, Spinner, ErrorText } from '../common';
import { DataAssetInline, NO_ASSET_STATES } from '../DataAssetState';
import { TextView } from '../TextView';
import { TraceImage } from '../TraceImage';
import { ArrayPreviewList, ValueView, recordedNpzStats, type EpisodeRef } from '../ValueView';
import type { TabProps } from '../DetailTabs';
import { RawJson } from './ModelInputTab';

export function DebugTab(props: TabProps) {
  return (
    <div>
      <DebugBanner />
      <DebugContent {...props} />
    </div>
  );
}

function DebugContent({ model, ep, summary, selection, onSelect, assetStates = NO_ASSET_STATES, onRetryAsset }: TabProps) {
  const node = selection.node;
  const [showRaw, setShowRaw] = useState(false);
  return (
    <div className="space-y-3">
      {(node.kind === 'code_block' || node.kind === 'api_call') && <GroundTruth ep={ep} node={node} />}
      <Evaluators model={model} node={node} onSelect={onSelect} />
      {(node.kind === 'episode_start' || node.kind === 'episode_end') && <EpisodeFacts model={model} ep={ep} summary={summary} />}
      {(node.kind === 'episode_start' || node.kind === 'episode_end') && <RecorderFacts model={model} summary={summary} />}
      {(node.kind === 'api_call' || node.kind === 'service_call' || node.kind === 'code_block') && (
        <Perception ep={ep} node={node} assetStates={assetStates} onRetryAsset={onRetryAsset} />
      )}
      {node.kind === 'model_call' && node.call && <ImageSources model={model} node={node} selectionAttempt={selection.attempt?.request ?? null} />}
      <div>
        <button type="button" onClick={() => setShowRaw(!showRaw)} className="text-[11px] text-accent-light hover:text-accent">
          {showRaw ? 'イベントの生データを隠す' : `イベントの生データを表示（${node.events.length} 件）`}
        </button>
        {showRaw && <RawJson value={node.events} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------

interface BodyPose {
  name: string;
  xpos: number[] | null;
  xquat: number[] | null;
}

/** Reads body poses from a state snapshot, whichever of the usual layouts it uses. */
function bodiesOf(state: unknown): BodyPose[] | null {
  if (!state || typeof state !== 'object') return null;
  const s = state as Record<string, unknown>;
  const nums = (v: unknown): number[] | null => (Array.isArray(v) && v.every((x) => typeof x === 'number') ? (v as number[]) : null);
  if (s.bodies && typeof s.bodies === 'object' && !Array.isArray(s.bodies)) {
    return Object.entries(s.bodies as Record<string, unknown>).map(([name, v]) => {
      const o = (v ?? {}) as Record<string, unknown>;
      return { name, xpos: nums(o.xpos ?? o.pos), xquat: nums(o.xquat ?? o.quat) };
    });
  }
  if (s.body_xpos && typeof s.body_xpos === 'object' && !Array.isArray(s.body_xpos)) {
    const xq = s.body_xquat && typeof s.body_xquat === 'object' && !Array.isArray(s.body_xquat) ? (s.body_xquat as Record<string, unknown>) : {};
    return Object.entries(s.body_xpos as Record<string, unknown>).map(([name, v]) => ({ name, xpos: nums(v), xquat: nums(xq[name]) }));
  }
  const names = Array.isArray(s.body_names) ? (s.body_names as unknown[]) : null;
  const xpos = Array.isArray(s.body_xpos) ? (s.body_xpos as unknown[]) : null;
  const xquat = Array.isArray(s.body_xquat) ? (s.body_xquat as unknown[]) : null;
  if (names && xpos) {
    return names.map((n, i) => ({ name: String(n), xpos: nums(xpos[i]), xquat: xquat ? nums(xquat[i]) : null }));
  }
  return null;
}

function frameOf(state: unknown): string | null {
  return state && typeof state === 'object' && typeof (state as Record<string, unknown>).frame === 'string' ? ((state as Record<string, unknown>).frame as string) : null;
}

function quatOrderOf(state: unknown): string | null {
  const v = state && typeof state === 'object' ? (state as Record<string, unknown>).quaternion_order : null;
  return typeof v === 'string' ? v : null;
}

function StateLoader({ ep, stateRef, children }: { ep: EpisodeRef; stateRef: StateRef | undefined; children: (state: unknown | null, note: string | null) => JSX.Element }) {
  const r = resolveStateRef(stateRef);
  const url = r?.path ? episodeFileUrl(ep, r.path) : null;
  const res = useAsync(url ? () => fetchJsonFile<unknown>(url) : null, [url]);
  if (!r) return children(null, '記録なし（CAPX_TRACE_STATE=0、または記録の欠落）');
  if (!r.path) return children(null, r.reason === 'unsupported_simulator' ? 'このシミュレータでは取得しません（unsupported_simulator）' : r.reason ?? LABELS.notCaptured);
  if (res.loading) return children(null, '読み込み中…');
  if (res.error) return children(null, `読み込めません: ${res.error}`);
  return children(res.data, null);
}

function GroundTruth({ ep, node }: { ep: EpisodeRef; node: TNode }) {
  const startRef = node.kind === 'code_block' ? dataOf(node.primary, 'code_block_start').state_ref : dataOf(node.primary, 'api_call_start').state_ref;
  const endRef = node.end ? (node.kind === 'code_block' ? dataOf(node.end, 'code_block_end').state_ref : dataOf(node.end, 'api_call_end').state_ref) : undefined;
  return (
    <div>
      <SectionTitle>シミュレータの正解状態（開始時 / 終了時）</SectionTitle>
      <StateLoader ep={ep} stateRef={startRef}>
        {(before, noteBefore) => (
          <StateLoader ep={ep} stateRef={endRef}>
            {(after, noteAfter) => <StatePair before={before} after={after} noteBefore={noteBefore} noteAfter={noteAfter} ep={ep} />}
          </StateLoader>
        )}
      </StateLoader>
    </div>
  );
}

function StatePair({ before, after, noteBefore, noteAfter, ep }: { before: unknown; after: unknown; noteBefore: string | null; noteAfter: string | null; ep: EpisodeRef }) {
  const bb = useMemo(() => bodiesOf(before), [before]);
  const ba = useMemo(() => bodiesOf(after), [after]);
  const rows = useMemo(() => {
    if (!bb && !ba) return null;
    const m = new Map<string, { before: BodyPose | null; after: BodyPose | null }>();
    for (const b of bb ?? []) m.set(b.name, { before: b, after: null });
    for (const a of ba ?? []) m.set(a.name, { before: m.get(a.name)?.before ?? null, after: a });
    return Array.from(m.entries())
      .map(([name, v]) => {
        const p0 = v.before?.xpos;
        const p1 = v.after?.xpos;
        const delta = p0 && p1 && p0.length === p1.length ? Math.sqrt(p0.reduce((acc, x, i) => acc + (x - p1[i]) ** 2, 0)) : null;
        return { name, before: p0 ?? null, after: p1 ?? null, delta };
      })
      .sort((a, b) => (b.delta ?? -1) - (a.delta ?? -1));
  }, [bb, ba]);
  const [showAll, setShowAll] = useState(false);
  const simTime = (s: unknown) => (s && typeof s === 'object' ? (s as Record<string, unknown>).time ?? (s as Record<string, unknown>).sim_time : undefined);
  return (
    <div className="space-y-2">
      <KV
        rows={[
          ['開始時', noteBefore ?? `sim time ${fmtNum(simTime(before))}`],
          ['終了時', noteAfter ?? `sim time ${fmtNum(simTime(after))}`],
        ]}
      />
      {rows && rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-left text-text-tertiary border-b border-surface-border">
                <th className="py-1 pr-2 font-normal">body</th>
                <th className="py-1 pr-2 font-normal">xpos（開始）</th>
                <th className="py-1 pr-2 font-normal">xpos（終了）</th>
                <th className="py-1 font-normal">移動量 |Δ|</th>
              </tr>
            </thead>
            <tbody>
              {(showAll ? rows : rows.slice(0, 15)).map((r) => (
                <tr key={r.name} className="border-b border-surface-border/50">
                  <td className="py-0.5 pr-2 text-sky-300">{r.name}</td>
                  <td className="py-0.5 pr-2">{r.before ? r.before.map((x) => fmtNum(x, 4)).join(', ') : '—'}</td>
                  <td className="py-0.5 pr-2">{r.after ? r.after.map((x) => fmtNum(x, 4)).join(', ') : '—'}</td>
                  <td className={`py-0.5 ${r.delta !== null && r.delta > 1e-3 ? 'text-accent' : 'text-text-tertiary'}`}>{r.delta === null ? '—' : fmtNum(r.delta, 5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 15 && (
            <button type="button" onClick={() => setShowAll(!showAll)} className="text-[11px] text-accent-light hover:text-accent mt-1">
              {showAll ? '上位 15 件だけ表示' : `全 ${rows.length} 件を表示`}
            </button>
          )}
          <div className="text-[11px] text-text-tertiary mt-1">
            {frameOf(before ?? after)
              ? `座標系: ${frameOf(before ?? after)}${quatOrderOf(before ?? after) ? ` · 四元数: ${quatOrderOf(before ?? after)}` : ''}（sim.data.body_xpos / body_xquat）`
              : `${LABELS.unitsUnknown}（記録された sim.data.body_xpos / body_xquat の値そのまま）`}
          </div>
        </div>
      )}
      {!rows && (before !== null || after !== null) && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
          <div>
            <div className="text-[11px] text-text-tertiary mb-1">開始時</div>
            {before !== null ? <ValueView value={before} ep={ep} /> : <span className="text-xs text-text-tertiary">{noteBefore}</span>}
          </div>
          <div>
            <div className="text-[11px] text-text-tertiary mb-1">終了時</div>
            {after !== null ? <ValueView value={after} ep={ep} /> : <span className="text-xs text-text-tertiary">{noteAfter}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evaluators and episode facts
// ---------------------------------------------------------------------------

function EvaluatorRow({ ev }: { ev: TraceEvent }) {
  const d = dataOf(ev, 'evaluator_result');
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge tone="neutral">scope: {d.scope ?? '?'}</Badge>
      <span>reward={fmtNum(d.reward)}</span>
      <Badge tone={d.task_completed === true ? 'green' : d.task_completed === false ? 'red' : 'neutral'}>task_completed={String(d.task_completed ?? '—')}</Badge>
      <span>terminated={String(d.terminated ?? '—')}</span>
      <span>truncated={String(d.truncated ?? '—')}</span>
      <span className="text-text-tertiary">source: {d.source ?? '不明'}</span>
      <Badge tone="sky">{NOT_MODEL_INPUT}</Badge>
    </div>
  );
}

function Evaluators({ model, node, onSelect }: { model: EpisodeModel; node: TNode; onSelect: TabProps['onSelect'] }) {
  let list: TraceEvent[] = [];
  let blockEnd: TraceEvent | null = null;
  let scopeNote = '';
  if (node.kind === 'evaluator') list = [node.primary];
  else if (node.kind === 'code_block') {
    list = node.children.filter((c) => c.kind === 'evaluator').map((c) => c.primary);
    blockEnd = node.end;
  } else if (node.kind === 'api_call' || node.kind === 'stdout' || node.kind === 'stderr' || node.kind === 'service_call') {
    const b = enclosingBlock(model, node);
    if (b) {
      list = b.children.filter((c) => c.kind === 'evaluator').map((c) => c.primary);
      blockEnd = b.end;
      scopeNote = 'このイベントを含むコードブロックの評価（ブロック終了時の値）';
    }
  } else if (node.kind === 'episode_end' || node.kind === 'episode_start') {
    list = model.evaluators.filter((e) => dataOf(e, 'evaluator_result').scope === 'episode');
  }
  if (list.length === 0 && !blockEnd) return null;
  const be = blockEnd ? dataOf(blockEnd, 'code_block_end') : null;
  const b = node.kind === 'code_block' ? node : enclosingBlock(model, node);
  return (
    <div>
      <SectionTitle>評価器の結果（ハーネスが計算した値のみ）</SectionTitle>
      {scopeNote && (
        <div className="text-[11px] text-text-tertiary mb-1">
          {scopeNote}
          {b && node !== b && (
            <>
              {' '}
              <EventLink onClick={() => onSelect(b)}>ブロックを開く</EventLink>
            </>
          )}
        </div>
      )}
      <div className="space-y-1">
        {list.map((e) => (
          <EvaluatorRow key={e.event_id} ev={e} />
        ))}
        {be && (
          <div className="text-xs text-text-secondary">
            code_block_end: reward={fmtNum(be.reward)} · task_completed={String(be.task_completed ?? '—')} · terminated={String(be.terminated ?? '—')} · truncated=
            {String(be.truncated ?? '—')}
          </div>
        )}
      </div>
    </div>
  );
}

/** A string field that may be a text asset ({truncated, preview, length, asset}); loaded on demand. */
function isTextNode(v: unknown): v is TextValue {
  return !!v && typeof v === 'object' && !Array.isArray(v) && (v as { truncated?: unknown }).truncated === true && typeof (v as { preview?: unknown }).preview === 'string';
}

function LongText({ ep, value }: { ep: EpisodeRef; value: TextValue }) {
  const t = useFullText(ep, value);
  if (t.text === null) return <span className="text-text-tertiary">{LABELS.notRecorded}</span>;
  return (
    <div>
      <TextView text={t.text} foldLines={8} />
      {t.partial && <div className="text-[11px] text-amber-300 mt-1">{t.loading ? '全文を読み込み中…' : t.error ? `全文を読み込めません: ${t.error}` : '先頭のみ'}</div>}
    </div>
  );
}

/** Old success_meaning prefix used by traces that described success as code running without an exception. */
const OLD_SUCCESS_MEANING = 'code ran without exception';

function EpisodeFacts({ model, ep, summary }: { model: EpisodeModel; ep: EpisodeRef; summary: EpisodeSummary | null }) {
  const end = model.episodeEnd;
  const d = end ? dataOf(end, 'episode_end') : null;
  const hs = d?.harness_summary ?? null;
  // TrialSummary.success: the batch harness and the web UI runner set it differently (5.2).
  const source = (model.episodeStart ? dataOf(model.episodeStart, 'episode_start').source : null) ?? summary?.source ?? null;
  const successMeaning = source === 'web_ui' ? LABELS.webUiSuccessMeaning : LABELS.harnessSuccessMeaning;
  // Fields the recorder adds beyond 5.2 (web UI session state, overhead on all threads).
  const extra = (end?.data ?? {}) as { session_state?: unknown; tracer_overhead_all_threads_s?: unknown };
  return (
    <div>
      <SectionTitle>エピソード終了の記録</SectionTitle>
      {!d ? (
        <Notice>episode_end がありません。</Notice>
      ) : (
        <KV
          rows={[
            ['end_reason', d.end_reason ?? '—'],
            ...(typeof d.end_note === 'string' && d.end_note ? [['end_note', d.end_note] as [string, string]] : []),
            [
              '例外',
              d.exception
                ? `${d.exception.type ?? ''}: ${textOf(d.exception.message)}${d.exception.message_redacted === true ? `（${REDACTED_MESSAGE_NOTE}）` : ''}`
                : 'なし',
            ],
            ['alarm_seen', String(d.alarm_seen ?? '—')],
            ['duration', fmtDuration(d.duration_s ?? null)],
            ['tracer_overhead_s', fmtDuration(d.tracer_overhead_s ?? null)],
            ['trace_complete', String(d.trace_complete ?? '—')],
            ['gaps', (d.gaps ?? []).map((g) => `${g.code}×${g.count ?? '?'}（最初 seq ${g.first_seq ?? '?'}）`).join(', ') || 'なし'],
            ['dropped / late / warnings', `${d.dropped_events ?? 0} / ${d.late_events ?? 0} / ${d.n_warnings ?? 0}`],
            ...(typeof extra.session_state === 'string' ? [['Web UI セッションの状態', extra.session_state] as [string, string]] : []),
            ...(typeof extra.tracer_overhead_all_threads_s === 'number'
              ? [['記録時間（全スレッド）', fmtDuration(extra.tracer_overhead_all_threads_s)] as [string, string]]
              : []),
          ]}
        />
      )}
      {hs && (
        <>
          <SectionTitle>harness_summary（TrialSummary の値）</SectionTitle>
          <KV
            rows={Object.entries(hs).map(([k, v]) => [
              k,
              k === 'success' ? (
                `${String(v)} — ${successMeaning}。${LABELS.successIsNotTaskVerdict}`
              ) : k === 'success_meaning' && typeof v === 'string' && v.startsWith(OLD_SUCCESS_MEANING) ? (
                `${v}（${LABELS.oldSuccessMeaningNote}）`
              ) : isTextNode(v) ? (
                // the harness log and other strings over 4000 characters are text assets
                <LongText ep={ep} value={v} />
              ) : typeof v === 'string' && (v.length > 200 || v.includes('\n')) ? (
                // the harness log (summary.txt text) and other long strings: folded
                <TextView text={v} foldLines={8} />
              ) : typeof v === 'object' ? (
                JSON.stringify(v)
              ) : (
                String(v)
              ),
            ])}
          />
        </>
      )}
    </div>
  );
}

/**
 * What the recorder installed and saw for this episode (spec 5.2 episode_start.tracer and
 * env; episode.json apis_wrapped and hooks_installed_during_episode). In the web UI the
 * env is created inside the episode, so episode_start has `env: null` and episode.json
 * holds it; trial role hook, trial lifecycle hook and environment hook are installed during the episode there.
 */
function RecorderFacts({ model, summary }: { model: EpisodeModel; summary: EpisodeSummary | null }) {
  const start = model.episodeStart ? dataOf(model.episodeStart, 'episode_start') : null;
  const epJson = summary?.episode ?? null;
  const tracer = start?.tracer ?? null;
  const env = start?.env ?? epJson?.env ?? null;
  const envFrom = start?.env ? 'episode_start' : epJson?.env ? 'episode.json（env はエピソード中に作られたため）' : null;
  const during = Array.isArray(epJson?.hooks_installed_during_episode) ? epJson.hooks_installed_during_episode : [];
  const failed = Array.isArray(tracer?.hooks_failed) ? tracer.hooks_failed : [];
  // hook_install_failed warnings: a hook that failed while the episode was open
  // (during_episode), or the API wrappers of one instance (the trace recorder).
  const failedWarnings = model.warnings.map((w) => dataOf(w, 'trace_warning')).filter((d) => d.code === 'hook_install_failed');
  const wrapped = Array.isArray(epJson?.apis_wrapped) ? epJson.apis_wrapped : [];
  const settings = tracer?.settings ?? null;
  const startExtra = (model.episodeStart?.data ?? {}) as { frames?: unknown; output_dir?: unknown };
  const frames = (startExtra.frames ?? (env as { frames?: unknown } | null)?.frames ?? null) as {
    installed?: unknown;
    buffers?: unknown;
    cameras?: unknown;
    backfilled?: unknown;
  } | null;
  const rc = start?.runner_condition ?? null;
  const hookName = (h: unknown) =>
    typeof h === 'string'
      ? h
      : h && typeof h === 'object'
        ? `${String((h as { hook?: unknown }).hook ?? '?')} ${String((h as { target?: unknown }).target ?? '')} ${String((h as { reason?: unknown }).reason ?? '')}${
            (h as { reason_redacted?: unknown }).reason_redacted === true ? `（${REDACTED_SHORT}）` : ''
          }`
        : '?';
  if (!start && !epJson) return null;
  return (
    <div>
      <SectionTitle>記録器（トレーサー）の状態</SectionTitle>
      <KV
        rows={[
          ['env', env ? `${String(env.class ?? '?')}${env.low_level_class ? ` / ${String(env.low_level_class)}` : ''}` : start?.env_pending ? `未記録（${start.env_pending}）` : '未記録'],
          ['env の記録元', envFrom ?? '—'],
          ['API', env?.apis?.join(', ') ?? '—'],
          ['privileged', String(env?.privileged ?? '—')],
          ['viser_port', String(env?.viser_port ?? '—')],
          [
            'フレームの取得（H10, frame hook）',
            frames
              ? `${frames.installed === true ? '導入済み' : '未導入'} · カメラ ${Array.isArray(frames.cameras) ? frames.cameras.join(', ') : '?'} · バッファ ${
                  Array.isArray(frames.buffers) ? frames.buffers.join(', ') : '?'
                }${typeof frames.backfilled === 'number' && frames.backfilled > 0 ? ` · 開始前のフレーム ${frames.backfilled} 枚（sim_step 不明）` : ''}`
              : '—',
          ],
          [
            'runner の条件（表示のみ）',
            rc
              ? Object.entries(rc)
                  .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
                  .join(' · ')
              : '—',
          ],
          ['output_dir', typeof startExtra.output_dir === 'string' ? startExtra.output_dir : '—'],
          ['tracer', `${tracer?.version ?? '?'}${typeof epJson?.tracer_overhead_s === 'number' ? ` · 試行スレッド上の記録時間 ${fmtDuration(epJson.tracer_overhead_s)}` : ''}`],
          [
            '設定',
            settings
              ? Object.entries(settings)
                  .filter(([k]) => k !== 'trace_dir')
                  .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
                  .join(' · ')
              : '—',
          ],
        ]}
      />
      <div className="mt-2 space-y-1 text-xs">
        <div className="text-text-tertiary">エピソード開始時に導入済みのフック（{(tracer?.hooks_installed ?? []).length}）</div>
        <div className="font-mono text-[11px] text-text-secondary break-all">{(tracer?.hooks_installed ?? []).join(' · ') || '—'}</div>
        {during.length > 0 && (
          <>
            <div className="text-text-tertiary">エピソード中に導入されたフック（{during.length}）: 対象のモジュールがエピソード中に初めて読み込まれたもの（Web UI など）</div>
            <div className="font-mono text-[11px] text-text-secondary break-all">{during.join(' · ')}</div>
          </>
        )}
        {(failed.length > 0 || failedWarnings.length > 0) && (
          <Notice tone="red">
            導入に失敗したフック:{' '}
            {[
              ...failed.map(hookName),
              ...failedWarnings.map((d) => `${d.hook ?? '?'} ${d.target ?? ''} ${textOf(d.message)}${d.during_episode ? '（エピソード中）' : ''}`),
            ].join(' / ')}
            。この記録は完全ではありません。
          </Notice>
        )}
      </div>
      {wrapped.length > 0 && (
        <div className="mt-2 space-y-1 text-xs">
          <div className="text-text-tertiary">API のラップ（H8, H9）</div>
          {wrapped.map((w, i) => (
            <div key={i} className="font-mono text-[11px] text-text-secondary">
              <span className="text-sky-300">{String(w.class ?? w.api ?? '?')}</span>: {(w.wrapped ?? []).join(', ') || '—'}
              {(w.services ?? []).length > 0 ? ` · サービス ${(w.services ?? []).join(', ')}` : ''}
              {w.installed_in_this_episode === false ? (
                <span className="ml-2 text-text-tertiary">（前のエピソードで導入済み。同じ env を再利用）</span>
              ) : w.installed_in_this_episode === true ? (
                <span className="ml-2 text-text-tertiary">（このエピソードで導入）</span>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Perception inputs and outputs
// ---------------------------------------------------------------------------

function isImage(a: Artifact): boolean {
  return (a.mime ?? '').startsWith('image/') || /\.(png|jpe?g|webp)$/i.test(a.path ?? '');
}

function isArrayFile(a: Artifact): boolean {
  return /\.(npz|npy)$/i.test(a.path ?? '') || a.kind === 'npz' || a.kind === 'npy';
}

/**
 * An npz of perception outputs. /array sends shapes and previews only (its `stats` are null);
 * the statistics shown are the ones the recorder saved in the service call's response
 * summary (`recordedStats`).
 */
function NpzArtifact({ ep, path, recordedStats }: { ep: EpisodeRef; path: string; recordedStats: Record<string, ArrayStats> }) {
  // Keyed by the generation too: a replaced trace reads its own file.
  const res = useAsync((signal) => traceApi.array(ep.runId, ep.episodeId, path, undefined, signal), [ep.runId, ep.episodeId, ep.generation ?? null, path]);
  if (res.loading) return <Spinner label="配列を読み込み中" />;
  if (res.error) return <ErrorText>{res.error}</ErrorText>;
  if (!res.data) return null;
  return <ArrayPreviewList arrays={res.data.arrays ?? {}} ep={ep} path={path} recordedStats={recordedStats} />;
}

function Perception({
  ep,
  node,
  assetStates,
  onRetryAsset,
}: {
  ep: EpisodeRef;
  node: TNode;
  assetStates: ReadonlyMap<string, DataAssetState>;
  onRetryAsset?: (ev: TraceEvent) => void;
}) {
  const services = node.kind === 'service_call' ? [node] : descendants(node).filter((c) => c.kind === 'service_call');
  const items = services.flatMap((s) =>
    s.events.flatMap((e) => (e.artifacts ?? []).filter((a) => a.label === 'perception_input' || a.label === 'perception_output').map((a) => ({ s, a }))),
  );
  if (services.length === 0) return null;
  return (
    <div>
      <SectionTitle>知覚サービスの入力と出力</SectionTitle>
      {items.length === 0 && <div className="text-xs text-text-tertiary">保存された入出力はありません（{services.length} 件の呼び出し）</div>}
      <div className="space-y-2">
        {items.map(({ s, a }, i) => {
          const sd = dataOf(s.primary, 'service_call_start');
          const label = a.label === 'perception_input' ? '入力（サービスが受け取ったデータ）' : '出力（デコード済み）';
          // The recorded statistics are in the service events' summaries (possibly moved to an asset).
          const unloaded = isArrayFile(a) ? s.events.filter((e) => pendingAsset(e, assetStates) !== null) : [];
          return (
            <div key={`${a.path}-${i}`} className="rounded-md border border-surface-border bg-surface-raised p-2">
              <div className="flex flex-wrap items-center gap-2 text-xs mb-1">
                <Badge tone="violet">
                  {sd.service ?? '?'}.{sd.function ?? ''}
                </Badge>
                <span className="text-text-secondary">{label}</span>
                <span className="font-mono text-[11px] text-text-tertiary">{a.path}</span>
              </div>
              {a.written === false || !a.path ? (
                <div className="text-xs text-amber-300">
                  {LABELS.notCaptured}: {a.reason ?? '保存されていません'}
                </div>
              ) : isImage(a) ? (
                <TraceImage src={episodeFileUrl(ep, a.path)} alt={a.path} provenance="perception" className="h-48 w-72" />
              ) : isArrayFile(a) ? (
                <>
                  {unloaded.map((e) => (
                    <div key={e.event_id} className="text-[11px] mb-1">
                      記録時の統計（#{e.seq} {e.event_type}）: <DataAssetInline ev={e} states={assetStates} onRetry={onRetryAsset} />
                    </div>
                  ))}
                  <NpzArtifact ep={ep} path={a.path} recordedStats={recordedNpzStats(s.events.map((e) => e.data), a.path)} />
                </>
              ) : (
                <a href={episodeFileUrl(ep, a.path)} target="_blank" rel="noreferrer" className="text-xs text-accent underline">
                  ファイルを開く
                </a>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model calls: where the sent images came from (exact hash matches only)
// ---------------------------------------------------------------------------

function ImageSources({ model, node, selectionAttempt }: { model: EpisodeModel; node: TNode; selectionAttempt: TraceEvent | null }) {
  const call = node.call!;
  const req = selectionAttempt ?? mainAttempt(call)?.request ?? null;
  const images = req ? dataOf(req, 'model_request').images ?? [] : [];
  if (images.length === 0) return null;
  return (
    <div>
      <SectionTitle>送信画像の由来（完全一致したものだけ）</SectionTitle>
      <div className="space-y-1 text-xs font-mono">
        {images.map((im, i) => (
          <div key={i} className="text-text-secondary">
            #{im.order ?? i} {im.path}: {imageSourceText(model, im, req?.seq ?? null)}
          </div>
        ))}
      </div>
    </div>
  );
}
