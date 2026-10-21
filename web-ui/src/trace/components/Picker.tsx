/**
 * Run and episode picker (docs/trace-format.md 9.3): runs newest first, then
 * the episodes of a run with status badges (attempts of one seed next to each
 * other). Traced runs come from /api/traces/runs; runner directories written
 * before the recorder existed are listed separately under
 * 「トレースなし（既存ログ）」 (9.4) from /api/traces/legacy/runs. A run's
 * unscoped logs (events with no episode, such as the runner's preflight
 * requests) are listed under the episodes and open like an episode; lines of
 * such a log that the recorder could not write (its unscoped.<pid>.status.json,
 * backend `write_status`) are named next to it, also when the log itself is
 * missing. An episode whose episode.json could not be used shows why.
 *
 * Related: TraceViewerPage.tsx (URL parameters), LegacyView.tsx.
 */

import { useMemo, useState } from 'react';
import { traceApi } from '../api';
import { fmtBytes, fmtWall } from '../format';
import { useAsync } from '../hooks';
import { LABELS, UNSCOPED_STATUS_LABEL, episodeStatusLabel, episodeStatusShortLabel } from '../labels';
import type { EpisodeSummary, TraceConfigResponse, UnscopedFileInfo } from '../types';
import { Badge, ErrorText, Notice, Spinner, SmallButton } from './common';
import { episodeStatusTone } from './TopBar';
import { episodeVerdict, VERDICT_LABELS, type TaskVerdict } from '../outcome';
import { TaskOutcomeBadge } from './TaskOutcomeBadge';
import { legacyRootsOf, legacyRunsOf, legacyTrialsOf, type LegacyRunRef, type LegacyTrialRef } from './LegacyView';

interface PickerProps {
  config: TraceConfigResponse | null;
  configError: string | null;
  currentRun: string | null;
  currentEpisode: string | null;
  onPick: (runId: string, episodeId: string) => void;
  onPickLegacy: (runId: string, trial: LegacyTrialRef) => void;
  onClose?: () => void;
}

export function Picker({ config, configError, currentRun, currentEpisode, onPick, onPickLegacy, onClose }: PickerProps) {
  const [mode, setMode] = useState<'traced' | 'legacy'>('traced');
  const [runSel, setRunSel] = useState<string | null>(currentRun);
  const [filter, setFilter] = useState('');
  const [resultFilter, setResultFilter] = useState<TaskVerdict | 'all'>('all');
  const runs = useAsync((s) => traceApi.runs(s), []);
  const detail = useAsync(runSel && mode === 'traced' ? (s) => traceApi.run(runSel, s) : null, [runSel, mode]);

  const episodes = useMemo(() => {
    const list = [...(detail.data?.episodes ?? [])].filter(episode => resultFilter === 'all' || episodeVerdict(episode, detail.data?.runner_records ?? []) === resultFilter);
    const q = filter.trim().toLowerCase();
    const filtered = q
      ? list.filter((e) => [e.episode_id, e.task, e.condition, e.condition_label, String(e.seed ?? '')].some((x) => (x ?? '').toString().toLowerCase().includes(q)))
      : list;
    filtered.sort((a, b) => {
      const k = (e: EpisodeSummary) => [e.task ?? '', e.condition ?? '', String(e.seed ?? '').padStart(8, '0'), String(e.attempt_index ?? '').padStart(4, '0'), e.episode_id].join('|');
      return k(a) < k(b) ? -1 : k(a) > k(b) ? 1 : 0;
    });
    return filtered;
  }, [detail.data, filter, resultFilter]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-surface">
      <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 border-b border-surface-border bg-surface-raised">
        <SmallButton active={mode === 'traced'} onClick={() => setMode('traced')}>
          トレースあり
        </SmallButton>
        <SmallButton active={mode === 'legacy'} onClick={() => setMode('legacy')}>
          {LABELS.legacySection}
        </SmallButton>
        <span className="text-[11px] text-text-tertiary font-mono truncate">
          trace root: {config?.trace_root ?? '—'}
          {config && config.exists === false ? '（存在しません）' : ''}
          {config?.viewer_only ? ' · viewer-only' : ''}
        </span>
        {configError && <ErrorText>{configError}</ErrorText>}
        <div className="ml-auto flex gap-2">
          <SmallButton onClick={runs.reload}>一覧を更新</SmallButton>
          {onClose && <SmallButton onClick={onClose}>閉じる</SmallButton>}
        </div>
      </div>
      {mode === 'legacy' ? (
        <LegacyPicker config={config} onPickLegacy={onPickLegacy} />
      ) : (
        <div className="flex-1 min-h-0 flex">
          <div className="w-80 flex-shrink-0 border-r border-surface-border overflow-y-auto">
            <div className="px-3 py-2 text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">run（新しい順）</div>
            {runs.loading && !runs.data && (
              <div className="px-3">
                <Spinner label="読み込み中" />
              </div>
            )}
            {runs.error && (
              <div className="px-3">
                <ErrorText>{runs.error}</ErrorText>
              </div>
            )}
            {runs.data && runs.data.runs.length === 0 && <div className="px-3 text-xs text-text-tertiary">トレースのある run がありません</div>}
            {runs.data?.runs.map((r) => (
              <button
                key={r.run_id}
                type="button"
                onClick={() => setRunSel(r.run_id)}
                className={`w-full text-left px-3 py-2 border-b border-surface-border/60 hover:bg-surface-overlay ${runSel === r.run_id ? 'bg-accent/10 border-l-2 border-l-accent' : ''}`}
              >
                <div className="text-xs font-mono text-text-primary break-all">{r.run_id}</div>
                <div className="flex flex-wrap items-center gap-1 mt-1 text-[11px] text-text-tertiary">
                  <span>エピソード {r.n_episodes ?? '?'}</span>
                  <span>· {fmtWall(r.last_modified ?? null)}</span>
                  {Object.entries(r.status_counts ?? {})
                    .filter(([, n]) => n > 0)
                    .map(([st, n]) => (
                      <Badge key={st} tone={episodeStatusTone(st)} title={episodeStatusLabel(st)}>
                        {episodeStatusShortLabel(st)} {n}
                      </Badge>
                    ))}
                </div>
              </button>
            ))}
          </div>
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex-shrink-0 flex items-center gap-2 px-3 py-2 border-b border-surface-border">
              <span className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">episode</span>
              <select aria-label="結果で絞り込み" value={resultFilter} onChange={event => setResultFilter(event.target.value as TaskVerdict | 'all')} className="rounded border border-surface-border bg-surface-sunken p-1 text-xs">
                <option value="all">すべての結果</option>{Object.entries(VERDICT_LABELS).map(([value,label]) => <option key={value} value={value}>{label}</option>)}
              </select>
              <input
                type="search"
                name="trace-episode-filter"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="絞り込み（task / condition / seed / id）"
                className="flex-1 px-2 py-1 bg-surface-sunken border border-surface-border rounded text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40"
              />
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto">
              {!runSel && <div className="p-4 text-xs text-text-tertiary">左の run を選んでください</div>}
              {detail.loading && !detail.data && (
                <div className="p-4">
                  <Spinner label="読み込み中" />
                </div>
              )}
              {detail.error && (
                <div className="p-4">
                  <ErrorText>{detail.error}</ErrorText>
                </div>
              )}
              {detail.data?.run_json_error && (
                <div className="px-3 pt-2">
                  <Notice tone="amber" className="text-xs">
                    この run の run.json を読めません（{detail.data.run_json_error}）。run の設定情報は表示できません。
                  </Notice>
                </div>
              )}
              {detail.data && (
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-surface-raised">
                    <tr className="text-left text-text-tertiary border-b border-surface-border">
                      <th className="py-1.5 px-3 font-normal">結果</th>
                      <th className="py-1.5 pr-2 font-normal">記録状態</th>
                      <th className="py-1.5 pr-2 font-normal">task</th>
                      <th className="py-1.5 pr-2 font-normal">condition</th>
                      <th className="py-1.5 pr-2 font-normal">seed</th>
                      <th className="py-1.5 pr-2 font-normal">attempt</th>
                      <th className="py-1.5 pr-2 font-normal">end_reason</th>
                      <th className="py-1.5 pr-2 font-normal">episode_id</th>
                    </tr>
                  </thead>
                  <tbody>
                    {episodes.map((e) => (
                      <tr
                        key={e.episode_id}
                        tabIndex={0}
                        role="button"
                        aria-label={`episode ${e.episode_id}`}
                        onClick={() => runSel && onPick(runSel, e.episode_id)}
                        onKeyDown={(ev) => {
                          if ((ev.key === 'Enter' || ev.key === ' ') && runSel) {
                            ev.preventDefault();
                            onPick(runSel, e.episode_id);
                          }
                        }}
                        className={`cursor-pointer border-b border-surface-border/50 hover:bg-surface-overlay focus:outline-none focus:bg-surface-overlay ${
                          runSel === currentRun && e.episode_id === currentEpisode ? 'bg-accent/10' : ''
                        }`}
                      >
                        <td className="py-1.5 px-3">
                          <TaskOutcomeBadge verdict={episodeVerdict(e, detail.data?.runner_records ?? [])} />
                        </td>
                        <td className="py-1.5 pr-2">
                          <EpisodeStatusCell episode={e} />
                        </td>
                        <td className="py-1.5 pr-2">{e.task ?? '—'}</td>
                        <td className="py-1.5 pr-2">
                          {e.condition ?? '—'}
                          {e.condition_label && e.condition_label !== e.condition ? <span className="text-text-tertiary">（{e.condition_label}）</span> : null}
                        </td>
                        <td className="py-1.5 pr-2 font-mono">{e.seed ?? '—'}</td>
                        <td className="py-1.5 pr-2 font-mono">{e.attempt_index ?? '—'}</td>
                        <td className="py-1.5 pr-2">{e.end_reason ?? '—'}</td>
                        <td className="py-1.5 pr-2 font-mono text-[11px] text-text-tertiary break-all">{e.episode_id}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {detail.data && episodes.length === 0 && <div className="p-4 text-xs text-text-tertiary">該当するエピソードがありません</div>}
              {detail.data && (detail.data.unscoped_files ?? []).length > 0 && (
                <UnscopedFileList
                  files={detail.data.unscoped_files ?? []}
                  runId={runSel}
                  currentRun={currentRun}
                  currentEpisode={currentEpisode}
                  onPick={onPick}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * An episode's status badge in the picker, with the reason when its episode.json could not be
 * used (`episode_json_error` other than "missing": unreadable, a symlink, invalid JSON, ...).
 */
export function EpisodeStatusCell({ episode: e }: { episode: EpisodeSummary }) {
  return (
    <>
      <Badge tone={episodeStatusTone(e.status)}>{episodeStatusLabel(e.status)}</Badge>
      {typeof e.episode_json_error === 'string' && e.episode_json_error !== 'missing' && (
        <Badge tone="amber" title={`episode.json を読めません（${e.episode_json_error}）`} className="ml-1">
          episode.json: {e.episode_json_error}
        </Badge>
      )}
    </>
  );
}

/**
 * The run's unscoped logs, each with a button that opens it, and what the recorder could not
 * write to it (unscoped.<pid>.status.json: `write_status`, or `write_status_error` when that
 * file could not be read). A log that exists only as its status file (`bytes` null) is listed
 * too, so its lost lines are never hidden by the missing log.
 */
export function UnscopedFileList({
  files,
  runId,
  currentRun,
  currentEpisode,
  onPick,
}: {
  files: UnscopedFileInfo[];
  runId: string | null;
  currentRun: string | null;
  currentEpisode: string | null;
  onPick: (runId: string, episodeId: string) => void;
}) {
  return (
    <div className="px-3 py-2 text-[11px] text-text-tertiary border-t border-surface-border space-y-1">
      <div>{UNSCOPED_STATUS_LABEL}: runner の preflight 要求や、所属するエピソードを決められなかったイベントです。</div>
      <div className="flex flex-wrap gap-2">
        {files.map((f, i) => {
          const size = f.bytes === null || f.bytes === undefined ? 'ファイルなし' : fmtBytes(f.bytes);
          const ws = f.write_status;
          const lost = ws && typeof ws.lines_not_written === 'number' ? ws.lines_not_written : null;
          const status =
            f.write_status_error ? (
              <span className="text-amber-300">書き込み状況ファイル（status.json）を読めません（{f.write_status_error}）</span>
            ) : ws ? (
              <span className="text-amber-300">
                書き込めなかった行 {lost ?? '?'} 件{ws.last_reason ? `（最後の理由: ${ws.last_reason}）` : ''}
              </span>
            ) : null;
          return (
            <span key={f.file ?? f.log_id ?? i} className="inline-flex flex-wrap items-center gap-1">
              {f.log_id && runId ? (
                <button
                  type="button"
                  onClick={() => onPick(runId, f.log_id!)}
                  className={`px-2 py-0.5 rounded border font-mono hover:border-accent/50 hover:text-accent ${
                    runId === currentRun && f.log_id === currentEpisode ? 'border-accent/60 text-accent' : 'border-surface-border text-text-secondary'
                  }`}
                >
                  {f.file}（{size}）を開く
                </button>
              ) : (
                <span className="font-mono">
                  {f.file ?? '?'}（{size}）
                </span>
              )}
              {status}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function LegacyPicker({ onPickLegacy }: { config: TraceConfigResponse | null; onPickLegacy: (runId: string, trial: LegacyTrialRef) => void }) {
  const [runSel, setRunSel] = useState<LegacyRunRef | null>(null);
  const [filter, setFilter] = useState('');
  const runs = useAsync((s) => traceApi.legacyRuns(s), []);
  const detail = useAsync(runSel ? (s) => traceApi.legacyRun(runSel.runId, runSel.root, s) : null, [runSel?.runId, runSel?.root]);
  const runList = useMemo(() => legacyRunsOf(runs.data), [runs.data]);
  const roots = useMemo(() => legacyRootsOf(runs.data), [runs.data]);
  const trials = useMemo(() => {
    const all = legacyTrialsOf(detail.data, runSel?.root ?? null);
    const q = filter.trim().toLowerCase();
    return q ? all.filter((t) => `${t.task} ${t.condition} ${t.trialDir}`.toLowerCase().includes(q)) : all;
  }, [detail.data, filter, runSel]);
  const isSel = (r: LegacyRunRef) => !!runSel && runSel.runId === r.runId && runSel.root === r.root;
  return (
    <div className="flex-1 min-h-0 flex">
      <div className="w-80 flex-shrink-0 border-r border-surface-border overflow-y-auto">
        <div className="px-3 py-2 text-[11px] text-text-tertiary">
          {LABELS.legacySection}: 記録器が無かった頃の runner 出力です。モデル入力は{LABELS.notRecorded}です。
          <div className="font-mono mt-1 space-y-0.5">
            {roots.map((r) => (
              <div key={r.index}>
                root {r.index}: {r.path}
                {r.exists ? '' : '（存在しません）'}
              </div>
            ))}
          </div>
        </div>
        {runs.loading && !runs.data && (
          <div className="px-3">
            <Spinner />
          </div>
        )}
        {runs.error && (
          <div className="px-3">
            <ErrorText>{runs.error}</ErrorText>
          </div>
        )}
        {runs.data !== null && runList.length === 0 && <div className="px-3 text-xs text-text-tertiary">runner の run がありません</div>}
        {runList.map((r) => (
          <button
            key={`${r.root}:${r.runId}`}
            type="button"
            onClick={() => setRunSel(r)}
            className={`w-full text-left px-3 py-2 border-b border-surface-border/60 hover:bg-surface-overlay ${isSel(r) ? 'bg-accent/10 border-l-2 border-l-accent' : ''}`}
          >
            <div className="text-xs font-mono text-text-primary break-all">{r.runId}</div>
            <div className="text-[11px] text-text-tertiary">
              {r.root !== null && roots.length > 1 ? `root ${r.root} · ` : ''}
              {r.detail}
            </div>
          </button>
        ))}
      </div>
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-shrink-0 px-3 py-2 border-b border-surface-border">
          <input
            type="search"
            name="trace-legacy-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="絞り込み（task / condition / trial）"
            className="w-full px-2 py-1 bg-surface-sunken border border-surface-border rounded text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40"
          />
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto">
          {!runSel && <div className="p-4 text-xs text-text-tertiary">左の run を選んでください</div>}
          {detail.loading && !detail.data && (
            <div className="p-4">
              <Spinner />
            </div>
          )}
          {detail.error && (
            <div className="p-4">
              <ErrorText>{detail.error}</ErrorText>
            </div>
          )}
          {runSel &&
            trials.map((t) => (
              <button
                key={`${t.task}/${t.condition}/${t.trialDir}`}
                type="button"
                onClick={() => onPickLegacy(runSel.runId, t)}
                className="w-full text-left px-3 py-1.5 border-b border-surface-border/50 hover:bg-surface-overlay text-xs"
              >
                <span className="mr-2"><TaskOutcomeBadge verdict={t.taskVerdict ?? 'unknown'} /></span>
                <span className="text-text-primary">{t.task}</span>
                <span className="mx-1 text-text-tertiary">/</span>
                <span className="text-text-primary">{t.condition}</span>
                <span className="mx-1 text-text-tertiary">/</span>
                <span className="font-mono text-text-secondary">{t.trialDir}</span>
                {t.note && <span className="ml-2 text-text-tertiary">{t.note}</span>}
              </button>
            ))}
          {runSel && detail.data !== null && trials.length === 0 && <div className="p-4 text-xs text-text-tertiary">trial がありません</div>}
        </div>
      </div>
    </div>
  );
}
