/** Simple episode picker and execution-by-execution viewer. */
import { useMemo, useState } from 'react';
import { traceApi, type EpisodeRef } from '../api';
import { useAsync } from '../hooks';
import type { EpisodeModel, Selection, TNode } from '../model';
import type { TrialVideo } from '../playback';
import { SmallButton } from './common';
import { episodeVerdict, VERDICT_LABELS, type TaskVerdict } from '../outcome';
import { TaskOutcomeBadge } from './TaskOutcomeBadge';
import { ExecutionViewer } from './ExecutionViewer';

export function SimpleViewer(props: {
  model: EpisodeModel; ep: EpisodeRef; selection: Selection | null; videos: TrialVideo[];
  onSelect: (node:TNode) => void; error: string | null;
}) { return <ExecutionViewer {...props} />; }

export function SimplePicker({onPick, onDetails}: {onPick:(run:string,episode:string)=>void; onDetails:()=>void}) {
  const runs = useAsync(s => traceApi.runs(s), []);
  const [resultFilter, setResultFilter] = useState<TaskVerdict | 'all'>('all');
  const [run, setRun] = useState<string|null>(null);
  const detail = useAsync(run ? s => traceApi.run(run,s) : null, [run]);
  const rows = useMemo(() => (detail.data?.episodes ?? []).map(episode => ({episode, verdict: episodeVerdict(episode, detail.data?.runner_records ?? [])})), [detail.data]);
  const counts = rows.reduce((total, row) => ({...total, [row.verdict]: total[row.verdict] + 1}), {success:0, failure:0, unknown:0, running:0});
  return <div className="h-full overflow-auto p-6 max-w-4xl mx-auto">
    <div className="flex items-center justify-between mb-4"><h1 className="text-xl font-semibold">見る試行を選ぶ</h1><SmallButton onClick={runs.reload}>一覧を更新</SmallButton></div>
    <p className="text-sm text-text-tertiary mb-5">実験を選び、その中の試行を開いてください。</p>
    <label className="text-sm block mb-5">実験
      <select aria-label="実験を選ぶ" className="mt-2 block w-full bg-surface-raised border border-surface-border rounded-lg p-3" value={run ?? ''} onChange={e=>setRun(e.target.value||null)}>
        <option value="">選択してください</option>{runs.data?.runs.map(r=><option key={r.run_id} value={r.run_id}>{r.run_id}（{r.n_episodes ?? '?'}試行）</option>)}
      </select>
    </label>
    {(runs.error || detail.error) && <p className="text-amber-200">一覧を読み込めません。再読み込みしてください。</p>}
    {(runs.loading || detail.loading) && <p className="text-text-tertiary">読み込み中…</p>}
    {runs.data?.runs.length === 0 && <p className="text-text-tertiary">記録された試行はまだありません。</p>}
    {detail.data && <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
      <span>成功 {counts.success} · 失敗 {counts.failure} · 未判定 {counts.unknown} · 実行中 {counts.running}</span>
      <label className="ml-auto">結果で絞り込み <select aria-label="結果で絞り込み" className="ml-2 rounded border border-surface-border bg-surface-raised p-2" value={resultFilter} onChange={event => setResultFilter(event.target.value as TaskVerdict | 'all')}>
        <option value="all">すべて</option>{Object.entries(VERDICT_LABELS).map(([value,label]) => <option key={value} value={value}>{label}</option>)}
      </select></label>
    </div>}
    <div className="space-y-2">{rows.filter(row => resultFilter === 'all' || row.verdict === resultFilter).map(({episode:e, verdict})=><button key={e.episode_id} data-task-outcome={verdict} onClick={()=>run&&onPick(run,e.episode_id)} className="flex w-full flex-wrap items-center justify-between gap-3 rounded-lg border border-surface-border bg-surface-raised hover:border-accent p-4 text-left">
      <span className="min-w-0"><span className="font-semibold">{e.task ?? '試行'} · {e.condition ?? ''}</span><span className="ml-3 text-sm text-text-tertiary">seed {e.seed ?? '?'} · 試行 {e.attempt_index ?? '?'}</span></span>
      <span className="ml-auto flex shrink-0 items-center gap-4"><span aria-label={`試行の結果：${VERDICT_LABELS[verdict]}`} className="flex items-center gap-2"><span className="text-sm text-text-tertiary">結果</span><TaskOutcomeBadge verdict={verdict} /></span><span className="text-accent">開く →</span></span>
    </button>)}</div>
    <button onClick={onDetails} className="mt-6 text-sm text-text-tertiary underline">過去のログ・詳しい情報を見る</button>
  </div>;
}
