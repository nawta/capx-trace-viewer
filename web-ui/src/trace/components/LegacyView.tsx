/**
 * Episodes without a trace (docs/trace-format.md 9.4): runner trial
 * directories written before the recorder existed, read through the legacy
 * endpoints of capx_trace_viewer/trace_api.py. Shows code.py, stdout/stderr parsed
 * from summary.txt, the videos, visual_feedback_*.png labelled
 * 「ハーネス保存画像（送信を検証していない）」 and trial_record.json.
 * Some trial directories also store sent codegen/VDM messages and state logs in
 * the model-input tab. Older trials still show 「未記録」.
 *
 * The same run id can exist under two runner roots, so every call carries the
 * root index that /legacy/runs reported.
 *
 * Related: Picker.tsx, api.ts (legacy endpoints).
 */

import { useState, type ReactNode } from 'react';
import { fetchText, traceApi } from '../api';
import { fmtBytes, fmtNum, fmtWall } from '../format';
import { useAsync } from '../hooks';
import { LABELS } from '../labels';
import { Badge, DebugBanner, ErrorText, KV, Notice, NotRecorded, SectionTitle, Spinner } from './common';
import { CodeView } from './CodeView';
import { TextView } from './TextView';
import { TraceImage } from './TraceImage';
import { RawJson } from './tabs/ModelInputTab';

export interface LegacyTrialRef {
  root: number | null;
  task: string;
  condition: string;
  trialDir: string;
  note?: string;
  taskVerdict?: 'success' | 'failure' | 'unknown';
}

export interface LegacyRunRef {
  runId: string;
  root: number | null;
  detail: string;
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string | null => (typeof v === 'string' ? v : typeof v === 'number' ? String(v) : null);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Runs of GET /legacy/runs: {roots, runs: [{run_id, root_index, n_trials, description, last_modified}]}. */
export function legacyRunsOf(data: unknown): LegacyRunRef[] {
  const list = isObj(data) && Array.isArray(data.runs) ? data.runs : [];
  const out: LegacyRunRef[] = [];
  for (const r of list) {
    if (!isObj(r)) continue;
    const id = str(r.run_id);
    if (!id) continue;
    const lm = num(r.last_modified);
    const parts = [
      r.n_trials !== undefined ? `${String(r.n_trials)} trials` : null,
      str(r.description),
      lm !== null ? fmtWall(lm) : null,
    ].filter(Boolean);
    out.push({ runId: id, root: num(r.root_index), detail: parts.join(' · ') });
  }
  return out;
}

/** Roots of GET /legacy/runs. */
export function legacyRootsOf(data: unknown): { index: number; path: string; exists: boolean }[] {
  const list = isObj(data) && Array.isArray(data.roots) ? data.roots : [];
  return list.filter(isObj).map((r) => ({ index: num(r.index) ?? -1, path: str(r.path) ?? '?', exists: r.exists === true }));
}

/** Trials of GET /legacy/runs/{run_id}: {trials: [{task, condition, trial_dir, sandbox_rc, reward, task_completed, record}]}. */
export function legacyTrialsOf(data: unknown, root: number | null): LegacyTrialRef[] {
  if (!isObj(data) || !Array.isArray(data.trials)) return [];
  const out: LegacyTrialRef[] = [];
  for (const t of data.trials) {
    if (!isObj(t)) continue;
    const task = str(t.task);
    const condition = str(t.condition);
    const dir = str(t.trial_dir);
    if (!task || !condition || !dir) continue;
    const rec = isObj(t.record) ? t.record : null;
    const outcome = rec && isObj(rec.outcome) ? rec.outcome : null;
    const note = [
      rec && rec.seed !== undefined ? `seed ${String(rec.seed)}` : null,
      outcome && outcome.task_success !== undefined ? `task_success=${String(outcome.task_success)}` : null,
      outcome && typeof outcome.termination_reason === 'string' ? outcome.termination_reason : null,
      rec && Array.isArray(rec.attempt_statuses) ? `attempts: ${(rec.attempt_statuses as unknown[]).join(',')}` : null,
      !rec ? 'runner の記録なし' : null,
    ]
      .filter(Boolean)
      .join(' · ');
    const success = outcome ? outcome.task_success : t.task_completed;
    const taskVerdict = success === true || success === 1 ? 'success' : success === false || success === 0 ? 'failure' : 'unknown';
    out.push({ taskVerdict, root: num(data.root_index) ?? root, task, condition, trialDir: dir, note: note || undefined });
  }
  return out;
}

interface FileRef {
  path: string;
  bytes: number | null;
  mime: string | null;
}

function filesOf(v: unknown): FileRef[] {
  if (!Array.isArray(v)) return [];
  return v.filter(isObj).map((f) => ({ path: str(f.path) ?? '', bytes: num(f.bytes), mime: str(f.mime) })).filter((f) => f.path);
}

export function LegacyPromptNotice({ modelInputRecorded }: { modelInputRecorded: boolean }) {
  return (
    <Notice tone="amber">
      以下はハーネスが保存した文字列です。送信した payload とは区別してください。
      {modelInputRecorded && '送信した payload は上の messages.json に記録されています。'}
    </Notice>
  );
}

export function LegacyTrialView({ runId, trial }: { runId: string; trial: LegacyTrialRef }) {
  const res = useAsync(
    (s) => traceApi.legacyTrial(runId, trial.task, trial.condition, trial.trialDir, trial.root, s),
    [runId, trial.task, trial.condition, trial.trialDir, trial.root],
  );
  const [tab, setTab] = useState(0);

  if (res.loading && !res.data) {
    return (
      <div className="p-6">
        <Spinner label="読み込み中" />
      </div>
    );
  }
  if (res.error) {
    return (
      <div className="p-6">
        <ErrorText>{res.error}</ErrorText>
      </div>
    );
  }
  const data = isObj(res.data) ? res.data : null;
  if (!data) return null;

  const url = (rel: string) => traceApi.legacyFileUrl(runId, trial.task, trial.condition, trial.trialDir, rel, trial.root);
  const code = isObj(data.code) ? data.code : null;
  const summary = isObj(data.summary) ? data.summary : null;
  const videos = filesOf(data.videos);
  const images = filesOf(data.harness_images);
  const prompts = filesOf(data.harness_prompts);
  const other = filesOf(data.other_files);
  const stateFiles = other.filter((f) => /^(state_snapshots\.jsonl|api_events\.jsonl|audit_events\.jsonl|state_prompt_usage\.jsonl|(?:codegen|vdm)_turn_\d+_(?:messages|images)\.json)$/.test(f.path));

  let body: ReactNode = null;
  if (tab === 0) {
    body = (
      <div className="space-y-3">
        {data.model_input_recorded === true ? (
          <>
            <SectionTitle>送信したメッセージと STATE</SectionTitle>
            {stateFiles.map((p) => <LazyText key={p.path} title={`${p.path}（${fmtBytes(p.bytes)}）`} url={url(p.path)} />)}
          </>
        ) : <NotRecorded reason="この trial にはトレースがありません（記録器の導入前）。送信した payload は記録されていません。" />}
        <SectionTitle>{LABELS.legacyHarnessPrompts}</SectionTitle>
        <LegacyPromptNotice modelInputRecorded={data.model_input_recorded === true} />
        {prompts.length === 0 && <div className="text-xs text-text-tertiary">保存されたプロンプト文字列はありません</div>}
        {prompts.map((p) => (
          <LazyText key={p.path} title={`${p.path}（${fmtBytes(p.bytes)}）`} url={url(p.path)} />
        ))}
      </div>
    );
  } else if (tab === 1) {
    body = code && typeof code.text === 'string' ? (
      <div className="space-y-1">
        <div className="text-[11px] font-mono text-text-tertiary">code.py{code.truncated ? '（先頭のみ）' : ''}</div>
        <CodeView code={code.text} />
      </div>
    ) : (
      <NotRecorded reason="code.py がありません" />
    );
  } else if (tab === 2) {
    body = (
      <div className="space-y-3">
        <div className="text-[11px] text-text-tertiary">
          summary.txt から読み取った値です（最後のブロックの stdout と、ターンをまたいで連結された stderr）。ブロックごとの記録ではありません。
        </div>
        {!summary && <NotRecorded reason="summary.txt がありません" />}
        {summary && !summary.parsed && (
          <>
            <Notice tone="amber">summary.txt の形式を読み取れませんでした。全文をそのまま表示します。</Notice>
            {typeof summary.raw === 'string' && <TextView text={summary.raw} />}
          </>
        )}
        {summary && summary.parsed === true && (
          <>
            <KV
              rows={[
                ['sandbox_rc', String(summary.sandbox_rc ?? '—')],
                ['reward', fmtNum(summary.reward)],
                ['task_completed', String(summary.task_completed ?? '—')],
                ['terminated / truncated', `${String(summary.terminated ?? '—')} / ${String(summary.truncated ?? '—')}`],
                ['code blocks / regenerations / finishes', `${String(summary.num_code_blocks ?? '—')} / ${String(summary.num_regenerations ?? '—')} / ${String(summary.num_finishes ?? '—')}`],
              ]}
            />
            {summary.ambiguous_split === true && <Notice tone="amber">stdout と stderr の境目が一意に決まりません（stdout に区切りの文字列が含まれています）。</Notice>}
            <SectionTitle>stdout</SectionTitle>
            {typeof summary.stdout === 'string' ? <TextView text={summary.stdout || '（空）'} /> : <div className="text-xs text-text-tertiary">なし</div>}
            <SectionTitle>stderr</SectionTitle>
            {typeof summary.stderr === 'string' ? <TextView text={summary.stderr || '（空）'} tone="stderr" /> : <div className="text-xs text-text-tertiary">なし</div>}
          </>
        )}
      </div>
    );
  } else if (tab === 3) {
    body = (
      <div className="space-y-3">
        <DebugBanner />
        <SectionTitle>trial_record.json</SectionTitle>
        {data.trial_record !== undefined && data.trial_record !== null ? <RawJson value={data.trial_record} /> : <NotRecorded />}
        {other.length > 0 && (
          <>
            <SectionTitle>その他のファイル</SectionTitle>
            <div className="space-y-0.5 text-xs font-mono">
              {other.map((f) => (
                <div key={f.path}>
                  <a href={url(f.path)} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-2">
                    {f.path}
                  </a>{' '}
                  <span className="text-text-tertiary">{fmtBytes(f.bytes)}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    );
  } else {
    body = <NotRecorded reason="トレースが無いため、入力の差分は計算できません" />;
  }

  return (
    <div className="flex-1 min-h-0 flex">
      <div className="w-[42%] min-w-[320px] border-r border-surface-border overflow-y-auto p-3 space-y-3">
        <KV
          rows={[
            ['run', runId],
            ['task', trial.task],
            ['condition', trial.condition],
            ['trial', trial.trialDir],
            ['ディレクトリ名から', `sandbox_rc=${String(data.sandbox_rc ?? '—')} · reward=${fmtNum(data.reward)} · task_completed=${String(data.task_completed ?? '—')}`],
          ]}
        />
        <Notice>{LABELS.past3dUnsupported}</Notice>
        <SectionTitle>動画</SectionTitle>
        {videos.length === 0 && <div className="text-xs text-text-tertiary">{LABELS.notCaptured}</div>}
        {videos.map((v) => (
          <div key={v.path}>
            <div className="text-[11px] font-mono text-text-tertiary mb-1">
              {v.path}（{fmtBytes(v.bytes)}）
            </div>
            <video src={url(v.path)} controls preload="metadata" className="w-full max-h-64 object-contain rounded border border-surface-border bg-black" />
          </div>
        ))}
        <SectionTitle right={<Badge tone="amber">{LABELS.legacyHarnessImage}</Badge>}>visual_feedback 画像</SectionTitle>
        <div className="grid grid-cols-2 gap-2">
          {images.map((im) => (
            <div key={im.path}>
              <TraceImage src={url(im.path)} alt={im.path} provenance="legacy_harness" className="h-32 w-full" />
              <div className="text-[10px] font-mono text-text-tertiary">{im.path}</div>
            </div>
          ))}
        </div>
        {images.length === 0 && <div className="text-xs text-text-tertiary">なし</div>}
      </div>
      <div className="flex-1 min-w-0 flex flex-col">
        <div className="flex-shrink-0 flex items-end gap-0.5 px-2 pt-1.5 border-b border-surface-border bg-surface-raised">
          {LABELS.tabs.map((label, i) => (
            <button
              key={label}
              type="button"
              onClick={() => setTab(i)}
              className={`px-3 py-1.5 text-xs font-display whitespace-nowrap rounded-t-md border-b-2 ${
                tab === i ? 'text-accent border-accent bg-surface' : 'text-text-tertiary border-transparent hover:text-text-primary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-3">{body}</div>
      </div>
    </div>
  );
}

function LazyText({ title, url }: { title: string; url: string }) {
  const res = useAsync(() => fetchText(url), [url]);
  return (
    <div>
      <div className="text-[11px] font-mono text-text-tertiary mb-1">{title}</div>
      {res.loading && <Spinner />}
      {res.error && <ErrorText>{res.error}</ErrorText>}
      {res.data !== null && <TextView text={res.data} />}
    </div>
  );
}
