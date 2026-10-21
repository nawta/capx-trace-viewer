/**
 * Contracts between the recorder, trace API and viewer (react-dom/server,
 * with effects disabled):
 * - `*_cut` (payload.redact read only the first 64 KB): the later-input report leaves a
 *   cut str(e) unchecked and the API log says only its start is stored;
 * - `*_redacted` flags beyond message_redacted (traceback, HTTP attempt errors, model
 *   error message and body excerpt, hook failure reasons) are labelled;
 * - runner_records_info: a runner_records.jsonl that exists but cannot be
 *   read, or has lines that did not parse, is never shown as 「runner の記録なし」;
 * - episode_json_error: why episode.json could not be used is listed with the gaps;
 * - unscoped.<pid>.status.json (recorder and backend write_status): lines of an unscoped
 *   log that could not be written are named in the picker, also without the log;
 * - harness_summary.success: the debug tab says what it means for the harness that set
 *   it (batch or web UI).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import type { ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { buildEpisodeModel, resolveSelection, type EpisodeModel, type Selection } from '../model';
import type { TabProps } from '../components/DetailTabs';
import { EpisodeStatusCell, UnscopedFileList } from '../components/Picker';
import { LegacyPromptNotice } from '../components/LegacyView';
import { Timeline } from '../components/Timeline';
import { TopBar } from '../components/TopBar';
import { ApiLogTab } from '../components/tabs/ApiLogTab';
import { DebugTab } from '../components/tabs/DebugTab';
import { ModelInputTab } from '../components/tabs/ModelInputTab';
import { ModelOutputTab } from '../components/tabs/ModelOutputTab';
import type { EpisodeSummary, RunnerRecord, RunnerRecordsInfo, TraceEvent } from '../types';
import { Rec, summaryWith } from './fixtures';

const consoleError = console.error;
console.error = (...args: unknown[]) => {
  if (typeof args[0] === 'string' && args[0].includes('useLayoutEffect does nothing on the server')) return;
  consoleError(...args);
};

const EP = { runId: 'run', episodeId: 'ep' };

function text(el: ReactElement): string {
  return renderToStaticMarkup(el)
    .replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&amp;/g, '&');
}

function props(m: EpisodeModel, s: Selection): TabProps {
  return { model: m, ep: EP, summary: null, selection: s, running: false, onSelect: () => {}, onFocusRange: () => {}, focus: null };
}

function sel(m: EpisodeModel, ev: TraceEvent): Selection {
  const s = resolveSelection(m, ev.event_id);
  assert.ok(s, `selection for ${ev.event_id}`);
  return s as Selection;
}

test('a cut str(e) (message_cut) is labelled in the API log and left unchecked in the later-input report', () => {
  const printed = `${'word '.repeat(14000)}end`;
  const stored = `${printed.slice(0, 65531)}\n[recorder: only the first 65536 of ${printed.length} characters were read]`;
  const r = new Rec();
  const req0 = r.request('code_generator', 'lc0');
  const block = r.blockStart(r.response(req0));
  const start = r.apiStart(block, 'get_object_pose', {});
  const err = r.apiError(start, 'ValueError', { truncated: true, preview: stored.slice(0, 4000), length: stored.length, asset: 'assets/aa/m.txt.gz' });
  Object.assign(err.data as Record<string, unknown>, { message_cut: true, traceback: 'Traceback (most recent call last):\nValueError: …', traceback_cut: true });
  const stderr = `Traceback (most recent call last):\nValueError: ${printed}\n`;
  r.stream(block, 'stdout', '');
  const se = r.stream(block, 'stderr', stderr);
  r.blockEnd(block, 0, [...stderr].length);
  r.request('code_generator', 'lc1', 1, {
    input_summary: summaryWith([{ kind: 'stderr', path: 'messages[1].content[0].text', start: 10, end: 10 + [...stderr].length, source_event_ids: [se.event_id, block.event_id] }]),
  });
  const m = buildEpisodeModel(r.events);
  const log = text(<ApiLogTab {...props(m, sel(m, start))} />);
  // once for the message, once for the traceback
  assert.equal(log.split('レコーダーが読んだのは先頭の 64 KB だけです').length - 1, 2);
  assert.match(log, /traceback: レコーダーが読んだのは先頭の 64 KB だけです/);
  const t = text(<ModelInputTab {...props(m, sel(m, start))} />);
  assert.match(t, /例外テキスト・最終トレースバック行: 未確認（レコーダーが例外テキスト str\(e\) の先頭 64 KB だけを注記付きで保存しており/);
  assert.ok(!t.includes('出力されていません'), 'no claim that the exception was not printed');
});

test('redacted tracebacks, HTTP attempt errors, model errors and hook failure reasons are labelled', () => {
  const r = new Rec();
  const startEv = r.episodeStart({ tracer: { version: '1', hooks_installed: [], hooks_failed: [{ hook: 'H9', target: 'sam3_seg_fn', reason: 'url: /x?[query removed]', reason_redacted: true }] } });
  const req = r.request('code_generator', 'lc0');
  const e = r.attemptError(req, 401, 'HTTPError: 401 for url http://h/v1?[query removed]');
  Object.assign(e.data as Record<string, unknown>, { message_redacted: true, body_excerpt: '{"error": "bad key ***masked***"}', body_excerpt_redacted: true });
  const block = r.blockStart(null);
  const start = r.apiStart(block, 'segment', {});
  const svc = r.serviceStart(start, 'sam3', 'sam3_seg_fn');
  r.serviceEnd(svc, { http_attempts: [{ status: null, latency_s: 0.1, error: 'ConnectionError: /seg?[query removed]', error_redacted: true }] });
  const err = r.apiError(start, 'ConnectionError', 'refused');
  Object.assign(err.data as Record<string, unknown>, { traceback: 'Traceback (most recent call last):\nConnectionError: refused', traceback_redacted: true });
  r.blockEnd(block, 0, 0);
  const m = buildEpisodeModel(r.events);
  const log = text(<ApiLogTab {...props(m, sel(m, start))} />);
  assert.match(log, /traceback: レコーダーが秘密情報/);
  const svcLog = text(<ApiLogTab {...props(m, sel(m, svc))} />);
  assert.match(svcLog, /ConnectionError: \/seg\?\[query removed\]（秘密情報を伏せて保存）/);
  const out = text(<ModelOutputTab {...props(m, sel(m, req))} />);
  assert.match(out, /秘密情報を伏せて保存/);
  const debug = text(<DebugTab {...props(m, sel(m, startEv))} />);
  assert.match(debug, /H9 sam3_seg_fn url: \/x\?\[query removed\]（秘密情報を伏せて保存）/);
});

test('debug recorder facts identify the API and service wrapper hooks', () => {
  const r = new Rec();
  const start = r.episodeStart();
  const m = buildEpisodeModel(r.events);
  const summary: EpisodeSummary = {
    episode_id: 'ep',
    episode: { apis_wrapped: [{ class: 'RobotEnv', wrapped: ['goto_pose'], services: ['sam3_seg_fn'] }] },
  };
  const debug = text(<DebugTab {...props(m, sel(m, start))} summary={summary} />);
  assert.match(debug, /API のラップ（H8, H9）/);
  assert.match(debug, /RobotEnv: goto_pose · サービス sam3_seg_fn/);
});

test('legacy prompts mention sent messages only when model input was recorded', () => {
  const unrecorded = text(<LegacyPromptNotice modelInputRecorded={false} />);
  assert.match(unrecorded, /ハーネスが保存した文字列/);
  assert.ok(!unrecorded.includes('上の messages.json'));
  const recorded = text(<LegacyPromptNotice modelInputRecorded={true} />);
  assert.match(recorded, /送信した payload は上の messages.json に記録されています/);
});

function bar(extra: { summary?: EpisodeSummary; records?: RunnerRecord[]; info?: RunnerRecordsInfo | null; readError?: string | null; count?: number }): string {
  const r = new Rec();
  r.episodeStart({}, { wall: 1000.5 });
  r.episodeEnd();
  const m = buildEpisodeModel(r.events);
  return text(
    <TopBar
      runLabel="run"
      episodeLabel="ep"
      episodeId="ep"
      summary={extra.summary ?? { episode_id: 'ep', status: 'ended' }}
      model={m}
      runnerRecords={extra.records ?? []}
      runnerRead={{ loaded: true, loading: false, error: null }}
      runnerRecordsInfo={extra.info ?? null}
      startWall={1000.5}
      live={false}
      onLive={() => {}}
      poll={{ text: '', tone: 'neutral' }}
      lastFetch={null}
      reader={{ corruptLines: 0, tornTail: false, eof: extra.readError ? false : true, count: extra.count ?? 2, replaced: false, error: extra.readError ?? null }}
      onOpenPicker={() => {}}
      onRefresh={() => {}}
    />,
  );
}

test('runner_records.jsonl that exists but cannot be read, or lost lines, is never 「runner の記録なし」', () => {
  const unreadable = bar({ info: { error: 'unreadable (EACCES)', corrupt_lines: 0, torn_tail: false, truncated: false } });
  assert.match(unreadable, /runner_records\.jsonl を読めません（unreadable \(EACCES\)）。runner の判定は不明です/);
  assert.ok(!unreadable.includes('runner の記録なし'));
  const corrupt = bar({ info: { corrupt_lines: 2, torn_tail: true, truncated: false } });
  assert.match(corrupt, /runner_records\.jsonl: 読めない行 2 件、末尾の行が書きかけ。この trial の記録がその中にある可能性があります/);
  // a clean read with no record stays as before
  const clean = bar({ info: { corrupt_lines: 0, torn_tail: false, truncated: false } });
  assert.match(clean, /runner の記録なし/);
  assert.ok(!clean.includes('runner_records.jsonl'));
});

test('the picker names why an episode.json could not be used, and not a missing one', () => {
  const t = text(<EpisodeStatusCell episode={{ episode_id: 'ep', status: 'incomplete', episode_json_error: 'unreadable (EACCES)' }} />);
  assert.match(t, /episode\.json: unreadable \(EACCES\)/);
  const missing = text(<EpisodeStatusCell episode={{ episode_id: 'ep', status: 'running', episode_json_error: 'missing' }} />);
  assert.ok(!missing.includes('episode.json'));
});

test('an episode.json that could not be used is listed with the recording gaps', () => {
  const t = bar({ summary: { episode_id: 'ep', status: 'incomplete', episode_json_error: 'unreadable (EACCES)' } });
  assert.match(t, /episode\.json を読めません（unreadable \(EACCES\)）/);
  // "missing" (not written yet) is not called unreadable
  const missing = bar({ summary: { episode_id: 'ep', status: 'running', episode_json_error: 'missing' } });
  assert.ok(!missing.includes('episode.json を読めません'));
});

test('unscoped logs name the lines the recorder could not write, also when only the status file exists', () => {
  const t = text(
    <UnscopedFileList
      files={[
        { file: 'unscoped.101.jsonl', bytes: 2048, log_id: 'unscoped.101', write_status: null },
        { file: 'unscoped.102.jsonl', bytes: 512, log_id: 'unscoped.102', write_status: { file: 'unscoped.102.jsonl', lines_not_written: 3, last_reason: 'OSError: [Errno 28] No space left on device' } },
        { file: 'unscoped.103.jsonl', bytes: null, log_id: null, write_status: { lines_not_written: 1, last_reason: 'OSError: [Errno 5] Input/output error' } },
        { file: 'unscoped.104.jsonl', bytes: 10, log_id: 'unscoped.104', write_status: null, write_status_error: 'unreadable (EACCES)' },
      ]}
      runId="run"
      currentRun={null}
      currentEpisode={null}
      onPick={() => {}}
    />,
  );
  assert.match(t, /unscoped\.101\.jsonl（2\.0 KB）を開く/);
  assert.match(t, /unscoped\.102\.jsonl（512 B）を開く書き込めなかった行 3 件（最後の理由: OSError: \[Errno 28\] No space left on device）/);
  assert.match(t, /unscoped\.103\.jsonl（ファイルなし）書き込めなかった行 1 件/);
  assert.match(t, /書き込み状況ファイル（status\.json）を読めません（unreadable \(EACCES\)）/);
});

test('an undone episode start in the unscoped log names the episode and what happened to the trial', () => {
  const r = new Rec();
  r.add(
    'trace_warning',
    {
      code: 'internal_error',
      message: '_episode_start: KeyError: task',
      outside_episode: true,
      episode_id: 'cube_lift__S1__s3__i1__a1',
      alarm: false,
      note: 'the recorder raised KeyError while the recorder was starting the episode; the trial ran untraced',
    },
    { status: 'warning' },
  );
  const tl = text(<Timeline model={buildEpisodeModel(r.events)} ep={EP} running={false} selectedKey={null} selectedEventId={null} onSelect={() => {}} scrollToken={0} showTaskText={false} />);
  assert.match(tl, /対象のエピソード: cube_lift__S1__s3__i1__a1/);
  assert.match(tl, /the trial ran untraced/);
});

test('an events.jsonl the server cannot read (403/500) is a recording gap, and the event count does not say loading', () => {
  const t = bar({ readError: 'アクセスできません (403): events.jsonl cannot be read: permission denied', count: 0 });
  assert.match(t, /記録の欠落: events\.jsonl を読み込めません/);
  assert.match(t, /events 0（読み込めません）/);
  assert.ok(!t.includes('（読み込み中）'));
  assert.ok(!t.includes('報告なし'));
  // a failed poll after events were read is not called a gap (the poll status names the error)
  const later = bar({ readError: 'HTTP 502: bad gateway', count: 2 });
  assert.ok(!later.includes('events.jsonl を読み込めません'));
});

test('the harness summary says what TrialSummary.success means for the harness that set it', () => {
  // The batch trial harness uses the last block's sandbox_rc after its override.
  const batch = new Rec();
  const startB = batch.episodeStart();
  batch.episodeEnd({ harness_summary: { success: true, sandbox_rc: 0, task_completed: true } });
  const mb = buildEpisodeModel(batch.events);
  const tb = text(<DebugTab {...props(mb, sel(mb, startB))} />);
  assert.ok(tb.includes('true — 最終ブロックの sandbox_rc == 0（ハーネスの補正後。途中のブロックの失敗は含まない）。タスク成功の判定とは別の値です'), tb);
  // The web trial harness uses task_completed or num_finishes > 0.
  const web = new Rec();
  const startW = web.episodeStart({ source: 'web_ui' });
  web.episodeEnd({ harness_summary: { success: true, sandbox_rc: 1, task_completed: false, num_finishes: 1 } });
  const mw = buildEpisodeModel(web.events);
  const tw = text(<DebugTab {...props(mw, sel(mw, startW))} />);
  assert.ok(tw.includes('true — task_completed または num_finishes > 0（Web UI の判定。sandbox_rc は見ない）。タスク成功の判定とは別の値です'), tw);
  assert.ok(!tw.includes('最終ブロックの sandbox_rc'));
  // Older traces that describe success as code running without an exception get a correction note.
  const old = new Rec();
  const startO = old.episodeStart();
  old.episodeEnd({ harness_summary: { success: true, success_meaning: 'code ran without exception (sandbox_rc == 0); not task success' } });
  const mo = buildEpisodeModel(old.events);
  const to = text(<DebugTab {...props(mo, sel(mo, startO))} />);
  assert.ok(to.includes('code ran without exception (sandbox_rc == 0); not task success（以前のレコーダーが保存した誤った説明。正しい意味は success の行を参照）'), to);
  assert.ok(!tb.includes('以前のレコーダー'));
});
