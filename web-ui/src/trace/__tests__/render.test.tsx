/**
 * Components rendered with react-dom/server (effects do not run, so nothing
 * is fetched): what the panes say for the regression cases.
 * - an unparsed request is never 「この呼び出しは画像入力なし」;
 * - long exception messages (text assets) render; error boundaries catch;
 * - the later-input report says 未確認 for unsearched text;
 * - logical failure after HTTP 200; a stopped stream's partial output;
 * - the runner verdict of the matching invocation only;
 * - an earlier attempt of a retried trial shows its own status, not the
 *   trial's final result as its verdict;
 * - array previews are keyed by the array they were loaded for;
 * - the notice for data moved to an asset;
 * - panes that read another event's asset distinguish loading, failed,
 *   never written, and not loaded from 未記録;
 * - the runner verdict says when the run's records are still being read
 *   or could not be read, never 「runner の記録なし」 for that;
 * - the note for a truncated 由来不明 list.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import type { ReactElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { buildEpisodeModel, resolveSelection, type EpisodeModel, type Selection } from '../model';
import type { TabProps } from '../components/DetailTabs';
import { DataAssetNotice, DetailTabs } from '../components/DetailTabs';
import { ErrorBoundary } from '../components/ErrorBoundary';
import { MediaPane } from '../components/MediaPane';
import { Timeline } from '../components/Timeline';
import { TopBar } from '../components/TopBar';
import { arraySource, valueFor } from '../components/ValueView';
import { ApiLogTab } from '../components/tabs/ApiLogTab';
import { InputDiffTab } from '../components/tabs/InputDiffTab';
import { ModelInputTab } from '../components/tabs/ModelInputTab';
import { ModelOutputTab } from '../components/tabs/ModelOutputTab';
import { MAX_NESTED_ASSET_TARGETS, applyDataAssets, dataAssetTargets, withTargetsLoading, type DataAssetState } from '../dataAsset';
import type { RunnerRecord, TraceEvent } from '../types';
import { Rec, summaryWith } from './fixtures';

// react-dom/server warns once per useLayoutEffect (VirtualList, DetailTabs); expected here.
const consoleError = console.error;
console.error = (...args: unknown[]) => {
  if (typeof args[0] === 'string' && args[0].includes('useLayoutEffect does nothing on the server')) return;
  consoleError(...args);
};

const EP = { runId: 'run', episodeId: 'ep' };
const NO_IMAGE = 'この呼び出しは画像入力なし';

function text(el: ReactElement): string {
  return renderToStaticMarkup(el)
    .replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&amp;/g, '&');
}

function props(m: EpisodeModel, sel: Selection, extra: Partial<TabProps> = {}): TabProps {
  return { model: m, ep: EP, summary: null, selection: sel, running: false, onSelect: () => {}, onFocusRange: () => {}, focus: null, ...extra };
}

function sel(m: EpisodeModel, ev: TraceEvent): Selection {
  const s = resolveSelection(m, ev.event_id);
  assert.ok(s, `selection for ${ev.event_id}`);
  return s as Selection;
}

function timeline(m: EpisodeModel): string {
  return text(<Timeline model={m} ep={EP} running={false} selectedKey={null} selectedEventId={null} onSelect={() => {}} scrollToken={0} showTaskText={false} />);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

function unparsedVdm() {
  const r = new Rec();
  const img = (sha: string, order: number) => ({ path: `messages[1].content[${order + 1}].image_url.url`, order, kind: 'image', asset: `assets/${sha}.png`, sha256: sha, mime: 'image/png' });
  const prev = r.request('vdm', 'lc_prev', 1, { n_images: 2, images: [img('aa', 0), img('bb', 1)], payload_path: 'payloads/p.request.json' });
  r.response(prev);
  // the writer queue was full: the body was never parsed (the trace recorder request capture)
  const cur = r.request('vdm', 'lc_cur', 1, { payload_path: undefined, reconstruction: { verified: false, reason: 'queue_full' } });
  const d = cur.data as Record<string, unknown>;
  delete d.n_images;
  delete d.n_videos;
  delete d.images;
  r.response(cur);
  return { m: buildEpisodeModel(r.events), cur };
}

test('an unparsed request shows 画像: 未記録 in every pane, never the no-image label or removed images', () => {
  const { m, cur } = unparsedVdm();
  const s = sel(m, cur);
  const input = text(<ModelInputTab {...props(m, s)} />);
  assert.ok(!input.includes(NO_IMAGE), 'model-input tab');
  assert.match(input, /画像・動画の数: 未記録/);
  assert.match(input, /queue_full/);
  const media = text(<MediaPane model={m} ep={EP} selection={s} live3d={{ available: false, reason: '' }} follow={false} />);
  assert.ok(!media.includes(NO_IMAGE), 'media pane');
  assert.match(media, /未記録/);
  const diff = text(<InputDiffTab {...props(m, s)} />);
  assert.ok(!diff.includes('削除 2'), 'input diff');
  assert.match(diff, /追加・削除は判定できません/);
  const tl = timeline(m);
  assert.match(tl, /画像数 未記録/);
});

test('an oversize request whose data is not loaded yet is "not loaded", not "body not parsed", in the image diff', () => {
  const r = new Rec();
  const prev = r.request('vdm', 'lc_prev', 1, { n_images: 1, images: [{ path: 'messages[1].content[1].image_url.url', order: 0, kind: 'image', sha256: 'aa' }] });
  r.response(prev);
  const cur = r.request('vdm', 'lc_cur', 1);
  cur.data = { role: 'vdm', call_kind: 'image_diff', logical_call_id: 'lc_cur', http_attempt: 1, n_images: 1, n_videos: 0, oversize: true, data_asset: 'assets/cc/c.json.gz', data_asset_written: true };
  r.response(cur);
  const m = buildEpisodeModel(r.events);
  const diff = text(<InputDiffTab {...props(m, sel(m, cur))} />);
  assert.match(diff, /今回の要求の data は 256 KB を超えたため別ファイルにあり、まだ読み込まれていません/);
  assert.ok(!diff.includes('送信本文を解析していない'));
});

test('zero measured images and videos still give the no-image label', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1', 1, { n_images: 0, n_videos: 0, payload_path: null });
  const m = buildEpisodeModel(r.events);
  const input = text(<ModelInputTab {...props(m, sel(m, req))} />);
  assert.match(input, new RegExp(NO_IMAGE));
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

const LONG = { truncated: true, preview: 'XLA error: ' + 'y'.repeat(3989), length: 8024, asset: 'assets/aa/msg.txt.gz', sha256: 'aa', written: true };

test('API and service errors with a text-asset message render (no crash) and show the preview', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'plan', {});
  const svc = r.serviceStart(api, 'curobo', 'trajopt_plan_fn');
  r.serviceEnd(svc, { error_type: 'RuntimeError', message: LONG }, 'service_call_error');
  r.apiError(api, 'RuntimeError', LONG, { stdout: [0, 0], stderr: [0, 0] });
  r.stream(block, 'stderr', 'RuntimeError: XLA error\n');
  r.blockEnd(block, 0, 24);
  const m = buildEpisodeModel(r.events);
  const tl = timeline(m);
  assert.match(tl, /RuntimeError: XLA error/);
  const log = text(<ApiLogTab {...props(m, sel(m, api))} />);
  assert.match(log, /RuntimeError: XLA error: y/);
  assert.match(log, /先頭のみ（全 8024 文字）|全文を読み込み中/);
  const svcLog = text(<ApiLogTab {...props(m, sel(m, svc))} />);
  assert.ok(!svcLog.includes('[object Object]'));
  assert.match(svcLog, /XLA error/);
  // the later-input report does not crash either (the full text is loaded on demand)
  assert.doesNotThrow(() => text(<ModelInputTab {...props(m, sel(m, api))} />));
});

test('a recording warning whose message is a text asset renders its preview in the model-input tab', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1', 1, { payload_path: null });
  r.add('trace_warning', { code: 'payload_unparsed', message: LONG, related_event_id: req.event_id }, { status: 'warning' });
  const m = buildEpisodeModel(r.events);
  const t = text(<ModelInputTab {...props(m, sel(m, req))} />);
  assert.match(t, /記録の警告 payload_unparsed: XLA error: y/);
  assert.ok(!t.includes('[object Object]'));
});

test('the error boundary turns a render error into a notice and clears it on a new selection', () => {
  const caught = ErrorBoundary.getDerivedStateFromError(new TypeError('s.slice is not a function'));
  assert.equal(caught.error, 'TypeError: s.slice is not a function');
  assert.deepEqual(ErrorBoundary.getDerivedStateFromProps({ label: 'x', resetKey: 'b', children: null }, { error: 'e', key: 'a' }), { error: null, key: 'b' });
  assert.equal(ErrorBoundary.getDerivedStateFromProps({ label: 'x', resetKey: 'a', children: null }, { error: 'e', key: 'a' }), null);
  assert.equal(text(<ErrorBoundary label="x">ok</ErrorBoundary>), 'ok');
  // the notice keeps a control of the part that failed (the top bar keeps the picker button)
  const b = new ErrorBoundary({ label: '上部バー', fallbackExtra: <button type="button">run / episode を選ぶ</button>, children: null });
  b.state = { error: 'TypeError: x', key: undefined };
  const notice = text(<>{b.render()}</>);
  assert.match(notice, /上部バーを表示できません（記録の形式がビューワーの想定と違います）: TypeError: x/);
  assert.match(notice, /run \/ episode を選ぶ/);
});

// ---------------------------------------------------------------------------
//   Later input in the model-input tab
// ---------------------------------------------------------------------------

test('a short stdout is shown as 未確認, not 含まれていません; the API\'s own traceback line is named', () => {
  const r = new Rec();
  const req0 = r.request('code_generator', 'lc0');
  const block = r.blockStart(r.response(req0));
  const api = r.apiStart(block, 'goto_pose', {});
  r.apiError(api, 'ValueError', 'cannot reshape array of size 2 into shape (3,)', { stdout: [0, 0], stderr: [0, 60] });
  const stderr = 'ValueError: cannot reshape array of size 2 into shape (3,)\nValueError: Invalid object name: blue cube\n';
  const out = r.stream(block, 'stdout', '0.12\n');
  const err = r.stream(block, 'stderr', stderr);
  r.blockEnd(block, 5, [...stderr].length);
  r.request('code_generator', 'lc1', 1, {
    input_summary: summaryWith([{ kind: 'stderr', path: 'messages[1].content[0].text', start: 10, end: 10 + [...stderr].length, source_event_ids: [err.event_id, block.event_id] }]),
  });
  void out;
  const m = buildEpisodeModel(r.events);
  const t = text(<ModelInputTab {...props(m, sel(m, api))} />);
  assert.match(t, /ブロックの stdout（5 文字）: 未確認（8 文字未満のため照合の対象外）/);
  assert.match(t, /この例外の最終トレースバック行 「ValueError: cannot reshape array of size 2 into shape \(3,\)」/);
  assert.ok(!t.includes('Invalid object name: blue cube」'), 'the other exception is not attributed');
});

test('a block whose stderr event is lost says the exception search is unchecked (not recorded), never "not printed"', () => {
  const r = new Rec();
  const req0 = r.request('code_generator', 'lc0');
  const block = r.blockStart(r.response(req0));
  const api = r.apiError(r.apiStart(block, 'goto_pose', {}), 'ValueError', 'cannot reshape array of size 2 into shape (3,)');
  r.stream(block, 'stdout', '');
  // the stderr line was lost (corrupt line); code_block_end still names it
  r.blockEnd(block, 0, 60, { stderr_event_id: 'ev_lost' });
  r.request('code_generator', 'lc1', 1, { input_summary: summaryWith([]) });
  const m = buildEpisodeModel(r.events);
  const t = text(<ModelInputTab {...props(m, sel(m, api))} />);
  assert.match(t, /例外テキスト・最終トレースバック行: 未確認（ブロックの stdout\/stderr のイベントが記録にない（未記録）ため）/);
  assert.ok(!t.includes('出力されていません'), 'no claim that the exception was not printed');
});

test('a redacted str(e) (message_redacted) is labelled in the API log and left unchecked in the later-input report', () => {
  const r = new Rec();
  const req0 = r.request('code_generator', 'lc0');
  const block = r.blockStart(r.response(req0));
  const start = r.apiStart(block, 'get_object_pose', {});
  const err = r.apiError(start, 'ConnectionError', 'Max retries exceeded with url: /ik?[query removed]');
  (err.data as Record<string, unknown>).message_redacted = true;
  const stderr = 'Traceback (most recent call last):\nConnectionError: Max retries exceeded with url: /ik?api_key=REDACTED-EXAMPLE\n';
  r.stream(block, 'stdout', '');
  const se = r.stream(block, 'stderr', stderr);
  r.blockEnd(block, 0, [...stderr].length);
  r.request('code_generator', 'lc1', 1, {
    input_summary: summaryWith([{ kind: 'stderr', path: 'messages[1].content[0].text', start: 10, end: 10 + [...stderr].length, source_event_ids: [se.event_id, block.event_id] }]),
  });
  const m = buildEpisodeModel(r.events);
  const log = text(<ApiLogTab {...props(m, sel(m, start))} />);
  assert.match(log, /秘密情報.*を伏せて保存した文です/);
  const t = text(<ModelInputTab {...props(m, sel(m, start))} />);
  assert.match(t, /例外テキスト・最終トレースバック行: 未確認（レコーダーが例外テキスト str\(e\) から秘密情報を伏せて保存しており/);
  assert.ok(!t.includes('出力されていません'), 'no claim that the exception was not printed');
});

test('warning cards name the new recorder codes and their flags (stale_context, state_unavailable, run_level, outside_episode)', () => {
  const r = new Rec();
  r.add('trace_warning', { code: 'stale_context', message: 'left over from an earlier call', counts_as_gap: false }, { status: 'warning' });
  r.add('trace_warning', { code: 'state_unavailable', message: 'ground-truth state not captured: no sim' }, { status: 'warning' });
  r.add('trace_warning', { code: 'hook_install_failed', message: 'H9 could not replace sam3_seg_fn', run_level: true }, { status: 'warning' });
  r.add('trace_warning', { code: 'internal_error', message: 'KeyError in _episode_start', outside_episode: true }, { status: 'warning' });
  const tl = timeline(buildEpisodeModel(r.events));
  assert.match(tl, /前の呼び出しの記録用コンテキストが残っていた/);
  assert.match(tl, /記録の欠落には数えない/);
  assert.match(tl, /正解状態を取得できず/);
  assert.match(tl, /プロセス全体（run 単位）/);
  assert.match(tl, /エピソード外/);
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

test('HTTP 200 then a logical error: the card and the output tab show a failed call', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1');
  r.response(req, 200, '{}');
  r.logicalError(req, 'KeyError', "'choices'");
  const m = buildEpisodeModel(r.events);
  const tl = timeline(m);
  assert.match(tl, /失敗/);
  assert.match(tl, /HTTP 200 の応答の後に失敗/);
  const out = text(<ModelOutputTab {...props(m, sel(m, req))} />);
  assert.match(out, /この論理呼び出しは失敗しました（HTTP 200 の応答を受け取った後）/);
  assert.match(out, /ハーネスはこの応答を使っていません/);
});

test('a stopped stream shows the partial output and no recording-gap claim', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1', 1, { stream: true });
  r.logicalError(req, 'GeneratorExit', '', { streamed: true, n_chunks: 3, partial_content: 'import numpy as np' });
  const m = buildEpisodeModel(r.events);
  const out = text(<ModelOutputTab {...props(m, sel(m, req))} />);
  assert.match(out, /中断までに受信した部分出力（chunk 3 件/);
  assert.match(out, /import numpy as np/);
  assert.match(out, /ストリーム中断（GeneratorExit/);
  assert.match(out, /論理呼び出しは中断/);
  assert.match(out, /この論理呼び出しは完了前に中断されました/);
  assert.ok(!out.includes('論理呼び出しは失敗'), 'a stop is not called a failure');
  assert.ok(!out.includes('記録の欠落'), 'not a recording gap');
  const tl = timeline(m);
  assert.match(tl, /中断/);
  assert.ok(!/失敗#/.test(tl.replace(/\s+/g, '')), 'the card badge says 中断, not 失敗');
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

test('the top bar shows the verdict of the matching invocation, or says it cannot tell', () => {
  const iso = (sec: number) => new Date(sec * 1000).toISOString();
  const rec = (start: number, success: boolean): RunnerRecord => ({
    task: 'cube_lift',
    condition: 'C',
    seed: 1,
    attempts: [{ attempt_id: 1, status: success ? 'completed' : 'timeout', started_at: iso(start), ended_at: iso(start + 100) }],
    outcome: { task_success: success },
  });
  const r = new Rec();
  r.episodeStart({}, { wall: 1000.5 });
  const m = buildEpisodeModel(r.events);
  const bar = (records: RunnerRecord[], startWall: number) =>
    text(
      <TopBar
        runLabel="run"
        episodeLabel="ep"
        episodeId="ep"
        summary={{ episode_id: 'ep', status: 'ended' }}
        model={m}
        runnerRecords={records}
        startWall={startWall}
        live={false}
        onLive={() => {}}
        poll={{ text: '', tone: 'neutral' }}
        lastFetch={null}
        reader={{ corruptLines: 0, tornTail: false, eof: true, count: 1, replaced: false }}
        onOpenPicker={() => {}}
        onRefresh={() => {}}
      />,
    );
  const own = bar([rec(1000, false), rec(5000, true)], 1000.5);
  assert.match(own, /task_success=false/);
  assert.ok(!own.includes('task_success=true'));
  const none = bar([rec(1000, false), rec(5000, true)], 3000);
  assert.match(none, /対応する runner 記録を特定できません/);
});

test('an attempt that timed out and was retried shows its own status; the trial\'s final result is named as attempt 2\'s', () => {
  const iso = (sec: number) => new Date(sec * 1000).toISOString().replace(/\.\d{3}Z$/, '+00:00');
  const record: RunnerRecord = {
    task: 'cube_lift',
    condition: 'C',
    seed: 1,
    trial: 1,
    attempts: [
      { attempt_id: 1, status: 'timeout', started_at: iso(1000), ended_at: iso(1600), timed_out_in_sandbox: true },
      { attempt_id: 2, status: 'completed', started_at: iso(1601), ended_at: iso(1700) },
    ],
    outcome: { task_success: true, code_execution_success: true, agent_finish: true, termination_reason: 'finish' },
  };
  const bar = (attemptIndex: number, wall: number) => {
    const r = new Rec();
    r.episodeStart({ attempt_index: attemptIndex }, { wall });
    r.episodeEnd({ end_reason: attemptIndex === 1 ? 'timeout_alarm' : 'completed' });
    const html = renderToStaticMarkup(
      <TopBar
        runLabel="run"
        episodeLabel="ep"
        episodeId="ep"
        summary={{ episode_id: 'ep', status: 'ended' }}
        model={buildEpisodeModel(r.events)}
        runnerRecords={[record]}
        startWall={wall}
        live={false}
        onLive={() => {}}
        poll={{ text: '', tone: 'neutral' }}
        lastFetch={null}
        reader={{ corruptLines: 0, tornTail: false, eof: true, count: 2, replaced: false }}
        onOpenPicker={() => {}}
        onRefresh={() => {}}
      />,
    );
    return { html };
  };
  const a1 = bar(1, 1000.5);
  const t1 = a1.html.replace(/<[^>]+>/g, '');
  assert.match(t1, /この試行（attempt 1）: timeout（生成コードの実行中）/);
  assert.match(t1, /trial の最終結果（attempt 2。このエピソードの試行ではありません）: task_success=true/);
  // the only task_success shown for attempt 1 is inside the labelled final result, in plain text
  assert.equal((t1.match(/task_success=true/g) ?? []).length, 1);
  assert.ok(t1.indexOf('task_success=true') > t1.indexOf('trial の最終結果'), 'no verdict before the label');
  assert.ok(!/bg-nv-green[^"]*"[^>]*>task_success=true/.test(a1.html), 'no green verdict badge for attempt 1');
  const a2 = bar(2, 1601.3);
  const t2 = a2.html.replace(/<[^>]+>/g, '');
  assert.match(t2, /task_success=true/);
  assert.match(t2, /この試行 attempt 2 が trial の最終試行/);
  assert.ok(!t2.includes('trial の最終結果（'), 'the final attempt shows the verdict as its own');
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

test('an array preview is read back only under the array it was loaded for', () => {
  const a = arraySource(EP, { type: 'ndarray', npz: 'assets/aa/a.npz', npz_key: 'a0' });
  const b = arraySource(EP, { type: 'ndarray', npz: 'assets/bb/b.npz', npz_key: 'a0' });
  const c = arraySource(EP, { type: 'ndarray', path: 'assets/cc/c.npy' });
  assert.notEqual(a.key, b.key);
  assert.equal(c.path, 'assets/cc/c.npy');
  const loadedForA = { key: a.key as string, value: { a0: { shape: [120, 4, 4] } } };
  assert.deepEqual(valueFor(loadedForA, a.key), { a0: { shape: [120, 4, 4] } });
  assert.equal(valueFor(loadedForA, b.key), null, 'another event\'s array never shows A\'s preview');
  assert.equal(valueFor(loadedForA, null), null);
});

test('the data-asset notice for loading, failed and unwritten data', () => {
  const r = new Rec();
  const mk = (path: string, written = true) => {
    const ev = r.add('api_call_start', { name: 'x', oversize: true, data_asset: path, data_asset_written: written });
    ev.artifacts = [{ path, label: 'payload', written, reason: written ? null : 'disk_low' }];
    return ev;
  };
  const a = mk('assets/aa/a.json.gz');
  const b = mk('assets/bb/b.json.gz');
  const c = mk('assets/cc/c.json.gz', false);
  const d = mk('assets/dd/d.json.gz');
  const states = new Map<string, DataAssetState>([
    [a.event_id, { status: 'loading' }],
    [b.event_id, { status: 'failed', error: 'HTTP 500' }],
    [d.event_id, { status: 'loaded', data: {} }],
  ]);
  const t = text(<DataAssetNotice events={[a, b, c, d]} states={states} onRetry={() => {}} />);
  assert.match(t, /assets\/aa\/a\.json\.gz に保存されています。読み込み中/);
  assert.match(t, /assets\/bb\/b\.json\.gz）を読み込めません: HTTP 500/);
  assert.match(t, /再読み込み/);
  assert.match(t, /そのファイルは未記録です（disk_low）/);
  assert.ok(!t.includes('d.json.gz'), 'loaded data needs no notice');
});

test('timeline cards say whether an oversize event\'s data can be loaded or was never written', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const mk = (name: string, path: string, written: boolean) => {
    const ev = r.apiStart(block, name, {});
    ev.data = { api_call_id: ev.event_id, name, depth: 0, oversize: true, data_asset: path, data_asset_written: written };
    ev.artifacts = [{ path, label: 'payload', written, reason: written ? null : 'disk_low' }];
    r.apiEnd(ev);
    return ev;
  };
  mk('saved_one', 'assets/aa/a.json.gz', true);
  mk('lost_one', 'assets/bb/b.json.gz', false);
  r.blockEnd(block, 0, 0);
  const tl = timeline(buildEpisodeModel(r.events));
  assert.equal((tl.match(/data は別ファイル（選択で読み込み）/g) ?? []).length, 1);
  assert.equal((tl.match(/data の本体は未記録（256 KB 超）/g) ?? []).length, 1);
});

test('a truncated unattributed list is said to be cut at 200', () => {
  const r = new Rec();
  const summary = { ...summaryWith([]), unattributed: [{ path: 'messages[0].content', start: 0, end: 5 }], unattributed_truncated: true };
  const req = r.request('code_generator', 'lc1', 1, { payload_path: 'payloads/x.request.json', input_summary: summary });
  const m = buildEpisodeModel(r.events);
  const t = text(<ModelInputTab {...props(m, sel(m, req))} />);
  assert.match(t, /由来不明の区間はレコーダーが 200 件で打ち切りました/);
});

// ---------------------------------------------------------------------------
// Panes that read an event whose data is in an asset distinguish loading,
// failed, and never written from 未記録.
// ---------------------------------------------------------------------------

/** What the recorder keeps inline when it limits line size: scalars, strings up to 200 characters, and markers. */
function makeOversize(ev: TraceEvent, path: string, written = true): Record<string, unknown> {
  const full = ev.data as Record<string, unknown>;
  const small: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(full)) if (v === null || typeof v === 'boolean' || typeof v === 'number' || (typeof v === 'string' && v.length <= 200)) small[k] = v;
  ev.data = { ...small, oversize: true, data_asset: path, data_asset_written: written };
  ev.artifacts = [{ path, label: 'payload', kind: 'json', written, reason: written ? null : 'disk_low' }];
  return full;
}

function laterOversize(apiPrints: boolean, written = true) {
  const r = new Rec();
  r.episodeStart();
  const req0 = r.request('code_generator', 'lc0');
  const block = r.blockStart(r.response(req0));
  const api = r.apiStart(block, 'goto_pose', {});
  r.apiEnd(api, null, apiPrints ? { stdout: [0, 20], stderr: [0, 0] } : { stdout: [0, 0], stderr: [0, 0] });
  const out = r.stream(block, 'stdout', 'moved to the target\n');
  r.stream(block, 'stderr', '');
  r.blockEnd(block, 20, 0);
  const next = r.request('code_generator', 'lc1', 1, {
    input_summary: summaryWith([{ kind: 'stdout', path: 'messages[1].content[0].text', start: 5, end: 25, source_event_ids: [out.event_id, block.event_id] }]),
  });
  const full = makeOversize(next, 'assets/ab/next.json.gz', written);
  return { r, api, next, full };
}

function detail(r: Rec, selected: TraceEvent, tab: number, states: Map<string, DataAssetState>): string {
  const m = buildEpisodeModel(applyDataAssets(r.events, states));
  const s = sel(m, selected);
  return text(
    <DetailTabs
      model={m}
      ep={EP}
      summary={null}
      selection={s}
      running={false}
      onSelect={() => {}}
      onFocusRange={() => {}}
      focus={null}
      tab={tab}
      onTab={() => {}}
      assetStates={states}
      onRetryAsset={() => {}}
      assetEvents={dataAssetTargets(m, s)}
    />,
  );
}

test('a later request whose input summary is in an asset is loading or failed (with a retry), not 未記録', () => {
  for (const prints of [true, false]) {
    const { r, api, next, full } = laterOversize(prints);
    const loading = detail(r, api, 0, new Map([[next.event_id, { status: 'loading' }]]));
    assert.ok(!loading.includes('未記録（input_summary なし）'), `loading (prints=${prints})`);
    assert.match(loading, prints ? /入力の内訳: 読み込み中…/ : /次のコード生成入力の内訳: 読み込み中…/);
    assert.match(loading, /この選択で参照する別のイベントの data/);
    const failed = detail(r, api, 0, new Map([[next.event_id, { status: 'failed', error: 'HTTP 500: boom' }]]));
    assert.ok(!failed.includes('未記録（input_summary なし）'), `failed (prints=${prints})`);
    assert.match(failed, /読み込めません（data は 256 KB を超えたため別ファイル assets\/ab\/next\.json\.gz にあります: HTTP 500: boom）/);
    assert.match(failed, /再読み込み/);
    const loaded = detail(r, api, 0, new Map([[next.event_id, { status: 'loaded', data: full }]]));
    assert.match(loaded, /含まれています/);
    assert.ok(!loaded.includes('この選択で参照する別のイベントの data'), 'nothing pending once loaded');
  }
  // the asset was never written: that is 未記録, with the reason
  const lost = laterOversize(true, false);
  const t = detail(lost.r, lost.api, 0, new Map());
  assert.match(t, /入力の内訳: 未記録（data は 256 KB を超えたため別ファイルに移されましたが、そのファイルは書かれていません: disk_low）/);
});

test('the recorded statistics of a nested service call whose data is in an asset say loading / failed in the debug tab', () => {
  const r = new Rec();
  r.episodeStart();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'get_object_3d_points_and_masks_from_language', {});
  const svc = r.serviceStart(api, 'sam3', 'sam3_seg_fn');
  const end = r.serviceEnd(svc, { response_summary: { type: 'list', items: [] } });
  makeOversize(end, 'assets/aa/svc.json.gz');
  end.artifacts = [...(end.artifacts ?? []), { path: 'assets/bb/out.npz', label: 'perception_output', kind: 'arrays', written: true }];
  r.apiEnd(api);
  r.blockEnd(block, 0, 0);
  const loading = detail(r, api, 3, new Map([[end.event_id, { status: 'loading' }]]));
  assert.match(loading, new RegExp(`記録時の統計（#${end.seq} service_call_end）: 読み込み中…`));
  const failed = detail(r, api, 3, new Map([[end.event_id, { status: 'failed', error: 'HTTP 404' }]]));
  assert.match(failed, /記録時の統計（#\d+ service_call_end）: 読み込めません/);
  assert.match(failed, /再読み込み/);
  const loaded = detail(r, api, 3, new Map([[end.event_id, { status: 'loaded', data: { service: 'sam3', function: 'sam3_seg_fn', response_summary: {} } }]]));
  assert.ok(!loaded.includes('記録時の統計（#'), 'nothing to say once loaded');
});

test('the input diff says the previous request\'s data failed to load (with a retry), never 未記録 for it', () => {
  const r = new Rec();
  const prev = r.request('vdm', 'lc_prev', 1, { n_images: 1, images: [{ path: 'messages[1].content[1].image_url.url', order: 0, kind: 'image', sha256: 'aa' }] });
  r.response(prev);
  makeOversize(prev, 'assets/pp/prev.json.gz');
  const cur = r.request('vdm', 'lc_cur', 1, { n_images: 1, images: [{ path: 'messages[1].content[1].image_url.url', order: 0, kind: 'image', sha256: 'bb' }] });
  r.response(cur);
  const failed = detail(r, cur, 4, new Map([[prev.event_id, { status: 'failed', error: 'HTTP 500' }]]));
  assert.match(failed, /前回の要求の data は 256 KB を超えたため別ファイルにあり、読み込めません（HTTP 500）。追加・削除は判定できません/);
  assert.match(failed, /再読み込み/);
  assert.ok(!/未記録前回の要求/.test(failed), 'not labelled 未記録');
  const loading = detail(r, cur, 4, new Map([[prev.event_id, { status: 'loading' }]]));
  assert.match(loading, /前回の要求の data は 256 KB を超えたため別ファイルにあり、まだ読み込まれていません/);
});

test('the top bar says episode_start\'s flags and models are loading, not 未記録, while its asset loads', () => {
  const r = new Rec();
  const start = r.episodeStart({ config_flags: { multi_turn: true }, models: { code_generator: { model: 'm' } } });
  makeOversize(start, 'assets/ee/start.json.gz');
  const m = buildEpisodeModel(r.events);
  const bar = (states: Map<string, DataAssetState>) =>
    text(
      <TopBar
        runLabel="run"
        episodeLabel="ep"
        episodeId="ep"
        summary={{ episode_id: 'ep', status: 'ended' }}
        model={m}
        runnerRecords={[]}
        live={false}
        onLive={() => {}}
        poll={{ text: '', tone: 'neutral' }}
        lastFetch={null}
        reader={{ corruptLines: 0, tornTail: false, eof: true, count: 1, replaced: false }}
        onOpenPicker={() => {}}
        onRefresh={() => {}}
        assetStates={states}
      />,
    );
  const loading = bar(new Map([[start.event_id, { status: 'loading' }]]));
  assert.match(loading, /設定フラグ: 読み込み中…/);
  assert.ok(!loading.includes('設定フラグ: 未記録'));
  assert.match(bar(new Map([[start.event_id, { status: 'failed', error: 'HTTP 500' }]])), /設定フラグ: 読み込めません/);
});

test('a nested service call beyond the loading bound says 未読み込み with a load button, the loaded-by-page ones say loading', () => {
  const r = new Rec();
  r.episodeStart();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'segment', {});
  const ends: TraceEvent[] = [];
  for (let i = 0; i < MAX_NESTED_ASSET_TARGETS + 1; i += 1) {
    const end = r.serviceEnd(r.serviceStart(api, 'sam3', 'sam3_seg_fn'), { response_summary: { type: 'list', items: [] } });
    makeOversize(end, `assets/s${i}/svc.json.gz`);
    end.artifacts = [...(end.artifacts ?? []), { path: `assets/o${i}/out.npz`, label: 'perception_output', kind: 'arrays', written: true }];
    ends.push(end);
  }
  r.apiEnd(api);
  r.blockEnd(block, 0, 0);
  const m = buildEpisodeModel(r.events);
  const s = sel(m, block);
  const targets = dataAssetTargets(m, s);
  // what the page passes: the targets it requests count as loading
  const shown = withTargetsLoading(new Map(), targets);
  const t = text(
    <DetailTabs
      model={m}
      ep={EP}
      summary={null}
      selection={s}
      running={false}
      onSelect={() => {}}
      onFocusRange={() => {}}
      focus={null}
      tab={3}
      onTab={() => {}}
      assetStates={shown}
      onRetryAsset={() => {}}
      assetEvents={targets}
    />,
  );
  const last = ends[ends.length - 1];
  assert.match(t, new RegExp(`記録時の統計（#${ends[0].seq} service_call_end）: 読み込み中…`));
  assert.match(t, new RegExp(`記録時の統計（#${last.seq} service_call_end）: 未読み込み（data は 256 KB を超えたため別ファイル assets/s${MAX_NESTED_ASSET_TARGETS}/svc\\.json\\.gz にあります。この選択では自動で読み込みません）読み込む`));
  assert.equal((t.match(/: 未読み込み/g) ?? []).length, 1, 'only the one beyond the bound');
});

test('the runner verdict says the records are being read or could not be read, not 「runner の記録なし」', () => {
  const r = new Rec();
  r.episodeStart({}, { wall: 1000.5 });
  r.episodeEnd();
  const m = buildEpisodeModel(r.events);
  const iso = (sec: number) => new Date(sec * 1000).toISOString().replace(/\.\d{3}Z$/, '+00:00');
  const rec: RunnerRecord = {
    task: 'cube_lift',
    condition: 'C',
    seed: 1,
    trial: 1,
    attempts: [{ attempt_id: 1, status: 'completed', started_at: iso(1000), ended_at: iso(1100) }],
    outcome: { task_success: true, code_execution_success: true, agent_finish: true, termination_reason: 'finish' },
  };
  const bar = (records: RunnerRecord[], read: { loaded: boolean; loading: boolean; error: string | null }) =>
    text(
      <TopBar
        runLabel="run"
        episodeLabel="ep"
        episodeId="ep"
        summary={{ episode_id: 'ep', status: 'ended' }}
        model={m}
        runnerRecords={records}
        runnerRead={read}
        startWall={1000.5}
        live={false}
        onLive={() => {}}
        poll={{ text: '', tone: 'neutral' }}
        lastFetch={null}
        reader={{ corruptLines: 0, tornTail: false, eof: true, count: 2, replaced: false }}
        onOpenPicker={() => {}}
        onRefresh={() => {}}
      />,
    );
  const loading = bar([], { loaded: false, loading: true, error: null });
  assert.match(loading, /runner 記録を読み込み中…/);
  assert.ok(!loading.includes('runner の記録なし'));
  const failed = bar([], { loaded: false, loading: false, error: 'HTTP 500: boom' });
  assert.match(failed, /runner 記録を読み込めません: HTTP 500: boom/);
  assert.ok(!failed.includes('runner の記録なし'));
  assert.match(bar([], { loaded: false, loading: true, error: 'HTTP 500: boom' }), /runner 記録を読み込めません: HTTP 500: boom（再取得中…）/);
  // a re-read failed: the records of the earlier read are shown, with the failure named
  const stale = bar([rec], { loaded: true, loading: false, error: 'HTTP 502: gone' });
  assert.match(stale, /task_success=true/);
  assert.match(stale, /runner 記録の再取得に失敗: HTTP 502: gone。前回の取得結果で表示/);
  assert.match(bar([], { loaded: true, loading: false, error: null }), /runner の記録なし/);
});
