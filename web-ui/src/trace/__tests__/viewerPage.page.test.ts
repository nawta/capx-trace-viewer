/**
 * The viewer page's own wiring, with its effects running:
 * TraceViewerPage and hooks.ts run under the hook runtime of harness/ against
 * a fake trace server on a virtual clock.
 * - a summary read slower than the poll interval is never
 *   aborted by the next tick, so the status still leaves 'running';
 * - runner records are read again on 更新, when another episode
 *   of the same run is opened, and while the run still runs (this episode
 *   included, until it stopped) and this episode has no record yet; the top
 *   bar is told when the records are still being read or could not be read;
 * - the grace period starts when the status leaves 'running', and
 *   Browse mode still re-reads the summary.
 */

import { advance, clock, net, propsOf, requests, resetBrowser, setLocation } from './harness/browser';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mount, type Mounted } from './harness/fakeReact';
import TraceViewerPage from '../TraceViewerPage';
import { Picker } from '../components/Picker';
import { TopBar } from '../components/TopBar';
import type { RunnerRecord } from '../types';
import { addEpisode, ev, makeRun, serve, type FakeRun } from './pageServer';

type TopBarProps = Parameters<typeof TopBar>[0];
type PickerProps = Parameters<typeof Picker>[0];

const RUNS = /^\/api\/traces\/runs\/r1$/;
const iso = (sec: number) => new Date(sec * 1000).toISOString().replace(/\.\d{3}Z$/, '+00:00');
const KEY = { task: 'cube_lift', condition: 'C', seed: 1 };

function bar(p: Mounted<Record<string, never>>): TopBarProps {
  return propsOf<TopBarProps>(p.output, TopBar);
}

function open(url: string): Mounted<Record<string, never>> {
  setLocation(url);
  return mount(TraceViewerPage, {});
}

/** An ended episode of attempt `attempt` that started at `start` (seconds). */
function endedEpisode(run: FakeRun, id: string, start: number, attempt = 1, seed = 1) {
  const data = { ...KEY, seed, attempt_index: attempt, runner_invocation: 1 };
  return addEpisode(run, id, { status: 'ended', ...data, start_wall_time: start, trace_complete: true }, [
    ev(id, 1, 'episode_start', data, { wall_time: start }),
    ev(id, 2, 'episode_end', { end_reason: 'completed', trace_complete: true }, { wall_time: start + 50 }),
  ]);
}

function recordFor(start: number, attempts: { status: string; from: number; to: number }[], success: boolean, seed = 1): RunnerRecord {
  return {
    ...KEY,
    seed,
    trial: seed,
    attempts: attempts.map((a, i) => ({ attempt_id: i + 1, status: a.status, started_at: iso(start + a.from), ended_at: iso(start + a.to) })),
    outcome: { task_success: success, code_execution_success: true, agent_finish: true, termination_reason: 'finish' },
  };
}

test('a summary read slower than the 1 s poll still arrives, and no read in flight is aborted', async () => {
  resetBrowser();
  const run = makeRun();
  run.latency.summary = 1200;
  const data = { ...KEY, attempt_index: 1 };
  const e1 = addEpisode(run, 'e1', { status: 'running', ...data }, [ev('e1', 1, 'episode_start', data)]);
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e1');
  await advance(10_000);
  assert.equal(bar(p).summary?.status, 'running');
  // the episode ends; the page must see it although every summary read takes 1.2 s
  e1.events.push(ev('e1', 2, 'episode_end', { end_reason: 'completed', trace_complete: true }));
  e1.summary = { ...e1.summary, status: 'ended', trace_complete: true };
  await advance(10_000);
  assert.equal(bar(p).summary?.status, 'ended');
  const reads = requests(/\/episodes\/e1$/);
  assert.equal(reads.aborted, 0, 'a poll tick never aborts the read in flight');
  assert.ok(reads.done >= 5, `summary reads completed: ${reads.done}`);
  p.unmount();
});

test('Browse mode (5 s) with a 5.2 s summary read also reaches the end', async () => {
  resetBrowser();
  const run = makeRun();
  run.latency.summary = 5200;
  const data = { ...KEY, attempt_index: 1 };
  const e1 = addEpisode(run, 'e1', { status: 'running', ...data }, [ev('e1', 1, 'episode_start', data)]);
  serve(run);
  // the URL names an event: the page opens in Browse mode
  const p = open('/traces?view=details&run=r1&episode=e1&event=e1_1');
  await advance(12_000);
  assert.equal(bar(p).live, false);
  e1.events.push(ev('e1', 2, 'episode_end', { end_reason: 'completed', trace_complete: true }));
  e1.summary = { ...e1.summary, status: 'ended', trace_complete: true };
  await advance(30_000);
  assert.equal(bar(p).summary?.status, 'ended');
  assert.equal(requests(/\/episodes\/e1$/).aborted, 0);
  p.unmount();
});

test('更新 re-reads the runner records, and so does opening another episode of the same run', async () => {
  resetBrowser();
  const run = makeRun();
  const start = clock.now / 1000 - 700;
  endedEpisode(run, 'e1', start);
  endedEpisode(run, 'e2', start + 100, 1, 2);
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e1');
  await advance(3000);
  assert.equal(requests(RUNS).sent, 1);
  assert.equal(bar(p).runnerRecords.length, 0);
  // the trial's record is appended after the page read the run
  run.records = [recordFor(start, [{ status: 'completed', from: -1, to: 60 }], true)];
  bar(p).onRefresh();
  await advance(3000);
  assert.equal(requests(RUNS).sent, 2, '更新 reads the run again');
  assert.equal(bar(p).runnerRecords.length, 1);
  // another episode of the same run: its record may be newer than the last read
  run.records = [...run.records, recordFor(start + 100, [{ status: 'completed', from: -1, to: 60 }], false, 2)];
  bar(p).onOpenPicker();
  await advance(50);
  propsOf<PickerProps>(p.output, Picker).onPick('r1', 'e2');
  await advance(3000);
  assert.equal(requests(RUNS).sent, 3, 'opening another episode of the run reads it again');
  assert.equal(bar(p).runnerRecords.length, 2);
  p.unmount();
});

test('while the run still runs and this episode has no record, the records are re-read slowly until it appears', async () => {
  resetBrowser();
  const run = makeRun();
  const start = clock.now / 1000 - 700;
  // attempt 1 timed out and ended; attempt 2 of the same trial is still running
  const a1 = endedEpisode(run, 'e_a1', start, 1);
  a1.summary = { ...a1.summary, end_reason: 'timeout_alarm' };
  const a2data = { ...KEY, attempt_index: 2, runner_invocation: 1 };
  const a2 = addEpisode(run, 'e_a2', { status: 'running', ...a2data, start_wall_time: start + 601 }, [ev('e_a2', 1, 'episode_start', a2data, { wall_time: start + 601 })]);
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e_a1');
  await advance(3000);
  assert.equal(requests(RUNS).sent, 1);
  await advance(40_000);
  const polled = requests(RUNS).sent;
  assert.ok(polled >= 2, `the run is re-read while it runs (${polled} reads)`);
  // attempt 2 ends; the runner appends the trial's record
  a2.summary = { ...a2.summary, status: 'ended' };
  run.records = [recordFor(start, [{ status: 'timeout', from: -1, to: 600 }, { status: 'completed', from: 601, to: 700 }], true)];
  await advance(40_000);
  assert.equal(bar(p).runnerRecords.length, 1, 'the record shows up without 更新');
  const after = requests(RUNS).sent;
  await advance(120_000);
  assert.equal(requests(RUNS).sent, after, 'no more reads once the record is there and nothing runs');
  p.unmount();
});

test('when the status leaves running, the page polls for the grace period, reads a late event, then stops', async () => {
  resetBrowser();
  const run = makeRun();
  const data = { ...KEY, attempt_index: 1 };
  const e1 = addEpisode(run, 'e1', { status: 'running', ...data }, [ev('e1', 1, 'episode_start', data)]);
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e1');
  await advance(5000);
  e1.events.push(ev('e1', 2, 'episode_end', { end_reason: 'completed', trace_complete: true }));
  e1.summary = { ...e1.summary, status: 'ended', trace_complete: true };
  await advance(3000);
  assert.match(bar(p).poll.text, /終了後に届くイベントを確認中/);
  // a thread still running writes after episode_end
  e1.events.push(ev('e1', 3, 'trace_warning', { code: 'late_event' }, { late: true }));
  e1.summary = { ...e1.summary, late_events: 1 };
  await advance(70_000);
  assert.equal(bar(p).reader.count, 3, 'the late event was read by the grace polls');
  assert.doesNotMatch(bar(p).poll.text, /確認中/);
  const events = requests(/\/events$/).sent;
  await advance(30_000);
  assert.equal(requests(/\/events$/).sent, events, 'polling stopped after the grace period');
  p.unmount();
});

test('opening after episode_end while video copy is pending keeps polling until trial_videos arrives', async () => {
  resetBrowser();
  const run = makeRun();
  const e1 = endedEpisode(run, 'e1', clock.now / 1000 - 100);
  e1.summary = {...e1.summary, episode: {video_copy_pending: true, updated_wall_time: clock.now / 1000}};
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e1');
  await advance(3000);
  assert.match(bar(p).poll.text, /動画を保存中/);
  const before = requests(/\/events$/).sent;
  await advance(65_000);
  assert.ok(requests(/\/events$/).sent > before + 10, 'pending keeps polling past the normal grace maximum');
  e1.events.push(ev('e1', 3, 'trial_videos', {}, {artifacts: [{path:'videos/video_combined.mp4',mime:'video/mp4',written:true}]}));
  e1.summary = {...e1.summary, episode: {video_copy_pending: false, updated_wall_time: clock.now / 1000}};
  await advance(3000);
  assert.equal(bar(p).reader.count, 3);
  assert.doesNotMatch(bar(p).poll.text, /動画を保存中/);
  p.unmount();
});

test('after watching an attempt end, the records are re-read although no other episode was running yet', async () => {
  resetBrowser();
  const run = makeRun();
  const start = clock.now / 1000;
  const a1data = { ...KEY, attempt_index: 1, runner_invocation: 1 };
  const a1 = addEpisode(run, 'e_a1', { status: 'running', ...a1data, start_wall_time: start }, [ev('e_a1', 1, 'episode_start', a1data, { wall_time: start })]);
  serve(run);
  const p = open('/traces?view=details&run=r1&episode=e_a1');
  // watched live for two minutes: the run was last read when the page opened
  await advance(120_000);
  assert.equal(requests(RUNS).sent, 1);
  // attempt 1 times out; nothing else runs yet (the retry resets the scene first)
  a1.events.push(ev('e_a1', 2, 'episode_end', { end_reason: 'timeout_alarm', trace_complete: true }));
  a1.summary = { ...a1.summary, status: 'ended', end_reason: 'timeout_alarm' };
  await advance(70_000); // the grace period ends; attempt 2 has not started
  assert.doesNotMatch(bar(p).poll.text, /確認中/);
  const a2data = { ...KEY, attempt_index: 2, runner_invocation: 1 };
  const a2 = addEpisode(run, 'e_a2', { status: 'running', ...a2data, start_wall_time: clock.now / 1000 }, [ev('e_a2', 1, 'episode_start', a2data, { wall_time: clock.now / 1000 })]);
  await advance(60_000);
  // attempt 2 ends; the runner appends the trial's record
  a2.summary = { ...a2.summary, status: 'ended' };
  run.records = [recordFor(start, [{ status: 'timeout', from: -1, to: 120 }, { status: 'completed', from: 200, to: 260 }], true)];
  await advance(90_000);
  assert.equal(bar(p).runnerRecords.length, 1, 'the record shows up without 更新');
  p.unmount();
});

test('the top bar is told that the records are being read, or that reading them failed', async () => {
  resetBrowser();
  const run = makeRun();
  const start = clock.now / 1000 - 700;
  endedEpisode(run, 'e1', start);
  run.latency.run = 2000;
  serve(run);
  const inner = net.handler;
  let failRuns = true;
  net.handler = (req) => (failRuns && RUNS.test(req.path) ? { status: 500, json: { detail: 'boom' }, latency: 2000 } : inner(req));
  const p = open('/traces?view=details&run=r1&episode=e1');
  await advance(1000);
  assert.deepEqual(bar(p).runnerRead, { loaded: false, loading: true, error: null });
  await advance(3000);
  assert.deepEqual(bar(p).runnerRead, { loaded: false, loading: false, error: 'HTTP 500: boom' });
  failRuns = false;
  run.records = [recordFor(start, [{ status: 'completed', from: -1, to: 60 }], true)];
  bar(p).onRefresh();
  await advance(3000);
  assert.deepEqual(bar(p).runnerRead, { loaded: true, loading: false, error: null });
  assert.equal(bar(p).runnerRecords.length, 1);
  // a later read fails: the records read before stay, and the failure is passed on
  failRuns = true;
  bar(p).onRefresh();
  await advance(1000);
  assert.deepEqual(bar(p).runnerRead, { loaded: true, loading: true, error: null }, 'data stays present during reload');
  await advance(3000);
  assert.deepEqual(bar(p).runnerRead, { loaded: true, loading: false, error: 'HTTP 500: boom' });
  assert.equal(bar(p).runnerRecords.length, 1);
  p.unmount();
});
