/**
 * When the page polls (live.ts):
 * - after the episode stops running, events and summary keep being polled
 *   until two quiet event polls and an unchanged summary, at most 60 s;
 * - in Browse mode a running episode's summary is still re-read (5 s),
 *   so the status and the live-3D tab follow the real state.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { BROWSE_SUMMARY_MS, GRACE_MAX_MS, GRACE_MIN_MS, INTERRUPTED_COPY_POLL_MS, graceActive, live3dState, pollPlan, summaryDigest, videoCopyFinished, videoCopyInterrupted, videoCopyMessage } from '../live';
import { EventsPager, type PagerSnapshot } from '../pager';
import type { EpisodeSummary, EventsPage, TraceEvent } from '../types';

test('poll plan per mode', () => {
  assert.deepEqual(pollPlan({ live: true, running: true, grace: false }), { events: true, summaryMs: 1000 });
  assert.deepEqual(pollPlan({ live: false, running: true, grace: false }), { events: false, summaryMs: BROWSE_SUMMARY_MS });
  assert.deepEqual(pollPlan({ live: false, running: false, grace: true }), { events: true, summaryMs: 1000 });
  assert.deepEqual(pollPlan({ live: true, running: false, grace: false }), { events: false, summaryMs: null });
  assert.deepEqual(pollPlan({ live: false, running: false, grace: false, videoPending: true }), { events: true, summaryMs: 1000 });
  assert.deepEqual(pollPlan({ live: false, running: false, grace: false, videoPending: true, videoCopyInterrupted: true }), { events: false, summaryMs: INTERRUPTED_COPY_POLL_MS });
});

test('copy pending keeps polling when an ended episode is opened and reports interrupted copies',()=>{
 const s:EpisodeSummary={episode_id:'e',status:'ended',episode:{video_copy_pending:true,updated_wall_time:1000}};
 assert.equal(videoCopyMessage(s,1001_000),'動画を保存中');
 assert.match(videoCopyMessage(s,1121_000)!,/保存されていません/);
 assert.match(videoCopyMessage({...s,heartbeat_age_s:121},1001_000)!,/保存されていません/);
 assert.equal(videoCopyInterrupted(s,1121_000),true);
 assert.equal(pollPlan({live:false,running:false,grace:false,videoPending:true,videoCopyInterrupted:videoCopyInterrupted(s,1121_000)}).summaryMs,20_000);
 const resumed={...s,heartbeat_age_s:1};
 assert.equal(videoCopyInterrupted(resumed,1122_000),false);
 assert.equal(pollPlan({live:false,running:false,grace:false,videoPending:true,videoCopyInterrupted:videoCopyInterrupted(resumed,1122_000)}).events,true);
 const recovered={...s,episode:{video_copy_pending:false,updated_wall_time:1122}};
 assert.equal(videoCopyInterrupted(recovered,1122_000),false);
 assert.equal(videoCopyFinished(true,recovered.episode.video_copy_pending),true);
 assert.equal(pollPlan({live:false,running:false,grace:false,videoPending:recovered.episode.video_copy_pending,videoCopyInterrupted:videoCopyInterrupted(recovered,1122_000)}).events,false);
 assert.equal(videoCopyMessage({...s,episode:{video_copy_pending:false}},1121_000),null);
 assert.notEqual(summaryDigest(s),summaryDigest({...s,episode:{video_copy_pending:false}}));
});

test('the grace period lasts at least 15 s, then until two quiet event polls and a stable summary', () => {
  const g = { startedAt: 0, passesAtStart: 10, summaryReadsAtStart: 4 };
  const at = (now: number, passes: number, quiet: number, reads: number, stable: number) =>
    graceActive(g, { now, passes, quietPasses: quiet, summaryReads: reads, stableSummaryReads: stable });
  assert.equal(at(4000, 14, 3, 8, 2), true, 'quiet, but late events can still come: not before the minimum');
  const t = GRACE_MIN_MS;
  assert.equal(at(t, 10, 5, 4, 3), true, 'quiet counters from before the end do not count');
  assert.equal(at(t + 1000, 11, 0, 5, 0), true, 'the tail just came in');
  assert.equal(at(t + 2000, 12, 1, 6, 0), true);
  assert.equal(at(t + 3000, 13, 2, 7, 0), true, 'the summary still changed (late events counted)');
  assert.equal(at(t + 4000, 14, 2, 8, 1), false, 'two quiet polls and an unchanged summary end it');
  assert.equal(at(GRACE_MAX_MS, 11, 0, 5, 0), false, 'never longer than the maximum');
  assert.equal(graceActive(null, { now: 0, passes: 0, quietPasses: 0, summaryReads: 0, stableSummaryReads: 0 }), false);
});

test('the summary digest changes when late events or gaps are counted', () => {
  const base: EpisodeSummary = { episode_id: 'e', status: 'ended', trace_complete: true, late_events: 0 };
  assert.equal(summaryDigest(base), summaryDigest({ ...base, task: 'other field' }));
  assert.notEqual(summaryDigest(base), summaryDigest({ ...base, late_events: 1, trace_complete: false }));
  assert.equal(summaryDigest(null), null);
});

test('live 3D only while the episode runs and the proxy serves its recorded port', () => {
  const ok = { running: true, viserPort: 8080, viewerOnly: false, viser: { available: true, port: 8080 }, viserError: null };
  assert.equal(live3dState(ok).available, true);
  assert.equal(live3dState({ ...ok, running: false }).available, false, 'an ended episode (status re-read in Browse) never offers it');
  assert.equal(live3dState({ ...ok, viser: { available: true, port: 8081 } }).available, false);
  assert.equal(live3dState({ ...ok, viewerOnly: true }).available, false);
  assert.equal(live3dState({ ...ok, viserPort: null }).available, false);
});

test('a late event written 6 s after episode_end is read by the grace-period polls, which then stop', async () => {
  // A fake events.jsonl (one line = 100 bytes) served the way /events pages it.
  const lines: TraceEvent[] = [
    { event_id: 'start', seq: 1, event_type: 'episode_start' },
    { event_id: 'end', seq: 2, event_type: 'episode_end' },
  ];
  const fetchPage = async (after: number): Promise<EventsPage> => ({
    events: lines.slice(after / 100),
    next_offset: lines.length * 100,
    eof: true,
    generation: 'g1',
  });
  let snap: PagerSnapshot | null = null;
  const pager = new EventsPager(fetchPage, (s) => (snap = s));
  await pager.pump();
  // The page saw the status leave 'running' at t = 0 (Live mode): the grace period starts.
  const grace = { startedAt: 0, passesAtStart: snap!.passes, summaryReadsAtStart: 0 };
  let summary: EpisodeSummary = { episode_id: 'e', status: 'ended', late_events: 0, trace_complete: true };
  let reads = 0;
  let stable = 0;
  let last: string | null = null;
  let stoppedAt: number | null = null;
  for (let now = 1000; now <= GRACE_MAX_MS + 5000; now += 1000) {
    if (now === 6000) {
      // a web UI thread writes after episode_end; the writer then updates episode.json
      lines.push({ event_id: 'late', seq: 3, event_type: 'trace_warning', late: true });
      summary = { ...summary, late_events: 1, trace_complete: false };
    }
    const s = snap as PagerSnapshot | null;
    const active = graceActive(grace, { now, passes: s!.passes, quietPasses: s!.quietPasses, summaryReads: reads, stableSummaryReads: stable });
    const plan = pollPlan({ live: true, running: false, grace: active });
    if (!plan.events) {
      stoppedAt = now;
      break;
    }
    await pager.pump();
    const d = summaryDigest(summary);
    stable = d === last ? stable + 1 : 0;
    last = d;
    reads += 1;
  }
  const final = snap as PagerSnapshot | null;
  assert.deepEqual(final!.events.map((e) => e.event_id), ['start', 'end', 'late']);
  assert.ok(stoppedAt !== null && stoppedAt >= GRACE_MIN_MS && stoppedAt < GRACE_MAX_MS, `polling stopped at ${stoppedAt} ms`);
});
