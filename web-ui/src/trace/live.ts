/**
 * When the viewer page polls (docs/trace-format.md 9.3 Live and Browse; 4.4
 * events that arrive after episode_end).
 *
 * - Live on a running episode: events and the episode summary about every
 *   second, and the newest event stays selected.
 * - Browse on a running episode: events are not polled (the view stays where
 *   the person left it), but the summary is re-read every 5 s, so the status,
 *   the 「終了イベントなし（実行中）」 markers and the 「ライブ3D」 tab follow the
 *   real state of the episode.
 * - After the episode stops running (in either mode): a grace period that
 *   polls events and summary every second, so the tail of events.jsonl and
 *   late events (written after episode_end, for example by web UI or ensemble
 *   threads still running) are read. It lasts at least 15 s; after that it
 *   ends when two consecutive event polls found nothing new and two summary
 *   reads in a row were identical, and it never lasts more than 60 s. Late
 *   events after that are read with the 更新 button.
 * - Runner records: the runner appends a trial's record only when its last
 *   attempt ends, which can be long after this episode ended (a timed-out
 *   attempt is retried). While this episode has no record and the run was
 *   seen with a running episode within the last minute (in the run's episode
 *   list, or this episode itself until it stopped running), the run is
 *   re-read every 30 s (runnerPollActive).
 *
 * A poll tick is skipped while the previous read is still in flight
 * (hooks.ts useAsync `poll`), so a slow read is never aborted by the next tick;
 * only a read that has been in flight for 60 s is given up and started again.
 *
 * Pure functions; TraceViewerPage.tsx holds the state.
 */

import type { EpisodeSummary } from './types';

export const LIVE_POLL_MS = 1000;
export const BROWSE_SUMMARY_MS = 5000;
export const GRACE_POLL_MS = 1000;
export const GRACE_MAX_MS = 60_000;
/** Late events can follow episode_end by seconds; the grace period never ends before this. */
export const GRACE_MIN_MS = 15_000;
/** Consecutive quiet polls that end the grace period. */
export const GRACE_QUIET_POLLS = 2;

/** Interval of runner-record reads while this episode's record may still be appended. */
export const RUNNER_POLL_MS = 30_000;
/** How long after the run was last seen with a running episode the records are still read. */
export const RUNNER_POLL_AFTER_MS = 2 * RUNNER_POLL_MS;

/**
 * True while the page should re-read the run for this episode's runner record: an episode
 * (not the unscoped log) is open, it is not running and not in its grace period (whose end
 * reads the records anyway), it has no record yet, and the run was seen with a running
 * episode less than RUNNER_POLL_AFTER_MS ago (a /runs read listed one, or this episode
 * stopped running then), so the record written after the trial's last attempt ended is
 * still read.
 */
export function runnerPollActive(o: { open: boolean; thisRunning: boolean; hasRecord: boolean; runSeenRunningAt: number | null; now: number }): boolean {
  if (!o.open || o.thisRunning || o.hasRecord || o.runSeenRunningAt === null) return false;
  return o.now - o.runSeenRunningAt < RUNNER_POLL_AFTER_MS;
}

export interface GraceStart {
  /** Date.now() when the episode was seen to stop running. */
  startedAt: number;
  /** Completed event passes and summary reads at that moment. */
  passesAtStart: number;
  summaryReadsAtStart: number;
}

export interface PollProgress {
  now: number;
  /** Completed event passes (pager.ts). */
  passes: number;
  /** Consecutive completed passes that found no new event. */
  quietPasses: number;
  /** Summary reads so far. */
  summaryReads: number;
  /** Consecutive summary reads identical to the one before. */
  stableSummaryReads: number;
}

/** True while the page should keep polling after the episode stopped running. */
export function graceActive(g: GraceStart | null, p: PollProgress): boolean {
  if (!g) return false;
  if (p.now - g.startedAt >= GRACE_MAX_MS) return false;
  if (p.now - g.startedAt < GRACE_MIN_MS) return true;
  const passes = p.passes - g.passesAtStart;
  const reads = p.summaryReads - g.summaryReadsAtStart;
  const quiet =
    passes >= GRACE_QUIET_POLLS &&
    p.quietPasses >= GRACE_QUIET_POLLS &&
    reads >= GRACE_QUIET_POLLS &&
    p.stableSummaryReads >= GRACE_QUIET_POLLS - 1;
  return !quiet;
}

export interface PollPlan {
  /** Poll events.jsonl. */
  events: boolean;
  /** Interval of summary reads, or null for none. */
  summaryMs: number | null;
}

export function pollPlan(o: { live: boolean; running: boolean; grace: boolean }): PollPlan {
  if (o.grace) return { events: true, summaryMs: GRACE_POLL_MS };
  if (o.running && o.live) return { events: true, summaryMs: LIVE_POLL_MS };
  if (o.running) return { events: false, summaryMs: BROWSE_SUMMARY_MS };
  return { events: false, summaryMs: null };
}

/** The summary fields whose change means the episode's record changed (status, gaps, late events). */
export function summaryDigest(s: EpisodeSummary | null | undefined): string | null {
  if (!s) return null;
  return JSON.stringify([
    s.status ?? null,
    s.trace_complete ?? null,
    s.end_reason ?? null,
    s.end_wall_time ?? null,
    s.late_events ?? null,
    s.dropped_events ?? null,
    s.n_warnings ?? null,
    s.gaps ?? null,
  ]);
}

export interface Live3dState {
  available: boolean;
  /** Why live 3D is not offered (shown as a tooltip). */
  reason: string;
}

/**
 * The 「ライブ3D」 tab (9.3): only while the selected episode is running (by the status
 * re-read above, also in Browse mode), its recorded Viser port is the one the server's
 * proxy serves, and the server is not viewer-only.
 */
export function live3dState(o: {
  running: boolean;
  viserPort: number | null | undefined;
  viewerOnly: boolean;
  viser: { available?: boolean | null; port?: number | null } | null;
  viserError: string | null;
}): Live3dState {
  if (!o.running) return { available: false, reason: 'エピソードは実行中ではありません' };
  if (o.viserPort === null || o.viserPort === undefined) return { available: false, reason: 'このエピソードは Viser のポートを記録していません' };
  if (o.viewerOnly) return { available: false, reason: 'viewer-only サーバーには /viser-proxy がありません' };
  if (!o.viser?.available) return { available: false, reason: o.viserError ?? 'サーバーのプロキシから Viser に接続できません' };
  if (o.viser.port !== o.viserPort) return { available: false, reason: `プロキシが中継しているのは別のポート（${o.viser.port}）です` };
  return { available: true, reason: '' };
}
