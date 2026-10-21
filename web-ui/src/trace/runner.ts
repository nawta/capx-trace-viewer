/**
 * Which runner record (runner_records.jsonl, docs/trace-format.md 5.2) holds
 * the result of this episode's trial, and which of its attempts is this
 * episode.
 *
 * The runner appends one record per trial when the trial's last attempt ends.
 * The record lists every attempt (a timed-out attempt is retried with a fresh
 * reset, up to --max-attempts), and its `outcome` is classified from the LAST
 * attempt only (the runner that wrote the trace reports only the last attempt as final). An
 * episode is one attempt (…__a1, …__a2), so `outcome` is this episode's result
 * only when this episode is the last attempt. For an earlier attempt the top
 * bar shows that attempt's own status and names the attempt the trial's final
 * result belongs to.
 *
 * The runner also appends a new record for the same (task, condition, seed)
 * when a later invocation redoes or retries a trial (--redo, --retry-timeouts,
 * --retry-infra-errors), and the trace run keeps the episodes of every
 * invocation (…__i1__a1, …__i2__a1). A record carries no invocation number,
 * so it is matched by time: the record's attempt with this episode's attempt
 * number must have started before the episode and ended after its start
 * (runner times are ISO strings cut to whole seconds, hence a small margin).
 * Zero or several matches are reported with the reason; the top bar then
 * shows no result instead of another invocation's.
 *
 * An attempt that starts and ends in the same second and does not contain the
 * episode start is a record the parent wrote when a worker failed or exited
 * (the runner error record, always attempt 1, stamped when the failure was
 * noticed). It cannot be checked by time, and it is reported as such, not as
 * another invocation's record.
 *
 * Related: components/TopBar.tsx, TraceViewerPage.tsx (re-reads the records).
 */

import type { EpisodeSummary, RunnerRecord } from './types';

/** Slack for the runner's whole-second timestamps. */
const MARGIN_MS = 2000;

export type RunnerAttempt = NonNullable<RunnerRecord['attempts']>[number];

export interface EpisodeKey {
  task: unknown;
  condition: unknown;
  seed: unknown;
  /** episode_start.attempt_index (1-based, the runner's attempt_id). */
  attemptIndex: number | null;
  /** Episode start, seconds since the epoch (episode_start wall_time or the summary). */
  startWall: number | null;
}

export interface RunnerMatch {
  record: RunnerRecord | null;
  /** The attempt of `record` that is this episode, when known. */
  attemptId: number | null;
  attempt: RunnerAttempt | null;
  /** The record's last attempt: `record.outcome` was classified from it. */
  finalAttempt: RunnerAttempt | null;
  /** This episode is the attempt that `record.outcome` describes. */
  isFinal: boolean;
  /** Records with the same task, condition and seed. */
  candidates: number;
  /** Why no record is shown although candidates exist (null otherwise). */
  problem: string | null;
  /** A record is shown without a time check (the episode's start time is unknown). */
  unverified: boolean;
}

/** The runner-record key of an episode: episode_start first, the summary when that is missing. */
export function episodeKeyOf(
  start: { task?: unknown; condition?: unknown; seed?: unknown; attempt_index?: unknown } | null | undefined,
  summary: EpisodeSummary | null | undefined,
  startWall: number | null | undefined,
): EpisodeKey {
  const attempt = start?.attempt_index ?? summary?.attempt_index ?? null;
  return {
    task: start?.task ?? summary?.task ?? null,
    condition: start?.condition ?? summary?.condition ?? null,
    seed: start?.seed ?? summary?.seed ?? null,
    attemptIndex: typeof attempt === 'number' ? attempt : null,
    startWall: typeof startWall === 'number' && Number.isFinite(startWall) ? startWall : null,
  };
}

function parseTime(v: unknown): number | null {
  if (typeof v !== 'string' || !v) return null;
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : null;
}

function attemptsOf(r: RunnerRecord): RunnerAttempt[] {
  return (r.attempts ?? []).filter((a): a is RunnerAttempt => !!a && typeof a === 'object');
}

function lastAttempt(r: RunnerRecord): RunnerAttempt | null {
  const all = attemptsOf(r);
  return all.length ? all[all.length - 1] : null;
}

/** Starts and ends in the same second (whole-second timestamps), as the runner error record writes it. */
function isPointInTime(a: RunnerAttempt): boolean {
  const s = parseTime(a.started_at);
  const e = parseTime(a.ended_at);
  return s !== null && e !== null && e - s < 1000;
}

type Fit = { fit: 'hit'; attempt: RunnerAttempt } | { fit: 'miss' | 'point' | 'untimed' | 'no_attempt'; attempt: null };

/** How one record's attempts relate to the episode start (`start` in ms). */
function fitOf(r: RunnerRecord, attemptIndex: number | null, start: number): Fit {
  const all = attemptsOf(r);
  const mine = attemptIndex === null ? all : all.filter((a) => a.attempt_id === attemptIndex);
  if (mine.length === 0) {
    // _error_record always says attempt 1, whichever attempt failed: it may still be this trial's.
    return { fit: all.some(isPointInTime) ? 'point' : 'no_attempt', attempt: null };
  }
  let timed = false;
  let point = false;
  for (const a of mine) {
    const s = parseTime(a.started_at);
    const e = parseTime(a.ended_at);
    if (s === null || e === null) continue;
    if (s - MARGIN_MS <= start && start <= e + MARGIN_MS) return { fit: 'hit', attempt: a };
    if (isPointInTime(a)) point = true;
    else timed = true;
  }
  if (point) return { fit: 'point', attempt: null };
  return { fit: timed ? 'miss' : 'untimed', attempt: null };
}

function matched(record: RunnerRecord, attempt: RunnerAttempt | null, candidates: number, unverified: boolean): RunnerMatch {
  const finalAttempt = lastAttempt(record);
  const isFinal =
    attempt !== null &&
    finalAttempt !== null &&
    (attempt === finalAttempt || (typeof attempt.attempt_id === 'number' && attempt.attempt_id === finalAttempt.attempt_id));
  return { record, attemptId: attempt?.attempt_id ?? null, attempt, finalAttempt, isFinal, candidates, problem: null, unverified };
}

export function runnerRecordFor(records: RunnerRecord[], ep: EpisodeKey): RunnerMatch {
  const keyed = records.filter(
    (r) => r && r.task === ep.task && r.condition === ep.condition && (r.seed === ep.seed || r.trial === ep.seed),
  );
  const none: RunnerMatch = {
    record: null,
    attemptId: null,
    attempt: null,
    finalAttempt: null,
    isFinal: false,
    candidates: keyed.length,
    problem: null,
    unverified: false,
  };
  if (keyed.length === 0) return none;
  const key = 'task/condition/seed';
  if (ep.startWall === null || !Number.isFinite(ep.startWall)) {
    if (keyed.length === 1) {
      const all = attemptsOf(keyed[0]);
      const a = ep.attemptIndex !== null ? all.find((x) => x.attempt_id === ep.attemptIndex) ?? null : all.length === 1 ? all[0] : null;
      return matched(keyed[0], a, 1, true);
    }
    return { ...none, problem: 'エピソードの開始時刻が記録にないため、どの記録か特定できません' };
  }
  const start = ep.startWall * 1000;
  const hits: { record: RunnerRecord; attempt: RunnerAttempt }[] = [];
  const counts = { miss: 0, point: 0, untimed: 0, no_attempt: 0 };
  for (const r of keyed) {
    const f = fitOf(r, ep.attemptIndex, start);
    if (f.fit === 'hit') hits.push({ record: r, attempt: f.attempt });
    else counts[f.fit] += 1;
  }
  if (hits.length === 1) return matched(hits[0].record, hits[0].attempt, keyed.length, false);
  if (hits.length > 1) {
    return { ...none, problem: `同じ ${key} の runner 記録 ${keyed.length} 件のうち ${hits.length} 件が当てはまり、どれか特定できません` };
  }
  const which = ep.attemptIndex !== null ? ` attempt ${ep.attemptIndex} ` : '試行';
  if (counts.point === 0 && counts.untimed === 0) {
    // Every record's attempts were timed and lie elsewhere, or the record has no such attempt.
    return { ...none, problem: `同じ ${key} の runner 記録 ${keyed.length} 件のどれも、このエピソードの開始時刻を含む${which}を持ちません（別の invocation の記録）` };
  }
  const parts: string[] = [];
  if (counts.point) {
    parts.push(
      `${counts.point} 件は試行の開始と終了が同じ時刻の記録で（試行の失敗や worker の終了の後に runner が書いた記録など。このエピソードの trial の記録かもしれません）、時刻では照合できません`,
    );
  }
  if (counts.untimed) parts.push(`${counts.untimed} 件は試行の時刻がなく、照合できません`);
  if (counts.miss) parts.push(`${counts.miss} 件は試行の時刻がこのエピソードの開始を含みません`);
  if (counts.no_attempt) parts.push(`${counts.no_attempt} 件は${which}を含みません`);
  return { ...none, problem: `同じ ${key} の runner 記録 ${keyed.length} 件のどれとも照合できません: ${parts.join('。')}` };
}
