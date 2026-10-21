/**
 * - The runner verdict shown for an episode comes from the record for the same
 *   task, condition, seed and attempt whose attempt window contains the episode's
 *   start, rather than the last record with that key from a later invocation.
 * - A record's outcome belongs to the trial's last attempt only.
 * - A point-in-time error record is not mistaken for another invocation's.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { runnerRecordFor } from '../runner';
import type { RunnerRecord } from '../types';

const iso = (sec: number) => new Date(sec * 1000).toISOString().replace(/\.\d{3}Z$/, '+00:00');

function record(start: number, end: number, success: boolean, attemptId = 1): RunnerRecord {
  return {
    task: 'cube_lift',
    condition: 'C',
    seed: 1,
    trial: 1,
    attempts: [{ attempt_id: attemptId, status: success ? 'completed' : 'timeout', started_at: iso(start), ended_at: iso(end) }],
    outcome: { task_success: success },
  };
}

const I1 = record(1000, 1100, false); // invocation 1: timed out
const I2 = record(5000, 5100, true); // invocation 2 (--retry-timeouts): success
const key = { task: 'cube_lift', condition: 'C', seed: 1, attemptIndex: 1 };

test('each invocation\'s episode gets its own record', () => {
  const m1 = runnerRecordFor([I1, I2], { ...key, startWall: 1000.4 });
  assert.equal(m1.record, I1);
  assert.equal(m1.record?.outcome?.task_success, false);
  assert.equal(m1.attemptId, 1);
  assert.equal(m1.candidates, 2);
  const m2 = runnerRecordFor([I1, I2], { ...key, startWall: 5001 });
  assert.equal(m2.record, I2);
});

test('no record contains the start: none is shown, and the reason is given', () => {
  const m = runnerRecordFor([I1, I2], { ...key, startWall: 3000 });
  assert.equal(m.record, null);
  assert.match(m.problem ?? '', /2 件/);
});

test('two records contain the start: ambiguous, none is shown', () => {
  const m = runnerRecordFor([I1, record(990, 1200, true)], { ...key, startWall: 1050 });
  assert.equal(m.record, null);
  assert.match(m.problem ?? '', /特定できません/);
});

test('the attempt number must match (a retry attempt is another episode)', () => {
  const two: RunnerRecord = { ...I1, attempts: [...(I1.attempts ?? []), { attempt_id: 2, status: 'completed', started_at: iso(1101), ended_at: iso(1200) }] };
  assert.equal(runnerRecordFor([two], { ...key, attemptIndex: 2, startWall: 1150 }).attemptId, 2);
  assert.equal(runnerRecordFor([two], { ...key, attemptIndex: 2, startWall: 1050 }).record, null);
});

test('without a start time: one record is shown as unverified, several are ambiguous; other keys never match', () => {
  const one = runnerRecordFor([I1], { ...key, startWall: null });
  assert.equal(one.record, I1);
  assert.equal(one.unverified, true);
  assert.equal(runnerRecordFor([I1, I2], { ...key, startWall: null }).record, null);
  const other = runnerRecordFor([I1], { ...key, seed: 2, startWall: 1000 });
  assert.equal(other.record, null);
  assert.equal(other.candidates, 0);
  assert.equal(other.problem, null);
});

// ---------------------------------------------------------------------------
//  the record's outcome is the trial's LAST attempt (the last attempt is final)
// ---------------------------------------------------------------------------

const RETRIED: RunnerRecord = {
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

test('an earlier attempt of a retried trial is matched to its own attempt, which is not the final one', () => {
  const a1 = runnerRecordFor([RETRIED], { ...key, attemptIndex: 1, startWall: 1000.5 });
  assert.equal(a1.record, RETRIED);
  assert.equal(a1.attempt?.status, 'timeout');
  assert.equal(a1.finalAttempt?.attempt_id, 2);
  assert.equal(a1.isFinal, false, 'the outcome is attempt 2\'s, not this episode\'s');
  const a2 = runnerRecordFor([RETRIED], { ...key, attemptIndex: 2, startWall: 1601.3 });
  assert.equal(a2.attemptId, 2);
  assert.equal(a2.isFinal, true);
  // one attempt: that attempt is the final one
  assert.equal(runnerRecordFor([I1], { ...key, startWall: 1000.4 }).isFinal, true);
  // without a start time the attempt still comes from the attempt number
  const u = runnerRecordFor([RETRIED], { ...key, attemptIndex: 1, startWall: null });
  assert.equal(u.unverified, true);
  assert.equal(u.isFinal, false);
  assert.equal(u.attempt?.attempt_id, 1);
});

// ---------------------------------------------------------------------------
//  the runner error record (one attempt, started_at == ended_at == when the parent noticed)
// ---------------------------------------------------------------------------

test('an error record written after the episode began is "cannot check by time", never "another invocation"', () => {
  const err: RunnerRecord = {
    task: 'cube_lift',
    condition: 'C',
    seed: 1,
    trial: 1,
    attempts: [{ attempt_id: 1, status: 'error', started_at: iso(1095), ended_at: iso(1095), duration_s: 0, error: 'worker exited before reporting this trial' }],
    outcome: { task_success: false, termination_reason: 'error' },
  };
  const m = runnerRecordFor([err], { ...key, startWall: 1000.2 });
  assert.equal(m.record, null);
  assert.doesNotMatch(m.problem ?? '', /別の invocation/);
  assert.match(m.problem ?? '', /開始と終了が同じ時刻の記録/);
  // the same holds for an attempt-2 episode (the error record always says attempt 1)
  assert.match(runnerRecordFor([err], { ...key, attemptIndex: 2, startWall: 1000.2 }).problem ?? '', /開始と終了が同じ時刻/);
  // an error record whose second contains the episode start is still that attempt
  assert.equal(runnerRecordFor([err], { ...key, startWall: 1095.4 }).record, err);
  // records whose timed attempts all lie elsewhere are another invocation's
  assert.match(runnerRecordFor([I1, I2], { ...key, startWall: 3000 }).problem ?? '', /別の invocation/);
  // a record without this attempt number
  assert.match(runnerRecordFor([I1], { ...key, attemptIndex: 3, startWall: 1000.4 }).problem ?? '', /attempt 3 を持ちません/);
});
