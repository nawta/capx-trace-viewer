/**
 * Line diff of the input-diff tab (diff.ts): the edit script turns the old
 * text into the new one, it is minimal (checked against an LCS table on
 * random inputs), and a diff over the size limit gives up and says so.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { diffLines, diffSequences, toHunks } from '../diff';

function lcs(a: string[], b: string[]): number {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  return dp[0][0];
}

/** Deterministic pseudo-random numbers (no Math.random, so failures reproduce). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

test('the edit script rebuilds both texts', () => {
  const res = diffLines('a\nb\nc\nd', 'a\nx\nc\nd\ne');
  const oldBack = res.ops.filter((o) => o.type !== 'insert').map((o) => o.text);
  const newBack = res.ops.filter((o) => o.type !== 'delete').map((o) => o.text);
  assert.deepEqual(oldBack, ['a', 'b', 'c', 'd']);
  assert.deepEqual(newBack, ['a', 'x', 'c', 'd', 'e']);
  assert.equal(res.added, 2);
  assert.equal(res.removed, 1);
  assert.equal(res.gaveUp, false);
});

test('random inputs: the script is valid and minimal (compared with an LCS table)', () => {
  const rand = rng(7);
  for (let n = 0; n < 1500; n += 1) {
    const alphabet = ['a', 'b', 'c', 'd'];
    const la = Math.floor(rand() * 12);
    const lb = Math.floor(rand() * 12);
    const a = Array.from({ length: la }, () => alphabet[Math.floor(rand() * alphabet.length)]);
    const b = Array.from({ length: lb }, () => alphabet[Math.floor(rand() * alphabet.length)]);
    const res = diffSequences(a, b);
    assert.deepEqual(res.ops.filter((o) => o.type !== 'insert').map((o) => o.text), a);
    assert.deepEqual(res.ops.filter((o) => o.type !== 'delete').map((o) => o.text), b);
    const equal = res.ops.filter((o) => o.type === 'equal').length;
    assert.equal(equal, lcs(a, b), `case ${n}: ${a.join('')} -> ${b.join('')}`);
  }
});

test('a diff over maxD gives up (all removed, all added) and says so', () => {
  const a = Array.from({ length: 50 }, (_, i) => `a${i}`);
  const b = Array.from({ length: 50 }, (_, i) => `b${i}`);
  const res = diffSequences(a, b, 10);
  assert.equal(res.gaveUp, true);
  assert.equal(res.removed, 50);
  assert.equal(res.added, 50);
});

test('hunks keep 3 context lines and count folded equal lines', () => {
  const old = Array.from({ length: 20 }, (_, i) => `l${i}`);
  const neu = [...old];
  neu[10] = 'changed';
  const { hunks, foldedAfter } = toHunks(diffSequences(old, neu).ops, 3);
  assert.equal(hunks.length, 1);
  assert.equal(hunks[0].foldedBefore, 7);
  assert.equal(hunks[0].ops.length, 8);
  assert.equal(foldedAfter, 6);
  assert.deepEqual(toHunks(diffSequences(old, old).ops).hunks, []);
});
