/**
 * - In the model-input tab, the newer of the page-level range focus
 *   and the tab's own position link wins.
 * - Only /traces and paths below it open the viewer.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { nextFocusToken, pickFocus } from '../focus';
import { isTraceViewerPath } from '../route';

test('a position link clicked after arriving from a range link wins; a newer range link wins again', () => {
  const page = { token: nextFocusToken() };
  assert.deepEqual(pickFocus(page, null), { id: 'focus', token: page.token, source: 'page' });
  const local = { id: 'seg:code_history:0', token: nextFocusToken() };
  assert.equal(pickFocus(page, local).source, 'local');
  assert.equal(pickFocus(page, local).id, 'seg:code_history:0');
  const newerPage = { token: nextFocusToken() };
  assert.equal(pickFocus(newerPage, local).source, 'page');
  assert.equal(pickFocus(null, null).id, null);
});

test('focus tokens strictly increase, also within one millisecond', () => {
  const a = nextFocusToken();
  const b = nextFocusToken();
  const c = nextFocusToken();
  assert.ok(a < b && b < c);
});

test('route check', () => {
  for (const p of ['/traces', '/traces/', '/traces/run/ep']) assert.equal(isTraceViewerPath(p), true, p);
  for (const p of ['/', '/traces-old', '/tracesX', '/api/traces', '/trace']) assert.equal(isTraceViewerPath(p), false, p);
});
