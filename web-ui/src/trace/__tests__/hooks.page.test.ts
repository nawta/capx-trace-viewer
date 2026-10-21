/**
 * hooks.ts useAsync with its effects running in the test harness:
 * - poll() while a read is in flight is skipped; that read is not aborted;
 * - reload() while a read is in flight reads once more after it (several
 *   reloads give one more read), so a change made before the call is seen;
 * - new deps (another episode) still abort the old read, and its result is
 *   never shown under the new deps;
 * - a read that never settles is given up after STALLED_READ_MS, so one hung
 *   request cannot stop the polling for good.
 */

import { advance, net, requests, resetBrowser } from './harness/browser';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mount } from './harness/fakeReact';
import { STALLED_READ_MS, useAsync, type AsyncState } from '../hooks';

interface Answer {
  key: string;
  n: number;
}

/** `latency.ms` can be changed while the probe is mounted (a hung server that recovers). */
function setup(latencyMs: number) {
  resetBrowser();
  let n = 0;
  const latency = { ms: latencyMs };
  net.handler = (req) => (req.path === '/x' ? { json: { key: req.query.get('k'), n: ++n }, latency: latency.ms } : null);
  const seen: { state: AsyncState<Answer> | null } = { state: null };
  function Probe(props: { k: string }) {
    seen.state = useAsync(
      (signal) => fetch(`/x?k=${props.k}`, { signal }).then((r) => r.json() as Promise<Answer>),
      [props.k],
    );
    return null;
  }
  return { seen, Probe, latency };
}

function state(seen: { state: AsyncState<Answer> | null }): AsyncState<Answer> {
  if (!seen.state) throw new Error('not rendered');
  return seen.state;
}

test('useAsync: a poll tick during a read in flight is skipped and never aborts it', async () => {
  const { seen, Probe } = setup(1500);
  const p = mount(Probe, { k: 'a' });
  await advance(500);
  state(seen).poll();
  await advance(500);
  state(seen).poll();
  await advance(600);
  assert.equal(state(seen).data?.n, 1);
  assert.deepEqual([requests(/^\/x$/).sent, requests(/^\/x$/).aborted], [1, 0]);
  state(seen).poll(); // nothing in flight now: a new read
  await advance(1600);
  assert.equal(state(seen).data?.n, 2);
  p.unmount();
});

test('useAsync: reloads during a read in flight give exactly one more read after it', async () => {
  const { seen, Probe } = setup(1000);
  const p = mount(Probe, { k: 'a' });
  await advance(200);
  state(seen).reload();
  state(seen).reload();
  await advance(900); // the first read is done; the follow-up read started
  assert.equal(state(seen).data?.n, 1);
  assert.equal(requests(/^\/x$/).sent, 2);
  await advance(1100);
  assert.equal(state(seen).data?.n, 2);
  await advance(3000);
  assert.deepEqual([requests(/^\/x$/).sent, requests(/^\/x$/).aborted], [2, 0], 'two reloads, one extra read');
  p.unmount();
});

test('useAsync: new deps abort the old read, and its result is never shown under the new deps', async () => {
  const { seen, Probe } = setup(1000);
  const p = mount(Probe, { k: 'a' });
  await advance(1100);
  assert.equal(state(seen).data?.key, 'a');
  state(seen).reload();
  await advance(300);
  p.update({ k: 'b' });
  await advance(10);
  assert.equal(state(seen).data, null, 'nothing of "a" under "b"');
  assert.equal(state(seen).loading, true);
  await advance(1100);
  assert.equal(state(seen).data?.key, 'b');
  assert.equal(requests(/^\/x$/).aborted, 1, 'the read for "a" was aborted');
  p.unmount();
});

test('useAsync: a read still in flight after STALLED_READ_MS is given up by the next poll, so polling goes on', async () => {
  const { seen, Probe, latency } = setup(10 * 60_000); // the server hangs: no answer for 10 minutes
  const p = mount(Probe, { k: 'a' });
  for (let t = 1000; t < STALLED_READ_MS; t += 1000) {
    await advance(1000);
    state(seen).poll();
  }
  assert.deepEqual([requests(/^\/x$/).sent, requests(/^\/x$/).aborted], [1, 0], 'kept while younger than STALLED_READ_MS');
  latency.ms = 200; // the server answers again
  await advance(1000);
  state(seen).poll(); // the first read is now STALLED_READ_MS old: given up, read again
  assert.deepEqual([requests(/^\/x$/).sent, requests(/^\/x$/).aborted], [2, 1]);
  await advance(300);
  assert.equal(state(seen).data?.n, 2);
  assert.equal(state(seen).loading, false);
  p.unmount();
});

test('useAsync: reload() gives up a stalled read at once instead of waiting for it', async () => {
  const { seen, Probe, latency } = setup(10 * 60_000);
  const p = mount(Probe, { k: 'a' });
  await advance(10_000);
  state(seen).reload(); // 10 s old: kept, one more read is queued after it
  assert.equal(requests(/^\/x$/).sent, 1);
  latency.ms = 200;
  await advance(STALLED_READ_MS);
  state(seen).reload(); // now older than STALLED_READ_MS: given up
  assert.deepEqual([requests(/^\/x$/).sent, requests(/^\/x$/).aborted], [2, 1]);
  await advance(300);
  assert.equal(state(seen).data?.n, 2);
  await advance(5000);
  assert.equal(requests(/^\/x$/).sent, 2, 'the queued re-read belonged to the read that was given up');
  p.unmount();
});
