/**
 * The file cache in api.ts. Files of an episode are fetched under the
 * events.jsonl generation (episodeFileUrl), so a replaced trace never gets the
 * old trace's cached file; a body that is not valid JSON is not kept; an
 * explicit retry (`fresh`) fetches again.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { episodeArrayImageUrl, episodeFileUrl, fetchJsonFile, fetchText } from '../api';

function stubFetch(bodies: string[]): { calls: string[] } {
  const calls: string[] = [];
  (globalThis as unknown as { fetch: unknown }).fetch = async (input: unknown) => {
    calls.push(String(input));
    const body = bodies[Math.min(calls.length, bodies.length) - 1];
    return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } });
  };
  return { calls };
}

test('episode file URLs carry the generation, so another trace in the same place gets other URLs', () => {
  const g1 = episodeFileUrl({ runId: 'r', episodeId: 'e', generation: '803-1a2b-0f0e0d0c0b0a0908' }, 'payloads/ev_1.request.json');
  const g2 = episodeFileUrl({ runId: 'r', episodeId: 'e', generation: '803-1a2c-1111111111111111' }, 'payloads/ev_1.request.json');
  assert.equal(g1, '/api/traces/runs/r/episodes/e/file/payloads/ev_1.request.json?g=803-1a2b-0f0e0d0c0b0a0908');
  assert.notEqual(g1, g2);
  assert.equal(episodeFileUrl({ runId: 'r', episodeId: 'e' }, 'a/b.json'), '/api/traces/runs/r/episodes/e/file/a/b.json', 'no generation yet: plain URL');
  assert.equal(
    episodeArrayImageUrl({ runId: 'r', episodeId: 'e', generation: 'g' }, 'assets/x.npz', 'a0', 2),
    '/api/traces/runs/r/episodes/e/array-image/assets/x.npz?name=a0&index=2&g=g',
  );
});

test('a file that is not valid JSON is not cached; the next read fetches it again', async () => {
  const net = stubFetch(['{broken', '{"fixed": true}']);
  const url = '/api/traces/runs/r/episodes/e/file/assets/aa/repaired.json.gz?g=t1';
  await assert.rejects(fetchJsonFile(url), /JSON として読めません/);
  assert.deepEqual(await fetchJsonFile(url), { fixed: true });
  assert.equal(net.calls.length, 2);
  // a good file stays cached
  assert.deepEqual(await fetchJsonFile(url), { fixed: true });
  assert.equal(net.calls.length, 2);
});

test('an explicit retry (fresh) fetches again past the cache', async () => {
  const net = stubFetch(['first', 'second']);
  const url = '/api/traces/runs/r/episodes/e/file/assets/bb/text.txt?g=t2';
  assert.equal(await fetchText(url), 'first');
  assert.equal(await fetchText(url), 'first', 'cached');
  assert.equal(await fetchText(url, { fresh: true }), 'second');
  assert.equal(net.calls.length, 2);
});
