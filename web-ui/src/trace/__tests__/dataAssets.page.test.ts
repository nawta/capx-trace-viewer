/**
 * Data moved to assets, with the page's effects running (harness/):
 * - Selecting an API call loads the asset of a service call nested
 *   below it (the debug tab's recorded statistics come from it);
 * - after events.jsonl is replaced (new generation, same event ids and
 *   paths), the asset is fetched again under the new generation and the new
 *   content is shown, never the old trace's cached file;
 * - an asset that was not valid JSON is fetched again by 再読み込み, and
 *   so is one that was valid JSON but not the saved object (再読み込み reads
 *   past the file cache).
 */

import { advance, propsOf, requests, resetBrowser, setLocation } from './harness/browser';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mount } from './harness/fakeReact';
import TraceViewerPage from '../TraceViewerPage';
import { DetailTabs } from '../components/DetailTabs';
import { TopBar } from '../components/TopBar';
import type { TraceEvent } from '../types';
import { addEpisode, ev, makeRun, serve, type FakeEpisode } from './pageServer';

type DetailProps = Parameters<typeof DetailTabs>[0];

const ASSET = 'assets/aa/svc.json.gz';

/** A block with one API call and a SAM3 service call whose end event's data moved to an asset. */
function episodeWithOversizeService(): { events: TraceEvent[]; api: TraceEvent; svcEnd: TraceEvent } {
  const E = 'e1';
  const start = ev(E, 1, 'episode_start', { task: 'cube_lift', condition: 'C', seed: 1, attempt_index: 1 });
  const block = ev(E, 2, 'code_block_start', { code_block_id: `${E}_2`, block_index: 0, code: 'x = 1' }, { refs: { code_block_id: `${E}_2` } });
  const api = ev(E, 3, 'api_call_start', { api_call_id: `${E}_3`, name: 'segment', depth: 0, call_site: { code_block_id: `${E}_2`, line: 1 } }, { parent_event_id: `${E}_2` });
  const svc = ev(E, 4, 'service_call_start', { service: 'sam3', function: 'sam3_seg_fn' }, { parent_event_id: `${E}_3` });
  const svcEnd = ev(E, 5, 'service_call_end', { service: 'sam3', function: 'sam3_seg_fn', oversize: true, data_asset: ASSET, data_asset_written: true }, {
    parent_event_id: `${E}_4`,
    artifacts: [
      { path: ASSET, label: 'payload', kind: 'json', written: true },
      { path: 'assets/bb/out.npz', label: 'perception_output', kind: 'arrays', written: true },
    ],
  });
  const apiEnd = ev(E, 6, 'api_call_end', { api_call_id: `${E}_3`, name: 'segment', duration_s: 0.1 }, { parent_event_id: `${E}_3` });
  const blockEnd = ev(E, 7, 'code_block_end', { code_block_id: `${E}_2`, sandbox_rc: 0, stdout_len: 0, stderr_len: 0 }, { parent_event_id: `${E}_2`, refs: { code_block_id: `${E}_2` } });
  const end = ev(E, 8, 'episode_end', { end_reason: 'completed', trace_complete: true });
  return { events: [start, block, api, svc, svcEnd, apiEnd, blockEnd, end], api, svcEnd };
}

/** `generation` differs per test: api.ts caches files per URL for the whole test file. */
function setup(generation: string): { e1: FakeEpisode; api: TraceEvent; svcEnd: TraceEvent } {
  resetBrowser();
  const run = makeRun();
  const { events, api, svcEnd } = episodeWithOversizeService();
  const e1 = addEpisode(run, 'e1', { status: 'ended', task: 'cube_lift', condition: 'C', seed: 1, attempt_index: 1, trace_complete: true }, events);
  e1.generation = generation;
  e1.files.set(ASSET, { json: { service: 'sam3', function: 'sam3_seg_fn', response_summary: { v: 'old' } } });
  serve(run);
  return { e1, api, svcEnd };
}

const assetReads = () => requests(/\/file\/assets\/aa\/svc\.json\.gz$/);

test('(a): selecting the API call loads the asset of the service call nested below it', async () => {
  const { api, svcEnd } = setup('t1-g1');
  setLocation(`/traces?view=details&run=r1&episode=e1&event=${api.event_id}`);
  const p = mount(TraceViewerPage, {});
  await advance(3000);
  assert.equal(assetReads().sent, 1);
  const st = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.equal(st?.status, 'loaded');
  p.unmount();
});

test('after events.jsonl is replaced, the asset is read under the new generation and the new content is shown', async () => {
  const { e1, svcEnd } = setup('t2-g1');
  setLocation(`/traces?view=details&run=r1&episode=e1&event=${svcEnd.event_id}`);
  const p = mount(TraceViewerPage, {});
  await advance(3000);
  const first = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.deepEqual(first?.status === 'loaded' ? first.data.response_summary : null, { v: 'old' });
  assert.equal(assetReads().list[0]?.query.get('g'), 't2-g1');
  // the trace is written again in the same place: same event ids and asset path, new content
  e1.generation = 't2-g2';
  e1.files.set(ASSET, { json: { service: 'sam3', function: 'sam3_seg_fn', response_summary: { v: 'new' } } });
  // 更新: the pager sends generation g1, the server answers 409 generation_changed, the pager reads g2
  propsOf<Parameters<typeof TopBar>[0]>(p.output, TopBar).onRefresh();
  await advance(3000);
  const again = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.deepEqual(again?.status === 'loaded' ? again.data.response_summary : null, { v: 'new' }, 'never the old trace\'s file');
  assert.equal(assetReads().list.at(-1)?.query.get('g'), 't2-g2');
  p.unmount();
});

test('an asset that was not valid JSON is fetched again by 再読み込み and then loads', async () => {
  const { e1, svcEnd } = setup('t3-g1');
  e1.files.set(ASSET, { text: '{"service": "sam3", "respo' });
  setLocation(`/traces?view=details&run=r1&episode=e1&event=${svcEnd.event_id}`);
  const p = mount(TraceViewerPage, {});
  await advance(3000);
  const failed = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.equal(failed?.status, 'failed');
  // the file is repaired; the person presses 再読み込み
  e1.files.set(ASSET, { json: { service: 'sam3', function: 'sam3_seg_fn', response_summary: { v: 'fixed' } } });
  const retry = propsOf<DetailProps>(p.output, DetailTabs).onRetryAsset;
  if (!retry) throw new Error('DetailTabs has no retry');
  retry(svcEnd);
  await advance(3000);
  const loaded = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.deepEqual(loaded?.status === 'loaded' ? loaded.data.response_summary : null, { v: 'fixed' });
  assert.equal(assetReads().sent, 2);
  p.unmount();
});

test('再読み込み reads past the file cache: an asset that was valid JSON but not an object loads after it is repaired', async () => {
  const { e1, svcEnd } = setup('t4-g1');
  e1.files.set(ASSET, { json: ['not', 'the', 'saved', 'object'] });
  setLocation(`/traces?view=details&run=r1&episode=e1&event=${svcEnd.event_id}`);
  const p = mount(TraceViewerPage, {});
  await advance(3000);
  const failed = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.equal(failed?.status, 'failed');
  e1.files.set(ASSET, { json: { service: 'sam3', function: 'sam3_seg_fn', response_summary: { v: 'fixed' } } });
  const retry = propsOf<DetailProps>(p.output, DetailTabs).onRetryAsset;
  if (!retry) throw new Error('DetailTabs has no retry');
  retry(svcEnd);
  await advance(3000);
  const loaded = propsOf<DetailProps>(p.output, DetailTabs).assetStates?.get(svcEnd.event_id);
  assert.deepEqual(loaded?.status === 'loaded' ? loaded.data.response_summary : null, { v: 'fixed' });
  assert.equal(assetReads().sent, 2, 'the retry fetched the file again');
  p.unmount();
});
