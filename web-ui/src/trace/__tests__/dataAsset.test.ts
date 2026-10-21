/**
 * - Events whose data the writer moved to an asset because a line exceeded
 *   256 KB are merged back when loaded. The events a selection needs are loaded.
 * - Nested service events are loaded too, and a pane can tell loading,
 *   failed and never written apart (pendingAsset).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MAX_NESTED_ASSET_TARGETS,
  applyDataAssets,
  dataAssetRef,
  dataAssetTargets,
  isOversize,
  mergeDataAsset,
  parseDataAsset,
  pendingAsset,
  withTargetsLoading,
  type DataAssetState,
} from '../dataAsset';
import { buildEpisodeModel, resolveSelection } from '../model';
import { dataOf } from '../types';
import { Rec } from './fixtures';

const ASSET = 'assets/ab/abcdef.json.gz';

/** An api_call_start as the writer leaves it: scalars inline, `args` only in the asset. */
function oversizeApi(r: Rec, block: ReturnType<Rec['blockStart']>, written = true) {
  const ev = r.apiStart(block, 'segment', {});
  const d = ev.data as Record<string, unknown>;
  const full = { ...d, args: { points: { type: 'list', length: 20000, items: [1, 2, 3] } } };
  ev.data = { api_call_id: d.api_call_id, name: 'segment', qualname: 'Api.segment', api_class: 'Api', depth: 0, oversize: true, data_asset: ASSET, data_asset_written: written };
  ev.artifacts = [{ path: ASSET, label: 'payload', kind: 'json', written, reason: written ? null : 'disk_low' }];
  return { ev, full };
}

test('oversize markers are recognised; an unwritten asset carries its reason', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const { ev } = oversizeApi(r, block, false);
  assert.equal(isOversize(ev), true);
  assert.deepEqual(dataAssetRef(ev), { path: ASSET, written: false, reason: 'disk_low' });
  assert.equal(dataAssetRef(block), null);
  assert.equal(isOversize(block), false);
});

test('merging puts the full data back and keeps the markers; a non-object leaves the event as it was', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const { ev, full } = oversizeApi(r, block);
  const merged = mergeDataAsset(ev, full);
  const d = dataOf(merged, 'api_call_start') as Record<string, unknown>;
  assert.deepEqual(d.args, full.args);
  assert.equal(d.oversize, true);
  assert.equal(d.data_asset, ASSET);
  assert.equal(d.data_asset_loaded, true);
  assert.equal(mergeDataAsset(ev, [1, 2]), ev);
  assert.equal(parseDataAsset('x'), null);
});

test('applyDataAssets returns the same array until something is loaded, then the tree sees the full data', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const { ev, full } = oversizeApi(r, block);
  r.apiEnd(ev);
  r.blockEnd(block, 0, 0);
  const none = new Map<string, DataAssetState>([[ev.event_id, { status: 'loading' }]]);
  assert.equal(applyDataAssets(r.events, none), r.events);
  const loaded = new Map<string, DataAssetState>([[ev.event_id, { status: 'loaded', data: full }]]);
  const merged = applyDataAssets(r.events, loaded);
  assert.notEqual(merged, r.events);
  const m = buildEpisodeModel(merged);
  const node = m.nodeOfEvent.get(ev.event_id)!;
  assert.deepEqual(dataOf(node.primary, 'api_call_start').args, full.args);
  // the ids stayed inline, so the tree is the same
  assert.equal(node.parent?.primary.event_id, block.event_id);
});

test('targets: the selected node, its block, the later requests, and episode_start', () => {
  const r = new Rec();
  const start = r.episodeStart();
  start.data = { oversize: true, data_asset: 'assets/ee/start.json.gz', data_asset_written: true };
  const block = r.blockStart(null);
  const { ev } = oversizeApi(r, block);
  r.apiEnd(ev);
  r.blockEnd(block, 0, 0);
  const later = r.request('code_generator', 'lc_next');
  later.data = { ...(later.data as object), oversize: true, data_asset: 'assets/ff/req.json.gz', data_asset_written: true };
  const unrelated = r.request('vdm', 'lc_far');
  unrelated.data = { ...(unrelated.data as object), oversize: true, data_asset: 'assets/99/far.json.gz' };
  const m = buildEpisodeModel(r.events);
  const sel = resolveSelection(m, ev.event_id);
  const ids = dataAssetTargets(m, sel).map((e) => e.event_id).sort();
  assert.deepEqual(ids, [start.event_id, ev.event_id, later.event_id].sort());
  assert.deepEqual(dataAssetTargets(m, null).map((e) => e.event_id), [start.event_id]);
});

test('(a): nested service events whose data moved to an asset are targets of the API and block selections, bounded', () => {
  const r = new Rec();
  r.episodeStart();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'get_object_3d_points_and_masks_from_language', {});
  const svc = r.serviceStart(api, 'sam3', 'sam3_seg_fn');
  const end = r.serviceEnd(svc, {});
  end.data = { service: 'sam3', function: 'sam3_seg_fn', latency_s: 0.1, oversize: true, data_asset: 'assets/aa/svc.json.gz', data_asset_written: true };
  end.artifacts = [{ path: 'assets/aa/svc.json.gz', label: 'payload', kind: 'json', written: true }, { path: 'assets/bb/out.npz', label: 'perception_output', kind: 'arrays', written: true }];
  r.apiEnd(api);
  r.blockEnd(block, 0, 0);
  const m = buildEpisodeModel(r.events);
  const idsFor = (id: string) => dataAssetTargets(m, resolveSelection(m, id)).map((e) => e.event_id);
  assert.ok(idsFor(api.event_id).includes(end.event_id), 'the API pane reads its service calls');
  assert.ok(idsFor(block.event_id).includes(end.event_id), 'the block pane reads every service call in it');
  assert.ok(idsFor(end.event_id).includes(end.event_id));
  // a block with many oversize service events loads at most MAX_NESTED_ASSET_TARGETS of them
  const r2 = new Rec();
  const b2 = r2.blockStart(null);
  const a2 = r2.apiStart(b2, 'segment', {});
  for (let i = 0; i < MAX_NESTED_ASSET_TARGETS + 10; i += 1) {
    const e = r2.serviceEnd(r2.serviceStart(a2, 'sam3', 'sam3_seg_fn'), {});
    e.data = { service: 'sam3', function: 'sam3_seg_fn', oversize: true, data_asset: `assets/x/${i}.json.gz`, data_asset_written: true };
  }
  r2.apiEnd(a2);
  r2.blockEnd(b2, 0, 0);
  const m2 = buildEpisodeModel(r2.events);
  assert.equal(dataAssetTargets(m2, resolveSelection(m2, b2.event_id)).length, MAX_NESTED_ASSET_TARGETS);
});

test('(b): pendingAsset tells loading, failed, never written and never requested apart; loaded and inline data are not pending', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const { ev, full } = oversizeApi(r, block);
  assert.deepEqual(pendingAsset(ev, new Map()), { kind: 'idle', path: ASSET }, 'never requested: idle, not loading');
  assert.deepEqual(pendingAsset(ev, new Map([[ev.event_id, { status: 'loading' }]])), { kind: 'loading', path: ASSET });
  assert.deepEqual(pendingAsset(ev, new Map([[ev.event_id, { status: 'failed', error: 'HTTP 500' }]])), { kind: 'failed', path: ASSET, error: 'HTTP 500' });
  assert.equal(pendingAsset(ev, new Map([[ev.event_id, { status: 'loaded', data: full }]])), null);
  assert.equal(pendingAsset(mergeDataAsset(ev, full), new Map()), null, 'a merged event has its data');
  assert.equal(pendingAsset(block, new Map()), null, 'inline data');
  const lost = oversizeApi(r, block, false).ev;
  assert.deepEqual(pendingAsset(lost, new Map()), { kind: 'unwritten', path: ASSET, reason: 'disk_low' });
});

test('(b): the targets the page is about to request count as loading; an event beyond the bound stays idle', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const a = oversizeApi(r, block).ev;
  const b = oversizeApi(r, block).ev;
  const plain = r.stream(block, 'stdout', 'x');
  const failed: DataAssetState = { status: 'failed', error: 'HTTP 500' };
  const states = new Map<string, DataAssetState>([[b.event_id, failed]]);
  const shown = withTargetsLoading(states, [a, b, plain]);
  assert.deepEqual(pendingAsset(a, shown), { kind: 'loading', path: ASSET }, 'a target without a state yet: loading');
  assert.equal(shown.get(b.event_id), failed, 'a state that exists is kept');
  assert.equal(shown.has(plain.event_id), false, 'inline data needs no state');
  assert.equal(states.has(a.event_id), false, 'the given map is not changed');
  assert.equal(withTargetsLoading(states, [b, plain]), states, 'nothing to add: the same map');
  // an oversize event that is not a target (beyond MAX_NESTED_ASSET_TARGETS) stays idle
  assert.deepEqual(pendingAsset(a, withTargetsLoading(states, [b])), { kind: 'idle', path: ASSET });
});
