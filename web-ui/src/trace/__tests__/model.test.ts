/**
 * Causal tree of model.ts: links by ids only, retries grouped under one
 * logical call, and a logical error after HTTP 200 wins over the HTTP answer.
 * A failed stream's logical error belongs to its attempt. It also covers
 * late events, orphans, and image counts from recorded fields.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildEpisodeModel,
  callOutcome,
  httpOkResponse,
  logicalErrorsOfAttempt,
  nodeDuration,
  okResponse,
  requestImageCounts,
  resolveSelection,
  textOf,
} from '../model';
import type { ModelRequestData } from '../types';
import { Rec } from './fixtures';

test('the tree links events by ids: call -> block -> API -> nested API / service / perception VLM, streams and evaluator under the block', () => {
  const r = new Rec();
  r.episodeStart();
  const req = r.request('code_generator', 'lc_code');
  const resp = r.response(req, 200, 'x = 1');
  r.parsed(resp);
  const block = r.blockStart(resp);
  const api = r.apiStart(block, 'get_object_pose', { object_name: 'red cube' });
  const nested = r.apiStart(block, 'inner', {}, api);
  const svc = r.serviceStart(nested, 'sam3', 'sam3_seg_fn');
  r.serviceEnd(svc);
  r.apiEnd(nested);
  const vlm = r.request('perception_vlm', 'lc_vlm', 1, { call_kind: 'point' }, { parent: api.event_id });
  r.response(vlm, 200, 'point');
  r.apiEnd(api);
  r.stream(block, 'stdout', 'hello world\n');
  r.stream(block, 'stderr', '');
  r.evaluator(block);
  r.blockEnd(block, 12, 0);

  const m = buildEpisodeModel(r.events);
  const callNode = m.nodeOfEvent.get(req.event_id)!;
  assert.equal(callNode.kind, 'model_call');
  assert.equal(callNode.call?.parsed.length, 1);
  const blockNode = m.nodeOfEvent.get(block.event_id)!;
  assert.equal(blockNode.sourceCall, callNode.call);
  assert.deepEqual(
    blockNode.children.map((c) => c.kind),
    ['api_call', 'stdout', 'stderr', 'evaluator'],
  );
  const apiNode = m.nodeOfEvent.get(api.event_id)!;
  assert.equal(apiNode.parent, blockNode);
  assert.ok(apiNode.end, 'api end paired by api_call_id');
  const kids = apiNode.children.map((c) => c.kind).sort();
  assert.deepEqual(kids, ['api_call', 'model_call']);
  const nestedNode = m.nodeOfEvent.get(nested.event_id)!;
  assert.equal(nestedNode.children[0].kind, 'service_call');
  assert.equal(m.nodeOfEvent.get(vlm.event_id)!.parent, apiNode);
  for (const n of m.allNodes) assert.equal(n.orphan, null, `${n.kind} #${n.seq} is linked`);
});

test('HTTP retries are one logical call; an answered retry makes the call ok', () => {
  const r = new Rec();
  const a1 = r.request('vdm', 'lc1', 1);
  r.attemptError(a1, 503);
  const a2 = r.request('vdm', 'lc1', 2);
  const resp = r.response(a2, 200, 'the cube moved');
  const m = buildEpisodeModel(r.events);
  assert.equal(m.calls.length, 1);
  const call = m.calls[0];
  assert.equal(call.attempts.length, 2);
  assert.equal(call.attempts[0].errors.length, 1);
  const out = callOutcome(call);
  assert.equal(out.state, 'ok');
  assert.equal(okResponse(call), resp);
  assert.equal(call.node.failed, false);
  assert.equal(call.node.open, false);
});

test('a logical model_error after HTTP 200 fails the call (the HTTP answer is kept apart)', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1');
  const resp = r.response(req, 200, '{}');
  const err = r.logicalError(req, 'KeyError', "'choices'");
  const m = buildEpisodeModel(r.events);
  const call = m.calls[0];
  const out = callOutcome(call);
  assert.equal(out.state, 'failed');
  assert.equal(out.finalError, err);
  assert.equal(out.stopped, false, 'a client error is a failure, not a stop');
  assert.equal(out.httpAnswer, resp, 'the HTTP 200 response is still recorded');
  assert.equal(httpOkResponse(call), resp);
  assert.equal(okResponse(call), null, 'no answer the harness used');
  assert.equal(call.node.failed, true);
  assert.equal(call.node.open, false);
  // latency of the HTTP exchange is still shown
  assert.equal(nodeDuration(call.node), 0.5);
});

test('attempt errors of earlier retries do not fail a call that got an answer; errors of a missing request are kept apart', () => {
  const r = new Rec();
  const a1 = r.request('code_generator', 'lc1', 1);
  r.attemptError(a1, 500);
  // an attempt error that names attempt 5, whose request is not in the record
  r.add('model_error', { scope: 'attempt', logical_call_id: 'lc1', http_attempt: 5, error_type: 'ConnectionError', message: 'x' }, { status: 'error', refs: { logical_call_id: 'lc1' } });
  const a2 = r.request('code_generator', 'lc1', 2);
  r.response(a2, 200, 'ok');
  const call = buildEpisodeModel(r.events).calls[0];
  assert.equal(call.strayErrors.length, 1);
  assert.equal(call.logicalErrors.length, 0);
  assert.equal(callOutcome(call).state, 'ok');
});

test('a stopped stream is a logical error of its attempt, with the partial text', () => {
  const r = new Rec();
  const req = r.request('code_generator', 'lc1', 1, { stream: true });
  const err = r.logicalError(req, 'GeneratorExit', '', { streamed: true, n_chunks: 3, partial_content: 'import num' });
  const m = buildEpisodeModel(r.events);
  const call = m.calls[0];
  assert.equal(callOutcome(call).state, 'failed');
  assert.equal(callOutcome(call).stopped, true, 'GeneratorExit: the caller stopped reading');
  assert.deepEqual(logicalErrorsOfAttempt(call, call.attempts[0]), [err]);
  assert.equal(textOf((err.data as { partial_content: unknown }).partial_content), 'import num');
});

test('an open call: the last attempt has no result yet', () => {
  const r = new Rec();
  r.request('code_generator', 'lc1');
  const call = buildEpisodeModel(r.events).calls[0];
  assert.equal(callOutcome(call).state, 'open');
  assert.equal(call.node.open, true);
  assert.equal(call.node.failed, false);
});

test('late events are marked; ids that do not resolve become orphans with a reason (never linked by time)', () => {
  const r = new Rec();
  const req = r.request('vdm', 'lc1');
  r.episodeEnd();
  r.response(req, 200, 'late answer');
  (r.events[r.events.length - 1] as { late?: boolean }).late = true;
  r.add('api_call_end', { api_call_id: 'ev_missing', duration_s: 1 }, { parent: 'ev_missing' });
  r.add('model_response', { http_status: 200 }, { parent: 'ev_nowhere' });
  const m = buildEpisodeModel(r.events);
  assert.equal(m.calls[0].node.late, true);
  const orphans = m.roots.filter((n) => n.orphan !== null).map((n) => n.orphan);
  assert.equal(orphans.length, 2);
  assert.ok(orphans.some((o) => /api_call_start/.test(o ?? '')));
  assert.ok(orphans.some((o) => /model_request/.test(o ?? '')));
});

test('selection resolves the attempt named by the event (response of attempt 2 -> attempt 2)', () => {
  const r = new Rec();
  const a1 = r.request('vdm', 'lc1', 1);
  r.attemptError(a1, 503);
  const a2 = r.request('vdm', 'lc1', 2);
  const resp = r.response(a2);
  const m = buildEpisodeModel(r.events);
  const sel = resolveSelection(m, resp.event_id)!;
  assert.equal(sel.attempt?.request, a2);
});

test('image counts only from recorded fields; a request whose body was not parsed is unknown, not "no images"', () => {
  const unparsed: ModelRequestData = { role: 'vdm', reconstruction: { verified: false, reason: 'queue_full' } };
  assert.deepEqual(requestImageCounts(unparsed), { images: null, videos: null, noImage: null });
  assert.equal(requestImageCounts({ n_images: 0, n_videos: 0 }).noImage, true);
  assert.equal(requestImageCounts({ n_images: 2, n_videos: 0 }).noImage, false);
  assert.equal(requestImageCounts({ input_summary: { no_image_input: true } }).noImage, true);
  const listed = requestImageCounts({ images: [{ kind: 'image' }, { kind: 'video' }, { kind: 'external_url' }] });
  assert.deepEqual(listed, { images: 2, videos: 1, noImage: false });
  assert.deepEqual(requestImageCounts(null), { images: null, videos: null, noImage: null });
});

test('textOf never throws on a text-asset reference and gives its preview', () => {
  assert.equal(textOf('abc'), 'abc');
  assert.equal(textOf({ truncated: true, preview: 'long…', length: 5000, asset: 'assets/aa/x.txt.gz' }), 'long…');
  assert.equal(textOf(null), '');
  assert.equal(textOf(undefined), '');
  assert.equal(textOf(['a']), '');
});

test('a Molmo parse of a call that got no response (connection refused) belongs to that call by its logical call id, not to "no parent"', () => {
  // the shape of the trace recorder (Molmo parser hook) after the Molmo server refused the connection
  const r = new Rec();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'get_object_3d_points_and_masks_from_language', {});
  const svc = r.serviceStart(api, 'molmo', 'molmo_point_fn');
  const req = r.request('perception_vlm', 'lc_molmo', 1, { call_kind: 'point', n_images: 1 }, { parent: svc.event_id });
  r.attemptError(req, null, "HTTPConnectionPool(host='127.0.0.1', port=8122): Max retries exceeded");
  const parse = r.add(
    'model_output_parsed',
    { parser: 'molmo._parse_points', model_event_id: null, logical_call_id: 'lc_molmo', content_empty: true, parsed_text: '', points: [] },
    { parent: svc.event_id, refs: { logical_call_id: 'lc_molmo' } },
  );
  r.serviceEnd(svc, { degraded_reason: 'transport error' });
  r.apiEnd(api);
  const m = buildEpisodeModel(r.events);
  const call = m.calls.find((c) => c.id === 'lc_molmo');
  assert.ok(call, 'the Molmo call is in the model');
  assert.ok(call!.parsed.includes(parse), 'the parse is attached to its call');
  const orphans = [...m.nodeOfEvent.values()].filter((n) => n.orphan && n.events.includes(parse));
  assert.equal(orphans.length, 0, 'no orphan card for the parse');
  // a parse that names neither a response nor a known call is still an orphan
  const r2 = new Rec();
  const lone = r2.add('model_output_parsed', { parser: 'molmo._parse_points', model_event_id: null, logical_call_id: 'lc_unknown' });
  const m2 = buildEpisodeModel(r2.events);
  assert.ok([...m2.nodeOfEvent.values()].some((n) => n.orphan && n.events.includes(lone)));
});
