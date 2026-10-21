/**
 * Array previews (components/ValueView.tsx) follow the server's image layout
 * (capx_trace_viewer/trace_arrays.py image_layout): a 3-D
 * bool array whose last axis is 1 is one H x W mask (the (H, W, 1)
 * segmentation layout); one whose last axis is larger than 1 is a stack of
 * masks along axis 0, so every mask can be reached with `index`. An older
 * server without `image` gets the same rules from shape and dtype.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { arrayImageLayout, isDrawnAsImage, recordedNpzStats, unsavedArrayHashText } from '../components/ValueView';
import type { ArrayInfo } from '../types';

test('the server layout is used as sent; null means the array is not drawn', () => {
  const stack = { shape: [3, 32, 32], dtype: 'bool', is_mask: true, image: { kind: 'mask_stack', count: 3, height: 32, width: 32 } };
  assert.deepEqual(arrayImageLayout(stack), { kind: 'mask_stack', count: 3, height: 32, width: 32 });
  // the server's answer wins over the client's own rule
  const one = { shape: [480, 640, 1], dtype: 'bool', image: { kind: 'mask', count: 1, height: 480, width: 640 } };
  assert.deepEqual(arrayImageLayout(one), { kind: 'mask', count: 1, height: 480, width: 640 });
  assert.equal(isDrawnAsImage(stack, arrayImageLayout(stack)), true, 'masks are drawn at any size');
  const notDrawable = { shape: [5], dtype: 'float32', image: null };
  assert.equal(arrayImageLayout(notDrawable), null);
  assert.equal(isDrawnAsImage(notDrawable, null), false);
  // a malformed layout from the server (fields missing) is not guessed from the shape
  const malformed = JSON.parse('{"shape": [4, 4], "dtype": "uint8", "image": {"kind": "gray"}}') as ArrayInfo;
  assert.equal(arrayImageLayout(malformed), null);
});

test('without a server layout, an (H, W, 1) bool array is one H x W mask; a larger last axis is a stack along axis 0', () => {
  assert.deepEqual(arrayImageLayout({ shape: [480, 640, 1], dtype: 'bool', is_mask: true }), { kind: 'mask', count: 1, height: 480, width: 640 });
  assert.deepEqual(arrayImageLayout({ shape: [2, 2, 1], dtype: 'bool', is_mask: true }), { kind: 'mask', count: 1, height: 2, width: 2 });
  assert.deepEqual(arrayImageLayout({ shape: [3, 32, 32], dtype: 'bool' }), { kind: 'mask_stack', count: 3, height: 32, width: 32 });
  assert.deepEqual(arrayImageLayout({ shape: [2, 480, 2], dtype: 'bool' }), { kind: 'mask_stack', count: 2, height: 480, width: 2 });
  assert.deepEqual(arrayImageLayout({ shape: [64, 64], dtype: 'bool' }), { kind: 'mask', count: 1, height: 64, width: 64 });
  assert.deepEqual(arrayImageLayout({ shape: [48, 64, 3], dtype: 'uint8' }), { kind: 'rgb', count: 1, height: 48, width: 64 });
  assert.deepEqual(arrayImageLayout({ shape: [48, 64, 1], dtype: 'float32' }), { kind: 'gray', count: 1, height: 48, width: 64 });
  assert.deepEqual(arrayImageLayout({ shape: [5, 48, 64], dtype: 'float32' }), { kind: 'stack', count: 5, height: 48, width: 64 });
  assert.equal(arrayImageLayout({ shape: [2, 3, 4, 5], dtype: 'float32' }), null);
  assert.equal(arrayImageLayout({ shape: [0, 4], dtype: 'float32' }), null);
  assert.equal(arrayImageLayout({ shape: [4, 4], dtype: 'object' }), null);
});

test('small numeric arrays (boxes, poses) are not drawn; depth maps are', () => {
  const box = { shape: [120, 4, 4], dtype: 'float32' };
  assert.equal(isDrawnAsImage(box, arrayImageLayout(box)), false);
  const depth = { shape: [480, 640], dtype: 'float32' };
  assert.equal(isDrawnAsImage(depth, arrayImageLayout(depth)), true);
});

test('arrays not saved for the budget show the bounded hash the recorder kept (recorder bounded hash)', () => {
  // at most 1 MiB: the whole array
  assert.equal(unsavedArrayHashText({ not_saved: 'array_budget', sha256: 'ab'.repeat(32) }), `sha256 ${'ab'.repeat(8)}`);
  // larger: only the first bytes, and the text says so
  const prefix = unsavedArrayHashText({ not_saved: 'array_budget', sha256: null, sha256_prefix: 'cd'.repeat(32), sha256_prefix_bytes: 1048576 });
  assert.match(prefix, /先頭 1048576 バイト/);
  assert.match(prefix, /cdcdcdcdcdcdcdcd/);
  // past the per-summary hash budget, or a failed hash: no hash, with the reason
  assert.match(unsavedArrayHashText({ not_saved: 'array_budget', sha256: null, sha256_skipped: 'hash_budget' }), /上限を超えたため計算せず/);
  assert.equal(unsavedArrayHashText({ not_saved: 'array_budget', sha256: null, sha256_skipped: 'hash_failed' }), 'sha256 なし（hash_failed）');
  assert.equal(unsavedArrayHashText({ not_saved: 'array_budget' }), 'sha256 —');
});

test('npz listings show the statistics recorded in the service response summary (/array sends stats: null)', () => {
  // the shape of a SAM3 service_call_end (the trace recorder array bundle writer)
  const end = {
    response_summary: {
      type: 'list',
      items: [
        { type: 'dict', items: { mask: { type: 'ndarray', shape: [512, 512], dtype: 'bool', npz: 'assets/c3/x.npz', npz_key: 'a0__packbits', packbits: true, is_mask: true, stats: { min: false, max: true, mean: 0.0103 } }, score: 0.9258 } },
        { type: 'dict', items: { mask: { type: 'ndarray', shape: [512, 512], dtype: 'bool', npz: 'assets/c3/x.npz', npz_key: 'a1__packbits', stats: { min: false, max: true, mean: 0.0674 } } } },
        { type: 'dict', items: { depth: { type: 'ndarray', shape: [64, 64], dtype: 'float32', npz: 'assets/c3/x.npz', npz_key: 'a2', stats: { min: 0.1, max: 2.5, mean: 1.2 } } } },
        // another file's array is not taken
        { type: 'dict', items: { mask: { type: 'ndarray', npz: 'assets/ff/other.npz', npz_key: 'a0__packbits', stats: { mean: 0.5 } } } },
      ],
    },
  };
  const stats = recordedNpzStats([{ request_summary: {} }, end], 'assets/c3/x.npz');
  // bit-packed masks are listed by /array under the name without __packbits
  assert.deepEqual(Object.keys(stats).sort(), ['a0', 'a1', 'a2']);
  assert.equal(stats.a0.mean, 0.0103);
  assert.equal(stats.a2.max, 2.5);
  assert.deepEqual(recordedNpzStats([end], 'assets/none.npz'), {});
  assert.deepEqual(recordedNpzStats([null, 3, 'x'], 'assets/c3/x.npz'), {});
});
