/**
 * Paging of events.jsonl (pager.ts):
 * - a refresh while a fetch is in flight is not dropped; the pass
 *   reads once more, so the tail written after the first read is fetched;
 * - when the file generation changes (409 generation_changed, or a
 *   page naming another generation), everything read is dropped and the file
 *   is read again from offset 0, so old and new events never mix;
 * - quiet passes are counted (the page's grace period after an episode ends).
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { ApiError } from '../api';
import { EventsPager, generationChangedOf, type PagerSnapshot } from '../pager';
import type { EventsPage, TraceEvent } from '../types';

const LINE = 100; // bytes per fake event line

function ev(id: string, seq: number): TraceEvent {
  return { event_id: id, seq, event_type: 'stdout' };
}

interface FakeFile {
  gen: string | null;
  events: TraceEvent[];
}

/** A fake /events endpoint over `file`; `mode` picks how the server treats generations. */
function server(file: FakeFile, mode: 'checks' | 'ignores' | 'none' = 'checks', gate?: { wait: Promise<void> | null }) {
  const calls: { after: number; generation: string | null }[] = [];
  const fetchPage = async (after: number, generation: string | null): Promise<EventsPage> => {
    calls.push({ after, generation });
    if (mode === 'checks' && generation && file.gen !== generation) {
      throw new ApiError(409, 'generation_changed', '/events', { error: 'generation_changed', generation: file.gen });
    }
    if (after % LINE !== 0 || after / LINE > file.events.length) {
      throw new ApiError(409, `after_offset ${after} is past the end of the file`, '/events', { detail: 'offset' });
    }
    // the file is read now; the answer may arrive later (gate)
    const page: EventsPage = {
      events: file.events.slice(after / LINE),
      next_offset: file.events.length * LINE,
      eof: true,
      ...(mode === 'none' ? {} : { generation: file.gen }),
    };
    if (gate?.wait) await gate.wait;
    return page;
  };
  return { fetchPage, calls };
}

function track() {
  const snaps: PagerSnapshot[] = [];
  return { snaps, onUpdate: (s: PagerSnapshot) => snaps.push(s), last: () => snaps[snaps.length - 1] };
}

test('a refresh during an in-flight fetch reads again after it (the tail is not lost)', async () => {
  const file: FakeFile = { gen: 'g1', events: [ev('A', 1)] };
  let open!: () => void;
  const gate = { wait: new Promise<void>((res) => (open = res)) as Promise<void> | null };
  const { fetchPage, calls } = server(file, 'checks', gate);
  const t = track();
  const pager = new EventsPager(fetchPage, t.onUpdate);
  const first = pager.pump(); // the poll: reads [A] now, answers later
  await Promise.resolve();
  file.events.push(ev('B', 2), ev('END', 3)); // the writer appends the tail and episode_end
  const second = pager.pump(); // the final refresh when the episode ended
  gate.wait = null;
  open();
  await Promise.all([first, second]);
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['A', 'B', 'END']);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].after, LINE);
});

test('409 generation_changed drops what was read and reads the new file from 0', async () => {
  const file: FakeFile = { gen: 'g1', events: [ev('A', 1), ev('B', 2)] };
  const { fetchPage, calls } = server(file, 'checks');
  const t = track();
  const pager = new EventsPager(fetchPage, t.onUpdate);
  await pager.pump();
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['A', 'B']);
  // replaced by a file whose saved offset (200) is still on a line boundary
  file.gen = 'g2';
  file.events = [ev('B2', 1), ev('C2', 2), ev('D2', 3)];
  await pager.pump();
  const s = t.last();
  assert.deepEqual(s.events.map((e) => e.event_id), ['B2', 'C2', 'D2']);
  assert.equal(s.replaced, true);
  assert.equal(s.resets, 1);
  assert.equal(s.generation, 'g2');
  assert.deepEqual(calls.map((c) => [c.after, c.generation]), [
    [0, null],
    [200, 'g1'],
    [0, 'g2'],
  ]);
});

test('a server that ignores the parameter but names the generation in the page still triggers the reset', async () => {
  const file: FakeFile = { gen: 'g1', events: [ev('A', 1), ev('B', 2)] };
  const { fetchPage } = server(file, 'ignores');
  const t = track();
  const pager = new EventsPager(fetchPage, t.onUpdate);
  await pager.pump();
  file.gen = 'g2';
  file.events = [ev('X', 1), ev('Y', 2), ev('Z', 3)];
  await pager.pump();
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['X', 'Y', 'Z']);
  assert.equal(t.last().replaced, true);
});

test('an older server without generations: a 409 on the saved offset reads again from 0', async () => {
  const file: FakeFile = { gen: null, events: [ev('A', 1), ev('B', 2)] };
  const { fetchPage } = server(file, 'none');
  const t = track();
  const pager = new EventsPager(fetchPage, t.onUpdate);
  await pager.pump();
  file.events = [ev('N', 1)]; // shorter: offset 200 is past the end
  await pager.pump();
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['N']);
  assert.equal(t.last().replaced, true);
});

test('a generation change before anything was read is adopted without calling it a replacement', async () => {
  const file: FakeFile = { gen: 'g-empty', events: [] };
  const { fetchPage } = server(file, 'checks');
  const t = track();
  const pager = new EventsPager(fetchPage, t.onUpdate);
  await pager.pump();
  file.gen = 'g-first-line';
  file.events = [ev('A', 1)];
  await pager.pump();
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['A']);
  assert.equal(t.last().replaced, false);
  assert.equal(t.last().resets, 0);
});

test('duplicates are dropped; quiet passes count up and reset on new events', async () => {
  const file: FakeFile = { gen: 'g1', events: [ev('A', 1)] };
  const t = track();
  const pager = new EventsPager(server(file).fetchPage, t.onUpdate);
  await pager.pump();
  assert.equal(t.last().quietPasses, 0);
  await pager.pump();
  await pager.pump();
  assert.equal(t.last().quietPasses, 2);
  assert.equal(t.last().passes, 3);
  file.events.push(ev('A', 1), ev('B', 2)); // a repeated id is not added twice
  await pager.pump();
  assert.deepEqual(t.last().events.map((e) => e.event_id), ['A', 'B']);
  assert.equal(t.last().quietPasses, 0);
});

test('errors are reported; a disposed pager publishes nothing', async () => {
  const t = track();
  const failing = new EventsPager(async () => {
    throw new ApiError(500, 'boom', '/events');
  }, t.onUpdate);
  await failing.pump();
  assert.match(t.last().error ?? '', /500/);
  assert.equal(t.last().loading, false);
  const t2 = track();
  const pager = new EventsPager(server({ gen: 'g', events: [ev('A', 1)] }).fetchPage, t2.onUpdate);
  pager.dispose();
  await pager.pump();
  assert.equal(t2.snaps.length, 0);
});

test('generationChangedOf reads both the plain body and a FastAPI detail wrapper', () => {
  assert.equal(generationChangedOf(new ApiError(409, 'x', 'u', { error: 'generation_changed', generation: 'g9' })), 'g9');
  assert.equal(generationChangedOf(new ApiError(409, 'x', 'u', { detail: { error: 'generation_changed', generation: 'g8' } })), 'g8');
  assert.equal(generationChangedOf(new ApiError(409, 'x', 'u', { detail: 'offset' })), null);
  assert.equal(generationChangedOf(new ApiError(404, 'x', 'u', { error: 'generation_changed', generation: 'g' })), null);
  assert.equal(generationChangedOf(new Error('x')), null);
});
