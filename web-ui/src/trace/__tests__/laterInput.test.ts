/**
 * The 6.4 later-input report (laterInput.ts):
 * - the final traceback line must match the exception's type AND message;
 *   another exception of the same type is never taken;
 * - a stream or exception text that was never searched is 未確認, not
 *   "not included"; a missing api_call_end leaves the API's output unknown;
 * - every HTTP attempt in the range is reported, retries included.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEpisodeModel } from '../model';
import { blockStreamText, checkStream, computeLaterInput, findTracebackLine, laterAttempts } from '../laterInput';
import type { TraceEvent } from '../types';
import { Rec, summaryWith } from './fixtures';

const STDERR_TWO_ERRORS =
  'Traceback (most recent call last):\n  File "<string>", line 12, in <module>\nValueError: cannot reshape array of size 2 into shape (3,)\n' +
  'Traceback (most recent call last):\n  File "<string>", line 18, in <module>\nValueError: Invalid object name: blue cube\n';

test('the traceback line needs the type and the whole message', () => {
  const a = findTracebackLine(STDERR_TWO_ERRORS, 'ValueError', 'cannot reshape array of size 2 into shape (3,)');
  assert.equal(a?.line, 'ValueError: cannot reshape array of size 2 into shape (3,)');
  assert.equal(a?.count, 1);
  const b = findTracebackLine(STDERR_TWO_ERRORS, 'ValueError', 'Invalid object name: blue cube');
  assert.equal(b?.line, 'ValueError: Invalid object name: blue cube');
  // same type, a message that was never printed: no line (the other ValueError is not taken)
  assert.equal(findTracebackLine(STDERR_TWO_ERRORS, 'ValueError', 'first failure'), null);
  // a message that only ends like the printed one is not this exception
  assert.equal(findTracebackLine('KeyError: x: shape (3,)\n', 'ValueError', 'shape (3,)'), null);
});

test('module-qualified type names, empty messages and multi-line messages', () => {
  const q = findTracebackLine('Traceback...\nrequests.exceptions.ConnectionError: refused\n', 'ConnectionError', 'refused');
  assert.equal(q?.line, 'requests.exceptions.ConnectionError: refused');
  // a prefix that is not a module path does not count
  assert.equal(findTracebackLine('note: ConnectionError: refused\n', 'ConnectionError', 'refused'), null);
  // str(e) == '': Python prints the type alone
  assert.equal(findTracebackLine('Traceback...\nStopIteration\n', 'StopIteration', '')?.line, 'StopIteration');
  assert.equal(findTracebackLine('Traceback...\nStopIteration: x\n', 'StopIteration', ''), null);
  // a message with a newline spans two lines
  const ml = findTracebackLine('Traceback...\nRuntimeError: line one\nline two\n', 'RuntimeError', 'line one\nline two');
  assert.equal(ml?.line, 'RuntimeError: line one\nline two');
  // the same final line twice: counted, so the report can say it is ambiguous
  assert.equal(findTracebackLine('E: bad input\nE: bad input\n', 'E', 'bad input')?.count, 2);
  // unknown type or message: nothing is matched
  assert.equal(findTracebackLine(STDERR_TWO_ERRORS, null, 'x'), null);
  assert.equal(findTracebackLine(STDERR_TWO_ERRORS, 'ValueError', null), null);
});

/** A block whose API raised, then a VDM call (with a retry) and the next code-generator call (with a retry). */
function scenario(opts: { stdout: string; stderr: string; message: unknown; stdoutFound?: boolean; stdoutSearched?: boolean; stderrFound?: boolean }) {
  const r = new Rec();
  const req0 = r.request('code_generator', 'lc0');
  const resp0 = r.response(req0);
  const block = r.blockStart(resp0);
  const api = r.apiStart(block, 'goto_pose', {});
  r.apiError(api, 'ValueError', opts.message, { stdout: [0, 0], stderr: [0, opts.stderr.length] });
  const out = r.stream(block, 'stdout', opts.stdout);
  const err = r.stream(block, 'stderr', opts.stderr);
  const end = r.blockEnd(block, [...opts.stdout].length, [...opts.stderr].length);
  const found = [];
  if (opts.stdoutFound) found.push({ kind: 'stdout', path: 'messages[1].content[0].text', start: 100, end: 100 + [...opts.stdout].length, source_event_ids: [out.event_id, block.event_id] });
  if (opts.stderrFound) found.push({ kind: 'stderr', path: 'messages[1].content[0].text', start: 500, end: 500 + [...opts.stderr].length, source_event_ids: [err.event_id, block.event_id] });
  const notFound = opts.stdoutSearched && !opts.stdoutFound ? [{ kind: 'stdout', source_event_ids: [out.event_id, block.event_id] }] : [];
  const summary = summaryWith(found, notFound);
  const v1 = r.request('vdm', 'lc_vdm', 1, { input_summary: summaryWith([]) });
  r.attemptError(v1, 503);
  const v2 = r.request('vdm', 'lc_vdm', 2, { input_summary: summaryWith([]) });
  r.response(v2, 200, 'moved');
  const c1 = r.request('code_generator', 'lc_next', 1, { input_summary: summary });
  r.attemptError(c1, 500);
  const c2 = r.request('code_generator', 'lc_next', 2, { input_summary: summary });
  r.response(c2);
  // a request after the next code-generator call is outside the range
  r.request('vdm', 'lc_after', 1, { input_summary: summaryWith([]) });
  const m = buildEpisodeModel(r.events);
  return { r, m, api, block, end, v1, v2, c1, c2 };
}

test('every HTTP attempt after the block end is reported, up to the next code-generator call and its resends', () => {
  const { m, api, end, v1, v2, c1, c2 } = scenario({ stdout: 'hello there\n', stderr: STDERR_TWO_ERRORS, message: 'cannot reshape array of size 2 into shape (3,)', stdoutFound: true, stderrFound: true });
  const { items, nextCodegen } = laterAttempts(m, end.seq);
  assert.deepEqual(items.map((x) => x.attempt.request), [v1, v2, c1, c2]);
  assert.equal(nextCodegen?.id, 'lc_next');
  const rep = computeLaterInput(m, m.nodeOfEvent.get(api.event_id)!, { stdout: 'hello there\n', stderr: STDERR_TWO_ERRORS });
  assert.deepEqual(rep.requests.map((q) => [q.call.role, q.httpAttempt, q.isRetry, q.isNextCodegen]), [
    ['vdm', 1, false, false],
    ['vdm', 2, true, false],
    ['code_generator', 1, false, true],
    ['code_generator', 2, true, true],
  ]);
});

test('in the report: the selected API error gets its own traceback line, not the last one of its type', () => {
  const { m, api } = scenario({ stdout: 'hello there\n', stderr: STDERR_TWO_ERRORS, message: 'cannot reshape array of size 2 into shape (3,)', stdoutFound: true, stderrFound: true });
  const rep = computeLaterInput(m, m.nodeOfEvent.get(api.event_id)!, { stdout: 'hello there\n', stderr: STDERR_TWO_ERRORS });
  assert.equal(rep.api?.tracebackLine, 'ValueError: cannot reshape array of size 2 into shape (3,)');
  const next = rep.requests.find((q) => q.isNextCodegen)!;
  const tb = next.errorFindings.find((f) => f.what === 'traceback_line')!;
  assert.equal(tb.text, 'ValueError: cannot reshape array of size 2 into shape (3,)');
  const idx = STDERR_TWO_ERRORS.indexOf('ValueError: cannot reshape');
  assert.equal(tb.streamOffset, idx);
  assert.deepEqual(tb.mapped, { path: 'messages[1].content[0].text', start: 500 + idx, end: 500 + idx + tb.text.length });
});

test('a 1-7 character stdout is unchecked, not "not included"; an empty one is empty', () => {
  const s = scenario({ stdout: '0.12\n', stderr: '', message: 'cannot reshape array of size 2 into shape (3,)' });
  const rep = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: '0.12\n', stderr: '' });
  const next = rep.requests.find((q) => q.isNextCodegen)!;
  assert.equal(next.stdout.state, 'short');
  assert.equal(next.stderr.state, 'empty');
});

test('a searched stream that is absent is "not included"; one never registered is unchecked', () => {
  const searched = scenario({ stdout: 'a long enough stdout\n', stderr: '', message: 'cannot reshape array of size 2 into shape (3,)', stdoutSearched: true });
  const rep1 = computeLaterInput(searched.m, searched.m.nodeOfEvent.get(searched.api.event_id)!, { stdout: 'a long enough stdout\n', stderr: '' });
  assert.equal(rep1.requests.find((q) => q.isNextCodegen)!.stdout.state, 'not_included');
  const unsearched = scenario({ stdout: 'a long enough stdout\n', stderr: '', message: 'cannot reshape array of size 2 into shape (3,)' });
  const rep2 = computeLaterInput(unsearched.m, unsearched.m.nodeOfEvent.get(unsearched.api.event_id)!, { stdout: 'a long enough stdout\n', stderr: '' });
  assert.equal(rep2.requests.find((q) => q.isNextCodegen)!.stdout.state, 'unsearched');
});

test('str(e) under 8 characters with no traceback line is unchecked, not "not included"', () => {
  const s = scenario({ stdout: '', stderr: 'nothing printed about it\n', message: 'bad', stderrFound: true });
  const rep = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: '', stderr: 'nothing printed about it\n' });
  assert.equal(rep.api?.messageTooShort, true);
  assert.equal(rep.api?.tracebackLine, null);
  const next = rep.requests.find((q) => q.isNextCodegen)!;
  assert.equal(next.errorFindings.length, 0);
  assert.equal(next.errorNote, 'unchecked');
});

test('a searchable str(e) that is in neither stream is reported as not printed', () => {
  const s = scenario({ stdout: '', stderr: 'unrelated output here\n', message: 'a message that was never printed', stderrFound: true });
  const rep = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: '', stderr: 'unrelated output here\n' });
  assert.equal(rep.requests.find((q) => q.isNextCodegen)!.errorNote, 'not_in_streams');
});

test('texts that are not loaded yet leave the exception search unchecked', () => {
  const s = scenario({ stdout: 'x', stderr: STDERR_TWO_ERRORS, message: 'cannot reshape array of size 2 into shape (3,)', stderrFound: true });
  const rep = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: null, stderr: null });
  assert.equal(rep.api?.missingStreamText, true);
  assert.equal(rep.requests.find((q) => q.isNextCodegen)!.errorNote, 'unchecked');
});

test('a str(e) the recorder stored with secrets removed (message_redacted) is not searched: unchecked, never "not printed"', () => {
  // The program printed the URL with its query; the recorder stored it redacted.
  const printed = 'HTTPConnectionPool: Max retries exceeded with url: /ik?api_key=REDACTED-EXAMPLE';
  const stored = 'HTTPConnectionPool: Max retries exceeded with url: /ik?[query removed]';
  const stderr = `Traceback (most recent call last):\n  File "<string>", line 3, in <module>\nConnectionError: ${printed}\n`;
  const s = scenario({ stdout: '', stderr, message: stored, stderrFound: true });
  // the fixture's api_call_error, marked as the recorder marks a redacted message
  (s.m.nodeOfEvent.get(s.api.event_id)!.end!.data as Record<string, unknown>).message_redacted = true;
  const rep = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: '', stderr });
  assert.equal(rep.api?.messageRedacted, true);
  assert.equal(rep.api?.errorMessage, null);
  assert.equal(rep.api?.tracebackLine, null);
  const next = rep.requests.find((q) => q.isNextCodegen)!;
  assert.equal(next.errorFindings.length, 0);
  assert.equal(next.errorNote, 'unchecked');
  // without the marker the same stored text would be reported as not printed
  (s.m.nodeOfEvent.get(s.api.event_id)!.end!.data as Record<string, unknown>).message_redacted = false;
  const plain = computeLaterInput(s.m, s.m.nodeOfEvent.get(s.api.event_id)!, { stdout: '', stderr });
  assert.equal(plain.requests.find((q) => q.isNextCodegen)!.errorNote, 'not_in_streams');
});

test('a long str(e) stored as a text asset is searched only with its full text', () => {
  const long = `${'x'.repeat(4100)} tail`;
  const stderr = `Traceback...\nValueError: ${long}\n`;
  const s = scenario({ stdout: '', stderr, message: { truncated: true, preview: long.slice(0, 4000), length: long.length, asset: 'assets/aa/m.txt.gz' }, stderrFound: true });
  const node = s.m.nodeOfEvent.get(s.api.event_id)!;
  const without = computeLaterInput(s.m, node, { stdout: '', stderr });
  assert.equal(without.api?.messageIncomplete, true);
  assert.equal(without.api?.errorMessage, null);
  assert.equal(without.requests.find((q) => q.isNextCodegen)!.errorNote, 'unchecked');
  const withFull = computeLaterInput(s.m, node, { stdout: '', stderr, errorMessage: long });
  assert.equal(withFull.api?.tracebackLine, `ValueError: ${long}`);
  assert.ok(withFull.requests.find((q) => q.isNextCodegen)!.errorFindings.some((f) => f.what === 'message' && f.mapped !== null));
});

test('without api_call_end the API output is unknown (null), not "did not print"', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'move', {});
  r.stream(block, 'stdout', 'something printed\n');
  r.blockEnd(block, 18, 0);
  const m = buildEpisodeModel(r.events);
  const rep = computeLaterInput(m, m.nodeOfEvent.get(api.event_id)!, { stdout: 'something printed\n', stderr: '' });
  assert.equal(rep.api?.endRecorded, false);
  assert.equal(rep.api?.printedStdout, null);
  assert.equal(rep.api?.printedStderr, null);
});

test('checkStream: a missing stream event with an unknown length is not recorded', () => {
  assert.equal(checkStream(summaryWith([]), 'stdout', new Set(['b']), null, null).state, 'not_recorded');
  assert.equal(checkStream(summaryWith([]), 'stdout', new Set(['b']), null, 0).state, 'empty');
});

test('a stream event that the block end names but the record lacks is unknown text, never empty', () => {
  const ev: TraceEvent = { event_id: 'ev_out', seq: 1, event_type: 'stdout' };
  assert.equal(blockStreamText(ev, { text: 'abc', partial: false }, 3, 'ev_out'), 'abc');
  assert.equal(blockStreamText(ev, { text: 'abc', partial: true }, 5000, 'ev_out'), null, 'only a preview is loaded');
  assert.equal(blockStreamText(ev, { text: null, partial: false }, 3, 'ev_out'), null, 'an event without text is not empty text');
  assert.equal(blockStreamText(null, { text: null, partial: false }, 12, 'ev_lost'), null, 'lost event with a length');
  assert.equal(blockStreamText(null, { text: null, partial: false }, null, 'ev_lost'), null, 'lost event named by id');
  assert.equal(blockStreamText(null, { text: null, partial: false }, 0, 'ev_lost'), '', 'the lost event was empty');
  assert.equal(blockStreamText(null, { text: null, partial: false }, null, null), '', 'step() returned no text');
});

test('with the stderr event lost, the exception search is unchecked, not "not printed"', () => {
  const r = new Rec();
  const block = r.blockStart(null);
  const api = r.apiStart(block, 'goto_pose', {});
  r.apiError(api, 'ValueError', 'cannot reshape array of size 2 into shape (3,)');
  r.stream(block, 'stdout', '');
  r.blockEnd(block, 0, 60, { stderr_event_id: 'ev_lost' });
  r.request('code_generator', 'lc1', 1, { input_summary: summaryWith([]) });
  const m = buildEpisodeModel(r.events);
  const stderr = blockStreamText(null, { text: null, partial: false }, 60, 'ev_lost');
  const rep = computeLaterInput(m, m.nodeOfEvent.get(api.event_id)!, { stdout: '', stderr });
  assert.equal(rep.api?.missingStreamText, true);
  assert.equal(rep.requests[0].errorNote, 'unchecked');
});

test('two blocks that printed the same stdout share one merged segment; it still maps exactly for each block', () => {
  const r = new Rec();
  const text = 'Moving to the target pose\n';
  const len = [...text].length;
  const b1 = r.blockStart(null);
  r.apiEnd(r.apiStart(b1, 'move', {}), null, { stdout: [0, len], stderr: [0, 0] });
  const o1 = r.stream(b1, 'stdout', text);
  r.blockEnd(b1, len, 0);
  const b2 = r.blockStart(null);
  const api2 = r.apiStart(b2, 'move', {});
  r.apiEnd(api2, null, { stdout: [0, len], stderr: [0, 0] });
  const o2 = r.stream(b2, 'stdout', text);
  r.blockEnd(b2, len, 0);
  // the recorder merges identical (kind, text): one segment naming both blocks' streams
  const merged = { kind: 'stdout', path: 'messages[1].content[0].text', start: 40, end: 40 + len, source_event_ids: [o1.event_id, b1.event_id, o2.event_id, b2.event_id] };
  r.request('code_generator', 'lc1', 1, { input_summary: summaryWith([merged]) });
  const m = buildEpisodeModel(r.events);
  const rep = computeLaterInput(m, m.nodeOfEvent.get(api2.event_id)!, { stdout: text, stderr: '' });
  const q = rep.requests[0];
  assert.equal(q.stdout.state, 'included');
  assert.equal(q.stdout.hits[0].exact, true);
  assert.deepEqual(q.apiSlices, [{ stream: 'stdout', range: { path: 'messages[1].content[0].text', start: 40, end: 40 + len } }]);
  assert.equal(q.sliceNote, null);
  // a segment of another length (a malformed record) is not mapped
  const odd = checkStream(summaryWith([{ ...merged, end: 41 }]), 'stdout', new Set([b2.event_id, o2.event_id]), o2, len);
  assert.equal(odd.hits[0].exact, false);
});

test('a str(e) the recorder cut to its first 64 KB (message_cut) is not searched: unchecked, never "not printed"', () => {
  // the trace recorder redact(text, None): a message longer than the window keeps its
  // start (minus the last token) and gets a note line, so the stored text is not in stderr.
  const printed = `${'word '.repeat(14000)}end`;
  const stored = `${printed.slice(0, 65536 - 5)}\n[recorder: only the first 65536 of ${printed.length} characters were read]`;
  const stderr = `Traceback (most recent call last):\n  File "<string>", line 3, in <module>\nValueError: ${printed}\n`;
  const s = scenario({ stdout: '', stderr, message: { truncated: true, preview: stored.slice(0, 4000), length: stored.length, asset: 'assets/aa/m.txt.gz' }, stderrFound: true });
  const node = s.m.nodeOfEvent.get(s.api.event_id)!;
  const data = node.end!.data as Record<string, unknown>;
  data.message_cut = true;
  const rep = computeLaterInput(s.m, node, { stdout: '', stderr, errorMessage: stored });
  assert.equal(rep.api?.messageCut, true);
  assert.equal(rep.api?.errorMessage, null);
  assert.equal(rep.api?.tracebackLine, null);
  const next = rep.requests.find((q) => q.isNextCodegen)!;
  assert.equal(next.errorFindings.length, 0);
  assert.equal(next.errorNote, 'unchecked');
  // without the marker the loaded stored text would be searched and reported as not printed
  data.message_cut = false;
  const plain = computeLaterInput(s.m, node, { stdout: '', stderr, errorMessage: stored });
  assert.equal(plain.requests.find((q) => q.isNextCodegen)!.errorNote, 'not_in_streams');
});
