/**
 * Line diff for the input-diff tab (docs/trace-format.md 9.3, tab 5).
 *
 * Myers' O((N+M)D) algorithm over interned lines, after trimming the common
 * prefix and suffix. No npm dependency (9.3). When the edit distance exceeds
 * `maxD`, the diff falls back to "all removed, all added" for the middle part
 * and says so, instead of freezing the page.
 *
 * Related: components/tabs/InputDiffTab.tsx (message alignment and rendering).
 */

export type DiffOpType = 'equal' | 'insert' | 'delete';

export interface DiffOp {
  type: DiffOpType;
  /** Line text. */
  text: string;
  /** 0-based line number in the old text (equal/delete). */
  aIndex: number | null;
  /** 0-based line number in the new text (equal/insert). */
  bIndex: number | null;
}

export interface DiffResult {
  ops: DiffOp[];
  /** True when the middle part exceeded `maxD` and was not diffed line by line. */
  gaveUp: boolean;
  added: number;
  removed: number;
}

/** Myers diff of two integer sequences. Returns null when D exceeds maxD. */
function myers(a: Int32Array, b: Int32Array, maxD: number): DiffOpType[] | null {
  const n = a.length;
  const m = b.length;
  const max = n + m;
  if (max === 0) return [];
  const limit = Math.min(max, maxD);
  const offset = limit + 1;
  const v = new Int32Array(2 * limit + 3);
  // trace[d] holds v[k] for k in [-d-1, d+1] as it was before step d.
  const trace: Int32Array[] = [];
  let found = -1;
  for (let d = 0; d <= limit; d += 1) {
    trace.push(v.slice(offset - d - 1, offset + d + 2));
    for (let k = -d; k <= d; k += 2) {
      let x: number;
      if (k === -d || (k !== d && v[offset + k - 1] < v[offset + k + 1])) {
        x = v[offset + k + 1];
      } else {
        x = v[offset + k - 1] + 1;
      }
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) {
        x += 1;
        y += 1;
      }
      v[offset + k] = x;
      if (x >= n && y >= m) {
        found = d;
        break;
      }
    }
    if (found >= 0) break;
  }
  if (found < 0) return null;

  // Backtrack.
  const ops: DiffOpType[] = [];
  let x = n;
  let y = m;
  for (let d = found; d >= 0; d -= 1) {
    const vd = trace[d];
    const at = (kk: number) => vd[kk + d + 1];
    const k = x - y;
    let prevK: number;
    if (k === -d || (k !== d && at(k - 1) < at(k + 1))) {
      prevK = k + 1;
    } else {
      prevK = k - 1;
    }
    const prevX = d === 0 ? 0 : at(prevK);
    const prevY = prevX - prevK;
    while (x > prevX && y > prevY) {
      ops.push('equal');
      x -= 1;
      y -= 1;
    }
    if (d > 0) {
      if (x === prevX) {
        ops.push('insert');
        y -= 1;
      } else {
        ops.push('delete');
        x -= 1;
      }
    }
  }
  while (x > 0 && y > 0) {
    ops.push('equal');
    x -= 1;
    y -= 1;
  }
  ops.reverse();
  return ops;
}

/** Diff of two sequences compared with `===` on their interned values. */
export function diffSequences(aItems: string[], bItems: string[], maxD = 1500): DiffResult {
  const intern = new Map<string, number>();
  const id = (s: string) => {
    let v = intern.get(s);
    if (v === undefined) {
      v = intern.size;
      intern.set(s, v);
    }
    return v;
  };
  const a = Int32Array.from(aItems, id);
  const b = Int32Array.from(bItems, id);

  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre += 1;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf += 1;

  const ops: DiffOp[] = [];
  for (let i = 0; i < pre; i += 1) ops.push({ type: 'equal', text: aItems[i], aIndex: i, bIndex: i });

  const midA = a.subarray(pre, a.length - suf);
  const midB = b.subarray(pre, b.length - suf);
  const path = myers(midA, midB, maxD);
  let gaveUp = false;
  let ai = pre;
  let bi = pre;
  if (path) {
    for (const t of path) {
      if (t === 'equal') {
        ops.push({ type: 'equal', text: aItems[ai], aIndex: ai, bIndex: bi });
        ai += 1;
        bi += 1;
      } else if (t === 'delete') {
        ops.push({ type: 'delete', text: aItems[ai], aIndex: ai, bIndex: null });
        ai += 1;
      } else {
        ops.push({ type: 'insert', text: bItems[bi], aIndex: null, bIndex: bi });
        bi += 1;
      }
    }
  } else {
    gaveUp = true;
    for (; ai < a.length - suf; ai += 1) ops.push({ type: 'delete', text: aItems[ai], aIndex: ai, bIndex: null });
    for (; bi < b.length - suf; bi += 1) ops.push({ type: 'insert', text: bItems[bi], aIndex: null, bIndex: bi });
  }
  for (let i = 0; i < suf; i += 1) {
    const aIdx = a.length - suf + i;
    const bIdx = b.length - suf + i;
    ops.push({ type: 'equal', text: aItems[aIdx], aIndex: aIdx, bIndex: bIdx });
  }
  let added = 0;
  let removed = 0;
  for (const o of ops) {
    if (o.type === 'insert') added += 1;
    else if (o.type === 'delete') removed += 1;
  }
  return { ops, gaveUp, added, removed };
}

export function diffLines(oldText: string, newText: string, maxD = 1500): DiffResult {
  return diffSequences(oldText.split('\n'), newText.split('\n'), maxD);
}

export interface Hunk {
  ops: DiffOp[];
  /** Number of equal lines folded before this hunk. */
  foldedBefore: number;
}

/** Groups ops into hunks with `context` equal lines around each change. */
export function toHunks(ops: DiffOp[], context = 3): { hunks: Hunk[]; foldedAfter: number } {
  const hunks: Hunk[] = [];
  const changeIdx: number[] = [];
  ops.forEach((o, i) => {
    if (o.type !== 'equal') changeIdx.push(i);
  });
  if (changeIdx.length === 0) return { hunks: [], foldedAfter: ops.length };
  let i = 0;
  let lastEnd = 0;
  while (i < changeIdx.length) {
    const start = Math.max(lastEnd, changeIdx[i] - context);
    let end = changeIdx[i] + 1;
    while (i + 1 < changeIdx.length && changeIdx[i + 1] - end <= context * 2) {
      i += 1;
      end = changeIdx[i] + 1;
    }
    const stop = Math.min(ops.length, end + context);
    hunks.push({ ops: ops.slice(start, stop), foldedBefore: start - lastEnd });
    lastEnd = stop;
    i += 1;
  }
  return { hunks, foldedAfter: ops.length - lastEnd };
}
