/**
 * Long text with fold/unfold, search marks and coloured ranges
 * (docs/trace-format.md 9.3: message text in the model-input tab with the
 * six-category segments drawn on it, stdout/stderr with the API's own slice
 * highlighted). Ranges are UTF-16 indices; convert recorder offsets with
 * offsets.ts first. Overlapping ranges are split into pieces; the shortest
 * covering range decides the colour and every covering range is listed in
 * the tooltip.
 *
 * Related: tabs/ModelInputTab.tsx, tabs/ApiLogTab.tsx.
 */

import { useEffect, useMemo, useRef, useState } from 'react';

export interface HighlightRange {
  start: number;
  end: number;
  className: string;
  title?: string;
  id?: string;
}

interface TextViewProps {
  text: string;
  ranges?: HighlightRange[];
  search?: string;
  foldLines?: number;
  /** Id of a range to reveal (the text unfolds when the range lies past the fold). */
  focusRangeId?: string | null;
  focusToken?: number;
  /**
   * Also scroll the enclosing panel to the focused range. Off where the text is not the
   * main content (it would push the rest of the panel out of view).
   */
  scrollToFocus?: boolean;
  mono?: boolean;
  className?: string;
  tone?: 'default' | 'stderr';
}

const NO_RANGES: HighlightRange[] = [];

interface Piece {
  start: number;
  end: number;
  range: HighlightRange | null;
  titles: string[];
  ids: string[];
  match: boolean;
}

function findMatches(text: string, q: string): [number, number][] {
  if (!q) return [];
  const out: [number, number][] = [];
  const hay = text.toLowerCase();
  const needle = q.toLowerCase();
  let i = hay.indexOf(needle);
  while (i >= 0 && out.length < 5000) {
    out.push([i, i + needle.length]);
    i = hay.indexOf(needle, i + Math.max(1, needle.length));
  }
  return out;
}

function buildPieces(ranges: HighlightRange[], matches: [number, number][], limit: number): Piece[] {
  const bounds = new Set<number>([0, limit]);
  const rs = ranges
    .map((r) => ({ ...r, start: Math.max(0, Math.min(r.start, limit)), end: Math.max(0, Math.min(r.end, limit)) }))
    .filter((r) => r.end > r.start);
  for (const r of rs) {
    bounds.add(r.start);
    bounds.add(r.end);
  }
  const ms = matches.filter(([s]) => s < limit).map(([s, e]) => [s, Math.min(e, limit)] as [number, number]);
  for (const [s, e] of ms) {
    bounds.add(s);
    bounds.add(e);
  }
  const sorted = Array.from(bounds).sort((a, b) => a - b);
  const pieces: Piece[] = [];
  let mi = 0;
  for (let i = 0; i + 1 < sorted.length; i += 1) {
    const s = sorted[i];
    const e = sorted[i + 1];
    if (e <= s) continue;
    let best: HighlightRange | null = null;
    const titles: string[] = [];
    const ids: string[] = [];
    for (const r of rs) {
      if (r.start <= s && r.end >= e) {
        if (!best || r.end - r.start < best.end - best.start) best = r;
        if (r.title) titles.push(r.title);
        if (r.id) ids.push(r.id);
      }
    }
    while (mi < ms.length && ms[mi][1] <= s) mi += 1;
    const match = mi < ms.length && ms[mi][0] <= s && ms[mi][1] >= e;
    pieces.push({ start: s, end: e, range: best, titles, ids, match });
  }
  return pieces;
}

export function TextView({
  text,
  ranges = NO_RANGES,
  search = '',
  foldLines = 40,
  focusRangeId = null,
  focusToken = 0,
  scrollToFocus = true,
  mono = true,
  className = '',
  tone = 'default',
}: TextViewProps) {
  const lineCount = useMemo(() => {
    let n = 1;
    for (let i = 0; i < text.length; i += 1) if (text.charCodeAt(i) === 10) n += 1;
    return n;
  }, [text]);
  const [open, setOpen] = useState(false);
  const foldAt = useMemo(() => {
    if (lineCount <= foldLines) return text.length;
    let n = 0;
    for (let i = 0; i < text.length; i += 1) {
      if (text.charCodeAt(i) === 10) {
        n += 1;
        if (n >= foldLines) return i;
      }
    }
    return text.length;
  }, [text, lineCount, foldLines]);

  const matches = useMemo(() => findMatches(text, search.trim()), [text, search]);
  const focusRange = focusRangeId ? ranges.find((r) => r.id === focusRangeId) ?? null : null;
  const needOpen = (focusRange !== null && focusRange.start >= foldAt) || (matches.length > 0 && matches[matches.length - 1][0] >= foldAt);
  const effectiveOpen = open || needOpen;
  const limit = effectiveOpen ? text.length : foldAt;
  const pieces = useMemo(() => buildPieces(ranges, matches, limit), [ranges, matches, limit]);

  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!scrollToFocus || !focusRangeId || !containerRef.current) return;
    const el = containerRef.current.querySelector(`[data-rid~="${CSS.escape(focusRangeId)}"]`);
    el?.scrollIntoView({ block: 'center' });
  }, [focusRangeId, focusToken, limit, scrollToFocus]);

  const base = tone === 'stderr' ? 'text-red-300' : 'text-text-primary';
  return (
    <div className={className}>
      <div
        ref={containerRef}
        className={`${mono ? 'font-mono text-xs' : 'text-sm'} ${base} whitespace-pre-wrap break-words leading-5 bg-surface-sunken border border-surface-border rounded-md px-3 py-2`}
      >
        {pieces.map((p) => {
          const content = text.slice(p.start, p.end);
          if (!p.range && !p.match) return <span key={p.start}>{content}</span>;
          return (
            <span
              key={p.start}
              data-rid={p.ids.length ? p.ids.join(' ') : undefined}
              title={p.titles.length ? p.titles.join('\n') : undefined}
              className={`${p.range ? p.range.className : ''} ${p.match ? 'outline outline-1 outline-accent bg-accent/30 text-white' : ''}`}
            >
              {content}
            </span>
          );
        })}
        {limit < text.length && <span className="text-text-muted">{'\n'}…</span>}
      </div>
      {lineCount > foldLines && (
        <div className="flex items-center gap-3 mt-1">
          <button
            type="button"
            onClick={() => setOpen(!effectiveOpen)}
            disabled={needOpen && !open}
            className="text-[11px] text-accent-light hover:text-accent disabled:opacity-50"
          >
            {effectiveOpen ? '折りたたむ' : `全文を表示（${lineCount} 行）`}
          </button>
          {search.trim() && <span className="text-[11px] text-text-tertiary">一致 {matches.length} 件</span>}
        </div>
      )}
      {lineCount <= foldLines && search.trim() && matches.length > 0 && (
        <div className="mt-1 text-[11px] text-text-tertiary">一致 {matches.length} 件</div>
      )}
    </div>
  );
}
