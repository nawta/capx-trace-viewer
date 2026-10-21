/**
 * Windowed list with variable row heights, for timelines with thousands of
 * events (docs/trace-format.md 9.3, left pane). Only rows near the viewport
 * are mounted; the rest are replaced by two spacers whose heights use
 * measured row heights where known and estimates elsewhere. Rows are measured
 * with one shared ResizeObserver. Native scroll anchoring keeps the view
 * steady when rows above the viewport change height.
 *
 * Related: Timeline.tsx.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';

interface VirtualListProps<T> {
  items: T[];
  itemKey: (item: T) => string;
  estimateHeight: (item: T) => number;
  renderItem: (item: T) => ReactNode;
  /** Key of the row to bring into view; changing `scrollToken` repeats the request. */
  scrollToKey: string | null;
  scrollToken: number;
  overscanPx?: number;
  className?: string;
  empty?: ReactNode;
}

/** How long a scroll-to-row request keeps correcting for rows that change height. */
const SCROLL_SETTLE_MS = 1500;

function lowerBound(arr: Float64Array, v: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function VirtualList<T>({
  items,
  itemKey,
  estimateHeight,
  renderItem,
  scrollToKey,
  scrollToken,
  overscanPx = 600,
  className = '',
  empty,
}: VirtualListProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const heights = useRef(new Map<string, number>());
  const [version, setVersion] = useState(0);
  const [view, setView] = useState({ top: 0, height: 800 });
  const rafRef = useRef<number | null>(null);
  const bumpRef = useRef<number | null>(null);

  const keys = useMemo(() => items.map(itemKey), [items, itemKey]);
  const offsets = useMemo(() => {
    const arr = new Float64Array(items.length + 1);
    for (let i = 0; i < items.length; i += 1) {
      arr[i + 1] = arr[i] + (heights.current.get(keys[i]) ?? estimateHeight(items[i]));
    }
    return arr;
    // `version` changes when measured heights change.
  }, [items, keys, estimateHeight, version]);

  const observer = useMemo(() => {
    if (typeof ResizeObserver === 'undefined') return null;
    return new ResizeObserver((entries) => {
      let changed = false;
      for (const e of entries) {
        const key = (e.target as HTMLElement).dataset.vkey;
        if (!key) continue;
        const h = Math.round((e.target as HTMLElement).getBoundingClientRect().height);
        if (h > 0 && heights.current.get(key) !== h) {
          heights.current.set(key, h);
          changed = true;
        }
      }
      if (changed && bumpRef.current === null) {
        bumpRef.current = window.requestAnimationFrame(() => {
          bumpRef.current = null;
          setVersion((v) => v + 1);
        });
      }
    });
  }, []);

  useEffect(
    () => () => {
      observer?.disconnect();
      if (bumpRef.current !== null) window.cancelAnimationFrame(bumpRef.current);
      if (rafRef.current !== null) window.cancelAnimationFrame(rafRef.current);
    },
    [observer],
  );

  const readView = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    setView((v) => (v.top === el.scrollTop && v.height === el.clientHeight ? v : { top: el.scrollTop, height: el.clientHeight }));
  }, []);

  useLayoutEffect(() => {
    readView();
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => readView());
    ro.observe(el);
    return () => ro.disconnect();
  }, [readView]);

  const onScroll = () => {
    if (rafRef.current !== null) return;
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      readView();
    });
  };

  // Bring the requested row into view. Row heights settle after mount (the
  // ResizeObserver measures rows, thumbnails load and grow the cards), so the
  // request is checked against the row's real position on every render for a
  // short while, and dropped as soon as the user scrolls by hand.
  // Declared before the effect below so both run in the same commit.
  const pending = useRef<{ key: string; until: number } | null>(null);
  useLayoutEffect(() => {
    pending.current = scrollToKey ? { key: scrollToKey, until: performance.now() + SCROLL_SETTLE_MS } : null;
  }, [scrollToKey, scrollToken]);
  useLayoutEffect(() => {
    const p = pending.current;
    const el = containerRef.current;
    if (!p || !el) return;
    const idx = keys.indexOf(p.key);
    if (idx < 0 || performance.now() > p.until) {
      pending.current = null;
      return;
    }
    const margin = Math.min(el.clientHeight * 0.3, 160);
    const row = el.querySelector<HTMLElement>(`[data-vkey="${CSS.escape(p.key)}"]`);
    if (!row) {
      el.scrollTop = Math.max(0, offsets[idx] - margin);
      readView();
      return;
    }
    const r = row.getBoundingClientRect();
    const c = el.getBoundingClientRect();
    const visible =
      r.height <= c.height ? r.top >= c.top - 1 && r.bottom <= c.bottom + 1 : r.top >= c.top - 1 && r.top <= c.bottom - 48;
    if (!visible) {
      el.scrollTop = Math.max(0, el.scrollTop + (r.top - c.top) - margin);
      readView();
    }
  });
  const cancelPending = () => {
    pending.current = null;
  };

  const n = items.length;
  const total = offsets[n];
  const start = Math.max(0, lowerBound(offsets, view.top - overscanPx) - 1);
  const end = Math.min(n, lowerBound(offsets, view.top + view.height + overscanPx) + 1);

  return (
    <div
      ref={containerRef}
      onScroll={onScroll}
      onWheel={cancelPending}
      onTouchStart={cancelPending}
      onPointerDown={cancelPending}
      onKeyDown={cancelPending}
      className={`overflow-y-auto ${className}`}
      style={{ overflowAnchor: 'auto' }}
    >
      {n === 0 && empty}
      <div style={{ height: offsets[start] }} />
      {items.slice(start, end).map((item, i) => {
        const key = keys[start + i];
        return (
          <MeasuredRow key={key} vkey={key} observer={observer}>
            {renderItem(item)}
          </MeasuredRow>
        );
      })}
      <div style={{ height: Math.max(0, total - offsets[end]) }} />
    </div>
  );
}

function MeasuredRow({ vkey, observer, children }: { vkey: string; observer: ResizeObserver | null; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !observer) return;
    observer.observe(el);
    return () => observer.unobserve(el);
  }, [observer]);
  return (
    <div ref={ref} data-vkey={vkey}>
      {children}
    </div>
  );
}
