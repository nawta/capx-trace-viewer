/**
 * Highlighted code with line numbers and one optional marked line (the API
 * call site, docs/trace-format.md 9.3 tab 3). It follows the CaP-X web UI code viewer, which has no line marking. `maxLines` folds long
 * code with a button to show the rest.
 *
 * Related: highlight.ts, Timeline.tsx (folded previews), tabs/*.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { highlight, splitHighlightedLines } from '../highlight';
import { CopyButton } from './common';

interface CodeViewProps {
  code: string;
  language?: 'python' | 'json';
  /** 1-based line to mark. */
  markLine?: number | null;
  /**
   * Scroll the enclosing panel so the marked line is visible. Only for views where the
   * code is the main content: elsewhere it would push the rest of the panel out of view.
   */
  scrollToMark?: boolean;
  maxLines?: number;
  compact?: boolean;
  showCopy?: boolean;
  className?: string;
}

export function CodeView({
  code,
  language = 'python',
  markLine = null,
  scrollToMark = false,
  maxLines,
  compact = false,
  showCopy = true,
  className = '',
}: CodeViewProps) {
  const [expanded, setExpanded] = useState(false);
  const allLines = useMemo(() => code.split('\n'), [code]);
  const limit = maxLines && !expanded && allLines.length > maxLines ? maxLines : allLines.length;
  // Keep the marked line visible when folding.
  const from = markLine && limit < allLines.length && markLine > limit ? Math.max(0, markLine - Math.ceil(limit / 2)) : 0;
  const shown = useMemo(() => allLines.slice(from, from + limit).join('\n'), [allLines, from, limit]);
  const html = useMemo(() => splitHighlightedLines(highlight(shown, language)), [shown, language]);
  const markRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (scrollToMark) markRef.current?.scrollIntoView({ block: 'center' });
  }, [markLine, code, scrollToMark]);

  const text = compact ? 'text-[11px] leading-[1.15rem]' : 'text-xs leading-5';
  return (
    <div className={`relative bg-surface-sunken border border-surface-border border-t-2 border-t-accent/20 rounded-md overflow-hidden ${className}`}>
      {showCopy && (
        <div className="absolute top-1.5 right-1.5 z-10">
          <CopyButton text={code} />
        </div>
      )}
      <div className="overflow-x-auto">
        <div className={`font-mono ${text} py-1.5 min-w-max`}>
          {from > 0 && <div className="pl-12 pr-3 text-text-muted">… {from} 行省略</div>}
          {html.map((line, i) => {
            const n = from + i + 1;
            const marked = markLine === n;
            return (
              <div key={n} ref={marked ? markRef : undefined} className={`flex ${marked ? 'bg-accent/15 border-l-2 border-accent' : 'border-l-2 border-transparent'}`}>
                <span className="w-10 flex-shrink-0 pr-2 text-right text-text-muted select-none">{n}</span>
                <span className="pr-4 whitespace-pre text-sand-200" dangerouslySetInnerHTML={{ __html: line || ' ' }} />
              </div>
            );
          })}
        </div>
      </div>
      {limit < allLines.length && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(true);
          }}
          className="w-full text-[11px] text-accent-light hover:text-accent py-1 border-t border-surface-border bg-surface-overlay/50"
        >
          残り {allLines.length - limit} 行を表示
        </button>
      )}
      {expanded && maxLines && allLines.length > maxLines && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(false);
          }}
          className="w-full text-[11px] text-accent-light hover:text-accent py-1 border-t border-surface-border bg-surface-overlay/50"
        >
          折りたたむ
        </button>
      )}
    </div>
  );
}
