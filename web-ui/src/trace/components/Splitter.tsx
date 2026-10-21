/**
 * Drag splitter for the resizable panes of docs/trace-format.md 9.3 (left
 * pane width, media/details split). While dragging, a transparent overlay
 * covers the page so an iframe (live 3D) cannot capture the pointer. Arrow
 * keys move the splitter when it has focus.
 *
 * Related: TraceViewerPage.tsx (owns the sizes).
 */

import { useEffect, useRef, useState } from 'react';

interface SplitterProps {
  /** 'vertical': a vertical bar between left and right (reports clientX). */
  orientation: 'vertical' | 'horizontal';
  onDrag: (clientPos: number) => void;
  onKeyStep: (deltaPx: number) => void;
  ariaLabel: string;
}

export function Splitter({ orientation, onDrag, onKeyStep, ariaLabel }: SplitterProps) {
  const [dragging, setDragging] = useState(false);
  const onDragRef = useRef(onDrag);
  onDragRef.current = onDrag;
  const vertical = orientation === 'vertical';
  const cursor = vertical ? 'col-resize' : 'row-resize';

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      e.preventDefault();
      onDragRef.current(vertical ? e.clientX : e.clientY);
    };
    const up = () => setDragging(false);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = cursor;
    document.body.style.userSelect = 'none';
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [dragging, vertical, cursor]);

  return (
    <>
      <div
        role="separator"
        aria-orientation={vertical ? 'vertical' : 'horizontal'}
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          setDragging(true);
        }}
        onKeyDown={(e) => {
          const back = vertical ? 'ArrowLeft' : 'ArrowUp';
          const fwd = vertical ? 'ArrowRight' : 'ArrowDown';
          if (e.key === back) {
            e.preventDefault();
            onKeyStep(-24);
          } else if (e.key === fwd) {
            e.preventDefault();
            onKeyStep(24);
          }
        }}
        className={`relative flex-shrink-0 z-20 group focus:outline-none ${
          vertical ? 'w-1 cursor-col-resize' : 'h-1 cursor-row-resize'
        } ${dragging ? 'bg-accent' : 'bg-surface-border hover:bg-accent/60 focus-visible:bg-accent/60'} transition-colors`}
      >
        <div className={vertical ? 'absolute inset-y-0 -left-1.5 -right-1.5' : 'absolute inset-x-0 -top-1.5 -bottom-1.5'} />
      </div>
      {dragging && <div className="fixed inset-0 z-50" style={{ cursor }} />}
    </>
  );
}
