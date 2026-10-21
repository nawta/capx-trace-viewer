/**
 * Small shared components of the trace viewer: badges, the fixed-label chips
 * of docs/trace-format.md 9.3 (「モデルに送信」, the debug banner, 「未記録」),
 * copy buttons with a fallback for plain-http pages, key/value tables and
 * links between events.
 *
 * Visual language follows the chat UI (web-ui/tailwind.config.js tokens
 * surface, sand, accent, nv and the display/mono fonts).
 */

import { useState, type ReactNode } from 'react';
import { LABELS } from '../labels';

export type Tone = 'neutral' | 'gold' | 'green' | 'red' | 'amber' | 'sky' | 'violet';

const TONES: Record<Tone, string> = {
  neutral: 'bg-surface-overlay text-text-secondary border-surface-border-light',
  gold: 'bg-accent/10 text-accent border-accent/30',
  green: 'bg-nv-green/10 text-nv-green border-nv-green/30',
  red: 'bg-red-500/10 text-red-400 border-red-500/30',
  amber: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
  sky: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
  violet: 'bg-violet-500/10 text-violet-300 border-violet-500/30',
};

export function Badge({ tone = 'neutral', children, title, className = '' }: { tone?: Tone; children: ReactNode; title?: string; className?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 px-1.5 py-px rounded border text-[11px] leading-4 font-medium whitespace-nowrap ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function statusTone(status: string | null | undefined): Tone {
  switch (status) {
    case 'ok':
      return 'green';
    case 'error':
      return 'red';
    case 'degraded':
    case 'warning':
      return 'amber';
    case 'started':
      return 'gold';
    default:
      return 'neutral';
  }
}

/** 「モデルに送信」: the item was part of a model input. */
export function SentToModelChip() {
  return <Badge tone="gold">{LABELS.sentToModel}</Badge>;
}

/** 「デバッグ用フレーム（モデル入力ではない）」. */
export function DebugFrameChip() {
  return <Badge tone="sky">{LABELS.debugFrame}</Badge>;
}

/** Fixed banner of the debug tab: 「モデル入力とは別（デバッグ用）」. */
export function DebugBanner() {
  return (
    <div className="sticky top-0 z-10 flex items-center gap-2 px-3 py-2 mb-3 rounded-md border border-sky-500/40 bg-sky-950/80 backdrop-blur text-sky-200 text-sm font-medium">
      <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      {LABELS.debugBanner}
    </div>
  );
}

/** 「未記録」 with the reason when one is known. */
export function NotRecorded({ reason, className = '' }: { reason?: string | null; className?: string }) {
  return (
    <div className={`px-3 py-2 rounded-md border border-dashed border-surface-border-light bg-surface-sunken text-sm ${className}`}>
      <span className="font-bold text-text-primary">{LABELS.notRecorded}</span>
      {reason && <span className="ml-2 text-text-tertiary">{reason}</span>}
    </div>
  );
}

export function Notice({ tone = 'neutral', children, className = '' }: { tone?: Tone; children: ReactNode; className?: string }) {
  const border: Record<Tone, string> = {
    neutral: 'border-surface-border-light bg-surface-sunken text-text-secondary',
    gold: 'border-accent/30 bg-accent/5 text-accent-light',
    green: 'border-nv-green/30 bg-nv-green/5 text-nv-green-light',
    red: 'border-red-500/40 bg-red-500/5 text-red-300',
    amber: 'border-amber-500/40 bg-amber-500/5 text-amber-200',
    sky: 'border-sky-500/40 bg-sky-500/5 text-sky-200',
    violet: 'border-violet-500/40 bg-violet-500/5 text-violet-200',
  };
  return <div className={`px-3 py-2 rounded-md border text-sm ${border[tone]} ${className}`}>{children}</div>;
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 mt-4 mb-2 first:mt-0">
      <h3 className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">{children}</h3>
      {right}
    </div>
  );
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <table className="w-full text-xs border-collapse">
      <tbody>
        {rows.map(([k, v], i) => (
          <tr key={i} className="border-b border-surface-border/60 last:border-0 align-top">
            <th className="text-left font-normal text-text-tertiary py-1 pr-3 whitespace-nowrap w-0">{k}</th>
            <td className="py-1 text-text-primary font-mono break-all">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-text-tertiary">
      <span className="w-3 h-3 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
      {label}
    </span>
  );
}

export function ErrorText({ children }: { children: ReactNode }) {
  return <div className="text-xs text-red-400 break-all">{children}</div>;
}

/** Copies text; falls back to a hidden textarea when the Clipboard API is missing (plain http). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the textarea path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

export function CopyButton({ text, label = 'コピー' }: { text: string | (() => string); label?: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle');
  return (
    <button
      type="button"
      onClick={async (e) => {
        e.stopPropagation();
        const ok = await copyText(typeof text === 'function' ? text() : text);
        setState(ok ? 'ok' : 'fail');
        window.setTimeout(() => setState('idle'), 1500);
      }}
      className="px-2 py-0.5 rounded border border-surface-border-light text-[11px] text-text-secondary hover:text-accent hover:border-accent/40 transition-colors"
    >
      {state === 'ok' ? 'コピーしました' : state === 'fail' ? 'コピー失敗' : label}
    </button>
  );
}

/** A link that selects another event in the viewer. */
export function EventLink({ onClick, children, title }: { onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button
      type="button"
      title={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className="text-accent hover:text-accent-light underline decoration-accent/40 underline-offset-2 text-left"
    >
      {children}
    </button>
  );
}

export function SmallButton({
  onClick,
  children,
  active = false,
  disabled = false,
  title,
}: {
  onClick: () => void;
  children: ReactNode;
  active?: boolean;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={`px-2 py-1 rounded border text-xs transition-colors disabled:opacity-40 ${
        active
          ? 'border-accent/60 bg-accent/15 text-accent'
          : 'border-surface-border-light bg-surface-sunken text-text-secondary hover:text-text-primary hover:border-surface-border-light'
      }`}
    >
      {children}
    </button>
  );
}
