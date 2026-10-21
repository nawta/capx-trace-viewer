/**
 * Formatting helpers of the trace viewer page (docs/trace-format.md 9.3):
 * numbers, durations, byte sizes, hashes and times. Pure functions; no React.
 * Related: components/common.tsx, components/TopBar.tsx.
 */

export function fmtDuration(s: number | null | undefined): string {
  if (typeof s !== 'number' || !Number.isFinite(s)) return '—';
  if (s < 1) return `${(s * 1000).toFixed(s < 0.01 ? 2 : 0)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  return `${m} 分 ${(s - m * 60).toFixed(0)} 秒`;
}

export function fmtBytes(n: number | null | undefined): string {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function shortSha(sha: string | null | undefined, n = 10): string {
  if (!sha) return '—';
  return sha.length > n ? sha.slice(0, n) : sha;
}

/** Wall time (seconds since epoch, or an ISO string) as local date-time. */
export function fmtWall(t: number | string | null | undefined): string {
  if (t === null || t === undefined || t === '') return '—';
  const d = typeof t === 'number' ? new Date(t * (t < 1e12 ? 1000 : 1)) : new Date(t);
  if (Number.isNaN(d.getTime())) return String(t);
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function fmtClock(ms: number | null | undefined): string {
  if (typeof ms !== 'number') return '—';
  const d = new Date(ms);
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function fmtNum(v: unknown, digits = 4): string {
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
    return String(Number(v.toFixed(digits)));
  }
  if (typeof v === 'string') return v;
  if (v === null || v === undefined) return '—';
  return String(v);
}

export function fmtBool(v: unknown): string {
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (v === null || v === undefined) return '—';
  return String(v);
}

export function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

export function firstLines(s: string, n: number): { text: string; more: number } {
  const lines = s.split('\n');
  return { text: lines.slice(0, n).join('\n'), more: Math.max(0, lines.length - n) };
}

/** Compact one-line JSON for card summaries. */
export function compactJson(v: unknown, max = 160): string {
  let s: string;
  try {
    s = typeof v === 'string' ? JSON.stringify(v) : JSON.stringify(v) ?? String(v);
  } catch {
    s = String(v);
  }
  return truncate(s, max);
}
