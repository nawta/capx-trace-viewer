/**
 * Which page a URL path opens (docs/trace-format.md 9.1): the trace viewer for
 * exactly `/traces` or a path below it (`/traces/...`), the chat UI for every
 * other path, including look-alikes such as `/traces-old`. Imported by
 * CaP-X route selection before the interactive page mounts, so it stays free of other imports.
 */
export function isTraceViewerPath(pathname: string): boolean {
  return pathname === '/traces' || pathname.startsWith('/traces/');
}
