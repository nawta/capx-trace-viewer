/**
 * Browser globals for the page tests (`*.page.test.ts`): `window` (location,
 * history, localStorage, and timers on a virtual clock), `document`,
 * `Date.now`, and `fetch` answered by a fake trace server with a latency per
 * request, so polling and in-flight requests can be tested deterministically.
 *
 * Import this module first in a page test: it installs the globals when it
 * is loaded. `advance(ms)` moves the virtual clock, firing due timers in
 * order and letting promises settle between them.
 *
 * Related: fakeReact.ts (the hook runtime), web-ui/scripts/test-trace.mjs.
 */

// Node's own setImmediate (not in the DOM typings this project compiles against).
declare function setImmediate(cb: () => void): unknown;

interface Timer {
  at: number;
  fn: () => void;
  repeat: number | null;
}

export const clock = { now: 1_790_000_000_000, timers: new Map<number, Timer>(), nextId: 1 };

function addTimer(fn: () => void, ms: unknown, repeat: boolean): number {
  const id = clock.nextId++;
  const d = Math.max(0, Number(ms) || 0);
  clock.timers.set(id, { at: clock.now + d, fn, repeat: repeat ? Math.max(1, d) : null });
  return id;
}

const listeners = new Map<string, (e: unknown) => void>();
const location = { pathname: '/traces', search: '', hash: '' };
const storage = new Map<string, string>();

function setUrl(url: string): void {
  const u = new URL(url, 'http://localhost');
  location.pathname = u.pathname;
  location.search = u.search;
  location.hash = u.hash;
}

const fakeWindow = {
  location,
  history: {
    pushState: (_s: unknown, _t: unknown, url: string) => setUrl(url),
    replaceState: (_s: unknown, _t: unknown, url: string) => setUrl(url),
  },
  addEventListener: (type: string, fn: (e: unknown) => void) => listeners.set(type, fn),
  removeEventListener: (type: string) => listeners.delete(type),
  setTimeout: (fn: () => void, ms?: number) => addTimer(fn, ms, false),
  setInterval: (fn: () => void, ms?: number) => addTimer(fn, ms, true),
  clearTimeout: (id: number) => clock.timers.delete(id),
  clearInterval: (id: number) => clock.timers.delete(id),
  requestAnimationFrame: (fn: () => void) => addTimer(fn, 16, false),
  cancelAnimationFrame: (id: number) => clock.timers.delete(id),
  localStorage: {
    getItem: (k: string) => (storage.has(k) ? (storage.get(k) as string) : null),
    setItem: (k: string, v: string) => storage.set(k, String(v)),
  },
  isSecureContext: false,
};

const g = globalThis as unknown as Record<string, unknown>;
g.window = fakeWindow;
g.document = { title: '' };
Date.now = () => clock.now;

/** Sets the page URL (path and query), as the address bar would. */
export function setLocation(url: string): void {
  setUrl(url);
}

/** The query parameters of the current URL. */
export function currentParams(): URLSearchParams {
  return new URLSearchParams(location.search);
}

/** Fires the popstate listener (the back button). */
export function popState(url: string): void {
  setUrl(url);
  listeners.get('popstate')?.({});
}

async function flush(): Promise<void> {
  for (let i = 0; i < 20; i += 1) await new Promise<void>((r) => setImmediate(r));
}

/** Advances the virtual clock by `ms`, firing timers in order and letting promises settle. */
export async function advance(ms: number): Promise<void> {
  const end = clock.now + ms;
  for (;;) {
    await flush();
    let next: [number, Timer] | null = null;
    for (const [id, t] of clock.timers) {
      if (t.at > end) continue;
      if (next === null || t.at < next[1].at || (t.at === next[1].at && id < next[0])) next = [id, t];
    }
    if (!next) break;
    const [id, t] = next;
    clock.now = t.at;
    if (t.repeat !== null) t.at += t.repeat;
    else clock.timers.delete(id);
    t.fn();
  }
  clock.now = end;
  await flush();
}

// ---------------------------------------------------------------------------
// Fake server
// ---------------------------------------------------------------------------

export interface FakeRequest {
  url: string;
  path: string;
  query: URLSearchParams;
  sentAt: number;
  doneAt: number | null;
  aborted: boolean;
}

export interface FakeAnswer {
  status?: number;
  /** Sent as application/json. */
  json?: unknown;
  /** Sent as text/plain (for example a file that is not valid JSON). */
  text?: string;
  /** Milliseconds on the virtual clock before the answer arrives. */
  latency?: number;
}

export type Handler = (req: FakeRequest) => FakeAnswer | null;

export const net: { handler: Handler; log: FakeRequest[]; latency: number } = {
  handler: () => null,
  log: [],
  latency: 5,
};

/** Requests whose path matches `re`: sent, answered and aborted counts. */
export function requests(re: RegExp): { sent: number; done: number; aborted: number; list: FakeRequest[] } {
  const list = net.log.filter((r) => re.test(r.path));
  return { sent: list.length, done: list.filter((r) => r.doneAt !== null).length, aborted: list.filter((r) => r.aborted).length, list };
}

function answerOf(a: FakeAnswer): Response {
  const status = a.status ?? 200;
  const isJson = a.text === undefined;
  const body = isJson ? JSON.stringify(a.json ?? {}) : (a.text as string);
  const resp = {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'ERR',
    headers: { get: (k: string) => (k.toLowerCase() === 'content-type' ? (isJson ? 'application/json' : 'text/plain') : null) },
    json: async () => JSON.parse(body) as unknown,
    text: async () => body,
    arrayBuffer: async () => new TextEncoder().encode(body).buffer,
  };
  return resp as unknown as Response;
}

function abortError(): Error {
  const e = new Error('The operation was aborted.');
  e.name = 'AbortError';
  return e;
}

g.fetch = (input: unknown, init?: { signal?: AbortSignal | null }): Promise<Response> => {
  const url = String(input);
  const u = new URL(url, 'http://localhost');
  const req: FakeRequest = { url, path: u.pathname, query: u.searchParams, sentAt: clock.now, doneAt: null, aborted: false };
  net.log.push(req);
  const signal = init?.signal ?? null;
  return new Promise<Response>((resolve, reject) => {
    if (signal?.aborted) {
      req.aborted = true;
      reject(abortError());
      return;
    }
    const a = net.handler(req) ?? { status: 404, json: { detail: 'not found' } };
    const id = addTimer(
      () => {
        req.doneAt = clock.now;
        resolve(answerOf(a));
      },
      a.latency ?? net.latency,
      false,
    );
    signal?.addEventListener('abort', () => {
      if (req.doneAt !== null) return;
      req.aborted = true;
      clock.timers.delete(id);
      reject(abortError());
    });
  });
};

/** Clears timers and the request log (between tests). */
export function resetBrowser(): void {
  clock.timers.clear();
  net.log.length = 0;
  net.handler = () => null;
  net.latency = 5;
  storage.clear();
  listeners.clear();
  setUrl('/traces');
}

// ---------------------------------------------------------------------------
// Element trees returned by the fake jsx runtime
// ---------------------------------------------------------------------------

export interface FakeElement {
  type: unknown;
  props: Record<string, unknown>;
}

function isElement(v: unknown): v is FakeElement {
  return !!v && typeof v === 'object' && 'type' in v && 'props' in v;
}

/** Every element of `type` in a tree (children and element-valued props are searched). */
export function findAll(node: unknown, type: unknown, out: FakeElement[] = []): FakeElement[] {
  if (node === null || node === undefined || typeof node !== 'object') return out;
  if (Array.isArray(node)) {
    for (const n of node) findAll(n, type, out);
    return out;
  }
  if (!isElement(node)) return out;
  if (node.type === type) out.push(node);
  for (const v of Object.values(node.props)) {
    if (Array.isArray(v) || isElement(v)) findAll(v, type, out);
  }
  return out;
}

/** The props of the one element of `type` in a tree. */
export function propsOf<P>(node: unknown, type: unknown): P {
  const all = findAll(node, type);
  if (all.length !== 1) throw new Error(`expected one element, found ${all.length}`);
  return all[0].props as P;
}
