/**
 * Reading events.jsonl page by page through `/events` (docs/trace-format.md
 * 9.2), without React, so the paging rules can be tested on their own.
 *
 * - One pass reads pages from the saved byte offset until eof. Only one pass
 *   runs at a time; a `pump()` call made while a pass runs is not dropped: the
 *   running pass reads once more when it finishes, so a final fetch after the
 *   episode ended always sees the lines written by then.
 * - File identity: every page carries `generation` (device, inode and a hash of
 *   the first line). The pager sends the generation it has with each request.
 *   When the file was replaced (the server answers 409 generation_changed, or a
 *   page names another generation), everything read so far is dropped: events,
 *   the de-duplication set and the offset. Reading starts again at 0 and
 *   `replaced` is set, so the page can say the file changed.
 * - An older server without generations answers 409 when the saved offset does
 *   not fit the file any more; the pager then also reads again from 0.
 * - `passes` and `quietPasses` (consecutive passes that found no new event)
 *   let the page keep polling after an episode ended until nothing new comes.
 *
 * Related: hooks.ts (useEpisodeEvents wraps this class), api.ts.
 */

import { ApiError, errorText } from './api';
import type { EventsPage, TraceEvent } from './types';

/** Resets allowed within one pass before the pager gives up (a file replaced again and again). */
const MAX_RESETS_PER_PASS = 3;
const MAX_PAGES_PER_PASS = 10_000;

export interface PagerSnapshot {
  events: TraceEvent[];
  nextOffset: number;
  eof: boolean;
  tornTail: boolean;
  corruptLines: number;
  /** True until the first pass (or the first pass after a reset) reached eof or failed. */
  loading: boolean;
  error: string | null;
  lastFetch: number | null;
  pages: number;
  /** events.jsonl was replaced while the page showed it and was read again from the start. */
  replaced: boolean;
  generation: string | null;
  /** How many times the pager dropped what it had read (file replaced). */
  resets: number;
  /** Completed passes. */
  passes: number;
  /** Consecutive completed passes that found no new event. */
  quietPasses: number;
}

export const EMPTY_SNAPSHOT: PagerSnapshot = {
  events: [],
  nextOffset: 0,
  eof: false,
  tornTail: false,
  corruptLines: 0,
  loading: false,
  error: null,
  lastFetch: null,
  pages: 0,
  replaced: false,
  generation: null,
  resets: 0,
  passes: 0,
  quietPasses: 0,
};

export type FetchPage = (afterOffset: number, generation: string | null) => Promise<EventsPage>;

/** The new generation named by a 409 `generation_changed` answer, or null for any other error. */
export function generationChangedOf(err: unknown): string | null {
  if (!(err instanceof ApiError) || err.status !== 409) return null;
  const body = err.body;
  const pick = (v: unknown): string | null => {
    if (!v || typeof v !== 'object') return null;
    const o = v as { error?: unknown; generation?: unknown };
    return o.error === 'generation_changed' && typeof o.generation === 'string' ? o.generation : null;
  };
  // FastAPI wraps an HTTPException detail as {"detail": {...}}; a plain JSON answer has the fields at the top.
  return pick(body) ?? pick(body && typeof body === 'object' ? (body as { detail?: unknown }).detail : null);
}

function isOffsetConflict(err: unknown): boolean {
  return err instanceof ApiError && err.status === 409;
}

export class EventsPager {
  private readonly fetchPage: FetchPage;
  private readonly onUpdate: (s: PagerSnapshot) => void;
  private state: PagerSnapshot = { ...EMPTY_SNAPSHOT, loading: true };
  private seen = new Set<string>();
  private running: Promise<void> | null = null;
  private rerun = false;
  private disposed = false;

  constructor(fetchPage: FetchPage, onUpdate: (s: PagerSnapshot) => void) {
    this.fetchPage = fetchPage;
    this.onUpdate = onUpdate;
  }

  get snapshot(): PagerSnapshot {
    return this.state;
  }

  /** Stops publishing; results of requests still in flight are ignored. */
  dispose(): void {
    this.disposed = true;
  }

  /**
   * Reads everything new. While a pass runs, a call marks it to read once more and returns
   * the running pass's promise (which includes that extra read).
   */
  pump(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (this.running) {
      this.rerun = true;
      return this.running;
    }
    const run = (async () => {
      try {
        do {
          this.rerun = false;
          await this.pass();
        } while (this.rerun && !this.disposed);
      } finally {
        this.running = null;
      }
    })();
    this.running = run;
    return run;
  }

  private publish(patch: Partial<PagerSnapshot>): void {
    this.state = { ...this.state, ...patch };
    if (!this.disposed) this.onUpdate(this.state);
  }

  /** Drops what was read (the file was replaced) and adopts `generation`. */
  private reset(generation: string | null): void {
    const hadData = this.state.nextOffset > 0 || this.state.events.length > 0;
    this.seen = new Set();
    this.publish({
      events: [],
      nextOffset: 0,
      eof: false,
      tornTail: false,
      corruptLines: 0,
      loading: true,
      generation,
      replaced: this.state.replaced || hadData,
      resets: this.state.resets + (hadData ? 1 : 0),
    });
  }

  private async pass(): Promise<void> {
    let resets = 0;
    let fresh = 0;
    for (let n = 0; n < MAX_PAGES_PER_PASS; n += 1) {
      if (this.disposed) return;
      const from = this.state.nextOffset;
      let page: EventsPage;
      try {
        page = await this.fetchPage(from, this.state.generation);
      } catch (err) {
        if (this.disposed) return;
        const changed = generationChangedOf(err);
        if (changed !== null && resets < MAX_RESETS_PER_PASS) {
          resets += 1;
          this.reset(changed);
          continue;
        }
        if (changed === null && isOffsetConflict(err) && from > 0 && resets < MAX_RESETS_PER_PASS) {
          // An older server: the saved offset does not fit the file any more, so it was replaced.
          resets += 1;
          this.reset(null);
          continue;
        }
        this.publish({ error: errorText(err), lastFetch: Date.now(), loading: false });
        return;
      }
      if (this.disposed) return;
      const gen = typeof page.generation === 'string' && page.generation ? page.generation : null;
      if (gen !== null && this.state.generation !== null && gen !== this.state.generation) {
        // A page of another file (a server that does not check the generation it is sent).
        if (resets >= MAX_RESETS_PER_PASS) {
          this.publish({ error: 'events.jsonl が繰り返し置き換えられたため読み込みを中断しました', lastFetch: Date.now(), loading: false });
          return;
        }
        resets += 1;
        this.reset(gen);
        continue;
      }
      const add: TraceEvent[] = [];
      for (const ev of page.events ?? []) {
        if (!ev || typeof ev.event_id !== 'string' || this.seen.has(ev.event_id)) continue;
        this.seen.add(ev.event_id);
        add.push(ev);
      }
      fresh += add.length;
      const next = typeof page.next_offset === 'number' ? Math.max(from, page.next_offset) : from;
      const done = !!page.eof || next <= from;
      this.publish({
        events: add.length ? this.state.events.concat(add) : this.state.events,
        nextOffset: next,
        eof: !!page.eof,
        tornTail: !!page.torn_tail,
        corruptLines: this.state.corruptLines + (page.corrupt_lines ?? 0),
        generation: gen ?? this.state.generation,
        error: null,
        lastFetch: Date.now(),
        pages: this.state.pages + 1,
        loading: done ? false : this.state.loading,
      });
      if (done) break;
    }
    this.publish({
      passes: this.state.passes + 1,
      quietPasses: fresh === 0 ? this.state.quietPasses + 1 : 0,
      loading: false,
    });
  }
}
