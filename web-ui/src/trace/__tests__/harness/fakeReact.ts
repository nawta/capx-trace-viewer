/**
 * A tiny stand-in for React used by the page tests (`*.page.test.ts`), so the
 * viewer page's own hooks and hooks.ts run under Node without a DOM library.
 * web-ui/scripts/test-trace.mjs bundles those tests with
 * `react` and `react/jsx-runtime` resolved to this file and fakeJsxRuntime.ts.
 *
 * It renders ONE function component: `mount(Page)` calls Page() and keeps what
 * it returns as a plain element tree (child components are never rendered; a
 * test reads their props, for example TopBar's `runnerRecords`). After each
 * render the effects whose deps changed run: every cleanup first, then every
 * body, in declaration order, as React does. setState schedules one re-render
 * in a microtask (updates are batched).
 *
 * Related: fakeJsxRuntime.ts, browser.ts (virtual clock and fake server).
 */

type Cleanup = (() => void) | void;

interface StateSlot {
  kind: 'state';
  value: unknown;
  set: (v: unknown) => void;
}
interface RefSlot {
  kind: 'ref';
  ref: { current: unknown };
}
interface MemoSlot {
  kind: 'memo';
  deps: readonly unknown[] | undefined;
  value: unknown;
}
interface EffectSlot {
  kind: 'effect';
  deps: readonly unknown[] | undefined;
  cleanup: (() => void) | undefined;
}
type Slot = StateSlot | RefSlot | MemoSlot | EffectSlot;

interface PendingEffect {
  index: number;
  fn: () => Cleanup;
  deps: readonly unknown[] | undefined;
}

export interface Mounted<P> {
  /** What the component returned from its last render. */
  output: unknown;
  renders: number;
  props: P;
  /** Renders again with new props (like a parent re-render). */
  update: (props: P) => void;
  unmount: () => void;
}

interface Instance {
  slots: Slot[];
  index: number;
  pending: PendingEffect[];
  scheduled: boolean;
  unmounted: boolean;
  schedule: () => void;
}

let current: Instance | null = null;

function inst(): Instance {
  if (!current) throw new Error('fake react: a hook was called outside a render');
  return current;
}

function changed(prev: readonly unknown[] | undefined, next: readonly unknown[] | undefined): boolean {
  if (!prev || !next) return true;
  if (prev.length !== next.length) return true;
  for (let i = 0; i < prev.length; i += 1) if (!Object.is(prev[i], next[i])) return true;
  return false;
}

export function useState<T>(init: T | (() => T)): [T, (v: T | ((prev: T) => T)) => void] {
  const it = inst();
  const i = it.index++;
  if (it.slots.length <= i) {
    const slot: StateSlot = {
      kind: 'state',
      value: typeof init === 'function' ? (init as () => T)() : init,
      set: (v: unknown) => {
        if (it.unmounted) return;
        const next = typeof v === 'function' ? (v as (p: unknown) => unknown)(slot.value) : v;
        if (Object.is(next, slot.value)) return;
        slot.value = next;
        it.schedule();
      },
    };
    it.slots.push(slot);
  }
  const s = it.slots[i] as StateSlot;
  return [s.value as T, s.set as (v: T | ((prev: T) => T)) => void];
}

export function useRef<T>(init: T): { current: T } {
  const it = inst();
  const i = it.index++;
  if (it.slots.length <= i) it.slots.push({ kind: 'ref', ref: { current: init } });
  return (it.slots[i] as RefSlot).ref as { current: T };
}

export function useMemo<T>(fn: () => T, deps: readonly unknown[] | undefined): T {
  const it = inst();
  const i = it.index++;
  if (it.slots.length <= i) {
    const slot: MemoSlot = { kind: 'memo', deps, value: fn() };
    it.slots.push(slot);
    return slot.value as T;
  }
  const s = it.slots[i] as MemoSlot;
  if (changed(s.deps, deps)) {
    s.deps = deps;
    s.value = fn();
  }
  return s.value as T;
}

export function useCallback<T>(fn: T, deps: readonly unknown[] | undefined): T {
  return useMemo(() => fn, deps);
}

export function useEffect(fn: () => Cleanup, deps?: readonly unknown[]): void {
  const it = inst();
  const i = it.index++;
  if (it.slots.length <= i) {
    it.slots.push({ kind: 'effect', deps: undefined, cleanup: undefined });
    it.pending.push({ index: i, fn, deps });
    return;
  }
  const s = it.slots[i] as EffectSlot;
  if (deps === undefined || changed(s.deps, deps)) it.pending.push({ index: i, fn, deps });
}

export const useLayoutEffect = useEffect;

/** Class components are only defined (ErrorBoundary), never rendered here. */
export class Component<P = unknown, S = unknown> {
  props: P;
  state: S;
  constructor(props: P) {
    this.props = props;
    this.state = {} as S;
  }
  setState(): void {}
  forceUpdate(): void {}
}

export const Fragment = Symbol.for('fake-react.fragment');

export function createElement(type: unknown, props: Record<string, unknown> | null, ...children: unknown[]): unknown {
  return { type, props: { ...(props ?? {}), children: children.length <= 1 ? children[0] : children }, key: null };
}

/** Renders `Comp` once, then again on every state change, until unmounted. */
export function mount<P>(Comp: (props: P) => unknown, props: P): Mounted<P> {
  const it: Instance = {
    slots: [],
    index: 0,
    pending: [],
    scheduled: false,
    unmounted: false,
    schedule: () => {},
  };
  const handle: Mounted<P> = {
    output: null,
    renders: 0,
    props,
    update: (p: P) => {
      handle.props = p;
      render();
    },
    unmount: () => {
      it.unmounted = true;
      for (const s of it.slots) if (s.kind === 'effect' && s.cleanup) s.cleanup();
    },
  };
  function render(): void {
    if (it.unmounted) return;
    current = it;
    it.index = 0;
    it.pending = [];
    try {
      handle.output = Comp(handle.props);
    } finally {
      current = null;
    }
    handle.renders += 1;
    const pending = it.pending;
    it.pending = [];
    for (const p of pending) {
      const s = it.slots[p.index] as EffectSlot;
      if (s.cleanup) {
        const c = s.cleanup;
        s.cleanup = undefined;
        c();
      }
    }
    for (const p of pending) {
      const s = it.slots[p.index] as EffectSlot;
      s.deps = p.deps;
      const c = p.fn();
      s.cleanup = typeof c === 'function' ? c : undefined;
    }
  }
  it.schedule = () => {
    if (it.unmounted || it.scheduled) return;
    it.scheduled = true;
    queueMicrotask(() => {
      it.scheduled = false;
      render();
    });
  };
  render();
  return handle;
}

export default { useState, useRef, useMemo, useCallback, useEffect, useLayoutEffect, Component, Fragment, createElement };
