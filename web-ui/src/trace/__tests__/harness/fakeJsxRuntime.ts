/**
 * `react/jsx-runtime` for the page tests: JSX becomes plain {type, props, key}
 * objects that a test can search (see fakeReact.ts and browser.ts findAll).
 */

import { Fragment } from './fakeReact';

export { Fragment };

export function jsx(type: unknown, props: Record<string, unknown> | null, key?: unknown): unknown {
  return { type, props: props ?? {}, key: key ?? null };
}

export const jsxs = jsx;
export const jsxDEV = jsx;
