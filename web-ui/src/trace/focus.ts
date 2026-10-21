/**
 * Which marked range the model-input tab scrolls to (docs/trace-format.md 9.3,
 * tab 1): a range link from another event's later-input report sets the
 * page-level focus; a position link in 「入力の内訳」 sets the tab's own focus.
 * The newer one wins, so a position link still works after arriving from a
 * range link, and a new range link wins over an older position click.
 *
 * Related: components/tabs/ModelInputTab.tsx, TraceViewerPage.tsx.
 */

let lastToken = 0;

/** A strictly increasing token (two clicks in the same millisecond still differ). */
export function nextFocusToken(): number {
  lastToken = Math.max(lastToken + 1, Date.now());
  return lastToken;
}

export interface FocusChoice {
  /** Id of the range to reveal: 'focus' for the page-level range, else a segment id. */
  id: string | null;
  token: number;
  source: 'page' | 'local' | null;
}

export function pickFocus(page: { token: number } | null, local: { id: string; token: number } | null): FocusChoice {
  if (local && (!page || local.token > page.token)) return { id: local.id, token: local.token, source: 'local' };
  if (page) return { id: 'focus', token: page.token, source: 'page' };
  return { id: null, token: 0, source: null };
}
