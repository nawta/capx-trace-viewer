/**
 * Syntax highlighting for the trace viewer (docs/trace-format.md 9.3: code
 * cards, the code with the API call line marked, stored payload JSON).
 *
 * Uses the same highlight.js core instance and colour theme (index.css) as the
 * CaP-X web UI code viewer. That component registers only python;
 * stored payloads need json as well, so both are registered here (registering
 * a language twice is harmless). Highlighted HTML is split per line so a
 * single line (an API call site) can be marked.
 *
 * Related: components/CodeView.tsx, components/common.tsx.
 */

import hljs from 'highlight.js/lib/core';
import json from 'highlight.js/lib/languages/json';
import python from 'highlight.js/lib/languages/python';

if (!hljs.getLanguage('python')) hljs.registerLanguage('python', python);
if (!hljs.getLanguage('json')) hljs.registerLanguage('json', json);

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function highlight(code: string, language: 'python' | 'json'): string {
  try {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value;
  } catch {
    return escapeHtml(code);
  }
}

/**
 * Splits highlighted HTML into lines. Spans that cross a newline are closed at
 * the end of the line and reopened on the next one, so every line is valid HTML.
 */
export function splitHighlightedLines(html: string): string[] {
  const lines: string[] = [];
  const open: string[] = [];
  let cur = '';
  let i = 0;
  while (i < html.length) {
    const ch = html[i];
    if (ch === '<') {
      const close = html.indexOf('>', i);
      if (close < 0) {
        cur += html.slice(i);
        break;
      }
      const tag = html.slice(i, close + 1);
      if (tag.startsWith('</')) open.pop();
      else if (!tag.endsWith('/>')) open.push(tag);
      cur += tag;
      i = close + 1;
    } else if (ch === '\n') {
      cur += '</span>'.repeat(open.length);
      lines.push(cur);
      cur = open.join('');
      i += 1;
    } else {
      cur += ch;
      i += 1;
    }
  }
  lines.push(cur);
  return lines;
}
