/**
 * Offset conversion between Python string offsets (code points), which the
 * recorder writes in input_summary and stdout/stderr slices
 * (docs/trace-format.md 5.2, 6.4), and JavaScript string indices (UTF-16 code
 * units). They differ only when the text holds characters outside the BMP.
 *
 * Related: laterInput.ts, components/TextView.tsx.
 */

const SURROGATE = /[\uD800-\uDFFF]/;

/** Converts a UTF-16 index of `text` to a code-point index. */
export function utf16ToCodePoint(text: string, index: number): number {
  if (!SURROGATE.test(text)) return Math.min(index, text.length);
  let cp = 0;
  for (let i = 0; i < index && i < text.length; i += 1) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
      const n = text.charCodeAt(i + 1);
      if (n >= 0xdc00 && n <= 0xdfff) i += 1;
    }
    cp += 1;
  }
  return cp;
}

/** Converts a code-point index to a UTF-16 index of `text` (clamped to the text). */
export function codePointToUtf16(text: string, cp: number): number {
  if (!SURROGATE.test(text)) return Math.max(0, Math.min(cp, text.length));
  let i = 0;
  let n = 0;
  while (i < text.length && n < cp) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
      const d = text.charCodeAt(i + 1);
      i += d >= 0xdc00 && d <= 0xdfff ? 2 : 1;
    } else {
      i += 1;
    }
    n += 1;
  }
  return i;
}

export function codePointLength(text: string): number {
  return utf16ToCodePoint(text, text.length);
}
