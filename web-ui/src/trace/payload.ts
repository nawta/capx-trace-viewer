/**
 * Reading the stored request payload (docs/trace-format.md 6.2-6.4).
 *
 * `payloads/<event_id>.request.json` is the parsed request body with every
 * `data:` URL replaced by `{"$asset", "sha256", "mime", "bytes"}`. This module
 * splits it into messages and parts and gives each part the JSON path the
 * recorder uses in `images[].path` and `input_summary` (for example
 * `messages[1].content[0].text`), so segment offsets can be drawn on the text.
 * Chat bodies (`messages`) and the Responses-API form used for codex models
 * (`input`, with `input_text` and `input_image` parts) are both handled.
 *
 * Related: components/tabs/ModelInputTab.tsx, components/tabs/InputDiffTab.tsx.
 */

import type { ImageInfo } from './types';

export interface AssetPlaceholder {
  $asset: string | null;
  sha256?: string | null;
  mime?: string | null;
  bytes?: number | null;
}

export interface MediaRef {
  /** Asset path relative to the episode dir, when the image was a data: URL and was written. */
  asset: string | null;
  /** Why the asset is missing (placeholder `written: false`). */
  reason: string | null;
  sha256: string | null;
  mime: string | null;
  bytes: number | null;
  /** External http(s) URL (kept, never fetched by the recorder or the viewer). */
  externalUrl: string | null;
  isVideo: boolean;
}

export interface PayloadPart {
  kind: 'text' | 'media' | 'other';
  /** JSON path of the text, or of the media URL. */
  path: string;
  partType: string;
  text: string;
  media: MediaRef | null;
  raw: unknown;
}

export interface PayloadMessage {
  index: number;
  role: string;
  path: string;
  parts: PayloadPart[];
}

export interface ParsedPayload {
  container: 'messages' | 'input' | 'none';
  messages: PayloadMessage[];
  /** Top-level keys other than the message list. */
  params: Record<string, unknown>;
}

export function isAssetPlaceholder(v: unknown): v is AssetPlaceholder {
  return !!v && typeof v === 'object' && !Array.isArray(v) && '$asset' in (v as object);
}

function mediaFrom(url: unknown): MediaRef | null {
  if (isAssetPlaceholder(url)) {
    const mime = url.mime ?? null;
    const p = url as AssetPlaceholder & { written?: unknown; reason?: unknown };
    const written = p.written !== false && typeof url.$asset === 'string';
    return {
      asset: written ? url.$asset : null,
      reason: written ? null : typeof p.reason === 'string' ? p.reason : '画像ファイルが保存されていません',
      sha256: url.sha256 ?? null,
      mime,
      bytes: typeof url.bytes === 'number' ? url.bytes : null,
      externalUrl: null,
      isVideo: !!mime && mime.startsWith('video/'),
    };
  }
  if (typeof url === 'string') {
    if (url.startsWith('data:')) {
      // A data: URL that the recorder did not move to an asset (should not happen).
      const mime = url.slice(5, url.indexOf(';') > 0 ? url.indexOf(';') : 5) || null;
      return { asset: null, reason: 'data: URL が assets に移されていません', sha256: null, mime, bytes: null, externalUrl: null, isVideo: !!mime && mime.startsWith('video/') };
    }
    return { asset: null, reason: null, sha256: null, mime: null, bytes: null, externalUrl: url, isVideo: false };
  }
  return null;
}

function parsePart(part: unknown, path: string): PayloadPart {
  if (typeof part === 'string') {
    return { kind: 'text', path, partType: 'string', text: part, media: null, raw: part };
  }
  if (!part || typeof part !== 'object') {
    return { kind: 'other', path, partType: typeof part, text: '', media: null, raw: part };
  }
  const p = part as Record<string, unknown>;
  const type = typeof p.type === 'string' ? p.type : '';
  if ((type === 'text' || type === 'input_text' || type === 'output_text') && typeof p.text === 'string') {
    return { kind: 'text', path: `${path}.text`, partType: type, text: p.text, media: null, raw: part };
  }
  if (type === 'image_url' || type === 'video_url') {
    const holder = p[type];
    const url = holder && typeof holder === 'object' ? (holder as Record<string, unknown>).url : holder;
    const media = mediaFrom(url);
    const urlPath = holder && typeof holder === 'object' ? `${path}.${type}.url` : `${path}.${type}`;
    if (media) {
      if (type === 'video_url') media.isVideo = true;
      return { kind: 'media', path: urlPath, partType: type, text: '', media, raw: part };
    }
  }
  if (type === 'input_image') {
    const media = mediaFrom(p.image_url);
    if (media) return { kind: 'media', path: `${path}.image_url`, partType: type, text: '', media, raw: part };
  }
  return { kind: 'other', path, partType: type || 'object', text: '', media: null, raw: part };
}

/** Splits a stored request body into messages and parts. */
export function parsePayload(body: unknown): ParsedPayload {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return { container: 'none', messages: [], params: {} };
  }
  const obj = body as Record<string, unknown>;
  const key = Array.isArray(obj.messages) ? 'messages' : Array.isArray(obj.input) ? 'input' : null;
  const params: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) if (k !== key) params[k] = v;
  if (!key) return { container: 'none', messages: [], params };
  const list = obj[key] as unknown[];
  const messages: PayloadMessage[] = list.map((m, i) => {
    const mpath = `${key}[${i}]`;
    if (!m || typeof m !== 'object') {
      return { index: i, role: '?', path: mpath, parts: [parsePart(m, mpath)] };
    }
    const mo = m as Record<string, unknown>;
    const role = typeof mo.role === 'string' ? mo.role : typeof mo.type === 'string' ? mo.type : '?';
    const content = mo.content;
    let parts: PayloadPart[];
    if (typeof content === 'string') {
      parts = [{ kind: 'text', path: `${mpath}.content`, partType: 'string', text: content, media: null, raw: content }];
    } else if (Array.isArray(content)) {
      parts = content.map((c, j) => parsePart(c, `${mpath}.content[${j}]`));
    } else if (content === undefined || content === null) {
      parts = [];
    } else {
      parts = [parsePart(content, `${mpath}.content`)];
    }
    return { index: i, role, path: mpath, parts };
  });
  return { container: key, messages, params };
}

/** Images metadata keyed by JSON path (model_request.images). */
export function imagesByPath(images: ImageInfo[] | null | undefined): Map<string, ImageInfo> {
  const m = new Map<string, ImageInfo>();
  for (const im of images ?? []) if (im.path) m.set(im.path, im);
  return m;
}

/** All text of a message, parts joined with blank lines (used by the diff). */
export function messageText(m: PayloadMessage): string {
  return m.parts
    .map((p) => (p.kind === 'text' ? p.text : p.kind === 'media' ? `[${p.media?.isVideo ? 'video' : 'image'} ${p.media?.sha256 ?? p.media?.externalUrl ?? ''}]` : `[${p.partType}]`))
    .join('\n\n');
}

/** Media hashes of a message in order (used by the diff). */
export function messageMedia(m: PayloadMessage): MediaRef[] {
  return m.parts.filter((p) => p.kind === 'media' && p.media).map((p) => p.media as MediaRef);
}
