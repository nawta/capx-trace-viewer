/**
 * Renders the argument and return summaries of docs/trace-format.md 5.3, in
 * the encoding of the trace recorder summarize():
 * - scalars inline (non-finite floats as "nan"/"inf"/"-inf"); strings inline
 *   up to 4000 characters, else {truncated, preview, length, asset|path};
 * - containers {type: dict|list|tuple|<mapping>, length, items, truncated?};
 *   namedtuples and dataclasses {type, fields, dataclass?}; depth-limited
 *   nodes {type, truncated: "depth"}; any other object {type} only;
 * - numpy arrays {type: "ndarray", shape, dtype} plus inline `values`
 *   (size <= 64), or a saved npy (`path`, `stats` over finite values) whose
 *   preview is loaded on demand through /array (9.2), or an `image` sub-node
 *   for image-like arrays a service received; torch tensors {shape, dtype,
 *   device}; PIL images {size, mode, image: {path}}.
 * Units and frames come from the recorder's table
 * ({args: {name: {unit, frame}}, returns: {field: {unit, frame}}}); where it
 * says nothing the viewer shows 「単位・座標系: 不明」.
 *
 * Related: tabs/ApiLogTab.tsx, tabs/DebugTab.tsx, Timeline.tsx (one-line
 * summaries), api.ts.
 */

import { useState, type ReactNode } from 'react';
import { episodeArrayImageUrl, episodeFileUrl, errorText, traceApi, type EpisodeRef } from '../api';
import { fmtNum } from '../format';
import { LABELS } from '../labels';
import type { ArrayImageLayout, ArrayInfo, ArrayStats } from '../types';
import { Badge, ErrorText, Spinner } from './common';
import { TraceImage } from './TraceImage';

/** An open episode (api.ts): run, episode and the events.jsonl generation the files belong to. */
export type { EpisodeRef } from '../api';

type Obj = Record<string, unknown>;

function isObj(v: unknown): v is Obj {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}

function typeOf(v: Obj): string | null {
  return typeof v.type === 'string' ? v.type : null;
}

export function isArraySummary(v: unknown): boolean {
  if (!isObj(v)) return false;
  const t = typeOf(v);
  return t === 'ndarray' || t === 'torch.Tensor' || (Array.isArray(v.shape) && typeof v.dtype === 'string' && t !== 'PIL.Image');
}

function isTextNode(v: Obj): boolean {
  return v.truncated === true && (typeof v.preview === 'string' || typeof v.text === 'string') && !typeOf(v);
}

function isContainer(v: Obj): boolean {
  return 'items' in v && (Array.isArray(v.items) || isObj(v.items));
}

function isFieldsNode(v: Obj): boolean {
  return isObj(v.fields);
}

function filePath(v: Obj): string | null {
  for (const k of ['path', 'asset']) if (typeof v[k] === 'string') return v[k] as string;
  return null;
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

interface UnitSpec {
  unit?: string | null;
  frame?: string | null;
}

function formatUnit(u: unknown): string {
  if (typeof u === 'string') return `単位・座標系: ${u}`;
  if (!isObj(u)) return LABELS.unitsUnknown;
  const s = u as UnitSpec;
  const unit = typeof s.unit === 'string' ? s.unit : null;
  const frame = typeof s.frame === 'string' ? s.frame : null;
  if (!unit && !frame) return LABELS.unitsUnknown;
  return `単位: ${unit ?? '不明'} / 座標系: ${frame ?? '不明'}`;
}

function unitsArgs(units: unknown): Obj | null {
  return isObj(units) && isObj(units.args) ? units.args : null;
}

function unitsReturns(units: unknown): Obj | null {
  if (!isObj(units)) return null;
  const r = units.returns ?? units.return;
  return isObj(r) ? r : null;
}

function isNumericLike(v: unknown): boolean {
  if (typeof v === 'number') return true;
  if (typeof v === 'string') return v === 'nan' || v === 'inf' || v === '-inf';
  if (isArraySummary(v)) return true;
  if (Array.isArray(v)) return v.length > 0 && v.every((x) => typeof x === 'number' || isNumericLike(x));
  if (isObj(v) && isContainer(v) && Array.isArray(v.items)) return v.items.length > 0 && v.items.every((x) => isNumericLike(x));
  return false;
}

/** Units of one argument; shown only for numeric values. */
export function ArgUnits({ units, name, value }: { units: unknown; name: string; value: unknown }) {
  if (!isNumericLike(value)) return null;
  const u = unitsArgs(units)?.[name];
  return <div className="text-[11px] text-text-tertiary mt-0.5">{u !== undefined ? formatUnit(u) : LABELS.unitsUnknown}</div>;
}

/** Units of the return value: per field for a returned dict, else for the whole value. */
export function ReturnUnits({ units, value }: { units: unknown; value: unknown }) {
  const table = unitsReturns(units);
  if (isObj(value) && isContainer(value) && isObj(value.items)) {
    const rows = Object.entries(value.items).filter(([, v]) => isNumericLike(v));
    if (rows.length === 0) return null;
    return (
      <div className="mt-1 text-[11px] text-text-tertiary space-y-0.5">
        {rows.map(([k]) => (
          <div key={k}>
            <span className="font-mono text-sky-300/80">{k}</span>: {table && table[k] !== undefined ? formatUnit(table[k]) : LABELS.unitsUnknown}
          </div>
        ))}
      </div>
    );
  }
  if (!isNumericLike(value)) return null;
  const entries = table ? Object.entries(table) : [];
  if (entries.length === 1) {
    return (
      <div className="text-[11px] text-text-tertiary mt-0.5">
        {entries[0][0]}: {formatUnit(entries[0][1])}
      </div>
    );
  }
  return <div className="text-[11px] text-text-tertiary mt-0.5">{LABELS.unitsUnknown}</div>;
}

// ---------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------

function formatNested(v: unknown, depth = 0): string {
  if (Array.isArray(v)) {
    if (depth > 3) return '[…]';
    return `[${v.map((x) => formatNested(x, depth + 1)).join(', ')}]`;
  }
  return typeof v === 'number' ? fmtNum(v) : JSON.stringify(v);
}

/**
 * Where an array summary's values are stored: a saved npy (`path`), or one member of the npz
 * that holds a service call's decoded outputs (`npz` + `npz_key`, 5.2; masks are stored
 * bit-packed as <key>__packbits). `key` names that array within the episode and the
 * events.jsonl generation; a preview loaded for one key is never shown under another.
 */
export function arraySource(ep: EpisodeRef | null, value: Obj): { path: string | null; npzKey: string | undefined; key: string | null } {
  const npyPath = filePath(value);
  const npzPath = !npyPath && typeof value.npz === 'string' ? value.npz : null;
  const npzKey = npzPath && typeof value.npz_key === 'string' ? value.npz_key : undefined;
  const path = npyPath ?? npzPath;
  return { path, npzKey, key: ep && path ? `${ep.runId}|${ep.episodeId}|${ep.generation ?? ''}|${path}|${npzKey ?? ''}` : null };
}

/**
 * The hash kept for an array whose values were not saved (the trace recorder
 * bounded_hash): `sha256` of the whole array (at most 1 MiB), else `sha256_prefix` over the
 * first `sha256_prefix_bytes` bytes in C order, else `sha256_skipped` says why there is none.
 */
export function unsavedArrayHashText(value: Obj): string {
  if (typeof value.sha256 === 'string' && value.sha256) return `sha256 ${value.sha256.slice(0, 16)}`;
  if (typeof value.sha256_prefix === 'string' && value.sha256_prefix) {
    const n = typeof value.sha256_prefix_bytes === 'number' ? value.sha256_prefix_bytes : null;
    return `sha256（先頭 ${n !== null ? `${n} バイト` : '一部'}のみ） ${value.sha256_prefix.slice(0, 16)}`;
  }
  if (value.sha256_skipped === 'hash_budget') return 'sha256 なし（1 回の要約でハッシュする量の上限を超えたため計算せず）';
  if (typeof value.sha256_skipped === 'string') return `sha256 なし（${value.sha256_skipped}）`;
  return 'sha256 —';
}

/** A value tagged with the key it was loaded for, read back only under that same key. */
export function valueFor<T>(tagged: { key: string; value: T } | null, key: string | null): T | null {
  return tagged !== null && key !== null && tagged.key === key ? tagged.value : null;
}

export function ArrayView({ value, ep }: { value: Obj; ep: EpisodeRef | null }) {
  const shape = Array.isArray(value.shape) ? (value.shape as unknown[]) : null;
  const dtype = typeof value.dtype === 'string' ? value.dtype : null;
  const t = typeOf(value) ?? 'ndarray';
  const stats = isObj(value.stats) ? value.stats : null;
  const { path, npzKey, key: arrayKey } = arraySource(ep, value);
  const image = isObj(value.image) ? value.image : null;
  const inline = value.values;
  const device = typeof value.device === 'string' ? value.device : null;
  // Everything loaded is tagged with the array it was loaded for: when React reuses this
  // component for another value (another event selected), the old preview is never shown.
  const [loadedState, setLoaded] = useState<{ key: string; value: Record<string, ArrayInfo> } | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const [errorState, setError] = useState<{ key: string; value: string } | null>(null);
  const loaded = valueFor(loadedState, arrayKey);
  const loading = loadingKey !== null && loadingKey === arrayKey;
  const error = valueFor(errorState, arrayKey);
  const load = async () => {
    if (!ep || !path || !arrayKey) return;
    const key = arrayKey;
    setLoadingKey(key);
    setError(null);
    try {
      const res = await traceApi.array(ep.runId, ep.episodeId, path, npzKey);
      setLoaded({ key, value: res.arrays ?? {} });
    } catch (e) {
      setError({ key, value: errorText(e) });
    } finally {
      setLoadingKey((k) => (k === key ? null : k));
    }
  };
  const statText = (k: string) => (stats && stats[k] !== undefined ? <span key={k}>{k}={stats[k] === null ? 'null' : fmtNum(stats[k])}</span> : null);
  return (
    <div className="text-xs font-mono">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Badge tone="violet">{t}</Badge>
        <span>shape=({shape ? shape.join(', ') : '?'})</span>
        <span>dtype={dtype ?? '?'}</span>
        {device && <span>device={device}</span>}
        {statText('min')}
        {statText('max')}
        {statText('mean')}
        {stats && typeof stats.n_nonfinite === 'number' && stats.n_nonfinite > 0 && <span className="text-amber-300">非有限 {String(stats.n_nonfinite)}</span>}
      </div>
      {Array.isArray(inline) || typeof inline === 'number' ? <div className="mt-1 text-text-secondary break-all">{formatNested(inline)}</div> : null}
      {typeof value.values_omitted === 'string' && <div className="mt-1 text-text-tertiary">値は記録していません（{value.values_omitted}）</div>}
      {typeof value.not_saved === 'string' && (
        <div className="mt-1 text-amber-300">
          {LABELS.notCaptured}（{value.not_saved}
          {value.not_saved === 'array_budget' ? ': 配列の保存上限を超えたため値は保存していません' : ''}）· {unsavedArrayHashText(value)}
        </div>
      )}
      {image && (
        <div className="mt-1">
          {typeof image.path === 'string' && image.written !== false && ep ? (
            <TraceImage src={episodeFileUrl(ep, image.path)} alt={image.path} provenance="perception" className="h-28 w-40" />
          ) : (
            <span className="text-amber-300">画像: {LABELS.notCaptured}{typeof image.reason === 'string' ? `（${image.reason}）` : ''}</span>
          )}
        </div>
      )}
      {path && (
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <span className="text-text-tertiary">
            {path}
            {npzKey ? `:${npzKey}` : ''}
          </span>
          {value.written === false ? (
            <span className="text-amber-300">未保存{typeof value.reason === 'string' ? `（${value.reason}）` : ''}</span>
          ) : (
            ep &&
            !loaded && (
              <button type="button" onClick={load} className="text-accent hover:text-accent-light underline underline-offset-2">
                プレビューを読み込む
              </button>
            )
          )}
          {loading && <Spinner />}
        </div>
      )}
      {error && <ErrorText>{error}</ErrorText>}
      {loaded && ep && path && <ArrayPreviewList arrays={loaded} ep={ep} path={path} recordedStats={recordedNpzStats([value], path)} />}
    </div>
  );
}

const DRAWABLE_DTYPE = /^(bool|u?int(8|16|32|64)|float(16|32|64))$/;

/**
 * How `/array-image` draws this array: the server's `image` answer when it sent one (null
 * there: it cannot be drawn), else the same rules from shape and dtype for an older server
 * (capx_trace_viewer/trace_arrays.py image_layout). A 2-D array is one image. A 3-D array whose last
 * axis has size 1 is one H x W image, a mask when it is bool (the (H, W, 1) segmentation
 * layout of the simulators). A 3-D bool array with a larger last axis is
 * a stack of masks along axis 0; 3-D uint8 with 3 or 4 channels is RGB(A); any other 3-D
 * array is a stack along axis 0.
 */
export function arrayImageLayout(a: ArrayInfo): ArrayImageLayout | null {
  if ('image' in a && a.image !== undefined) {
    const im = a.image;
    if (!im || typeof im !== 'object') return null;
    const { kind, count, height, width } = im as Partial<ArrayImageLayout>;
    if (typeof kind !== 'string' || typeof count !== 'number' || typeof height !== 'number' || typeof width !== 'number') return null;
    return count > 0 && height > 0 && width > 0 ? { kind, count, height, width } : null;
  }
  const shape = a.shape ?? [];
  const dtype = a.dtype ?? '';
  if (!DRAWABLE_DTYPE.test(dtype)) return null;
  const isBool = dtype === 'bool';
  let layout: ArrayImageLayout | null = null;
  if (shape.length === 2) {
    layout = { kind: isBool ? 'mask' : 'gray', count: 1, height: shape[0], width: shape[1] };
  } else if (shape.length === 3) {
    if (shape[2] === 1) layout = { kind: isBool ? 'mask' : 'gray', count: 1, height: shape[0], width: shape[1] };
    else if (isBool) layout = { kind: 'mask_stack', count: shape[0], height: shape[1], width: shape[2] };
    else if (dtype === 'uint8' && (shape[2] === 3 || shape[2] === 4)) layout = { kind: 'rgb', count: 1, height: shape[0], width: shape[1] };
    else layout = { kind: 'stack', count: shape[0], height: shape[1], width: shape[2] };
  }
  return layout && layout.count > 0 && layout.height > 0 && layout.width > 0 ? layout : null;
}

/**
 * True when the array is worth drawing: masks always; other arrays only when both image
 * sides have at least 16 pixels (small numeric arrays such as boxes, poses and joint
 * vectors are not images).
 */
export function isDrawnAsImage(a: ArrayInfo, layout: ArrayImageLayout | null): boolean {
  if (!layout) return false;
  const mask = a.is_mask === true || layout.kind === 'mask' || layout.kind === 'mask_stack';
  return mask || (layout.height >= 16 && layout.width >= 16);
}

/** First arrays shown of a file with many (a SAM3 call stores up to 50 masks in one npz). */
const PREVIEW_LIST_LIMIT = 6;
const naturalOrder = new Intl.Collator(undefined, { numeric: true }).compare;

/**
 * Statistics the recorder saved for each array of an npz file, from the summary nodes that
 * point into it (`npz` = the file, `npz_key` = the member; a bit-packed mask is listed by
 * `/array` under the name without `__packbits`). The /array endpoint returns `stats: null`
 * (9.2: statistics are computed at record time), so listings show these instead.
 */
export function recordedNpzStats(values: unknown[], path: string): Record<string, ArrayStats> {
  const out: Record<string, ArrayStats> = {};
  let nodes = 0;
  const walk = (v: unknown, depth: number) => {
    if (depth > 12 || nodes > 20000 || v === null || typeof v !== 'object') return;
    nodes += 1;
    if (Array.isArray(v)) {
      for (const x of v) walk(x, depth + 1);
      return;
    }
    const o = v as Record<string, unknown>;
    if (o.npz === path && typeof o.npz_key === 'string' && isObj(o.stats)) {
      out[o.npz_key.replace(/__packbits$/, '')] = o.stats as ArrayStats;
    }
    for (const x of Object.values(o)) walk(x, depth + 1);
  };
  for (const v of values) walk(v, 0);
  return out;
}

export function ArrayPreviewList({
  arrays,
  ep,
  path,
  recordedStats,
}: {
  arrays: Record<string, ArrayInfo>;
  ep: EpisodeRef;
  path: string;
  /** Per-array statistics recorded in the events (recordedNpzStats), used when /array has none. */
  recordedStats?: Record<string, ArrayStats>;
}) {
  const names = Object.keys(arrays).sort(naturalOrder);
  const [all, setAll] = useState(false);
  if (names.length === 0) return <div className="mt-1 text-text-tertiary text-xs">配列がありません</div>;
  const shown = all ? names : names.slice(0, PREVIEW_LIST_LIMIT);
  // Several masks (a SAM3 call) are laid out side by side.
  const masks = names.length > 1 && names.every((n) => arrays[n].is_mask === true);
  return (
    <div className={`mt-2 ${masks ? 'grid grid-cols-[repeat(auto-fill,minmax(210px,1fr))] gap-2' : 'space-y-2'}`}>
      {shown.map((name) => (
        <ArrayPreviewItem key={`${path}|${name}`} name={name} a={arrays[name]} ep={ep} path={path} recorded={recordedStats?.[name] ?? null} />
      ))}
      {names.length > PREVIEW_LIST_LIMIT && (
        <button type="button" onClick={() => setAll(!all)} className="text-[11px] text-accent-light hover:text-accent text-left">
          {all ? `先頭 ${PREVIEW_LIST_LIMIT} 件だけ表示` : `残り ${names.length - PREVIEW_LIST_LIMIT} 件を表示（全 ${names.length} 件）`}
        </button>
      )}
    </div>
  );
}

function ArrayPreviewItem({ name, a, ep, path, recorded }: { name: string; a: ArrayInfo; ep: EpisodeRef; path: string; recorded: ArrayStats | null }) {
  const shape = a.shape ?? [];
  const stats = a.stats ?? recorded;
  // Masks, depth maps and images, laid out as the server draws them (a stack of masks or
  // images gets a stepper, so `index` reaches every one).
  const layout = arrayImageLayout(a);
  const imageable = isDrawnAsImage(a, layout);
  const stack = layout ? layout.count : 1;
  const [index, setIndex] = useState(0);
  const [showValues, setShowValues] = useState(!a.is_mask);
  return (
    <div className="border border-surface-border rounded p-2 bg-surface-sunken">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs font-mono">
        <span className="text-text-primary">{name}</span>
        <span>shape=({shape.join(', ')})</span>
        <span>dtype={a.dtype ?? '?'}</span>
        {stats &&
          ['min', 'max', 'mean'].map((k) =>
            stats[k] !== undefined ? (
              <span key={k}>
                {k}={stats[k] === null ? 'null' : fmtNum(stats[k])}
              </span>
            ) : null,
          )}
        {!a.stats && recorded && <span className="text-text-tertiary">（記録時の統計）</span>}
        {a.is_mask && <Badge tone="sky">mask</Badge>}
        {a.packed === true && <Badge tone="neutral">packbits</Badge>}
      </div>
      {imageable && (
        <div className="mt-2 flex items-end gap-3">
          <TraceImage
            src={episodeArrayImageUrl(ep, path, name, index)}
            alt={`${path}:${name}${stack > 1 ? `[${index}]` : ''}`}
            provenance="perception"
            className="h-32 w-48"
          />
          {stack > 1 && (
            <div className="flex items-center gap-1 text-[11px] text-text-tertiary">
              <button type="button" onClick={() => setIndex(Math.max(0, index - 1))} className="px-1.5 border border-surface-border rounded hover:text-accent">
                ◀
              </button>
              <span className="font-mono">
                {index + 1}/{stack}
              </span>
              <button type="button" onClick={() => setIndex(Math.min(stack - 1, index + 1))} className="px-1.5 border border-surface-border rounded hover:text-accent">
                ▶
              </button>
            </div>
          )}
        </div>
      )}
      {typeof a.preview_omitted === 'string' && (
        <div className="mt-1 text-[11px] text-text-tertiary">
          値のプレビューは省略しました（
          {a.preview_omitted === 'listing_budget' ? 'ファイル内の全配列を一覧したため、読み込み量の上限に達しました' : a.preview_omitted}）
        </div>
      )}
      {a.preview !== undefined && a.preview !== null && (
        <div className="mt-1">
          <button type="button" onClick={() => setShowValues(!showValues)} className="text-[11px] text-accent-light hover:text-accent">
            {showValues ? '値のプレビューを隠す' : '値のプレビューを表示'}
            {a.truncated ? '（先頭のみ）' : ''}
          </button>
          {showValues && (
            <pre className="mt-1 text-[11px] text-text-secondary whitespace-pre-wrap break-all max-h-40 overflow-auto">
              {typeof a.preview === 'string' ? a.preview : formatNested(a.preview)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic value
// ---------------------------------------------------------------------------

function TextNodeView({ value, ep }: { value: Obj; ep: EpisodeRef | null }) {
  const text = typeof value.preview === 'string' ? value.preview : typeof value.text === 'string' ? value.text : '';
  const path = filePath(value);
  const length = typeof value.length === 'number' ? value.length : null;
  return (
    <div className="text-xs font-mono">
      <span className="text-emerald-300 whitespace-pre-wrap break-all">{JSON.stringify(text)}</span>
      <span className="ml-2 text-amber-300">…（全 {length ?? '?'} 文字のうち先頭のみ）</span>
      {path && ep && value.written !== false && (
        <a className="ml-2 text-accent underline" href={episodeFileUrl(ep, path)} target="_blank" rel="noreferrer">
          全文（txt.gz）
        </a>
      )}
    </div>
  );
}

const MAX_RENDER_NODES = 800;

export function ValueView({ value, ep }: { value: unknown; ep: EpisodeRef | null }) {
  const counter = { n: 0 };
  return <ValueInner value={value} ep={ep} counter={counter} />;
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="flex gap-2 items-start">
      <span className="text-sky-300 font-mono text-xs flex-shrink-0">{k}:</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function ValueInner({ value, ep, counter }: { value: unknown; ep: EpisodeRef | null; counter: { n: number } }): ReactNode {
  counter.n += 1;
  if (counter.n > MAX_RENDER_NODES) return <span className="text-text-muted">…</span>;
  if (value === null || value === undefined) return <span className="text-text-tertiary font-mono text-xs">{value === null ? 'None' : '—'}</span>;
  if (typeof value === 'boolean') return <span className="text-orange-300 font-mono text-xs">{value ? 'True' : 'False'}</span>;
  if (typeof value === 'number') return <span className="text-orange-300 font-mono text-xs">{fmtNum(value, 6)}</span>;
  if (typeof value === 'string') {
    if (value === '***masked***') return <Badge tone="amber">***masked***</Badge>;
    return <span className="text-emerald-300 font-mono text-xs whitespace-pre-wrap break-all">{JSON.stringify(value)}</span>;
  }
  if (Array.isArray(value)) {
    if (value.every((x) => typeof x === 'number' || x === null)) {
      return <span className="font-mono text-xs text-orange-300 break-all">{formatNested(value)}</span>;
    }
    return (
      <div className="pl-3 border-l border-surface-border space-y-0.5">
        {value.map((x, i) => (
          <Row key={i} k={String(i)}>
            <ValueInner value={x} ep={ep} counter={counter} />
          </Row>
        ))}
      </div>
    );
  }
  if (!isObj(value)) return <span className="font-mono text-xs">{String(value)}</span>;

  if (isTextNode(value)) return <TextNodeView value={value} ep={ep} />;
  if (isArraySummary(value)) return <ArrayView value={value} ep={ep} />;
  const t = typeOf(value);
  if (t === 'PIL.Image') {
    const img = isObj(value.image) ? value.image : null;
    return (
      <div className="text-xs font-mono">
        <Badge tone="violet">PIL.Image</Badge> size={Array.isArray(value.size) ? (value.size as unknown[]).join('×') : '?'} mode={String(value.mode ?? '?')}
        {img && typeof img.path === 'string' && img.written !== false && ep ? (
          <TraceImage src={episodeFileUrl(ep, img.path)} alt={img.path} provenance="perception" className="mt-1 h-28 w-40" />
        ) : (
          <span className="ml-2 text-text-tertiary">
            {value.image_not_loaded ? '（画像は未ロードのため保存していません）' : value.image_not_saved ? `（${String(value.image_not_saved)}）` : ''}
          </span>
        )}
      </div>
    );
  }
  if (isContainer(value)) {
    const length = typeof value.length === 'number' ? value.length : null;
    const trunc = typeof value.truncated === 'number' ? value.truncated : null;
    const items = value.items;
    const header = (
      <span className="text-[11px] text-text-tertiary font-mono">
        {t ?? 'container'}
        {length !== null ? `（${length} 件${trunc ? `、うち ${trunc} 件は省略` : ''}）` : ''}
      </span>
    );
    if (Array.isArray(items)) {
      if (items.length > 0 && items.every((x) => typeof x === 'number' || x === null || typeof x === 'string')) {
        return (
          <span className="font-mono text-xs text-orange-300 break-all">
            {t === 'tuple' ? `(${items.map((x) => formatNested(x)).join(', ')})` : formatNested(items)}
            {trunc ? <span className="text-amber-300"> …+{trunc}</span> : null}
          </span>
        );
      }
      return (
        <div>
          {header}
          <div className="pl-3 border-l border-surface-border space-y-0.5">
            {items.map((x, i) => (
              <Row key={i} k={String(i)}>
                <ValueInner value={x} ep={ep} counter={counter} />
              </Row>
            ))}
          </div>
        </div>
      );
    }
    return (
      <div>
        {header}
        <div className="pl-3 border-l border-surface-border space-y-0.5">
          {Object.entries(items as Obj).map(([k, v]) => (
            <Row key={k} k={k}>
              <ValueInner value={v} ep={ep} counter={counter} />
            </Row>
          ))}
        </div>
      </div>
    );
  }
  if (isFieldsNode(value)) {
    return (
      <div>
        <span className="text-[11px] text-text-tertiary font-mono">
          {t ?? 'object'}
          {value.dataclass ? '（dataclass）' : ''}
        </span>
        <div className="pl-3 border-l border-surface-border space-y-0.5">
          {Object.entries(value.fields as Obj).map(([k, v]) => (
            <Row key={k} k={k}>
              <ValueInner value={v} ep={ep} counter={counter} />
            </Row>
          ))}
        </div>
      </div>
    );
  }
  if (t && value.truncated === 'depth') {
    return <span className="font-mono text-xs text-text-secondary">&lt;{t}&gt; <span className="text-text-muted">（深さの上限で省略）</span></span>;
  }
  if (t === 'bytes' || t === 'bytearray' || t === 'memoryview') {
    return <span className="font-mono text-xs text-text-secondary">&lt;{t}&gt; {typeof value.length === 'number' ? `${value.length} バイト` : ''}</span>;
  }
  if (t && Object.keys(value).length === 1) {
    return (
      <span className="font-mono text-xs text-text-secondary">
        &lt;{t}&gt; <span className="text-text-muted">（型名のみ記録。値は評価していません）</span>
      </span>
    );
  }
  // A plain JSON object (for example a service request summary).
  return (
    <div className="pl-3 border-l border-surface-border space-y-0.5">
      {Object.entries(value).map(([k, v]) => (
        <Row key={k} k={k}>
          <ValueInner value={v} ep={ep} counter={counter} />
        </Row>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// One-line summaries for cards
// ---------------------------------------------------------------------------

function oneLine(v: unknown, budget: { left: number }): string {
  if (budget.left <= 0) return '…';
  budget.left -= 1;
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'number') return fmtNum(v, 4);
  if (typeof v === 'string') return v === '***masked***' ? v : JSON.stringify(v.length > 60 ? `${v.slice(0, 60)}…` : v);
  if (Array.isArray(v)) return `[${v.slice(0, 8).map((x) => oneLine(x, budget)).join(', ')}${v.length > 8 ? ', …' : ''}]`;
  if (!isObj(v)) return String(v);
  if (isTextNode(v)) return `"${String(v.preview ?? v.text ?? '').slice(0, 40)}…"(${String(v.length ?? '?')}字)`;
  const t = typeOf(v);
  if (isArraySummary(v)) {
    if (Array.isArray(v.values) && JSON.stringify(v.values).length <= 60) return formatNested(v.values);
    return `${t ?? 'ndarray'}(${Array.isArray(v.shape) ? (v.shape as unknown[]).join(',') : '?'}) ${typeof v.dtype === 'string' ? v.dtype : ''}`.trim();
  }
  if (isContainer(v)) {
    const items = v.items;
    const more = typeof v.truncated === 'number' ? ', …' : '';
    if (Array.isArray(items)) {
      const inner = items.slice(0, 8).map((x) => oneLine(x, budget)).join(', ') + (items.length > 8 || more ? ', …' : '');
      return t === 'tuple' ? `(${inner})` : `[${inner}]`;
    }
    const entries = Object.entries(items as Obj);
    return `{${entries
      .slice(0, 6)
      .map(([k, x]) => `${k}: ${oneLine(x, budget)}`)
      .join(', ')}${entries.length > 6 || more ? ', …' : ''}}`;
  }
  if (isFieldsNode(v)) {
    const entries = Object.entries(v.fields as Obj);
    return `${(t ?? '').split('.').pop()}(${entries
      .slice(0, 6)
      .map(([k, x]) => `${k}=${oneLine(x, budget)}`)
      .join(', ')}${entries.length > 6 ? ', …' : ''})`;
  }
  if (t === 'PIL.Image') return `PIL.Image(${Array.isArray(v.size) ? (v.size as unknown[]).join('×') : '?'})`;
  if (t) return `<${t}>`;
  const entries = Object.entries(v);
  return `{${entries
    .slice(0, 6)
    .map(([k, x]) => `${k}: ${oneLine(x, budget)}`)
    .join(', ')}${entries.length > 6 ? ', …' : ''}}`;
}

/** One-line summary of a value (cards). */
export function valueSummary(v: unknown, max = 120): string {
  const s = oneLine(v, { left: 60 });
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

/** One-line summary of named arguments: `a=1, b=[…]`. */
export function argsSummary(args: unknown, max = 140): string {
  if (!isObj(args)) return valueSummary(args, max);
  const budget = { left: 60 };
  const s = Object.entries(args)
    .filter(([k]) => k !== '<truncated>')
    .map(([k, v]) => `${k}=${oneLine(v, budget)}`)
    .join(', ');
  return s.length > max ? `${s.slice(0, max)}…` : s;
}
