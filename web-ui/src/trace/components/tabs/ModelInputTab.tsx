/**
 * Tab 1 「実際のモデル入力」 (docs/trace-format.md 9.3, 6.2-6.5).
 *
 * Model calls: the stored request in message order (role, full text with
 * fold/unfold, search, copy), images inline with their metadata, the
 * six-category summary with payload positions drawn on the text,
 * 「保存用正規化済みpayload」 with masked and moved fields, the proxy note for
 * `endpoint.known_proxy`, and 「この呼び出しは画像入力なし」. 「未記録」 when
 * no payload was stored.
 *
 * API and code-block events: 「このAPI実行時に新しいLLM呼び出しはありません」
 * or the nested perception VLM calls, the statement that no code-generator or
 * VDM call happened during the event, links to the call that produced the
 * code and to the next one, and the 6.4 later-input report (laterInput.ts),
 * computed only from recorded input_summary matches. Each later request is one
 * HTTP attempt (retries are listed); a text that was never searched is shown
 * as 未確認, never as "not included". A later request whose data (with its
 * input summary) was moved to an asset says whether that asset is loading,
 * failed (with a retry) or was never written, not 未記録.
 *
 * 「この呼び出しは画像入力なし」 is shown only when zero images and videos were
 * measured; a request whose body was not parsed shows 画像: 未記録.
 *
 * Related: payload.ts, laterInput.ts, TextView.tsx, InputDiffTab.tsx.
 */

import { useMemo, useState, type ReactNode } from 'react';
import { pendingAsset, type DataAssetState } from '../../dataAsset';
import { nextFocusToken, pickFocus } from '../../focus';
import { episodeFileUrl, fetchJsonFile } from '../../api';
import { compactJson, fmtBytes, shortSha } from '../../format';
import { highlight } from '../../highlight';
import { useAsync, useFullText } from '../../hooks';
import {
  CATEGORY_LABELS,
  LABELS,
  NORMALIZED_PAYLOAD_NOTE,
  NO_CODEGEN_OR_VDM_DURING_EVENT,
  PROXY_NOTE,
  SEGMENT_KIND_LABELS,
  STREAM_STATE_LABELS,
  STREAM_UNCHECKED_REASONS,
} from '../../labels';
import {
  blockStreamText,
  computeLaterInput,
  MIN_SEGMENT_CHARS,
  type LaterInputReport,
  type LaterRequestReport,
  type MappedRange,
  type StreamCheck,
  type StreamTexts,
} from '../../laterInput';
import {
  ambiguousSource,
  codegenOrVdmDuring,
  enclosingBlock,
  imageSourceText,
  mainAttempt,
  nestedModelCalls,
  nextCallAfter,
  requestImageCounts,
  resolveText,
  textOf,
  type Attempt,
  type EpisodeModel,
  type LogicalCall,
  type TNode,
} from '../../model';
import { codePointLength, codePointToUtf16 } from '../../offsets';
import { imagesByPath, parsePayload, type PayloadMessage, type PayloadPart } from '../../payload';
import { dataOf, type FeedbackSegment, type ImageInfo, type InputSummary, type SegmentCategory, type TraceEvent } from '../../types';
import { Badge, CopyButton, EventLink, KV, Notice, NotRecorded, SectionTitle, SentToModelChip, Spinner, ErrorText } from '../common';
import { DataAssetInline, NO_ASSET_STATES } from '../DataAssetState';
import { SourceSnapshotLink, imageMeta } from '../MediaPane';
import { TextView, type HighlightRange } from '../TextView';
import { roleTone } from '../Timeline';
import { TraceImage } from '../TraceImage';
import type { RangeFocus, TabProps } from '../DetailTabs';
import type { EpisodeRef } from '../ValueView';

const SEGMENT_CATEGORIES: SegmentCategory[] = ['task_api_description', 'code_history', 'stdout_stderr', 'vdm_description', 'other'];

type RangeCat = SegmentCategory | 'unattributed' | 'whitespace';
type PartRange = { cat: RangeCat; start: number; end: number; id: string; title: string };
const NO_RANGES: PartRange[] = [];

const CAT_CLASS: Record<RangeCat, string> = {
  task_api_description: 'bg-sky-500/15',
  code_history: 'bg-violet-500/20',
  stdout_stderr: 'bg-emerald-500/20',
  vdm_description: 'bg-pink-500/20',
  other: 'bg-sand-500/40',
  unattributed: 'bg-red-500/10 underline decoration-dotted decoration-red-400/70',
  // unattributed ranges that hold only whitespace (the newline between two segments)
  whitespace: 'bg-surface-overlay/60',
};

const CAT_SWATCH: Record<SegmentCategory | 'unattributed', string> = {
  task_api_description: 'bg-sky-500/40',
  code_history: 'bg-violet-500/50',
  stdout_stderr: 'bg-emerald-500/50',
  vdm_description: 'bg-pink-500/50',
  other: 'bg-sand-500',
  unattributed: 'bg-red-500/40',
};

export function ModelInputTab(props: TabProps) {
  const { selection } = props;
  const node = selection.node;
  if (node.kind === 'model_call' && node.call) return <ModelCallInput {...props} call={node.call} />;
  if (node.kind === 'code_block' || node.kind === 'api_call' || node.kind === 'stdout' || node.kind === 'stderr' || node.kind === 'service_call') {
    return <EventInputContext {...props} />;
  }
  if (node.kind === 'feedback') return <FeedbackInput {...props} />;
  return (
    <Notice>
      このイベントはモデル呼び出しではありません。モデルへの入力はモデル呼び出しのカードを選ぶと表示されます。
    </Notice>
  );
}

// ---------------------------------------------------------------------------
// Model calls
// ---------------------------------------------------------------------------

function payloadPathOf(req: TraceEvent): { path: string | null; reason: string | null } {
  const d = dataOf(req, 'model_request');
  const art = (req.artifacts ?? []).find((a) => a.label === 'payload');
  if (art && art.written === false) return { path: null, reason: `保存に失敗しました: ${art.reason ?? '理由不明'}` };
  const path = d.payload_path ?? art?.path ?? null;
  return { path, reason: path ? null : 'payload_path がありません（フック未導入、保存失敗、ディスク不足のいずれか）' };
}

function relatedWarnings(model: EpisodeModel, ev: TraceEvent): TraceEvent[] {
  return model.warnings.filter((w) => dataOf(w, 'trace_warning').related_event_id === ev.event_id);
}

function AttemptChips({ call, attempt, onSelect }: { call: LogicalCall; attempt: Attempt | null; onSelect: TabProps['onSelect'] }) {
  if (call.attempts.length <= 1) return null;
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-[11px] text-text-tertiary">HTTP 試行:</span>
      {call.attempts.map((a, i) => {
        const r = a.responses[0];
        const e = a.errors[0];
        const st = r ? dataOf(r, 'model_response').http_status : e ? dataOf(e, 'model_error').http_status : null;
        return (
          <button
            key={a.request.event_id}
            type="button"
            onClick={() => onSelect(call.node, a.request.event_id)}
            className={`px-1.5 py-px rounded border text-[11px] font-mono ${
              a === attempt ? 'border-accent text-accent bg-accent/10' : 'border-surface-border text-text-secondary hover:border-surface-border-light'
            }`}
          >
            attempt {dataOf(a.request, 'model_request').http_attempt ?? i + 1}: {st ?? (r ? 'ok' : e ? 'error' : '応答なし')}
          </button>
        );
      })}
      <span className="text-[11px] text-text-tertiary">（同じ論理呼び出しの再送。再計画ではありません）</span>
    </div>
  );
}

function ModelCallInput({ model, ep, selection, onSelect, focus, call }: TabProps & { call: LogicalCall }) {
  const attempt = selection.attempt ?? mainAttempt(call);
  if (!attempt) {
    return <NotRecorded reason="この論理呼び出しの model_request が記録されていません" />;
  }
  const req = attempt.request;
  const d = dataOf(req, 'model_request');
  const { path, reason } = payloadPathOf(req);
  const warnings = relatedWarnings(model, req);
  const endpoint = d.endpoint ?? null;
  const images = [...(d.images ?? [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  // Counts from recorded fields only: a body that was never parsed has no counts (unknown).
  const counts = requestImageCounts(d);
  const f = focus && (focus.eventId === req.event_id || call.node.events.some((e) => e.event_id === focus.eventId)) ? focus : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={roleTone(call.role)}>{call.role}</Badge>
        {call.callKind && <Badge tone="neutral">{call.callKind}</Badge>}
        {call.turnIndex !== null && <span className="text-xs text-text-tertiary">ターン {call.turnIndex}</span>}
        <span className="text-xs font-mono text-text-secondary">{d.model ?? call.model ?? 'model ?'}</span>
        <span className="text-[11px] text-text-tertiary">
          {call.role === 'vdm' ? 'VDM への入力（コード生成モデルとは別の呼び出し）' : call.role === 'code_generator' ? 'コード生成モデルへの入力' : ''}
        </span>
      </div>
      <AttemptChips call={call} attempt={attempt} onSelect={onSelect} />
      <KV
        rows={[
          ['endpoint', endpoint ? `${endpoint.url ?? `${endpoint.host ?? '?'}:${endpoint.port ?? '?'}${endpoint.path ?? ''}`}` : '—'],
          ['body', `${fmtBytes(d.body_bytes)} · sha256 ${shortSha(d.body_sha256, 16)}`],
          ['messages / 画像 / 動画', `${d.n_messages ?? '?'} / ${counts.images ?? '?'} / ${counts.videos ?? '?'}`],
          ['stream', String(d.stream ?? '—')],
          ['logical_call_id', call.synthetic ? '（記録なし）' : call.id],
          ['request event', req.event_id],
        ]}
      />
      {counts.noImage === true && <Notice tone="neutral"><span className="font-bold text-text-primary">{LABELS.noImageInput}</span></Notice>}
      {counts.noImage === false && counts.images === 0 && (counts.videos ?? 0) > 0 && <Notice>動画入力のみ（画像なし）</Notice>}
      {counts.noImage === null && (
        <Notice tone="amber">
          画像・動画の数: {LABELS.notRecorded}（送信本文を解析していないため、画像の有無は分かりません
          {d.reconstruction?.reason ? `: ${d.reconstruction.reason}` : ''}）
        </Notice>
      )}
      {endpoint?.known_proxy ? <Notice tone="amber">{PROXY_NOTE}</Notice> : null}
      {warnings.map((w) => (
        <Notice key={w.event_id} tone="amber">
          記録の警告 {dataOf(w, 'trace_warning').code}: {textOf(dataOf(w, 'trace_warning').message)}
        </Notice>
      ))}
      {path ? (
        <PayloadView ep={ep} path={path} req={req} images={images} summary={d.input_summary ?? null} model={model} onSelect={onSelect} focus={f} />
      ) : (
        <NotRecorded reason={reason} />
      )}
    </div>
  );
}

function PayloadView({
  ep,
  path,
  req,
  images,
  summary,
  model,
  onSelect,
  focus,
}: {
  ep: EpisodeRef;
  path: string;
  req: TraceEvent;
  images: ImageInfo[];
  summary: InputSummary | null;
  model: EpisodeModel;
  onSelect: TabProps['onSelect'];
  focus: RangeFocus | null;
}) {
  const url = episodeFileUrl(ep, path);
  const payload = useAsync(() => fetchJsonFile<unknown>(url), [url]);
  const parsed = useMemo(() => (payload.data !== null ? parsePayload(payload.data) : null), [payload.data]);
  const [search, setSearch] = useState('');
  const [localFocus, setLocalFocus] = useState<{ id: string; token: number } | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const d = dataOf(req, 'model_request');

  // Ranges per text path, from input_summary (code-point offsets converted per text).
  const rangesByPath = useMemo(() => {
    const m = new Map<string, PartRange[]>();
    const push = (p: string | null | undefined, v: PartRange) => {
      if (!p) return;
      const list = m.get(p) ?? [];
      list.push(v);
      m.set(p, list);
    };
    const cats = summary?.categories ?? {};
    for (const cat of SEGMENT_CATEGORIES) {
      (cats[cat] ?? []).forEach((s, i) => {
        if (typeof s.start !== 'number' || typeof s.end !== 'number') return;
        push(s.path, {
          cat,
          start: s.start,
          end: s.end,
          id: `seg:${cat}:${i}`,
          title: `${CATEGORY_LABELS[cat]}${s.kind ? ` (${s.kind})` : ''} ${s.start}–${s.end}`,
        });
      });
    }
    (summary?.unattributed ?? []).forEach((u, i) => {
      if (typeof u.start !== 'number' || typeof u.end !== 'number') return;
      const ws = u.whitespace_only === true;
      push(u.path, {
        cat: ws ? 'whitespace' : 'unattributed',
        start: u.start,
        end: u.end,
        id: `seg:unattributed:${i}`,
        title: `${LABELS.unattributed}${ws ? '（空白のみ）' : ''} ${u.start}–${u.end}`,
      });
    });
    return m;
  }, [summary]);

  const imgByPath = useMemo(() => imagesByPath(images), [images]);
  // The newer of the page-level range focus and this tab's own position links wins.
  const chosen = pickFocus(focus, localFocus);
  const focusRangeId = chosen.id;
  const focusToken = chosen.token;

  const moved = images.filter((i) => i.asset && i.written !== false);
  const unsaved = images.filter((i) => !i.url && (!i.asset || i.written === false));
  const masked = d.masked_fields ?? d.reconstruction?.masked ?? [];
  const recon = d.reconstruction;

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-accent/30 bg-accent/5 p-3 space-y-1.5">
        <div className="flex items-center gap-2">
          <Badge tone="gold">{LABELS.normalizedPayload}</Badge>
          <a href={url} target="_blank" rel="noreferrer" className="text-[11px] text-accent underline underline-offset-2">
            保存ファイルを開く
          </a>
        </div>
        <div className="text-xs text-text-secondary">{NORMALIZED_PAYLOAD_NOTE}</div>
        <div className="text-xs text-text-secondary">
          assets に移した画像・動画: {moved.length === 0 ? 'なし' : `${moved.length} 件`}
          {moved.length > 0 && (
            <ul className="mt-0.5 font-mono text-[11px] text-text-tertiary">
              {moved.map((im, i) => (
                <li key={i}>
                  {im.path} → {im.asset} ({im.mime ?? '?'}, {fmtBytes(im.bytes)}, sha256 {shortSha(im.sha256)})
                </li>
              ))}
            </ul>
          )}
        </div>
        {unsaved.length > 0 && (
          <div className="text-xs text-amber-300">
            保存できなかった画像・動画: {unsaved.length} 件
            <ul className="mt-0.5 font-mono text-[11px]">
              {unsaved.map((im, i) => (
                <li key={i}>
                  {im.path}: {im.reason ?? '理由不明'}
                </li>
              ))}
            </ul>
          </div>
        )}
        <div className="text-xs text-text-secondary">
          マスクした項目: {masked.length === 0 ? 'なし' : <span className="font-mono">{masked.join(', ')}</span>}
        </div>
        <div className="text-xs">
          {recon?.verified === true ? (
            <span className="text-nv-green">
              送信本文と一致を確認済み（{recon.method ?? 'json_roundtrip'}
              {recon.method === 'masked_compare' ? '、マスク前の本文と比較' : ''}）
            </span>
          ) : recon?.verified === false ? (
            <span className="text-amber-300">送信本文との一致を確認できません: {recon.reason ?? '理由不明'}</span>
          ) : (
            <span className="text-text-tertiary">再構成の検証結果は記録されていません</span>
          )}
        </div>
      </div>

      <CategorySummary summary={summary} images={images} model={model} requestSeq={req.seq} onSelect={onSelect} onFocus={(id) => setLocalFocus({ id, token: nextFocusToken() })} />

      <SectionTitle
        right={
          <div className="flex items-center gap-2">
            <input
              type="search"
              name="trace-input-search"
              aria-label="入力内を検索"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="入力内を検索"
              className="px-2 py-0.5 bg-surface-sunken border border-surface-border rounded text-[11px] text-text-primary w-40 focus:outline-none focus:ring-1 focus:ring-accent/40"
            />
            {parsed && <CopyButton label="全文をコピー" text={() => parsed.messages.map((m) => `[${m.role}]\n${m.parts.map((p) => p.text).join('\n')}`).join('\n\n')} />}
          </div>
        }
      >
        メッセージ（送信順）
      </SectionTitle>
      {payload.loading && <Spinner label="payload を読み込み中" />}
      {payload.error && (
        <>
          <NotRecorded reason={`payload を読み込めません: ${payload.error}`} />
        </>
      )}
      {parsed && parsed.container === 'none' && (
        <>
          <Notice tone="amber">messages / input が見つかりません。保存ファイルをそのまま表示します。</Notice>
          <RawJson value={payload.data} />
        </>
      )}
      {parsed && parsed.messages.length > 0 && <div data-playback-input />}
      {parsed &&
        parsed.messages.map((m) => (
          <MessageView
            key={m.index}
            m={m}
            ep={ep}
            imgByPath={imgByPath}
            rangesByPath={rangesByPath}
            search={search}
            focus={focus}
            focusRangeId={focusRangeId}
            focusToken={focusToken}
            model={model}
            requestSeq={req.seq}
          />
        ))}
      {parsed && (
        <>
          <SectionTitle>その他の本文フィールド（messages 以外）</SectionTitle>
          <KV rows={Object.entries(parsed.params).map(([k, v]) => [k, compactJson(v, 400)])} />
        </>
      )}
      {payload.data !== null && (
        <div>
          <button type="button" onClick={() => setShowRaw(!showRaw)} className="text-[11px] text-accent-light hover:text-accent">
            {showRaw ? '保存ファイルの JSON を隠す' : '保存ファイルの JSON を表示'}
          </button>
          {showRaw && <RawJson value={payload.data} />}
        </div>
      )}
    </div>
  );
}

export function RawJson({ value }: { value: unknown }) {
  const text = useMemo(() => JSON.stringify(value, null, 2) ?? '', [value]);
  const html = useMemo(() => (text.length < 300_000 ? highlight(text, 'json') : null), [text]);
  return (
    <pre className="mt-1 max-h-96 overflow-auto text-[11px] font-mono bg-surface-sunken border border-surface-border rounded-md p-2 whitespace-pre-wrap break-all">
      {html !== null ? <code dangerouslySetInnerHTML={{ __html: html }} /> : text}
    </pre>
  );
}

function MessageView({
  m,
  ep,
  imgByPath,
  rangesByPath,
  search,
  focus,
  focusRangeId,
  focusToken,
  model,
  requestSeq,
}: {
  m: PayloadMessage;
  ep: EpisodeRef;
  imgByPath: Map<string, ImageInfo>;
  rangesByPath: Map<string, PartRange[]>;
  search: string;
  focus: RangeFocus | null;
  focusRangeId: string | null;
  focusToken: number;
  model: EpisodeModel;
  requestSeq: number;
}) {
  const roleClass = m.role === 'system' ? 'neutral' : m.role === 'user' ? 'gold' : m.role === 'assistant' ? 'violet' : 'neutral';
  return (
    <div className="rounded-md border border-surface-border bg-surface-raised">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-surface-border">
        <Badge tone={roleClass}>{m.role}</Badge>
        <span className="text-[11px] text-text-tertiary font-mono">{m.path}</span>
        <span className="ml-auto text-[11px] text-text-tertiary">{m.parts.length} パート</span>
      </div>
      <div className="p-2 space-y-2">
        {m.parts.length === 0 && <div className="text-[11px] text-text-tertiary">（content なし）</div>}
        {m.parts.map((p, i) => (
          <PartView
            key={i}
            part={p}
            ep={ep}
            image={imgByPath.get(p.path) ?? null}
            ranges={rangesByPath.get(p.path) ?? NO_RANGES}
            search={search}
            focus={focus && focus.path === p.path ? focus : null}
            focusRangeId={focusRangeId}
            focusToken={focusToken}
            model={model}
            requestSeq={requestSeq}
          />
        ))}
      </div>
    </div>
  );
}

function PartView({
  part,
  ep,
  image,
  ranges,
  search,
  focus,
  focusRangeId,
  focusToken,
  model,
  requestSeq,
}: {
  part: PayloadPart;
  ep: EpisodeRef;
  image: ImageInfo | null;
  ranges: PartRange[];
  search: string;
  focus: RangeFocus | null;
  focusRangeId: string | null;
  focusToken: number;
  model: EpisodeModel;
  requestSeq: number;
}) {
  const hl = useMemo<HighlightRange[]>(() => {
    if (part.kind !== 'text') return [];
    const out: HighlightRange[] = ranges.map((r) => ({
      start: codePointToUtf16(part.text, r.start),
      end: codePointToUtf16(part.text, r.end),
      className: CAT_CLASS[r.cat],
      title: r.title,
      id: r.id,
    }));
    if (focus) {
      out.push({
        start: codePointToUtf16(part.text, focus.start),
        end: codePointToUtf16(part.text, focus.end),
        className: 'outline outline-2 outline-accent bg-accent/25',
        title: '選択した位置',
        id: 'focus',
      });
    }
    return out;
  }, [part, ranges, focus]);

  if (part.kind === 'text') {
    return (
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] text-text-muted font-mono">{part.path}</span>
          <span className="text-[10px] text-text-muted">{part.text.length} 文字</span>
          <span className="ml-auto">
            <CopyButton text={part.text} />
          </span>
        </div>
        <TextView text={part.text} ranges={hl} search={search} foldLines={30} focusRangeId={focusRangeId} focusToken={focusToken} />
      </div>
    );
  }
  if (part.kind === 'media' && part.media) {
    const media = part.media;
    if (media.externalUrl) {
      return (
        <div className="text-xs font-mono break-all p-2 rounded border border-surface-border bg-surface-sunken">
          外部URL（ビューワーもレコーダーも取得しません。送信時の画像と同一とは断定できません）: {media.externalUrl}
        </div>
      );
    }
    const info: ImageInfo = image ?? { path: part.path, asset: media.asset, sha256: media.sha256, mime: media.mime, bytes: media.bytes };
    const url = media.asset ? episodeFileUrl(ep, media.asset) : null;
    const sourceText = imageSourceText(model, info, requestSeq, false);
    const amb = ambiguousSource(model, info);
    return (
      <div className="flex gap-3 items-start">
        <TraceImage src={url} alt={part.path} provenance="model_input" video={media.isVideo} meta={imageMeta(info, amb ? `${sourceText}（${amb}）` : sourceText)} className="h-40 w-56 flex-shrink-0" />
        <div className="text-[11px] font-mono text-text-secondary space-y-0.5 min-w-0">
          <SentToModelChip />
          {amb && <div className="text-amber-300 font-sans">{amb}</div>}
          {imageMeta(info, sourceText)}
          <SourceSnapshotLink ep={ep} im={info} />
          {!media.asset && (
            <div className="text-amber-300">
              {LABELS.notRecorded}（{media.reason ?? 'asset がありません'}）
            </div>
          )}
        </div>
      </div>
    );
  }
  return <pre className="text-[11px] font-mono text-text-secondary whitespace-pre-wrap break-all">{compactJson(part.raw, 2000)}</pre>;
}

function CategorySummary({
  summary,
  images,
  model,
  requestSeq,
  onSelect,
  onFocus,
}: {
  summary: InputSummary | null;
  images: ImageInfo[];
  model: EpisodeModel;
  requestSeq: number;
  onSelect: TabProps['onSelect'];
  onFocus: (id: string) => void;
}) {
  if (!summary) {
    return (
      <div>
        <SectionTitle>入力の内訳（6分類）</SectionTitle>
        <NotRecorded reason="input_summary が記録されていません" />
      </div>
    );
  }
  const cats = summary.categories ?? {};
  const imgs = cats.images ?? images;
  const rows: ReactNode[] = [];
  rows.push(
    <tr key="images" className="border-b border-surface-border/60 align-top">
      <td className="py-1 pr-2 whitespace-nowrap">
        <span className="inline-block w-2.5 h-2.5 rounded-sm bg-accent/60 mr-1.5" />
        {CATEGORY_LABELS.images}
      </td>
      <td className="py-1 pr-2 font-mono whitespace-nowrap">
        画像 {summary.n_images ?? '?'} / 動画 {summary.n_videos ?? 0}
      </td>
      <td className="py-1 font-mono text-[11px] text-text-secondary">
        {summary.no_image_input ? LABELS.noImageInput : imgs.map((im, i) => (
          <div key={i}>
            #{im.order ?? i} {im.path} · 由来: {imageSourceText(model, im, requestSeq)}
          </div>
        ))}
      </td>
    </tr>,
  );
  for (const cat of SEGMENT_CATEGORIES) {
    const segs = cats[cat] ?? [];
    const chars = segs.reduce((acc, s) => acc + Math.max(0, (s.end ?? 0) - (s.start ?? 0)), 0);
    rows.push(
      <tr key={cat} className="border-b border-surface-border/60 align-top">
        <td className="py-1 pr-2 whitespace-nowrap">
          <span className={`inline-block w-2.5 h-2.5 rounded-sm mr-1.5 ${CAT_SWATCH[cat]}`} />
          {CATEGORY_LABELS[cat]}
        </td>
        <td className="py-1 pr-2 font-mono whitespace-nowrap">
          {segs.length} 件 · {chars} 字
        </td>
        <td className="py-1 font-mono text-[11px] text-text-secondary">
          {segs.length === 0 && <span className="text-text-muted">なし</span>}
          {segs.map((s, i) => (
            <div key={i} className="flex flex-wrap gap-x-2">
              <button type="button" onClick={() => onFocus(`seg:${cat}:${i}`)} className="text-accent hover:text-accent-light underline underline-offset-2">
                {s.path} {s.start}–{s.end}
              </button>
              {s.kind && (
                <span className="text-text-tertiary" title={SEGMENT_KIND_LABELS[s.kind] ?? undefined}>
                  kind={s.kind}
                </span>
              )}
              {(s.source_event_ids ?? []).slice(0, 3).map((sid) => (
                <SourceLink key={sid} id={sid} model={model} onSelect={onSelect} />
              ))}
            </div>
          ))}
        </td>
      </tr>,
    );
  }
  // Ranges that hold only whitespace (the newline between two segments) are counted apart.
  const allUn = (summary.unattributed ?? []).map((s, i) => ({ s, i }));
  const un = allUn.filter((x) => x.s.whitespace_only !== true);
  const ws = allUn.filter((x) => x.s.whitespace_only === true);
  const chars = (list: typeof allUn) => list.reduce((acc, x) => acc + Math.max(0, (x.s.end ?? 0) - (x.s.start ?? 0)), 0);
  rows.push(
    <tr key="unattributed" className="align-top">
      <td className="py-1 pr-2 whitespace-nowrap">
        <span className={`inline-block w-2.5 h-2.5 rounded-sm mr-1.5 ${CAT_SWATCH.unattributed}`} />
        {LABELS.unattributed}
      </td>
      <td className="py-1 pr-2 font-mono whitespace-nowrap">
        {un.length} 件 · {chars(un)} 字
      </td>
      <td className="py-1 font-mono text-[11px] text-text-secondary">
        {un.length === 0 && <span className="text-text-muted">なし</span>}
        {un.slice(0, 20).map(({ s, i }) => (
          <div key={i}>
            <button type="button" onClick={() => onFocus(`seg:unattributed:${i}`)} className="text-accent hover:text-accent-light underline underline-offset-2">
              {s.path} {s.start}–{s.end}
            </button>
          </div>
        ))}
        {un.length > 20 && <div className="text-text-muted">…ほか {un.length - 20} 件</div>}
        {summary.unattributed_truncated === true && (
          <div className="text-amber-300 font-sans">
            由来不明の区間はレコーダーが 200 件で打ち切りました。それ以降の区間は記録されておらず、本文にも印を付けていません。
          </div>
        )}
        {ws.length > 0 && (
          <div className="text-text-muted" title={ws.map(({ s }) => `${s.path} ${s.start}–${s.end}`).join('\n')}>
            ほかに空白だけの区間 {ws.length} 件（{chars(ws)} 字。セグメントの間の改行など。由来は記録されていません）
          </div>
        )}
      </td>
    </tr>,
  );
  const missing = summary.registered_not_found ?? [];
  return (
    <div>
      <SectionTitle>入力の内訳（6分類・payload 内の位置）</SectionTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <tbody>{rows}</tbody>
        </table>
      </div>
      <div className="mt-1 text-[11px] text-text-tertiary">
        位置は記録済みの文字列と完全一致した範囲だけです。意味から推定した由来は表示しません。
      </div>
      {missing.length > 0 && (
        <div className="mt-2 text-[11px] text-text-secondary">
          <div className="text-text-tertiary mb-0.5">この入力に含まれていない、以前に生成されたテキスト（registered_not_found）:</div>
          {missing.map((r, i) => (
            <div key={i} className="flex flex-wrap gap-2 font-mono">
              <span>{r.kind ?? '?'}</span>
              {(r.source_event_ids ?? []).slice(0, 4).map((sid) => (
                <SourceLink key={sid} id={sid} model={model} onSelect={onSelect} />
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SourceLink({ id, model, onSelect }: { id: string; model: EpisodeModel; onSelect: TabProps['onSelect'] }) {
  const n = model.nodeOfEvent.get(id);
  if (!n) return <span className="text-text-muted" title="記録にないイベント">{shortSha(id, 14)}</span>;
  return (
    <EventLink onClick={() => onSelect(n, id)} title={id}>
      {n.kind === 'model_call' ? `${n.call?.role ?? 'model'}` : n.kind} #{n.seq}
    </EventLink>
  );
}

// ---------------------------------------------------------------------------
// API and code-block events
// ---------------------------------------------------------------------------

function EventInputContext({ model, ep, selection, running, onSelect, onFocusRange, assetStates = NO_ASSET_STATES, onRetryAsset }: TabProps) {
  const node = selection.node;
  const block = enclosingBlock(model, node);
  const nested = nestedModelCalls(node);
  const during = codegenOrVdmDuring(model, node);
  const source = block?.sourceCall ?? null;
  const codeSource = block ? dataOf(block.primary, 'code_block_start').code_source : null;
  const next = block ? nextCallAfter(model, block.end?.seq ?? block.lastSeq, 'code_generator') : null;

  return (
    <div className="space-y-3">
      {nested.length === 0 ? (
        <Notice tone="neutral">
          <span className="font-bold text-text-primary">{LABELS.noNewLlmCall}</span>
        </Notice>
      ) : (
        <div>
          <SectionTitle>このイベントの中で行われたモデル呼び出し（知覚 VLM など）</SectionTitle>
          <div className="space-y-1">
            {nested.map((c) => {
              const a = mainAttempt(c);
              const rd = a ? dataOf(a.request, 'model_request') : null;
              return (
                <div key={c.id} className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge tone={roleTone(c.role)}>{c.role}</Badge>
                  {c.callKind && <span className="text-text-tertiary">{c.callKind}</span>}
                  <span className="font-mono text-text-secondary">{c.model ?? rd?.model ?? ''}</span>
                  <span className="text-text-tertiary">画像 {rd?.n_images ?? '?'}</span>
                  <EventLink onClick={() => onSelect(c.node)}>入力と出力を見る</EventLink>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {during.length === 0 ? (
        <Notice tone="neutral">{NO_CODEGEN_OR_VDM_DURING_EVENT}</Notice>
      ) : (
        <Notice tone="red">
          このイベントの記録範囲（seq {node.seq}–{node.lastSeq}）にコード生成または VDM の要求があります:{' '}
          {during.map((c) => (
            <EventLink key={c.id} onClick={() => onSelect(c.node)}>
              {c.role} #{c.node.seq}{' '}
            </EventLink>
          ))}
        </Notice>
      )}
      <div>
        <SectionTitle>関連するモデル呼び出し</SectionTitle>
        <KV
          rows={[
            [
              'このコードを生成した呼び出し',
              source ? (
                <EventLink onClick={() => onSelect(source.node)}>
                  {source.role} {source.callKind} {source.turnIndex !== null ? `ターン ${source.turnIndex}` : ''} (#{source.node.seq})
                </EventLink>
              ) : codeSource === 'oracle' ? (
                'oracle コード（モデル呼び出しなし）'
              ) : (
                '不明（source_model_event_id が記録にありません）'
              ),
            ],
            [
              '次のコード生成呼び出し',
              next ? (
                <EventLink onClick={() => onSelect(next.node)}>
                  {next.role} {next.callKind} {next.turnIndex !== null ? `ターン ${next.turnIndex}` : ''} (#{next.node.seq})
                </EventLink>
              ) : (
                <span>
                  {LABELS.noLaterModelCall}
                  {running ? '（エピソード実行中のため、今後追加される可能性があります）' : ''}
                </span>
              ),
            ],
          ]}
        />
      </div>
      <LaterInputSection
        model={model}
        ep={ep}
        node={node}
        running={running}
        onSelect={onSelect}
        onFocusRange={onFocusRange}
        assetStates={assetStates}
        onRetryAsset={onRetryAsset}
      />
    </div>
  );
}

function LaterInputSection({
  model,
  ep,
  node,
  running,
  onSelect,
  onFocusRange,
  assetStates,
  onRetryAsset,
}: {
  model: EpisodeModel;
  ep: EpisodeRef;
  node: TNode;
  running: boolean;
  onSelect: TabProps['onSelect'];
  onFocusRange: TabProps['onFocusRange'];
  assetStates: ReadonlyMap<string, DataAssetState>;
  onRetryAsset?: (ev: TraceEvent) => void;
}) {
  const block = enclosingBlock(model, node);
  const out = block?.children.find((c) => c.kind === 'stdout')?.primary ?? null;
  const err = block?.children.find((c) => c.kind === 'stderr')?.primary ?? null;
  const outText = useFullText(ep, out ? dataOf(out, 'stdout').text : null);
  const errText = useFullText(ep, err ? dataOf(err, 'stderr').text : null);
  // A long str(e) is a text asset: load it, so the exception search uses the whole text.
  const apiError = node.kind === 'api_call' && node.end?.event_type === 'api_call_error' ? node.end : null;
  const msgValue = apiError ? dataOf(apiError, 'api_call_error').message ?? null : null;
  const msgText = useFullText(ep, msgValue);
  // A stream event that the block end names but the record lacks is unknown text (null), not ''.
  const endData = block?.end ? dataOf(block.end, 'code_block_end') : null;
  const texts: StreamTexts = {
    stdout: blockStreamText(out, outText, endData?.stdout_len, endData?.stdout_event_id),
    stderr: blockStreamText(err, errText, endData?.stderr_len, endData?.stderr_event_id),
    errorMessage: msgValue === null ? undefined : msgText.partial ? null : msgText.text,
  };
  const streamsLost = (!out && texts.stdout === null) || (!err && texts.stderr === null);
  const report = useMemo(
    () => computeLaterInput(model, node, texts),
    [model, node, texts.stdout, texts.stderr, texts.errorMessage],
  );
  return (
    <div>
      <SectionTitle>以後のモデル入力への反映（記録された一致のみ）</SectionTitle>
      <LaterInputReportView
        report={report}
        running={running}
        onSelect={onSelect}
        onFocusRange={onFocusRange}
        loading={outText.loading || errText.loading || msgText.loading}
        streamsLost={streamsLost}
        assetStates={assetStates}
        onRetryAsset={onRetryAsset}
      />
    </div>
  );
}

function RangeLink({ r, range, onFocusRange }: { r: LaterRequestReport; range: MappedRange; onFocusRange: TabProps['onFocusRange'] }) {
  return (
    <EventLink onClick={() => onFocusRange(r.attempt.request.event_id, range.path, range.start, range.end)} title="モデル入力タブでこの位置を表示">
      {range.path} {range.start}–{range.end}
    </EventLink>
  );
}

function stateTone(state: StreamCheck['state']): string {
  return state === 'not_included' || state === 'empty' ? 'text-text-tertiary' : 'text-amber-300';
}

/** One stream of the block in one request: found (with links), searched and not found, or unchecked. */
function StreamLine({
  label,
  check,
  len,
  r,
  stream,
  onFocusRange,
}: {
  label: string;
  check: StreamCheck;
  len: number | null;
  r: LaterRequestReport;
  stream: 'stdout' | 'stderr';
  onFocusRange: TabProps['onFocusRange'];
}) {
  return (
    <div className="text-text-secondary">
      {label}（{len ?? '?'} 文字）:{' '}
      {check.state === 'included' ? (
        check.hits.map((h, i) => (
          <span key={i} className="mr-2">
            {STREAM_STATE_LABELS.included} <RangeLink r={r} range={h} onFocusRange={onFocusRange} />
            {!h.exact && <span className="text-amber-300">（セグメントはブロックの {stream} 全体と一致しません）</span>}
          </span>
        ))
      ) : (
        <span className={stateTone(check.state)}>{STREAM_STATE_LABELS[check.state]}</span>
      )}
    </div>
  );
}

/** What the non-printing API notice says about the block's stdout in the next code-generator input. */
function stdoutInNextText(check: StreamCheck, len: number | null): string {
  const what = `このブロックの stdout（${len ?? '?'} 文字、生成コードによる print を含む）`;
  switch (check.state) {
    case 'not_included':
      return `${what}は、次のコード生成入力に含まれていませんでした（照合済み）。`;
    case 'empty':
      return 'このブロックの stdout は空でした。';
    case 'short':
      return `${what}は ${MIN_SEGMENT_CHARS} 文字未満のため照合の対象外で、次のコード生成入力に含まれたかは未確認です。`;
    case 'unsearched':
      return `${what}が次のコード生成入力に含まれたかは未確認です（照合の対象に登録されていません）。`;
    case 'not_recorded':
      return `このブロックの stdout は${LABELS.notRecorded}です。`;
    default:
      return '';
  }
}

function LaterInputReportView({
  report,
  running,
  onSelect,
  onFocusRange,
  loading,
  streamsLost = false,
  assetStates = NO_ASSET_STATES,
  onRetryAsset,
}: {
  report: LaterInputReport;
  running: boolean;
  onSelect: TabProps['onSelect'];
  onFocusRange: TabProps['onFocusRange'];
  loading: boolean;
  /** A stream event of the block is missing from the record although the block end names it. */
  streamsLost?: boolean;
  /** Loading state of requests whose data (with the input summary) is in an asset. */
  assetStates?: ReadonlyMap<string, DataAssetState>;
  onRetryAsset?: (ev: TraceEvent) => void;
}) {
  if (!report.applicable) return <Notice>{report.reason}</Notice>;
  const api = report.api;
  const next = report.requests.find((r) => r.isNextCodegen && !r.isRetry) ?? report.requests.find((r) => r.isNextCodegen) ?? null;
  const uncheckedReason = api
    ? api.missingStreamText
      ? streamsLost
        ? `ブロックの stdout/stderr のイベントが記録にない（${LABELS.notRecorded}）ため`
        : 'ブロックの stdout/stderr の全文を読み込めていないため'
      : api.messageIncomplete
        ? '例外テキストの全文を読み込めていないため'
        : api.messageRedacted
          ? 'レコーダーが例外テキスト str(e) から秘密情報を伏せて保存しており、プログラムが出力した文とは一致しないため（照合していません）'
          : api.messageCut
            ? 'レコーダーが例外テキスト str(e) の先頭 64 KB だけを注記付きで保存しており、プログラムが出力した文とは一致しないため（照合していません）'
          : api.errorMessage === null
          ? `例外テキスト str(e) が${LABELS.notRecorded}で、この例外の最終トレースバック行を特定できないため`
          : `str(e) が ${MIN_SEGMENT_CHARS} 文字未満で、stderr にこの例外の最終トレースバック行（型とメッセージが一致する行）も見つからないため`
    : '';
  return (
    <div className="space-y-2">
      <div className="text-[11px] text-text-tertiary">
        対象: このブロックの code_block_end（seq {report.blockEnd?.seq}）より後のモデル要求（HTTP 試行ごと）から、次のコード生成要求まで（その再送を含む）。数値や座標を値で検索することはしません。
      </div>
      {loading && <Spinner label="ブロックの stdout/stderr を読み込み中" />}
      {api && !api.endRecorded && (
        <Notice tone="amber">api_call_end がないため、この API の戻り値と出力の範囲は{LABELS.notRecorded}です。</Notice>
      )}
      {api && api.endRecorded && (api.printedStdout === null || api.printedStderr === null) && (
        <Notice tone="amber">
          この API 自身の出力範囲は{LABELS.notRecorded}です（api_call_end に
          {api.printedStdout === null ? ' stdout_slice' : ''}
          {api.printedStderr === null ? ' stderr_slice' : ''} がありません）。
        </Notice>
      )}
      {report.requests.length === 0 && (
        <Notice>
          {LABELS.noLaterModelCall}
          {running ? '（エピソード実行中のため、今後追加される可能性があります）' : ''}
        </Notice>
      )}
      {report.requests.map((r) => (
        <div key={r.attempt.request.event_id} className="rounded-md border border-surface-border bg-surface-raised p-2 space-y-1 text-xs">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone={roleTone(r.call.role)}>{r.call.role}</Badge>
            {r.call.callKind && <span className="text-text-tertiary">{r.call.callKind}</span>}
            {r.call.turnIndex !== null && <span className="text-text-tertiary">ターン {r.call.turnIndex}</span>}
            {r.call.attempts.length > 1 && (
              <Badge tone="amber">
                HTTP 試行 {r.httpAttempt}
                {r.isRetry ? '（同じ論理呼び出しの再送）' : ''}
              </Badge>
            )}
            <EventLink onClick={() => onSelect(r.call.node, r.attempt.request.event_id)}>#{r.attempt.request.seq} を開く</EventLink>
            {r.isNextCodegen && <Badge tone="gold">次のコード生成入力</Badge>}
          </div>
          {!r.recorded ? (
            <div className="text-text-secondary">
              入力の内訳:{' '}
              {pendingAsset(r.attempt.request, assetStates) ? (
                <DataAssetInline ev={r.attempt.request} states={assetStates} onRetry={onRetryAsset} />
              ) : (
                <>
                  <span className="font-bold">{LABELS.notRecorded}</span>（input_summary なし）
                </>
              )}
            </div>
          ) : (
            <>
              <StreamLine label="ブロックの stdout" check={r.stdout} len={report.stdoutLen} r={r} stream="stdout" onFocusRange={onFocusRange} />
              <StreamLine label="ブロックの stderr" check={r.stderr} len={report.stderrLen} r={r} stream="stderr" onFocusRange={onFocusRange} />
              {api && (api.printedStdout === true || api.printedStderr === true) && (
                <div className="text-text-secondary">
                  この API 自身の出力
                  {api.stdoutSlice && api.printedStdout ? `（stdout ${api.stdoutSlice.start}–${api.stdoutSlice.end}）` : ''}
                  {api.stderrSlice && api.printedStderr ? `（stderr ${api.stderrSlice.start}–${api.stderrSlice.end}）` : ''}:{' '}
                  {r.apiSlices.length > 0 ? (
                    r.apiSlices.map((s, i) => (
                      <span key={i} className="mr-2">
                        {s.stream} → <RangeLink r={r} range={s.range} onFocusRange={onFocusRange} />
                      </span>
                    ))
                  ) : r.sliceNote ? (
                    <span className="text-amber-300">{r.sliceNote}</span>
                  ) : (
                    <ApiOutputState r={r} api={api} />
                  )}
                </div>
              )}
              {api?.isError && (
                <div className="text-text-secondary space-y-0.5">
                  {api.messageTooShort && api.errorMessage !== null && (
                    <div className="text-text-tertiary">
                      例外テキスト str(e)（{codePointLength(api.errorMessage)} 文字）は {MIN_SEGMENT_CHARS} 文字未満のため検索していません
                    </div>
                  )}
                  {api.tracebackMatches > 1 && (
                    <div className="text-amber-300">
                      同じ最終トレースバック行が stderr に {api.tracebackMatches} 回あります。どれがこの例外のものかは区別できません（位置は最後のもの）
                    </div>
                  )}
                  {r.errorNote === 'unchecked' && (
                    <div>
                      例外テキスト・最終トレースバック行: <span className="text-amber-300">未確認</span>（{uncheckedReason}）
                    </div>
                  )}
                  {r.errorNote === 'not_in_streams' && (
                    <div>
                      例外テキスト・最終トレースバック行:{' '}
                      <span className="text-text-tertiary">
                        ブロックの stdout/stderr に出力されていません。stdout/stderr 経由ではこの入力に含まれません
                      </span>
                    </div>
                  )}
                  {r.errorFindings.map((f, i) => (
                    <div key={i}>
                      {f.what === 'message' ? '例外テキスト（str(e)）' : 'この例外の最終トレースバック行'} 「{f.text.length > 80 ? `${f.text.slice(0, 80)}…` : f.text}」 ブロック {f.stream} の {f.streamOffset} 文字目:{' '}
                      {f.mapped ? (
                        <>
                          {STREAM_STATE_LABELS.included} <RangeLink r={r} range={f.mapped} onFocusRange={onFocusRange} />
                        </>
                      ) : f.inUnmappedSegment ? (
                        <span className="text-amber-300">{f.stream} のセグメントは含まれていますが、その中の位置は算出していません</span>
                      ) : f.streamState === 'not_included' ? (
                        <span className="text-text-tertiary">ブロックの {f.stream} がこの入力に含まれていないため、{f.stream} 経由では含まれていません</span>
                      ) : (
                        <span className="text-amber-300">
                          未確認（ブロックの {f.stream}: {STREAM_UNCHECKED_REASONS[f.streamState] ?? f.streamState}）
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      ))}
      {api && api.endRecorded && !api.isError && api.printedStdout === false && api.printedStderr === false && (
        <Notice>
          {next ? (
            !next.recorded ? (
              pendingAsset(next.attempt.request, assetStates) ? (
                <>
                  API 自身は出力していません。次のコード生成入力の内訳:{' '}
                  <DataAssetInline ev={next.attempt.request} states={assetStates} onRetry={onRetryAsset} />
                </>
              ) : (
                <>API 自身は出力していません。次のコード生成入力の内訳は{LABELS.notRecorded}です。</>
              )
            ) : next.stdout.state === 'included' ? (
              <>
                API 自身は出力していません。このブロックの stdout（{report.stdoutLen ?? '?'} 文字、生成コードによる print を含む）は、次のコード生成入力の{' '}
                {next.stdout.hits.map((h, i) => (
                  <span key={i} className="mr-1">
                    <RangeLink r={next} range={h} onFocusRange={onFocusRange} />
                  </span>
                ))}
                に含まれていました。この値がそこに含まれるかは確認していません。
              </>
            ) : (
              <>
                API 自身は出力していません。{stdoutInNextText(next.stdout, report.stdoutLen)}
                {next.stdout.state !== 'empty' && next.stdout.state !== 'not_recorded' ? 'この値がそこに含まれるかは確認していません。' : ''}
              </>
            )
          ) : (
            <>
              API 自身は出力していません。{LABELS.noLaterModelCall}
              {running ? '（エピソード実行中）' : ''}。
            </>
          )}
        </Notice>
      )}
      {report.requests.length > 0 && !next && (
        <Notice>
          {LABELS.noLaterModelCall}（コード生成）
          {running ? '（エピソード実行中のため、今後追加される可能性があります）' : ''}
        </Notice>
      )}
    </div>
  );
}

/** The API printed, but no position was mapped: say why from the stream checks. */
function ApiOutputState({ r, api }: { r: LaterRequestReport; api: NonNullable<LaterInputReport['api']> }) {
  const streams: ('stdout' | 'stderr')[] = [];
  if (api.printedStdout) streams.push('stdout');
  if (api.printedStderr) streams.push('stderr');
  return (
    <>
      {streams.map((st) => {
        const check = st === 'stdout' ? r.stdout : r.stderr;
        return (
          <span key={st} className={`mr-2 ${stateTone(check.state)}`}>
            {st}:{' '}
            {check.state === 'not_included'
              ? `ブロックの ${st} がこの入力に含まれていないため、含まれていません`
              : check.state === 'included'
                ? '位置を算出できません'
                : `未確認（${STREAM_UNCHECKED_REASONS[check.state] ?? check.state}）`}
          </span>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Feedback built for the next input
// ---------------------------------------------------------------------------

function FeedbackInput({ model, ep, selection, onSelect }: TabProps) {
  const ev = selection.node.primary;
  const d = dataOf(ev, 'feedback_built');
  const next =
    typeof d.turn_index === 'number'
      ? model.calls.find((c) => c.role === 'code_generator' && c.turnIndex === d.turn_index && c.node.seq > ev.seq) ?? null
      : null;
  const vdm = d.vdm_model_event_id ? model.callOfEvent.get(d.vdm_model_event_id) ?? null : null;
  const textsArt = (ev.artifacts ?? []).find((a) => a.label === 'text' || a.label === 'payload') ?? null;
  const payloadPath =
    d.texts_path ?? d.payload_path ?? (textsArt && textsArt.written !== false ? textsArt.path ?? null : null);
  const textsMissing = !payloadPath && textsArt && textsArt.written === false ? textsArt.reason ?? '理由不明' : null;
  const user = useFullText(ep, d.user_feedback ?? null);
  const segFile = useAsync(
    payloadPath ? () => fetchJsonFile<unknown>(episodeFileUrl(ep, payloadPath)) : null,
    [payloadPath, ep.runId, ep.episodeId, ep.generation ?? null],
  );
  return (
    <div className="space-y-3">
      <Notice>
        ハーネスが次の入力のために組み立てた内容です（{d.builder ?? 'builder 不明'}）。実際に送られた内容は次のコード生成呼び出しの入力で確認できます。
      </Notice>
      <KV
        rows={[
          ['turn_index', String(d.turn_index ?? '—')],
          ['次のコード生成呼び出し', next ? <EventLink onClick={() => onSelect(next.node)}>#{next.node.seq} を開く</EventLink> : '見つかりません'],
          ['VDM 呼び出し', vdm ? <EventLink onClick={() => onSelect(vdm.node)}>#{vdm.node.seq} を開く</EventLink> : '—'],
          ['画像 pixel sha256', Array.isArray(d.image_pixel_sha256) ? d.image_pixel_sha256.map((s) => shortSha(s)).join(', ') : shortSha(d.image_pixel_sha256 ?? null)],
        ]}
      />
      <SectionTitle>セグメント</SectionTitle>
      {(d.segments ?? []).length === 0 && <div className="text-xs text-text-tertiary">なし</div>}
      {(d.segments ?? []).map((s, i) => (
        <div key={i} className="space-y-1">
          <FeedbackSegmentRow s={s} model={model} onSelect={onSelect} />
          {(s.contains ?? []).length > 0 && (
            <div className="pl-4 border-l border-surface-border space-y-1">
              <div className="text-[11px] text-text-tertiary">この文字列に含まれていた、記録済みのテキスト（完全一致）:</div>
              {(s.contains ?? []).map((c, j) => (
                <FeedbackSegmentRow key={j} s={c} model={model} onSelect={onSelect} />
              ))}
            </div>
          )}
        </div>
      ))}
      {d.user_feedback !== undefined && d.user_feedback !== null && (
        <>
          <SectionTitle>ユーザーのフィードバック（Web UI）</SectionTitle>
          {user.text !== null ? <TextView text={user.text} /> : <NotRecorded />}
        </>
      )}
      <SectionTitle>ビルダーに渡されたテキスト（保存ファイル）</SectionTitle>
      {payloadPath ? (
        <>
          {segFile.loading && <Spinner />}
          {segFile.error && <ErrorText>{segFile.error}</ErrorText>}
          {segFile.data !== null && <SegmentTexts value={segFile.data} model={model} onSelect={onSelect} />}
        </>
      ) : (
        <NotRecorded reason={textsMissing ? `保存に失敗しました: ${textsMissing}` : 'texts_path がありません'} />
      )}
      {resolveText(d.user_feedback).truncated && user.loading && <Spinner />}
    </div>
  );
}

function FeedbackSegmentRow({ s, model, onSelect }: { s: FeedbackSegment; model: EpisodeModel; onSelect: TabProps['onSelect'] }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <Badge tone="neutral">{s.kind ?? '?'}</Badge>
      <span className="font-mono text-text-secondary">
        {s.length ?? '?'} 字 · sha256 {shortSha(s.text_sha256)}
      </span>
      {(s.source_event_ids ?? [])
        .filter((sid): sid is string => typeof sid === 'string')
        .slice(0, 4)
        .map((sid) => (
          <SourceLink key={sid} id={sid} model={model} onSelect={onSelect} />
        ))}
    </div>
  );
}

/** The builder's text arguments as the recorder stores them (the trace recorder). */
const FEEDBACK_TEXT_LABELS: Record<string, string> = {
  complete_multi_turn_prompt: 'マルチターン用プロンプト（complete_multi_turn_prompt）',
  visual_differencing_feedback: 'VDM の説明（visual_differencing_feedback）',
  user_feedback: 'ユーザーのフィードバック（user_feedback）',
};

/**
 * Texts of a feedback_built file: the recorder's {complete_multi_turn_prompt,
 * visual_differencing_feedback, user_feedback}, or {segments: [{kind, text,
 * source_event_ids}]}; any other shape as JSON.
 */
function SegmentTexts({ value, model, onSelect }: { value: unknown; model: EpisodeModel; onSelect: TabProps['onSelect'] }) {
  const obj = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
  if (obj && !Array.isArray(obj.segments) && Object.keys(obj).length > 0 && Object.values(obj).every((v) => v === null || typeof v === 'string')) {
    return (
      <div className="space-y-2">
        <div className="text-[11px] text-text-tertiary">
          ハーネスのビルダーが受け取った文字列です。モデルに送られた本文は次のコード生成呼び出しの「実際のモデル入力」で確認してください。
        </div>
        {Object.entries(obj).map(([k, v]) => (
          <div key={k}>
            <div className="mb-1 text-xs text-text-secondary">{FEEDBACK_TEXT_LABELS[k] ?? k}</div>
            {typeof v === 'string' ? <TextView text={v} foldLines={20} /> : <div className="text-xs text-text-tertiary">なし（None）</div>}
          </div>
        ))}
      </div>
    );
  }
  const segs =
    obj && Array.isArray(obj.segments)
      ? ((obj.segments as unknown[]).filter((x) => x && typeof x === 'object') as { kind?: unknown; text?: unknown; source_event_ids?: unknown }[])
      : null;
  if (!segs) return <RawJson value={value} />;
  return (
    <div className="space-y-2">
      {segs.map((sg, i) => (
        <div key={i}>
          <div className="flex flex-wrap items-center gap-2 mb-1 text-xs">
            <Badge tone="neutral">{typeof sg.kind === 'string' ? sg.kind : '?'}</Badge>
            {Array.isArray(sg.source_event_ids) &&
              (sg.source_event_ids as unknown[])
                .filter((x): x is string => typeof x === 'string')
                .slice(0, 4)
                .map((sid) => <SourceLink key={sid} id={sid} model={model} onSelect={onSelect} />)}
          </div>
          {typeof sg.text === 'string' ? <TextView text={sg.text} foldLines={20} /> : <RawJson value={sg} />}
        </div>
      ))}
    </div>
  );
}
