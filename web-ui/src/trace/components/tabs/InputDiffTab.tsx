/**
 * Tab 5 「入力差分」 (docs/trace-format.md 9.3).
 *
 * Compares the selected request with the previous request of the same role
 * and call-kind family in the same episode (code-generator calls with
 * code-generator calls, VDM calls with VDM calls; see model.callFamily).
 * Messages are aligned with a diff over (role, text); unmatched messages of
 * the same role in the same gap are shown as changed with a line diff computed
 * in the browser (diff.ts, Myers). Images are compared by sha256: added,
 * removed and unchanged. A request whose image list was not recorded (body not
 * parsed) is shown as 「未記録」, never as a request without images; one whose
 * data is in an asset (line over 256 KB) that is loading or failed says so,
 * with a retry.
 *
 * Related: diff.ts, payload.ts, model.ts.
 */

import { useMemo } from 'react';
import { episodeFileUrl, fetchJsonFile } from '../../api';
import { diffLines, diffSequences, toHunks, type DiffOp } from '../../diff';
import { shortSha } from '../../format';
import { useAsync } from '../../hooks';
import { LABELS } from '../../labels';
import { pendingAsset, type DataAssetState } from '../../dataAsset';
import { callFamily, mainAttempt, previousSameFamily, type LogicalCall } from '../../model';
import { messageText, parsePayload, type PayloadMessage } from '../../payload';
import { dataOf, type ImageInfo, type TraceEvent } from '../../types';
import { Badge, EventLink, NotRecorded, Notice, SectionTitle, Spinner } from '../common';
import { NO_ASSET_STATES } from '../DataAssetState';
import { TraceImage } from '../TraceImage';
import { roleTone } from '../Timeline';
import type { TabProps } from '../DetailTabs';
import type { EpisodeRef } from '../ValueView';

export function InputDiffTab({ model, ep, selection, onSelect, assetStates = NO_ASSET_STATES, onRetryAsset }: TabProps) {
  const node = selection.node;
  if (node.kind !== 'model_call' || !node.call) {
    return <Notice>入力差分はモデル呼び出しを選ぶと表示されます。</Notice>;
  }
  const call = node.call;
  const attempt = selection.attempt ?? mainAttempt(call);
  const prev = previousSameFamily(model, call);
  const family = callFamily(call);
  const httpAttempt = attempt ? dataOf(attempt.request, 'model_request').http_attempt ?? 1 : 1;
  const first = call.attempts[0]?.request ?? null;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-text-tertiary">比較する系列:</span>
        <Badge tone={roleTone(call.role)}>{family}</Badge>
        {prev ? (
          <>
            <span className="text-text-tertiary">前回:</span>
            <EventLink onClick={() => onSelect(prev.node)}>
              {prev.role} {prev.callKind} {prev.turnIndex !== null ? `ターン ${prev.turnIndex}` : ''} (#{prev.node.seq})
            </EventLink>
            <span className="text-text-tertiary">→ 今回 #{call.node.seq}</span>
          </>
        ) : null}
      </div>
      {attempt && httpAttempt > 1 && first && (
        <Notice>
          この要求は同じ論理呼び出しの再試行（attempt {httpAttempt}）です。本文 sha256:{' '}
          {(() => {
            const a = dataOf(first, 'model_request').body_sha256;
            const b = dataOf(attempt.request, 'model_request').body_sha256;
            if (!a || !b) return '比較できません（記録なし）';
            return a === b ? 'attempt 1 と一致' : 'attempt 1 と異なる';
          })()}
        </Notice>
      )}
      {!prev ? (
        <Notice>同じ役割・種類（{family}）の前回の呼び出しはこのエピソードにありません。</Notice>
      ) : !attempt ? (
        <NotRecorded />
      ) : (
        <DiffBody ep={ep} prev={prev} cur={attempt.request} assetStates={assetStates} onRetryAsset={onRetryAsset} />
      )}
    </div>
  );
}

function usePayload(ep: EpisodeRef, req: TraceEvent | null) {
  const path = req ? dataOf(req, 'model_request').payload_path ?? (req.artifacts ?? []).find((a) => a.label === 'payload' && a.written !== false)?.path ?? null : null;
  const url = path ? episodeFileUrl(ep, path) : null;
  return useAsync(url ? () => fetchJsonFile<unknown>(url) : null, [url]);
}

type MsgEntry =
  | { type: 'same'; a: PayloadMessage; b: PayloadMessage }
  | { type: 'changed'; a: PayloadMessage; b: PayloadMessage }
  | { type: 'added'; b: PayloadMessage }
  | { type: 'removed'; a: PayloadMessage };

function alignMessages(a: PayloadMessage[], b: PayloadMessage[]): MsgEntry[] {
  const keyOf = (m: PayloadMessage) => `${m.role}\u0000${messageText(m)}`;
  const res = diffSequences(a.map(keyOf), b.map(keyOf), 400);
  const out: MsgEntry[] = [];
  let dels: PayloadMessage[] = [];
  let ins: PayloadMessage[] = [];
  const flush = () => {
    const used = new Set<number>();
    for (const d of dels) {
      const j = ins.findIndex((m, idx) => !used.has(idx) && m.role === d.role);
      if (j >= 0) {
        used.add(j);
        out.push({ type: 'changed', a: d, b: ins[j] });
      } else {
        out.push({ type: 'removed', a: d });
      }
    }
    ins.forEach((m, idx) => {
      if (!used.has(idx)) out.push({ type: 'added', b: m });
    });
    dels = [];
    ins = [];
  };
  for (const op of res.ops) {
    if (op.type === 'equal') {
      flush();
      out.push({ type: 'same', a: a[op.aIndex!], b: b[op.bIndex!] });
    } else if (op.type === 'delete') {
      dels.push(a[op.aIndex!]);
    } else {
      ins.push(b[op.bIndex!]);
    }
  }
  flush();
  return out;
}

function DiffBody({
  ep,
  prev,
  cur,
  assetStates,
  onRetryAsset,
}: {
  ep: EpisodeRef;
  prev: LogicalCall;
  cur: TraceEvent;
  assetStates: ReadonlyMap<string, DataAssetState>;
  onRetryAsset?: (ev: TraceEvent) => void;
}) {
  const prevReq = mainAttempt(prev)?.request ?? null;
  const pa = usePayload(ep, prevReq);
  const pb = usePayload(ep, cur);
  const entries = useMemo(() => {
    if (pa.data === null || pb.data === null) return null;
    return alignMessages(parsePayload(pa.data).messages, parsePayload(pb.data).messages);
  }, [pa.data, pb.data]);
  // null: the request's image list was not recorded (not "no images").
  const imagesOf = (req: TraceEvent | null): ImageInfo[] | null => {
    const v = req ? dataOf(req, 'model_request').images : null;
    return Array.isArray(v) ? v : null;
  };
  const imgsA = imagesOf(prevReq);
  const imgsB = imagesOf(cur);

  const counts = useMemo(() => {
    const c = { same: 0, changed: 0, added: 0, removed: 0 };
    for (const e of entries ?? []) c[e.type] += 1;
    return c;
  }, [entries]);

  return (
    <div className="space-y-3">
      <SectionTitle>メッセージ</SectionTitle>
      {(pa.loading || pb.loading) && <Spinner label="payload を読み込み中" />}
      {(pa.error || pb.error) && <NotRecorded reason={`payload を読み込めません: ${pa.error ?? pb.error}`} />}
      {!prevReq && <NotRecorded reason="前回の要求が記録されていません" />}
      {prevReq && !pa.loading && pa.data === null && !pa.error && <NotRecorded reason="前回の payload がありません" />}
      {!pb.loading && pb.data === null && !pb.error && <NotRecorded reason="今回の payload がありません" />}
      {entries && (
        <>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge tone="green">追加 {counts.added}</Badge>
            <Badge tone="red">削除 {counts.removed}</Badge>
            <Badge tone="amber">変更 {counts.changed}</Badge>
            <Badge tone="neutral">同一 {counts.same}</Badge>
          </div>
          <div className="space-y-2">
            {entries.map((e, i) => (
              <MessageDiff key={i} entry={e} />
            ))}
          </div>
        </>
      )}
      <SectionTitle>画像・動画（sha256 で比較）</SectionTitle>
      <ImageDiff ep={ep} a={imgsA} b={imgsB} reqA={prevReq} reqB={cur} assetStates={assetStates} onRetryAsset={onRetryAsset} />
    </div>
  );
}

function MessageDiff({ entry }: { entry: MsgEntry }) {
  if (entry.type === 'same') {
    return (
      <div className="flex items-center gap-2 px-2 py-1 rounded border border-surface-border text-[11px] text-text-tertiary">
        <Badge tone="neutral">{entry.b.role}</Badge>
        <span className="font-mono">
          {entry.a.path} → {entry.b.path}
        </span>
        同一（{messageText(entry.b).length} 文字）
      </div>
    );
  }
  if (entry.type === 'added' || entry.type === 'removed') {
    const m = entry.type === 'added' ? entry.b : entry.a;
    const text = messageText(m);
    return (
      <div className={`rounded border ${entry.type === 'added' ? 'border-nv-green/40' : 'border-red-500/40'}`}>
        <div className="flex items-center gap-2 px-2 py-1 text-[11px]">
          <Badge tone={entry.type === 'added' ? 'green' : 'red'}>{entry.type === 'added' ? '追加' : '削除'}</Badge>
          <Badge tone="neutral">{m.role}</Badge>
          <span className="font-mono text-text-tertiary">{m.path}</span>
        </div>
        <pre
          className={`px-3 py-2 text-[11px] font-mono whitespace-pre-wrap break-all max-h-72 overflow-auto ${
            entry.type === 'added' ? 'bg-nv-green/5 text-nv-green-light' : 'bg-red-500/5 text-red-300'
          }`}
        >
          {text}
        </pre>
      </div>
    );
  }
  return <ChangedMessage a={entry.a} b={entry.b} />;
}

function ChangedMessage({ a, b }: { a: PayloadMessage; b: PayloadMessage }) {
  const result = useMemo(() => diffLines(messageText(a), messageText(b)), [a, b]);
  const { hunks, foldedAfter } = useMemo(() => toHunks(result.ops, 3), [result]);
  return (
    <div className="rounded border border-amber-500/40">
      <div className="flex items-center gap-2 px-2 py-1 text-[11px]">
        <Badge tone="amber">変更</Badge>
        <Badge tone="neutral">{b.role}</Badge>
        <span className="font-mono text-text-tertiary">
          {a.path} → {b.path}
        </span>
        <span className="text-nv-green">+{result.added}</span>
        <span className="text-red-400">-{result.removed}</span>
        {result.gaveUp && <Badge tone="amber">差分が大きいため行単位の対応付けを省略</Badge>}
      </div>
      <div className="font-mono text-[11px] bg-surface-sunken max-h-[28rem] overflow-auto">
        {hunks.map((h, i) => (
          <div key={i}>
            {h.foldedBefore > 0 && <div className="px-3 py-0.5 text-text-muted bg-surface-overlay/60">… {h.foldedBefore} 行同一</div>}
            {h.ops.map((op, j) => (
              <DiffLine key={j} op={op} />
            ))}
          </div>
        ))}
        {foldedAfter > 0 && <div className="px-3 py-0.5 text-text-muted bg-surface-overlay/60">… {foldedAfter} 行同一</div>}
      </div>
    </div>
  );
}

function DiffLine({ op }: { op: DiffOp }) {
  const cls = op.type === 'insert' ? 'bg-nv-green/10 text-nv-green-light' : op.type === 'delete' ? 'bg-red-500/10 text-red-300' : 'text-text-secondary';
  const sign = op.type === 'insert' ? '+' : op.type === 'delete' ? '-' : ' ';
  return (
    <div className={`flex ${cls}`}>
      <span className="w-10 flex-shrink-0 text-right pr-1 text-text-muted select-none">{op.aIndex !== null ? op.aIndex + 1 : ''}</span>
      <span className="w-10 flex-shrink-0 text-right pr-1 text-text-muted select-none">{op.bIndex !== null ? op.bIndex + 1 : ''}</span>
      <span className="w-4 flex-shrink-0 select-none">{sign}</span>
      <span className="whitespace-pre-wrap break-all pr-3">{op.text || ' '}</span>
    </div>
  );
}

function ImageDiff({
  ep,
  a: aList,
  b: bList,
  reqA = null,
  reqB = null,
  assetStates = NO_ASSET_STATES,
  onRetryAsset,
}: {
  ep: EpisodeRef;
  a: ImageInfo[] | null;
  b: ImageInfo[] | null;
  /** The two requests: one whose data is in an asset (line over 256 KB) has no image list until it is merged. */
  reqA?: TraceEvent | null;
  reqB?: TraceEvent | null;
  assetStates?: ReadonlyMap<string, DataAssetState>;
  onRetryAsset?: (ev: TraceEvent) => void;
}) {
  if (aList === null || bList === null) {
    const which = aList === null && bList === null ? '前回・今回とも' : aList === null ? '前回の要求' : '今回の要求';
    const side = [
      { list: aList, ev: reqA, label: '前回の要求' },
      { list: bList, ev: reqB, label: '今回の要求' },
    ]
      .filter((x) => x.list === null && x.ev)
      .map((x) => ({ ...x, pending: pendingAsset(x.ev as TraceEvent, assetStates) }))
      .find((x) => x.pending !== null);
    if (side && side.pending && side.pending.kind !== 'unwritten') {
      const ev = side.ev as TraceEvent;
      const p = side.pending;
      return (
        <Notice tone={p.kind === 'failed' ? 'red' : 'gold'}>
          {side.label}の data は 256 KB を超えたため別ファイルにあり、
          {p.kind === 'failed' ? `読み込めません（${p.error}）` : p.kind === 'idle' ? 'まだ読み込まれていません（自動では読み込みません）' : 'まだ読み込まれていません（読み込み中）'}
          。追加・削除は判定できません
          {(p.kind === 'failed' || p.kind === 'idle') && onRetryAsset && (
            <button type="button" onClick={() => onRetryAsset(ev)} className="ml-2 underline underline-offset-2">
              {p.kind === 'failed' ? '再読み込み' : '読み込む'}
            </button>
          )}
        </Notice>
      );
    }
    return (
      <NotRecorded
        reason={
          side && side.pending && side.pending.kind === 'unwritten'
            ? `${side.label}の data は 256 KB を超えたため別ファイルに移されましたが、そのファイルは書かれていません${side.pending.reason ? `（${side.pending.reason}）` : ''}。追加・削除は判定できません`
            : `${which}の画像の記録がありません（送信本文を解析していない）。追加・削除は判定できません`
        }
      />
    );
  }
  const a = aList;
  const b = bList;
  const key = (im: ImageInfo) => im.sha256 ?? im.url ?? im.path ?? '';
  const setA = new Set(a.map(key));
  const setB = new Set(b.map(key));
  const added = b.filter((im) => !setA.has(key(im)));
  const removed = a.filter((im) => !setB.has(key(im)));
  const same = b.filter((im) => setA.has(key(im)));
  const cell = (im: ImageInfo, i: number, tone: 'green' | 'red' | 'neutral') => (
    <div key={`${key(im)}-${i}`} className="w-44 space-y-1">
      <TraceImage
        src={im.asset ? episodeFileUrl(ep, im.asset) : null}
        alt={im.path ?? ''}
        provenance="model_input"
        video={(im.mime ?? '').startsWith('video/')}
        compact
        className="h-28 w-44"
      />
      <div className="text-[10px] font-mono text-text-tertiary">
        <Badge tone={tone}>{tone === 'green' ? '追加' : tone === 'red' ? '削除' : '変化なし'}</Badge> #{im.order ?? '?'} {shortSha(im.sha256)}
      </div>
    </div>
  );
  if (a.length === 0 && b.length === 0) return <div className="text-xs text-text-tertiary">どちらも{LABELS.noImageInput}</div>;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2 text-xs">
        <Badge tone="green">追加 {added.length}</Badge>
        <Badge tone="red">削除 {removed.length}</Badge>
        <Badge tone="neutral">変化なし {same.length}</Badge>
      </div>
      <div className="flex flex-wrap gap-3">
        {added.map((im, i) => cell(im, i, 'green'))}
        {removed.map((im, i) => cell(im, i, 'red'))}
        {same.map((im, i) => cell(im, i, 'neutral'))}
      </div>
    </div>
  );
}
