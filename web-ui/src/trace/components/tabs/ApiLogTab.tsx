/**
 * Tab 3 「API/実行ログ」 (docs/trace-format.md 9.3, 5.2, 5.3).
 *
 * API calls: arguments and return value (arrays with shape, dtype, stats and
 * a preview; units from the recorder's table or 「単位・座標系: 不明」), the
 * block's stdout/stderr with this API's own slice highlighted, exception and
 * traceback, duration, simulator steps, and links to the code block (call
 * line marked) and to the model call that produced the code. Code blocks,
 * service calls and stdout/stderr events get their execution records.
 * Exception messages over 4000 characters are text assets and are loaded on
 * demand like the other long texts.
 *
 * Related: ValueView.tsx, TextView.tsx, model.ts.
 */

import { useMemo } from 'react';
import { fmtDuration } from '../../format';
import { useFullText } from '../../hooks';
import { CUT_TEXT_NOTE, REDACTED_MESSAGE_NOTE, REDACTED_SHORT, arrayBudgetLabel } from '../../labels';
import { apiDepth, arrayBudgetOf, enclosingBlock, nestedModelCalls, nodeDuration, type TNode } from '../../model';
import { codePointToUtf16 } from '../../offsets';
import { dataOf, type OutputSlice, type TextValue } from '../../types';
import { Badge, EventLink, KV, Notice, NotRecorded, SectionTitle, Spinner } from '../common';
import { CodeView } from '../CodeView';
import { TextView, type HighlightRange } from '../TextView';
import { roleTone } from '../Timeline';
import { ArgUnits, ReturnUnits, ValueView, type EpisodeRef } from '../ValueView';
import type { TabProps } from '../DetailTabs';

export function ApiLogTab(props: TabProps) {
  const node = props.selection.node;
  switch (node.kind) {
    case 'api_call':
      return <ApiCallLog {...props} node={node} />;
    case 'code_block':
      return <BlockLog {...props} block={node} />;
    case 'service_call':
      return <ServiceLog {...props} node={node} />;
    case 'stdout':
    case 'stderr':
      return <StreamLog ep={props.ep} node={node} />;
    case 'model_call':
      return <Notice>モデル呼び出しです。HTTP の試行と応答は「モデル出力/コード」タブにあります。</Notice>;
    default:
      return <Notice>このイベントには API の実行記録がありません。</Notice>;
  }
}

function sliceRange(text: string, slice: OutputSlice | null | undefined, cls: string, title: string): HighlightRange[] {
  if (!slice || typeof slice.start !== 'number' || typeof slice.end !== 'number' || slice.end <= slice.start) return [];
  return [{ start: codePointToUtf16(text, slice.start), end: codePointToUtf16(text, slice.end), className: cls, title, id: 'api-slice' }];
}

function StreamWithSlice({ ep, value, slice, stream, title }: { ep: EpisodeRef; value: TextValue | null | undefined; slice: OutputSlice | null | undefined; stream: 'stdout' | 'stderr'; title: string }) {
  const t = useFullText(ep, value ?? null);
  const ranges = useMemo(
    () => (t.text !== null ? sliceRange(t.text, slice, 'bg-accent/25 outline outline-1 outline-accent/60', 'この API 自身の出力') : []),
    [t.text, slice],
  );
  if (value === null || value === undefined) {
    return (
      <div>
        <div className="text-[11px] text-text-tertiary mb-1">{title}</div>
        <div className="text-xs text-text-tertiary">（{stream} イベントなし）</div>
      </div>
    );
  }
  return (
    <div>
      <div className="flex items-center gap-2 text-[11px] text-text-tertiary mb-1">
        {title}
        {ranges.length > 0 && <Badge tone="gold">強調部分 = この API の出力（{slice?.start}–{slice?.end}）</Badge>}
        {t.loading && <Spinner />}
      </div>
      {t.text === '' ? (
        <div className="text-xs text-text-tertiary">（空）</div>
      ) : t.text !== null ? (
        <TextView
          text={t.text}
          ranges={ranges}
          tone={stream === 'stderr' ? 'stderr' : 'default'}
          focusRangeId={ranges.length ? 'api-slice' : null}
          scrollToFocus={false}
          foldLines={40}
        />
      ) : (
        <NotRecorded />
      )}
      {t.partial && !t.loading && <div className="text-[11px] text-amber-300">先頭のみ（全文を読み込めませんでした{t.error ? `: ${t.error}` : ''}）</div>}
    </div>
  );
}

function ApiCallLog({ model, ep, onSelect, node }: TabProps & { node: TNode }) {
  const d = dataOf(node.primary, 'api_call_start');
  const endEv = node.end;
  const ed = endEv ? dataOf(endEv, 'api_call_end') : null;
  const isErr = endEv?.event_type === 'api_call_error';
  const units = ed?.units ?? d.units;
  const block = enclosingBlock(model, node);
  const out = block?.children.find((c) => c.kind === 'stdout')?.primary ?? null;
  const err = block?.children.find((c) => c.kind === 'stderr')?.primary ?? null;
  const blockCode = block ? dataOf(block.primary, 'code_block_start').code : null;
  const code = useFullText(ep, blockCode ?? null);
  const nestedApis = node.children.filter((c) => c.kind === 'api_call');
  const services = node.children.filter((c) => c.kind === 'service_call');
  const vlm = nestedModelCalls(node);
  const args = d.args;
  const argObj = args && typeof args === 'object' && !Array.isArray(args) ? (args as Record<string, unknown>) : null;
  const argEntries = argObj ? Object.entries(argObj).filter(([k]) => k !== '<truncated>') : null;
  const argsTruncated = argObj ? argObj['<truncated>'] : undefined;
  const budget = arrayBudgetOf(node);

  return (
    <div className="space-y-3">
      {budget > 0 && <Notice tone="amber">{arrayBudgetLabel(budget)}。該当する値は「未取得（array_budget）」と表示されます。</Notice>}
      <KV
        rows={[
          ['name', d.qualname ?? d.name ?? '?'],
          ['api_class', d.api_class ?? '—'],
          ['depth', `${apiDepth(node)}${apiDepth(node) > 0 ? '（別の API の内部から呼ばれた）' : ''}`],
          ['privileged', String(d.privileged ?? '—')],
          ['状態', node.open ? '終了イベントなし' : isErr ? 'エラー' : 'ok'],
          ['所要時間', fmtDuration(nodeDuration(node))],
          ['sim_step', `${node.primary.sim_step ?? '?'} → ${endEv?.sim_step ?? '?'}${typeof ed?.sim_step_delta === 'number' ? `（Δ ${ed.sim_step_delta}）` : ''}`],
          ['録画バッファ', Array.isArray(ed?.buffer_frames) ? `${ed?.buffer_frames?.[0] ?? '?'} – ${ed?.buffer_frames?.[1] ?? '?'}` : '—'],
          ['api_call_id', d.api_call_id ?? node.primary.event_id],
        ]}
      />

      <SectionTitle>引数</SectionTitle>
      {argsTruncated !== undefined && <div className="text-[11px] text-amber-300">要約の上限に達したため、以降の引数は省略されています（{String(argsTruncated)}）</div>}
      {args === undefined ? (
        <NotRecorded />
      ) : argEntries ? (
        argEntries.length === 0 ? (
          <div className="text-xs text-text-tertiary">引数なし</div>
        ) : (
          <div className="space-y-1.5">
            {argEntries.map(([k, v]) => (
              <div key={k} className="flex gap-2 items-start">
                <span className="text-sky-300 font-mono text-xs flex-shrink-0 min-w-[6rem]">{k}</span>
                <div className="min-w-0 flex-1">
                  <ValueView value={v} ep={ep} />
                  <ArgUnits units={units} name={k} value={v} />
                </div>
              </div>
            ))}
          </div>
        )
      ) : (
        <ValueView value={args} ep={ep} />
      )}

      {!isErr && (
        <>
          <SectionTitle>戻り値</SectionTitle>
          {!endEv ? (
            <div className="text-xs text-text-tertiary">終了イベントがありません</div>
          ) : ed && 'return' in ed ? (
            <>
              <ValueView value={ed.return} ep={ep} />
              <ReturnUnits units={units} value={ed.return} />
            </>
          ) : (
            <NotRecorded />
          )}
        </>
      )}

      {isErr && ed && (
        <>
          <SectionTitle>例外</SectionTitle>
          <ExceptionMessage
            ep={ep}
            type={ed.error_type ?? null}
            message={ed.message ?? null}
            alarm={ed.alarm_timeout === true}
            redacted={ed.message_redacted === true}
            cut={ed.message_cut === true}
          />
          <TracebackView ep={ep} value={ed.traceback ?? null} redacted={ed.traceback_redacted === true} cut={ed.traceback_cut === true} />
        </>
      )}

      <SectionTitle>ブロックの stdout / stderr</SectionTitle>
      <div className="space-y-2">
        <StreamWithSlice ep={ep} value={out ? dataOf(out, 'stdout').text : null} slice={ed?.stdout_slice} stream="stdout" title="stdout（ブロック全体）" />
        <StreamWithSlice ep={ep} value={err ? dataOf(err, 'stderr').text : null} slice={ed?.stderr_slice} stream="stderr" title="stderr（ブロック全体）" />
      </div>

      <SectionTitle>コード内の呼び出し位置</SectionTitle>
      <div className="flex flex-wrap gap-3 text-xs">
        {block ? <EventLink onClick={() => onSelect(block)}>コードブロック #{dataOf(block.primary, 'code_block_start').block_index ?? '?'} を開く</EventLink> : <span className="text-text-tertiary">コードブロック不明</span>}
        {block?.sourceCall ? <EventLink onClick={() => onSelect(block.sourceCall!.node)}>このコードを生成したモデル呼び出し</EventLink> : null}
        {d.call_site?.line ? <span className="text-text-tertiary">行 {d.call_site.line}: <span className="font-mono">{d.call_site.source_line}</span></span> : null}
      </div>
      {code.text !== null && d.call_site?.line ? <CodeView code={code.text} markLine={d.call_site.line} maxLines={14} /> : null}

      {(nestedApis.length > 0 || services.length > 0 || vlm.length > 0) && <SectionTitle>内部の呼び出し</SectionTitle>}
      <div className="space-y-1 text-xs">
        {nestedApis.map((c) => (
          <div key={c.key} className="flex items-center gap-2">
            <Badge tone="sky">API</Badge>
            <EventLink onClick={() => onSelect(c)}>{dataOf(c.primary, 'api_call_start').name ?? '?'}</EventLink>
            <span className="text-text-tertiary">{fmtDuration(nodeDuration(c))}</span>
            {c.failed && <Badge tone="red">失敗</Badge>}
          </div>
        ))}
        {services.map((c) => (
          <div key={c.key} className="flex items-center gap-2">
            <Badge tone="violet">サービス</Badge>
            <EventLink onClick={() => onSelect(c)}>
              {dataOf(c.primary, 'service_call_start').service ?? '?'}.{dataOf(c.primary, 'service_call_start').function ?? ''}
            </EventLink>
            <span className="text-text-tertiary">{fmtDuration(nodeDuration(c))}</span>
            {c.degraded && <Badge tone="amber">degraded</Badge>}
            {c.failed && <Badge tone="red">失敗</Badge>}
          </div>
        ))}
        {vlm.map((c) => (
          <div key={c.id} className="flex items-center gap-2">
            <Badge tone={roleTone(c.role)}>{c.role}</Badge>
            <EventLink onClick={() => onSelect(c.node)}>知覚 VLM の入力と出力</EventLink>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * `<type>: str(e)`; a message over 4000 characters is a text asset, loaded here. `redacted`:
 * the recorder removed secrets from the stored text (`message_redacted`), so it differs from
 * what the program printed. `cut`: only the first 64 KB of the message were read
 * (`message_cut`).
 */
function ExceptionMessage({
  ep,
  type,
  message,
  alarm,
  redacted = false,
  cut = false,
}: {
  ep: EpisodeRef;
  type: string | null;
  message: TextValue | null;
  alarm: boolean;
  redacted?: boolean;
  cut?: boolean;
}) {
  const t = useFullText(ep, message);
  return (
    <Notice tone="red">
      <div className="font-mono text-xs whitespace-pre-wrap break-all">
        {type ?? 'Error'}: {t.text ?? ''}
      </div>
      {redacted && <div className="mt-1 text-[11px] text-amber-300">{REDACTED_MESSAGE_NOTE}</div>}
      {cut && <div className="mt-1 text-[11px] text-amber-300">{CUT_TEXT_NOTE}</div>}
      {t.partial && (
        <div className="mt-1 text-[11px] text-amber-300">
          {t.loading ? '全文を読み込み中…' : t.error ? `全文を読み込めません: ${t.error}` : `先頭のみ（全 ${t.length ?? '?'} 文字）`}
        </div>
      )}
      {alarm && <div className="mt-1 text-xs">wall-clock の alarm（TimeoutError）による中断です</div>}
    </Notice>
  );
}

/** The recorded traceback; `redacted` / `cut`: traceback_redacted / traceback_cut of the event. */
function TracebackView({ ep, value, redacted = false, cut = false }: { ep: EpisodeRef; value: TextValue | null; redacted?: boolean; cut?: boolean }) {
  const t = useFullText(ep, value);
  if (t.text === null) return <div className="text-xs text-text-tertiary">traceback は記録されていません</div>;
  return (
    <div>
      <div className="text-[11px] text-text-tertiary mb-1">traceback（レコーダーのフレームを除いたもの）</div>
      {redacted && <div className="text-[11px] text-amber-300 mb-1">traceback: {REDACTED_MESSAGE_NOTE}</div>}
      {cut && <div className="text-[11px] text-amber-300 mb-1">traceback: {CUT_TEXT_NOTE}</div>}
      <TextView text={t.text} tone="stderr" foldLines={40} />
    </div>
  );
}

function BlockLog({ ep, onSelect, block }: TabProps & { block: TNode }) {
  const ed = block.end ? dataOf(block.end, 'code_block_end') : null;
  const out = block.children.find((c) => c.kind === 'stdout')?.primary ?? null;
  const err = block.children.find((c) => c.kind === 'stderr')?.primary ?? null;
  const apis = block.children.filter((c) => c.kind === 'api_call');
  const budget = arrayBudgetOf(block);
  return (
    <div className="space-y-3">
      {budget > 0 && <Notice tone="amber">step() の結果（result_summary）: {arrayBudgetLabel(budget)}</Notice>}
      {ed?.tracer_frame_visible && (
        <Notice tone="red">レコーダーのパスがこのブロックの stdout/stderr に現れています（tracer_frame_visible）。本来は起きないはずの状態です。</Notice>
      )}
      <KV
        rows={[
          ['sandbox_rc', ed ? String(ed.sandbox_rc ?? '—') : '終了イベントなし'],
          ['所要時間', fmtDuration(ed?.duration_s ?? null)],
          ['sim_step の範囲', Array.isArray(ed?.sim_step_range) ? `${ed?.sim_step_range?.[0] ?? '?'} – ${ed?.sim_step_range?.[1] ?? '?'}` : '—'],
          ['stdout / stderr の長さ', `${ed?.stdout_len ?? '?'} / ${ed?.stderr_len ?? '?'} 文字`],
          ['code_block_id', dataOf(block.primary, 'code_block_start').code_block_id ?? block.primary.event_id],
        ]}
      />
      {ed?.result_summary !== undefined && (
        <>
          <SectionTitle>step() の結果（result_summary）</SectionTitle>
          <ValueView value={ed.result_summary} ep={ep} />
        </>
      )}
      <SectionTitle>このブロックの API 呼び出し（depth 0）</SectionTitle>
      {apis.length === 0 && <div className="text-xs text-text-tertiary">記録された API 呼び出しはありません（APIS[...] 以外の経路の呼び出しは記録されません）</div>}
      <div className="space-y-0.5 text-xs">
        {apis.map((c) => {
          const cd = dataOf(c.primary, 'api_call_start');
          return (
            <div key={c.key} className="flex items-center gap-2">
              <EventLink onClick={() => onSelect(c)}>{cd.name ?? '?'}</EventLink>
              {cd.call_site?.line ? <span className="text-text-tertiary">行 {cd.call_site.line}</span> : null}
              <span className="text-text-tertiary">{fmtDuration(nodeDuration(c))}</span>
              {c.open ? <Badge tone="gold">終了なし</Badge> : c.failed ? <Badge tone="red">失敗</Badge> : <Badge tone="green">ok</Badge>}
            </div>
          );
        })}
      </div>
      <SectionTitle>stdout / stderr（step() が返したもの）</SectionTitle>
      <StreamWithSlice ep={ep} value={out ? dataOf(out, 'stdout').text : null} slice={null} stream="stdout" title="stdout" />
      <StreamWithSlice ep={ep} value={err ? dataOf(err, 'stderr').text : null} slice={null} stream="stderr" title="stderr" />
    </div>
  );
}

function ServiceLog({ ep, node }: TabProps & { node: TNode }) {
  const sd = dataOf(node.primary, 'service_call_start');
  const ed = node.end ? dataOf(node.end, 'service_call_end') : null;
  // The recorder learns the endpoint from the HTTP attempt, so it is on the end event.
  const epInfo = ed?.endpoint ?? sd.endpoint ?? null;
  const endpoint =
    typeof epInfo === 'string' ? epInfo : epInfo ? epInfo.url ?? `${epInfo.host ?? '?'}:${epInfo.port ?? '?'}${epInfo.path ?? ''}` : null;
  const attempts = ed?.http_attempts ?? sd.http_attempts ?? [];
  const arts = node.events.flatMap((e) => e.artifacts ?? []);
  const budget = arrayBudgetOf(node);
  return (
    <div className="space-y-3">
      {budget > 0 && <Notice tone="amber">{arrayBudgetLabel(budget)}。該当する値は「未取得（array_budget）」と表示されます。</Notice>}
      {node.degraded && (
        <Notice tone="amber">
          degraded: HTTP の試行が失敗しましたが、関数は値を返しました（SAM3 と Molmo は通信エラーを内部で握りつぶします）。
          {ed?.degraded_reason ? `（記録: ${ed.degraded_reason}）` : ''}
        </Notice>
      )}
      {node.orphan && <Notice tone="red">{node.orphan}</Notice>}
      <KV
        rows={[
          ['service', `${sd.service ?? ed?.service ?? '?'}.${sd.function ?? ed?.function ?? ''}`],
          ['endpoint', endpoint || '—'],
          ['latency', fmtDuration(ed?.latency_s ?? null)],
          ['状態', node.open ? '終了イベントなし' : node.end?.event_type === 'service_call_error' ? `エラー ${ed?.error_type ?? ''}` : 'ok'],
        ]}
      />
      {node.end?.event_type === 'service_call_error' && ed && (
        <ExceptionMessage ep={ep} type={ed.error_type ?? null} message={ed.message ?? null} alarm={false} redacted={ed.message_redacted === true} cut={ed.message_cut === true} />
      )}
      <SectionTitle>リクエストの要約</SectionTitle>
      {sd.request_summary === undefined ? <NotRecorded /> : <ValueView value={sd.request_summary} ep={ep} />}
      <SectionTitle>レスポンスの要約</SectionTitle>
      {ed?.response_summary === undefined ? <NotRecorded /> : <ValueView value={ed.response_summary} ep={ep} />}
      <SectionTitle>HTTP 試行</SectionTitle>
      {attempts.length === 0 && <div className="text-xs text-text-tertiary">記録なし</div>}
      {attempts.map((a, i) => (
        <div key={i} className="text-xs font-mono text-text-secondary">
          #{i + 1} status {a.status ?? '—'} · {fmtDuration(a.latency_s ?? null)} {a.error ? `· ${a.error}` : ''}
          {a.error && a.error_redacted === true ? `（${REDACTED_SHORT}）` : ''}
        </div>
      ))}
      {arts.length > 0 && (
        <>
          <SectionTitle>保存した入出力（デバッグ状態タブで表示）</SectionTitle>
          {arts.map((a, i) => (
            <div key={i} className="text-xs font-mono text-text-secondary">
              {a.label} · {a.path} {a.written === false ? `（未保存: ${a.reason ?? ''}）` : ''}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function StreamLog({ ep, node }: { ep: EpisodeRef; node: TNode }) {
  const d = dataOf(node.primary, 'stdout');
  return (
    <div className="space-y-2">
      <div className="text-[11px] text-text-tertiary">step() が返した {node.kind} そのもの（ブロック終了時に 1 回記録）</div>
      <StreamWithSlice ep={ep} value={d.text ?? null} slice={null} stream={node.kind === 'stderr' ? 'stderr' : 'stdout'} title={node.kind} />
    </div>
  );
}
