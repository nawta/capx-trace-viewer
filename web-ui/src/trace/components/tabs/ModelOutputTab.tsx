/**
 * Tab 2 「モデル出力/コード」 (docs/trace-format.md 9.3, 5.2).
 *
 * Model calls: the raw response text, reasoning only when the API returned it
 * (labelled with its field; never inferred), the parsed code blocks and the
 * harness's reading of the reply ("harness: no REGENERATE -> finish"; 「モデル
 * は完了と主張」 only when model_said_finish), usage, latency and every HTTP
 * attempt. A logical failure is the call's result even after an HTTP 200; a
 * failed or stopped stream shows the text received until then
 * (`partial_content`). Code blocks: the executed code. API calls: the code
 * with the call line marked, and a link to the model call.
 *
 * Related: model.ts, CodeView.tsx, ModelInputTab.tsx (RawJson).
 */

import { useState } from 'react';
import { episodeFileUrl, fetchJsonFile } from '../../api';
import { fmtDuration, shortSha } from '../../format';
import { useAsync, useFullText } from '../../hooks';
import { LABELS, REDACTED_MESSAGE_NOTE, REDACTED_SHORT, harnessReadingLabel, modelErrorLabel } from '../../labels';
import {
  attemptNumber,
  callOutcome,
  enclosingBlock,
  logicalErrorsOfAttempt,
  mainAttempt,
  okResponse,
  textOf,
  type Attempt,
  type LogicalCall,
  type TNode,
} from '../../model';
import { dataOf, type TextValue, type TraceEvent } from '../../types';
import { Badge, EventLink, KV, Notice, NotRecorded, SectionTitle, Spinner, ErrorText } from '../common';
import { CodeView } from '../CodeView';
import { TextView } from '../TextView';
import { roleTone } from '../Timeline';
import type { TabProps } from '../DetailTabs';
import type { EpisodeRef } from '../ValueView';
import { RawJson } from './ModelInputTab';

export function ModelOutputTab(props: TabProps) {
  const node = props.selection.node;
  if (node.kind === 'model_call' && node.call) return <ModelCallOutput {...props} call={node.call} />;
  if (node.kind === 'code_block') return <BlockCode {...props} block={node} markLine={null} />;
  if (node.kind === 'api_call') {
    const block = enclosingBlock(props.model, node);
    const line = dataOf(node.primary, 'api_call_start').call_site?.line ?? null;
    return block ? <BlockCode {...props} block={block} markLine={line} /> : <Notice>この API 呼び出しのコードブロックが記録にありません。</Notice>;
  }
  return <Notice>このイベントにはモデル出力もコードもありません。</Notice>;
}

function FullTextBlock({ ep, value, empty }: { ep: EpisodeRef; value: TextValue | null | undefined; empty: string }) {
  const t = useFullText(ep, value);
  if (t.text === null || t.text === '') return <div className="text-xs text-text-tertiary">{empty}</div>;
  return (
    <div>
      <TextView text={t.text} foldLines={60} />
      {t.partial && <div className="text-[11px] text-amber-300 mt-1">{t.loading ? '全文を読み込み中…' : t.error ? `全文を読み込めません: ${t.error}` : '先頭のみ'}</div>}
    </div>
  );
}

function ModelCallOutput({ model, ep, selection, running, onSelect, call }: TabProps & { call: LogicalCall }) {
  const attempt: Attempt | null = selection.attempt ?? mainAttempt(call);
  // Only the selected attempt's own response: a response of another HTTP attempt is never
  // shown as if this attempt had received it.
  const resp = attempt?.responses[0] ?? null;
  const rd = resp ? dataOf(resp, 'model_response') : null;
  const outcome = callOutcome(call);
  const answer = okResponse(call);
  const httpAnswer = outcome.httpAnswer;
  const answerAttempt = answer ? call.attempts.find((a) => a.responses.includes(answer)) ?? null : null;
  const attemptNo = (a: Attempt | null) => (a ? attemptNumber(call, a) : null);
  const ownLogical = attempt ? logicalErrorsOfAttempt(call, attempt) : [];
  // Logical errors shown as this attempt's result are not listed again under the attempts table.
  const shownAsAttemptResult = !resp && attempt && attempt.errors.length === 0 ? ownLogical : [];
  const streamedRequest = attempt ? dataOf(attempt.request, 'model_request').stream === true : false;
  const attemptOpen =
    !!attempt && attempt.responses.length === 0 && attempt.errors.length === 0 && ownLogical.length === 0 && outcome.state === 'open' && attempt === call.attempts[call.attempts.length - 1];
  const isAnswer = !!resp && resp === answer;
  const failedAfterThisResponse = !!resp && resp === httpAnswer && outcome.finalError !== null;
  const [showRaw, setShowRaw] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge tone={roleTone(call.role)}>{call.role}</Badge>
        {call.callKind && <Badge tone="neutral">{call.callKind}</Badge>}
        <span className="text-xs font-mono text-text-secondary">{rd?.response_model ?? call.model ?? ''}</span>
        {call.attempts.length > 1 && attempt && <Badge tone="amber">HTTP 試行 {attemptNo(attempt)} / {call.attempts.length}</Badge>}
        {outcome.state === 'failed' && <Badge tone={outcome.stopped ? 'amber' : 'red'}>{outcome.stopped ? '論理呼び出しは中断' : '論理呼び出しは失敗'}</Badge>}
      </div>
      {outcome.state === 'failed' && (
        <CallFailure
          call={call}
          stopped={outcome.stopped}
          httpAnswerStatus={httpAnswer ? dataOf(httpAnswer, 'model_response').http_status ?? null : null}
        />
      )}

      <SectionTitle>応答本文（生）{call.attempts.length > 1 && attempt ? `: attempt ${attemptNo(attempt)}` : ''}</SectionTitle>
      {!resp ? (
        <div className="space-y-2">
          {attempt && attempt.errors.length > 0 ? (
            attempt.errors.map((e) => <ErrorView key={e.event_id} ep={ep} ev={e} title={`この試行（attempt ${attemptNo(attempt)}）は応答を受け取っていません`} />)
          ) : ownLogical.length > 0 ? (
            ownLogical.map((e) => <ErrorView key={e.event_id} ep={ep} ev={e} title={`この試行（attempt ${attemptNo(attempt)}）の結果`} />)
          ) : attemptOpen ? (
            <Notice>応答はまだ記録されていません（{running ? LABELS.openWhileRunning : LABELS.openProcessEnded}）。</Notice>
          ) : streamedRequest ? (
            <Notice>
              ストリーミング要求の HTTP 試行ごとの応答は記録していません（応答はストリームの最後に論理呼び出しの結果として記録されます）。
              {outcome.state === 'failed' ? 'この論理呼び出しの結果は上の「論理呼び出しは失敗」です。' : ''}
            </Notice>
          ) : (
            <NotRecorded reason="この試行の model_response がありません（記録の欠落）" />
          )}
        </div>
      ) : (
        <>
          <FullTextBlock ep={ep} value={rd?.content} empty="（空の応答）" />
          {failedAfterThisResponse && (
            <Notice tone="red">
              この試行は HTTP {rd?.http_status ?? '2xx'} の応答を受け取りましたが、その後に論理呼び出しが失敗しました（上記）。ハーネスはこの応答を使っていません。
            </Notice>
          )}
        </>
      )}
      {answerAttempt && answerAttempt !== attempt && (
        <Notice>
          同じ論理呼び出しの正常な応答は attempt {attemptNo(answerAttempt)} で受け取りました:{' '}
          <EventLink onClick={() => onSelect(call.node, answerAttempt.request.event_id)}>attempt {attemptNo(answerAttempt)} を開く</EventLink>
        </Notice>
      )}

      {resp && (
        <>
          <SectionTitle>API が返した reasoning</SectionTitle>
          {rd?.reasoning && (rd.reasoning.text !== undefined && rd.reasoning.text !== null) ? (
            <div className="space-y-1">
              <Badge tone="violet">API 提供（field: {rd.reasoning.field ?? '不明'}）· モデル内部の推論を推測したものではありません</Badge>
              <FullTextBlock ep={ep} value={rd.reasoning.text} empty="（空）" />
            </div>
          ) : (
            <div className="text-xs text-text-tertiary">API は reasoning を返していません。</div>
          )}
        </>
      )}

      {(isAnswer || !answer) && call.parsed.length > 0 && <SectionTitle>ハーネスによる解析（model_output_parsed）</SectionTitle>}
      {(isAnswer || !answer) &&
        call.parsed.map((p) => (
          <ParsedView key={p.event_id} ev={p} ep={ep} />
        ))}

      {isAnswer && call.producedBlocks.length > 0 && (
        <>
          <SectionTitle>このモデル呼び出しから実行されたコードブロック</SectionTitle>
          <div className="flex flex-wrap gap-2 text-xs">
            {call.producedBlocks.map((b) => (
              <EventLink key={b.key} onClick={() => onSelect(b)}>
                コードブロック #{dataOf(b.primary, 'code_block_start').block_index ?? '?'} (seq {b.seq})
              </EventLink>
            ))}
          </div>
        </>
      )}

      {rd && (
        <>
          <SectionTitle>usage・時間</SectionTitle>
          <KV
            rows={[
              ['HTTP status', String(rd.http_status ?? '—')],
              [
                'latency',
                `${fmtDuration(rd.latency_s)}${
                  !rd.streamed && dataOf(attempt?.request ?? call.node.primary, 'model_request').endpoint?.known_proxy
                    ? '（プロキシ内部の処理時間を含む場合があります）'
                    : ''
                }`,
              ],
              ['streamed', `${String(rd.streamed ?? '—')}${rd.n_chunks ? ` · ${rd.n_chunks} chunks` : ''}${rd.ttft_s ? ` · ttft ${fmtDuration(rd.ttft_s)}` : ''}`],
              ['finish_reason', rd.finish_reason ?? '—'],
              ['usage', rd.usage ? JSON.stringify(rd.usage) : 'API は usage を返していません'],
              ['response_id', rd.response_id ?? '—'],
            ]}
          />
        </>
      )}

      <SectionTitle>HTTP 試行（{call.attempts.length}）</SectionTitle>
      <AttemptsTable call={call} model={model} />
      {call.logicalErrors
        .filter((e) => !shownAsAttemptResult.includes(e))
        .map((e) => (
          <ErrorView key={e.event_id} ep={ep} ev={e} />
        ))}
      {call.strayErrors.map((e) => (
        <ErrorView key={e.event_id} ep={ep} ev={e} title="要求が記録にない HTTP 試行のエラー" />
      ))}

      {rd?.payload_path && (
        <div>
          <button type="button" onClick={() => setShowRaw(!showRaw)} className="text-[11px] text-accent-light hover:text-accent">
            {showRaw
              ? '保存した応答を隠す'
              : assembledFromStream(resp)
                ? 'ストリームの chunk から組み立てた応答（保存ファイル、JSON）を表示'
                : '保存したレスポンス本文（JSON）を表示'}
          </button>
          {showRaw && assembledFromStream(resp) && (
            <div className="mt-1 text-[11px] text-text-tertiary">
              ストリーミング応答の本文そのものではなく、レコーダーが受け取った chunk を組み立てた結果です。
            </div>
          )}
          {showRaw && <StoredJson ep={ep} path={rd.payload_path} />}
        </div>
      )}
    </div>
  );
}

/** The call's final result when it failed: the logical error, or every attempt failing. */
function CallFailure({ call, stopped, httpAnswerStatus }: { call: LogicalCall; stopped: boolean; httpAnswerStatus: number | null }) {
  const fin = call.logicalErrors[call.logicalErrors.length - 1] ?? null;
  if (!fin) return <Notice tone="red">この論理呼び出しは失敗しました: すべての HTTP 試行がエラーで終わっています。</Notice>;
  const d = dataOf(fin, 'model_error');
  const msg = textOf(d.message);
  return (
    <Notice tone={stopped ? 'amber' : 'red'}>
      {stopped ? 'この論理呼び出しは完了前に中断されました' : 'この論理呼び出しは失敗しました'}
      {httpAnswerStatus !== null ? `（HTTP ${httpAnswerStatus} の応答を受け取った後）` : ''}: {modelErrorLabel(d)}
      {msg ? `: ${msg}` : ''}
    </Notice>
  );
}

/** The stored response of a streamed call is assembled from the chunks (web trial hook), not an HTTP body. */
function assembledFromStream(resp: TraceEvent | null): boolean {
  if (!resp) return false;
  if (dataOf(resp, 'model_response').streamed === true) return true;
  return (resp.artifacts ?? []).some((a) => (a as { assembled_from_stream?: unknown }).assembled_from_stream === true);
}

function StoredJson({ ep, path }: { ep: EpisodeRef; path: string }) {
  const url = episodeFileUrl(ep, path);
  const res = useAsync(() => fetchJsonFile<unknown>(url), [url]);
  if (res.loading) return <Spinner />;
  if (res.error) return <ErrorText>{res.error}</ErrorText>;
  return res.data !== null ? <RawJson value={res.data} /> : null;
}

function ParsedBlock({ ep, value }: { ep: EpisodeRef; value: TextValue | null }) {
  const t = useFullText(ep, value);
  if (t.text === null) return <NotRecorded reason="コードブロックの文字列が記録されていません" />;
  return (
    <div>
      <CodeView code={t.text} maxLines={40} />
      {t.partial && <div className="text-[11px] text-amber-300 mt-1">{t.loading ? '全文を読み込み中…' : t.error ? `全文を読み込めません: ${t.error}` : '先頭のみ'}</div>}
    </div>
  );
}

function ParsedView({ ev, ep }: { ev: TraceEvent; ep: EpisodeRef }) {
  const d = dataOf(ev, 'model_output_parsed');
  const blocks = d.code_blocks ?? [];
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        {d.parser && <span className="font-mono text-text-tertiary">{d.parser}</span>}
        {d.decision && (
          <Badge tone="neutral" title="ハーネスの読み取りです。モデルの主張ではありません。">
            {harnessReadingLabel(d.decision, d.harness_reading)}
          </Badge>
        )}
        {d.model_said_finish === true && <Badge tone="green">{LABELS.modelClaimsFinish}</Badge>}
        {d.content_empty === true && <Badge tone="red">応答が空</Badge>}
        {d.code_source && <Badge tone={d.code_source === 'oracle' ? 'amber' : 'neutral'}>code_source: {d.code_source}</Badge>}
        <span className="text-text-tertiary">コード {blocks.length} ブロック</span>
      </div>
      {blocks.map((b, i) => (
        <ParsedBlock key={i} ep={ep} value={b ?? null} />
      ))}
    </div>
  );
}

function AttemptsTable({ call }: { call: LogicalCall; model: TabProps['model'] }) {
  const outcome = callOutcome(call);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-text-tertiary border-b border-surface-border">
            <th className="py-1 pr-2 font-normal">attempt</th>
            <th className="py-1 pr-2 font-normal">seq</th>
            <th className="py-1 pr-2 font-normal">結果</th>
            <th className="py-1 pr-2 font-normal">時間</th>
            <th className="py-1 font-normal">詳細</th>
          </tr>
        </thead>
        <tbody>
          {call.attempts.map((a) => {
            const r = a.responses[0];
            const e = a.errors[0];
            const rd = r ? dataOf(r, 'model_response') : null;
            const ed = e ? dataOf(e, 'model_error') : null;
            const logical = !r && !e ? logicalErrorsOfAttempt(call, a)[0] ?? null : null;
            const ld = logical ? dataOf(logical, 'model_error') : null;
            const streamed = dataOf(a.request, 'model_request').stream === true;
            const failedLater = !!r && r === outcome.httpAnswer && outcome.finalError !== null;
            return (
              <tr key={a.request.event_id} className="border-b border-surface-border/60 align-top">
                <td className="py-1 pr-2 font-mono">{attemptNumber(call, a)}</td>
                <td className="py-1 pr-2 font-mono text-text-tertiary">{a.request.seq}</td>
                <td className="py-1 pr-2">
                  {r ? (
                    <Badge tone={failedLater ? 'amber' : 'green'}>HTTP {rd?.http_status ?? 'ok'}{failedLater ? '（その後に失敗）' : ''}</Badge>
                  ) : e ? (
                    <Badge tone="red">{ed?.http_status ? `HTTP ${ed.http_status}` : ed?.error_type ?? 'error'}</Badge>
                  ) : ld ? (
                    <Badge tone="red">{modelErrorLabel(ld)}</Badge>
                  ) : streamed && outcome.state !== 'open' ? (
                    <Badge tone="neutral">試行ごとの結果は記録なし（ストリーム）</Badge>
                  ) : (
                    <Badge tone="gold">応答なし</Badge>
                  )}
                </td>
                <td className="py-1 pr-2 font-mono">{fmtDuration(rd?.latency_s ?? ed?.latency_s ?? ld?.latency_s ?? null)}</td>
                <td className="py-1 text-text-secondary break-all">
                  {ed ? `${ed.error_type ?? ''} ${textOf(ed.message)}${ed.body_excerpt ? ` — ${textOf(ed.body_excerpt).slice(0, 300)}` : ''}` : ''}
                  {ed && (ed.message_redacted === true || ed.body_excerpt_redacted === true) ? `（${REDACTED_SHORT}）` : ''}
                  {ld ? textOf(ld.message) : ''}
                  {ld && ld.message_redacted === true ? `（${REDACTED_SHORT}）` : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ErrorView({ ev, title, ep }: { ev: TraceEvent; title?: string; ep: EpisodeRef }) {
  const d = dataOf(ev, 'model_error');
  const msg = textOf(d.message);
  const excerpt = textOf(d.body_excerpt);
  const hasPartial = d.partial_content !== undefined && d.partial_content !== null;
  return (
    <div className="space-y-1">
      <Notice tone="red">
        {title ?? `論理呼び出しの失敗（scope: ${d.scope ?? '?'}）`}: {d.http_status ? `HTTP ${d.http_status} ` : ''}
        {modelErrorLabel({ error_type: d.error_type, scope: null })}
        {msg ? `: ${msg}` : ''}
        {d.streamed ? `（ストリーム · 受信した chunk ${d.n_chunks ?? '?'}）` : ''}
        {excerpt && <pre className="mt-1 text-[11px] whitespace-pre-wrap break-all">{excerpt}</pre>}
        {(d.message_redacted === true || d.body_excerpt_redacted === true) && (
          <div className="mt-1 text-[11px] text-amber-300">
            {d.message_redacted === true ? 'メッセージ' : ''}
            {d.message_redacted === true && d.body_excerpt_redacted === true ? 'と' : ''}
            {d.body_excerpt_redacted === true ? '応答本文の抜粋' : ''}: {REDACTED_MESSAGE_NOTE}
          </div>
        )}
      </Notice>
      {hasPartial && (
        <div>
          <div className="text-[11px] text-text-tertiary mb-1">
            中断までに受信した部分出力（chunk {d.n_chunks ?? '?'} 件。組み立てた最終応答ではありません）
          </div>
          <FullTextBlock ep={ep} value={d.partial_content} empty="（受信したテキストはありません）" />
        </div>
      )}
    </div>
  );
}

/**
 * code_block_start.direct_access (5.2): which of env, obs, APIS and INPUTS the code names,
 * as the recorder writes it ({names, complete, note}; the trace recorder), or a
 * plain list. Calls made through these names, other than APIS[...], are not recorded.
 */
function directAccessText(v: unknown): string {
  const note = '（APIS[...] 以外の経路の呼び出しは記録されません）';
  if (Array.isArray(v)) return v.length ? `${v.join(', ')}${note}` : 'なし';
  if (!v || typeof v !== 'object') return '未記録';
  const o = v as { names?: unknown; complete?: unknown };
  const names = Array.isArray(o.names) ? o.names.filter((x): x is string => typeof x === 'string') : [];
  const partial = o.complete === false ? '（コードを最後まで字句解析できなかったため、一部だけの可能性があります）' : '';
  return names.length ? `${names.join(', ')}${note}${partial}` : `なし${partial}`;
}

function BlockCode({ ep, onSelect, block, markLine }: TabProps & { block: TNode; markLine: number | null }) {
  const d = dataOf(block.primary, 'code_block_start');
  const code = useFullText(ep, d.code ?? null);
  const direct = directAccessText(d.direct_access);
  return (
    <div className="space-y-3">
      <KV
        rows={[
          ['コードブロック', `#${d.block_index ?? '?'}${d.turn_index !== undefined && d.turn_index !== null ? ` · ターン ${d.turn_index}` : ''}`],
          ['code_source', d.code_source ?? '—'],
          [
            '生成元のモデル呼び出し',
            block.sourceCall ? (
              <EventLink onClick={() => onSelect(block.sourceCall!.node)}>
                {block.sourceCall.role} {block.sourceCall.callKind} (#{block.sourceCall.node.seq})
              </EventLink>
            ) : d.code_source === 'oracle' ? (
              'oracle コード（モデル呼び出しなし）'
            ) : (
              '不明'
            ),
          ],
          ['code_sha256', shortSha(d.code_sha256 ?? null, 16)],
          ['env / obs / APIS / INPUTS の直接参照', direct],
        ]}
      />
      {markLine !== null && <div className="text-[11px] text-accent">API の呼び出し行（{markLine} 行目）を強調しています</div>}
      {code.text === null ? <NotRecorded reason="コードが記録されていません" /> : <CodeView code={code.text} markLine={markLine} scrollToMark={markLine !== null} />}
      {code.partial && <div className="text-[11px] text-amber-300">{code.loading ? '全文を読み込み中…' : '先頭のみ'}</div>}
    </div>
  );
}
