import { useEffect, useRef } from 'react';
import { episodeFileUrl, fetchJsonFile, type EpisodeRef } from '../api';
import { useAsync } from '../hooks';
import { parsePayload, type PayloadMessage } from '../payload';
import { dataOf, type TraceEvent } from '../types';
import { SmallButton } from './common';
import { TraceImage } from './TraceImage';

export function requestHeading(role: string) {
  return role === 'vdm' ? '映像を読み取るための入力' : role === 'perception_vlm' ? '物体を探すための入力' : 'ロボットへの指示';
}

function Message({message, ep}: {message: PayloadMessage; ep: EpisodeRef}) {
  return <div className="space-y-4">
    {message.parts.map((part, i) => part.kind === 'text' ?
      <p key={i} className="whitespace-pre-wrap break-words text-sm leading-7 text-text-primary">{part.text}</p> :
      part.kind === 'media' && part.media?.asset ?
        <TraceImage key={i} src={episodeFileUrl(ep, part.media.asset)} alt="モデルに送信した画像" provenance="model_input" video={part.media.isVideo} className="h-52 w-full" /> :
        <p key={i} className="text-sm text-text-tertiary">{part.kind === 'media' ? '画像・動画は未記録、または外部URLのため表示しません。' : 'この形式の入力は詳細表示で確認できます。'}</p>)}
  </div>;
}

export function SimplePrompt({ep, request}: {ep: EpisodeRef; request: TraceEvent | null}) {
  const artifact = request?.artifacts?.find(a => a.label === 'payload');
  const path = artifact?.written === false ? null : dataOf(request, 'model_request').payload_path ?? artifact?.path;
  const url = path ? episodeFileUrl(ep, path) : null;
  const result = useAsync(url ? () => fetchJsonFile<unknown>(url) : null, [url]);
  const parsed = result.data ? parsePayload(result.data) : null;
  const messages = parsed?.messages ?? [];
  const primary = messages.filter(m => m.role === 'user').at(-1) ?? messages.at(-1);
  const previous = messages.filter(m => m !== primary);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => { if (body.current) body.current.scrollTop = 0; }, [request?.event_id]);
  return <section className="min-h-0 flex flex-col rounded-xl border border-surface-border bg-surface-raised overflow-hidden">
    <div className="px-5 py-4 border-b border-surface-border">
      <h2 className="font-semibold">{requestHeading(dataOf(request, 'model_request').role ?? '')}</h2>
      <p className="text-xs text-text-tertiary mt-1">この実行コードを生成したモデルへの入力です。</p>
    </div>
    <div ref={body} className="flex-1 min-h-0 overflow-y-auto p-5" aria-label="現在のプロンプト">
      {!request ? <p className="text-text-tertiary">モデルへの入力は記録されていません。</p> :
        !url ? <p className="text-text-tertiary">この入力の本文は未記録です。詳細表示で記録状況を確認できます。</p> :
        result.loading ? <p className="text-text-tertiary">入力を読み込み中…</p> :
        result.error ? <div><p className="text-amber-200">入力を読み込めませんでした。</p><SmallButton onClick={result.reload}>再読み込み</SmallButton></div> :
        primary ? <div key={request.event_id} className="space-y-6">
          <Message message={primary} ep={ep} />
          {previous.length > 0 && <details className="border-t border-surface-border pt-4">
            <summary className="cursor-pointer text-sm text-text-tertiary">共通の指示・会話履歴（{previous.length}件）</summary>
            <div className="mt-4 space-y-6">{previous.map(m => <div key={m.index}><div className="text-xs text-text-tertiary mb-2">{m.role}</div><Message message={m} ep={ep}/></div>)}</div>
          </details>}
        </div> : <p className="text-text-tertiary">この形式の入力は詳細表示で確認できます。</p>}
    </div>
  </section>;
}
