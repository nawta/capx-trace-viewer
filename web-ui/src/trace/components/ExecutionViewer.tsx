/** One execution: its generating request/response, clip, stdout/stderr and outcome. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { episodeFileUrl, type EpisodeRef } from '../api';
import { executionFrame, executionRange, executionResult, executionSource, selectedExecution } from '../executions';
import { useFullText } from '../hooks';
import type { EpisodeModel, Selection, TNode } from '../model';
import { playbackMap, type TrialVideo } from '../playback';
import { dataOf, type TextValue } from '../types';
import { SmallButton } from './common';
import { SimplePrompt } from './SimplePrompt';
import { TraceImage } from './TraceImage';

function RecordedText({ep, value}: {ep:EpisodeRef;value:TextValue|null|undefined}) {
  const text=useFullText(ep,value);
  return <div>
    {text.loading && <p className="text-xs text-text-tertiary">全文を読み込み中…</p>}
    {(text.partial || text.error) && <p className="text-xs text-amber-200">全文を表示できていません。記録された範囲を表示します。</p>}
    <pre className="text-sm leading-6 whitespace-pre-wrap break-words">{text.text ?? '未記録'}</pre>
    {text.text === '' && <p className="text-text-tertiary">出力なし</p>}
  </div>;
}

function ExecutionClip({model,block,ep,videos}:{model:EpisodeModel;block:TNode;ep:EpisodeRef;videos:TrialVideo[]}) {
  const video=videos.find(v=>v.name==='video_combined.mp4');
  const range=executionRange(block);
  const noMotion=!!range && range.start===range.end;
  const frame=executionFrame(model,block);
  const [duration,setDuration]=useState(0);
  const [failed,setFailed]=useState(false);
  const player=useRef<HTMLVideoElement>(null);
  const mapping=useMemo(()=>playbackMap(model,video?.name??'',duration),[model,video?.name,duration]);
  const clip=range && mapping.timeForStep ? {start:mapping.timeForStep(range.start),end:mapping.timeForStep(range.end)} : null;
  const start=clip?.start ?? null, end=clip?.end ?? null;
  useEffect(()=>{
    const el=player.current;
    if(!el || start===null || end===null || end<=start)return;
    el.pause();el.currentTime=start;el.playbackRate=.5;
    let handle=0;
    const tick=()=>{
      if(el.currentTime>=end){el.pause();el.currentTime=Math.max(start,end-.001);}
      if(el.requestVideoFrameCallback)handle=el.requestVideoFrameCallback(tick);
    };
    if(el.requestVideoFrameCallback)handle=el.requestVideoFrameCallback(tick);
    return ()=>{if(handle)el.cancelVideoFrameCallback(handle);};
  },[start,end]);
  if(noMotion)return <div className="p-4 space-y-3" data-motion="none">
    <p className="font-semibold">動作なし</p><p className="text-sm text-text-tertiary">この回はシミュレーションが進んでいません。下は実行開始時の記録画像です。</p>
    {frame ? <TraceImage src={episodeFileUrl(ep,frame)} alt="実行開始時" provenance="debug_frame" className="h-64 w-full"/> : <p className="text-text-tertiary">開始時の画像は未記録です。</p>}
  </div>;
  if(!video)return <p className="p-4 text-text-tertiary">この回に対応する動画は見つかりません。</p>;
  const ready=!!clip && clip.end>clip.start;
  return <div className="p-4 space-y-2">
    {!ready && <p className="text-sm text-amber-200">{failed?'動画を読み込めません。':!range?'この回の動作範囲は未記録です。':mapping.reason ?? 'この回に対応する動画区間を確認できません。'}</p>}
    <video ref={player} src={video.url} controls preload="metadata" aria-label="選んだ実行の動画" className={`w-full max-h-[45vh] bg-black rounded ${ready?'':'hidden'}`}
      onLoadedMetadata={e=>setDuration(e.currentTarget.duration)} onError={()=>setFailed(true)}
      onSeeking={e=>{if(clip && (e.currentTarget.currentTime<clip.start || e.currentTarget.currentTime>clip.end))e.currentTarget.currentTime=Math.min(Math.max(e.currentTarget.currentTime,clip.start),clip.end);}}
      onPlay={e=>{if(clip && e.currentTarget.currentTime>=clip.end-.03)e.currentTarget.currentTime=clip.start;}}
      onTimeUpdate={e=>{if(clip && e.currentTarget.currentTime>=clip.end){e.currentTarget.pause();e.currentTarget.currentTime=Math.max(clip.start,clip.end-.001);}}}/>
    {ready && <p className="text-xs text-text-tertiary">この回の区間のみ再生（全体動画の {clip.start.toFixed(2)}〜{clip.end.toFixed(2)} 秒）。記録フレーム間は補間しています。</p>}
  </div>;
}

export function ExecutionViewer({model,ep,selection,videos,onSelect,error}:{model:EpisodeModel;ep:EpisodeRef;selection:Selection|null;videos:TrialVideo[];onSelect:(node:TNode)=>void;error:string|null}) {
  const blocks=model.blocks;
  const block=selectedExecution(model,selection);
  const index=block?blocks.indexOf(block):-1;
  const [tab,setTab]=useState<'input'|'output'|'result'>('input');
  const source=block?executionSource(model,block):{request:null,response:null};
  const end=block?dataOf(block.end,'code_block_end'):{};
  const stdout=end.stdout_event_id?model.byId.get(end.stdout_event_id):undefined;
  const stderr=end.stderr_event_id?model.byId.get(end.stderr_event_id):undefined;
  if(!block)return <main className="p-6"><p>コード実行はまだ記録されていません。モデル呼び出しは「詳細を見る」で確認できます。</p></main>;
  return <main className="flex-1 min-h-0 overflow-auto p-4 md:p-6 flex flex-col gap-4">
    {error&&<p className="text-amber-200">記録を読み込めません。詳細表示で確認してください。</p>}
    <nav className="flex flex-wrap items-center gap-3" aria-label="コード実行を選ぶ">
      <SmallButton disabled={index<=0} onClick={()=>onSelect(blocks[index-1])}>前の実行</SmallButton>
      <select aria-label="実行回" value={block.key} onChange={e=>{const next=blocks.find(b=>b.key===e.target.value);if(next)onSelect(next);}} className="bg-surface-raised border border-surface-border rounded p-2">
        {blocks.map((b,i)=><option key={b.key} value={b.key}>{i+1}回目 · {executionResult(b)}{executionRange(b)?.start===executionRange(b)?.end && executionRange(b)?' · 動作なし':''}</option>)}
      </select>
      <span className="text-sm text-text-tertiary">全{blocks.length}回のコード実行</span>
      <SmallButton disabled={index>=blocks.length-1} onClick={()=>onSelect(blocks[index+1])}>次の実行</SmallButton>
    </nav>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 flex-1 min-h-0">
      <section className="rounded-xl border border-surface-border bg-surface-raised overflow-y-auto">
        <div className="px-5 py-4 border-b border-surface-border"><h2 className="font-semibold">{index+1}回目の実行：{executionResult(block)}</h2><p className="text-sm text-text-tertiary mt-1">入力から生成したコードを1回実行した結果です。</p></div>
        <ExecutionClip key={`${ep.episodeId}/${block.key}`} model={model} block={block} ep={ep} videos={videos}/>
        <div className="p-4 border-t border-surface-border space-y-2 text-sm" aria-label="この実行の結果">
          <p>タスク達成：{end.task_completed===true?'達成':end.task_completed===false?'未達成':'未記録'} · 実行：{end.sandbox_rc===0?'正常終了':typeof end.sandbox_rc==='number'?'エラー':'結果未記録'}</p>
          <p className="text-text-tertiary">{typeof end.duration_s==='number'?`実行時間 ${end.duration_s.toFixed(2)} 秒`:'実行時間は未記録'}</p>
          {stderr&&<div className="text-amber-200 max-h-32 overflow-auto"><RecordedText ep={ep} value={dataOf(stderr,'stderr').text}/></div>}
        </div>
      </section>
      <section className="min-h-[400px] lg:min-h-0 flex flex-col rounded-xl border border-surface-border bg-surface-raised overflow-hidden">
        <div className="flex gap-2 p-3 border-b border-surface-border"><SmallButton active={tab==='input'} onClick={()=>setTab('input')}>入力プロンプト</SmallButton><SmallButton active={tab==='output'} onClick={()=>setTab('output')}>モデル出力</SmallButton><SmallButton active={tab==='result'} onClick={()=>setTab('result')}>実行ログ</SmallButton></div>
        {tab==='input'?<div className="flex-1 min-h-0 flex flex-col [&>section]:flex-1"><SimplePrompt key={`${ep.episodeId}/${block.key}/${source.request?.event_id??''}`} ep={ep} request={source.request}/></div>:
        <div key={block.key} className="flex-1 min-h-0 overflow-auto p-5 space-y-4">
          {tab==='output'?<><h2 className="font-semibold">このコードを生成したモデルの応答</h2><RecordedText ep={ep} value={dataOf(source.response,'model_response').content}/><details><summary className="cursor-pointer">実際に実行したコード</summary><RecordedText ep={ep} value={dataOf(block.primary,'code_block_start').code}/></details></>:
          <><h2 className="font-semibold">標準出力</h2><RecordedText ep={ep} value={dataOf(stdout,'stdout').text}/><h2 className="font-semibold">エラー出力</h2><RecordedText ep={ep} value={dataOf(stderr,'stderr').text}/></>}
        </div>}
      </section>
    </div>
  </main>;
}
