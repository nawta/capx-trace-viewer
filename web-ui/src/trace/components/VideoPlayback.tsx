/** In-page MP4 playback and prompt following (docs/trace-format.md §9.3). */
import { useEffect, useMemo, useRef, useState } from 'react';
import { nodeTitle, type EpisodeModel, type TNode } from '../model';
import { cueAt, playbackMap, type TrialVideo, type PlaybackCue } from '../playback';
import { SmallButton } from './common';

export function VideoPlayback({model, videos, onSelect, simple = false}: {simple?: boolean; model: EpisodeModel; videos: TrialVideo[]; onSelect: (node:TNode)=>void}) {
  const [name,setName]=useState(videos[0]?.name ?? '');
  const chosen=videos.find(v=>v.name===name) ?? videos[0];
  const [duration,setDuration]=useState(0);
  const [time,setTime]=useState(0);
  const [follow,setFollow]=useState(true);
  const [manualKey,setManualKey]=useState<string|null>(null);
  const [mediaError,setMediaError]=useState(false);
  const [mode,setMode]=useState<'prompts'|'actions'>('prompts');
  const video=useRef<HTMLVideoElement>(null);
  const list=useRef<HTMLDivElement>(null);
  const selected=useRef<string|null>(null);
  const map=useMemo(()=>playbackMap(model,chosen?.name??'',duration),[model,chosen?.name,duration]);
  const cues=map[mode];
  const current=(!follow && manualKey ? cues.find(c=>c.node.key===manualKey) : null) ?? cueAt(cues,time);
  useEffect(()=>{setDuration(0);setTime(0);setMediaError(false);setManualKey(null);selected.current=null;},[chosen?.url]);
  useEffect(()=>{
    if (!follow || !current || selected.current===current.node.key) return;
    selected.current=current.node.key;
    onSelect(current.node);
    const row=list.current?.querySelector<HTMLElement>('[data-current="true"]');
    if(row && list.current) list.current.scrollTop=row.offsetTop-list.current.clientHeight/2+row.clientHeight/2;
  },[current,follow,onSelect]);
  const jump=(cue:PlaybackCue)=>{
    setFollow(false);setManualKey(cue.node.key);
    if(video.current){video.current.pause();video.current.currentTime=Math.min(cue.time,duration);}
    setTime(cue.time);onSelect(cue.node);
  };
  const currentIndex = cues.findIndex(c => c.node.key === current?.node.key);
  if(!chosen)return <div className="p-3 text-xs text-text-tertiary">この試行の動画は見つかりません。「詳細を見る」から記録された画像を確認できます。</div>;
  return <div className="h-full min-h-0 flex flex-col p-2 gap-2">
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {(!simple || videos.length > 1) && <select aria-label="録画カメラ" value={chosen.name} onChange={e=>setName(e.target.value)} className="bg-surface-sunken">{videos.map(v=><option key={v.url}>{v.name}</option>)}</select>}
      <label><input type="checkbox" checked={follow} onChange={e=>{selected.current=null;setFollow(e.target.checked);}}/> 再生位置に追従</label>
      {!simple && <SmallButton active={mode==='prompts'} onClick={()=>{selected.current=null;setMode('prompts');}}>プロンプト</SmallButton>}
      {!simple && <SmallButton active={mode==='actions'} onClick={()=>{selected.current=null;setMode('actions');}}>コード・API</SmallButton>}
      <select aria-label="動画の再生速度" defaultValue="0.5" onChange={e=>{if(video.current)video.current.playbackRate=Number(e.target.value);}} className="bg-surface-sunken">{[0.25,0.5,1,2].map(x=><option key={x} value={x}>{x}×</option>)}</select>
    </div>
    <div className="flex flex-1 min-h-0 gap-2">
      <video key={chosen.url} ref={video} src={chosen.url} controls preload="metadata" className={`${simple ? "w-full" : "w-[58%]"} min-w-0 h-full object-contain bg-black rounded`}
        onLoadedMetadata={e=>{setDuration(e.currentTarget.duration);e.currentTarget.playbackRate=0.5;}}
        onTimeUpdate={e=>setTime(e.currentTarget.currentTime)} onSeeked={e=>setTime(e.currentTarget.currentTime)}
        onEnded={()=>setTime(duration)} onError={()=>{setDuration(0);setMediaError(true);}}/>
      {!simple && <div ref={list} className="relative flex-1 min-w-0 overflow-y-auto space-y-1" aria-label="動画に対応するイベント">
        {mediaError ? <div className="text-xs text-amber-200 p-2">動画を読み込めません。保存ファイルとサーバーの登録先を確認してください。</div> : map.reason ? <div className="text-xs text-amber-200 p-2">{map.reason}</div> : cues.length===0 ? <div className="text-xs">対応するイベントは未記録です。</div> : cues.map(c=><button key={c.node.key} data-current={current?.node.key===c.node.key} onClick={()=>jump(c)} className={`block w-full text-left p-2 rounded text-xs border ${current?.node.key===c.node.key?'border-accent bg-accent/10':'border-surface-border'}`}>
          <span className="font-mono">{c.time.toFixed(2)}s</span> {nodeTitle(c.node)}<br/><span className="text-text-tertiary">イベント #{c.node.seq} · クリックで入力と動画位置を表示</span>
        </button>)}
      </div>}
    </div>
    {simple && <div className="flex items-center justify-between gap-2 p-2 text-sm">
      <SmallButton disabled={currentIndex <= 0} onClick={()=>jump(cues[currentIndex-1])}>前の入力</SmallButton>
      <span>{cues.length ? `入力 ${currentIndex+1} / ${cues.length}` : '入力との同期なし'}</span>
      <SmallButton disabled={currentIndex < 0 || currentIndex >= cues.length-1} onClick={()=>jump(cues[currentIndex+1])}>次の入力</SmallButton>
    </div>}
    {simple && (mediaError || map.reason) && <p className="text-xs text-amber-200 px-2">{mediaError ? '動画を読み込めませんでした。' : map.reason}</p>}
    <div className="text-[11px] text-text-tertiary">{map.reason ? 'MP4 はこの画面内で再生できます。' : '記録フレーム間は補間。モデルの待ち時間は動画に含まれず、同じ位置に複数のプロンプトが並ぶことがあります。'}</div>
  </div>;
}
