/** Video lookup and frame-based playback mapping; docs/trace-format.md §9.3. */
import { episodeFileUrl, traceApi, type EpisodeRef } from './api';
import type { EpisodeModel, FrameRec, TNode } from './model';
import { dataOf } from './types';
import type { RunnerMatch } from './runner';

export interface TrialVideo { url: string; name: string; turnIndex?: number; source?: 'episode' | 'runner' }
const object = (x: unknown): Record<string, unknown> => x && typeof x === 'object' ? x as Record<string, unknown> : {};
const safe = (x: string) => !!x && x !== '.' && x !== '..' && !/[\\/\0]/.test(x);

/** Only a time-verified final attempt may use the record's final artifacts. */
export function trialVideos(run: string, match: RunnerMatch, roots: string[]): TrialVideo[] {
  if (!match.record || !match.isFinal || match.unverified) return [];
  const r = match.record;
  const a = object(r.artifacts);
  if (typeof a.trial_dir !== 'string' || !Array.isArray(a.videos) || !r.task || !r.condition) return [];
  const dir = a.trial_dir;
  const trial = dir.split('/').at(-1)!;
  if (![run, r.task, r.condition, trial].every(safe)) return [];
  const root = roots.findIndex(x => `${x.replace(/\/$/, '')}/${run}/${r.task}/${r.condition}/${trial}` === dir);
  if (root < 0) return [];
  return a.videos.flatMap((p: unknown) => {
    if (typeof p !== 'string' || !p.startsWith(`${dir}/`)) return [];
    const name = p.slice(dir.length + 1);
    if (name !== 'video_combined.mp4') return [];
    return [{ name, url: traceApi.legacyFileUrl(run, r.task!, r.condition!, trial, name, root), source: 'runner' as const }];
  });
}

/** Videos copied into an upstream episode by H4; only declared, written files are shown. */
export function episodeVideos(ep: EpisodeRef, model: EpisodeModel): TrialVideo[] {
  return model.events.filter(e => e.event_type === 'trial_videos').flatMap(e => e.artifacts ?? []).flatMap(art => {
    if (art.written !== true || typeof art.path !== 'string' || art.mime !== 'video/mp4') return [];
    const match = /^videos\/(video_(?:turn_(\d+)(?:_[\w-]+)?|combined(?:_[\w-]+)?)\.mp4)$/.exec(art.path);
    if (!match || (match[2] !== undefined &&
      (typeof art.turn_index !== 'number' || art.turn_index !== Number(match[2])))) return [];
    return [{name: match[1], url: episodeFileUrl(ep, art.path), source: 'episode' as const,
      ...(match[2] !== undefined ? {turnIndex: Number(match[2])} : {})}];
  });
}

/** A missing code block event must never shift a later turn onto an earlier video. */
export function chooseExecutionVideo(block: TNode, videos: TrialVideo[]): {video: TrialVideo | undefined; wholeTrial: boolean} {
  const index = dataOf(block.primary, 'code_block_start').block_index;
  if (typeof index === 'number' && Number.isInteger(index) && index >= 0) {
    const turn = videos.find(v => v.turnIndex === index && !v.name.includes('_wrist'));
    if (turn) return {video: turn, wholeTrial: false};
  }
  const combined = videos.filter(v => /^video_combined(?:_[\w-]+)?\.mp4$/.test(v.name))
    .sort((a,b) => (a.name === 'video_combined.mp4' ? -1 : b.name === 'video_combined.mp4' ? 1 : a.name.localeCompare(b.name)))[0];
  return {video: combined, wholeTrial: true};
}

export function cameraVideos(videos: TrialVideo[]): TrialVideo[] {
  return videos.filter(v => v.turnIndex === undefined);
}

export function canPlayExecutionVideo(video: TrialVideo | undefined, turnVideo: boolean,
  clip: {start:number;end:number} | null): boolean {
  return !!video && (turnVideo || video.source === 'episode' || !!clip && clip.end > clip.start);
}

export function validClip(clip: {start:number;end:number} | null): {start:number;end:number} | null {
  return clip && Number.isFinite(clip.start) && Number.isFinite(clip.end) && clip.end > clip.start ? clip : null;
}

export function videosForEpisode(ep: EpisodeRef, model: EpisodeModel, match: RunnerMatch | null,
  roots: string[]): TrialVideo[] {
  const runner = match ? trialVideos(ep.runId, match, roots) : [];
  return runner.length ? runner : episodeVideos(ep, model);
}

// the runner that wrote the trace encodes combined videos at 30 fps.
// Do not use wall-clock duration: inference delays are absent from the MP4.
export const RECORDED_VIDEO_FPS = 30;
export interface PlaybackCue { time: number; node: TNode }
export interface PlaybackMap { reason: string | null; prompts: PlaybackCue[]; actions: PlaybackCue[]; timeForStep?: (step: number) => number }
export function playbackMap(model: EpisodeModel, name: string, duration: number): PlaybackMap {
  const fail = (reason: string): PlaybackMap => ({reason, prompts: [], actions: []});
  if (name !== 'video_combined.mp4') return fail('このカメラの動画とイベントの対応は未記録です。');
  if (!Number.isFinite(duration) || duration <= 0) return fail('動画を読み込み中です。');
  if (model.events.some(e => e.event_type === 'env_reset')) return fail('環境リセットを含むため、自動同期はできません。');
  const cameras = [...model.framesByCamera.values()].filter(fs => fs.length > 0);
  if (cameras.length !== 1) return fail('動画と記録カメラを一意に対応付けられません。');
  const frames = cameras[0].filter(f => f.simStep !== null && f.bufferIndex >= 0);
  if (frames.length < 2 || frames.some(f => f.epoch !== 0)) return fail('動画との対応に必要なフレーム記録が不足しています。');
  const unique = new Map<number, FrameRec>();
  for (const f of frames) unique.set(f.bufferIndex, f);
  const anchors = [...unique.values()].sort((a,b) => a.bufferIndex-b.bufferIndex);
  if (anchors[0].bufferIndex !== 0 || Math.abs((anchors.at(-1)!.bufferIndex + 1) - duration * RECORDED_VIDEO_FPS) > 1.1 ||
      anchors.some((f,i) => i > 0 && f.simStep! < anchors[i-1].simStep!)) {
    return fail('動画全体とフレーム記録の対応を確認できません。動画のみ再生できます。');
  }
  const seconds = (step: number): number => {
    const hi = anchors.findIndex(f => f.simStep! >= step);
    if (hi < 0) return duration;
    if (hi === 0) return anchors[0].bufferIndex / RECORDED_VIDEO_FPS;
    const lo = anchors[hi-1], up = anchors[hi];
    const fraction = (step-lo.simStep!) / Math.max(1,up.simStep!-lo.simStep!);
    return (lo.bufferIndex + fraction*(up.bufferIndex-lo.bufferIndex))/RECORDED_VIDEO_FPS;
  };
  const cues = (kind: 'model_call' | 'api_call' | 'code_block') => model.allNodes
    .filter(n => n.kind === kind && typeof n.primary.sim_step === 'number' && n.primary.sim_epoch === 0)
    .map(node => ({node, time: seconds(node.primary.sim_step!)}))
    .sort((a,b) => a.time-b.time || a.node.seq-b.node.seq);
  return {reason:null, timeForStep: seconds, prompts:cues('model_call'), actions:[...cues('code_block'),...cues('api_call')].sort((a,b)=>a.time-b.time||a.node.seq-b.node.seq)};
}
export function cueAt(cues: PlaybackCue[], time: number): PlaybackCue | null {
  let last: PlaybackCue | null = null;
  for (const c of cues) { if (c.time > time + 0.001) break; last=c; }
  return last;
}
