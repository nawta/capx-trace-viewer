/**
 * Media part of the right pane (docs/trace-format.md 9.3):
 * - model calls: the images and videos that were sent, in payload order,
 *   labelled 「モデルに送信」, with hash, size and exact-match source;
 * - code blocks and API calls: before/after key frames side by side with
 *   their sim_step and the offset 「このフレームは N step 前」 (8.2);
 * - every event: a flipbook over the recorded frames around it (play, pause,
 *   step, speed), labelled 「デバッグ用フレーム（モデル入力ではない）」;
 * - a camera switch and click-to-enlarge;
 * - the 「ライブ3D」 tab only when the selected episode is running and its
 *   recorded Viser port is the one the server's proxy serves; otherwise
 *   「過去3D未対応：記録済みフレームを表示しています」.
 * Play/pause only affects this page; the viewer cannot stop an experiment.
 *
 * Related: model.ts (keyFramesOf, framesAround), TraceImage.tsx.
 */

import { useEffect, useMemo, useState } from 'react';
import { episodeFileUrl } from '../api';
import { fmtBytes, shortSha } from '../format';
import { LABELS, frameOffsetLabel } from '../labels';
import type { Live3dState } from '../live';
import {
  ambiguousSource,
  camerasOf,
  framesAround,
  imageSourceText,
  keyFramesOf,
  mainAttempt,
  requestImageCounts,
  type EpisodeModel,
  type FrameRec,
  type KeyFrame,
  type Selection,
} from '../model';
import { dataOf, type ImageInfo } from '../types';
import { Badge, DebugFrameChip, SentToModelChip, SmallButton } from './common';
import { TraceImage } from './TraceImage';
import { VideoPlayback } from './VideoPlayback';
import type { TrialVideo } from '../playback';
import type { TNode } from '../model';
import type { EpisodeRef } from './ValueView';

export type { Live3dState } from '../live';

interface MediaPaneProps {
  videos?: TrialVideo[];
  onPlaybackSelect?: (node: TNode) => void;
  model: EpisodeModel;
  ep: EpisodeRef;
  selection: Selection | null;
  live3d: Live3dState;
  /** Live mode on a running episode: the flipbook of an open event shows the newest frame. */
  follow: boolean;
}

export function MediaPane({ model, ep, selection, live3d, follow, videos = [], onPlaybackSelect }: MediaPaneProps) {
  const [preferredTab, setTab] = useState<'video' | 'recorded' | 'live'>('video');
  const tab = preferredTab === 'video' && videos.length === 0 ? 'recorded' : preferredTab;
  const cameras = useMemo(() => camerasOf(model), [model]);
  const [camera, setCamera] = useState<string | null>(null);
  const node = selection?.node ?? null;
  const kf = useMemo(() => (node ? keyFramesOf(model, node) : { before: [], after: [] }), [model, node]);
  const kfCameras = useMemo(
    () => Array.from(new Set([...kf.before, ...kf.after].map((k) => k.camera).filter((c): c is string => c !== null))),
    [kf],
  );
  const allCameras = useMemo(() => Array.from(new Set([...cameras, ...kfCameras])).sort(), [cameras, kfCameras]);
  const cam = camera && allCameras.includes(camera) ? camera : kfCameras[0] ?? allCameras[0] ?? null;

  useEffect(() => {
    if (!live3d.available && tab === 'live') setTab('recorded');
  }, [live3d.available, tab]);

  return (
    <div className="flex flex-col h-full min-h-0 bg-surface-raised">
      <div className="flex-shrink-0 flex flex-wrap items-center gap-2 px-3 py-1.5 border-b border-surface-border">
        <div className="flex gap-1">
          <SmallButton active={tab === 'video'} onClick={() => setTab('video')}>
            動画＋プロンプト
          </SmallButton>
          <SmallButton active={tab === 'recorded'} onClick={() => setTab('recorded')}>
            記録フレーム
          </SmallButton>
          {live3d.available && (
            <SmallButton active={tab === 'live'} onClick={() => setTab('live')}>
              {LABELS.live3dTab}
            </SmallButton>
          )}
        </div>
        {!live3d.available && (
          <span className="text-[11px] text-text-tertiary" title={live3d.reason}>
            {LABELS.past3dUnsupported}
          </span>
        )}
        {tab === 'recorded' && allCameras.length > 0 && (
          <label className="ml-auto flex items-center gap-1 text-[11px] text-text-tertiary">
            カメラ
            <select
              name="trace-camera"
              value={cam ?? ''}
              onChange={(e) => setCamera(e.target.value)}
              className="bg-surface-sunken border border-surface-border rounded px-1.5 py-0.5 text-[11px] text-text-primary"
            >
              {allCameras.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {tab === 'video' ? (
        <div className="flex-1 min-h-0">
          <VideoPlayback key={`${ep.runId}/${ep.episodeId}/${ep.generation ?? ''}`} model={model} videos={videos} onSelect={onPlaybackSelect ?? (() => {})} />
        </div>
      ) : tab === 'live' && live3d.available ? (
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="flex-shrink-0 px-3 py-1 text-[11px] text-amber-200 bg-amber-500/10 border-b border-amber-500/30">{LABELS.live3dCaption}</div>
          <iframe src="/viser-proxy/" title={LABELS.live3dCaption} className="flex-1 w-full border-0 bg-surface-sunken" allow="fullscreen" />
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
          {!node && <div className="text-xs text-text-tertiary">イベントを選択すると、対応する画像とフレームを表示します。</div>}
          {node?.kind === 'model_call' && node.call && <ModelImages ep={ep} selection={selection!} model={model} />}
          {node && (node.kind === 'code_block' || node.kind === 'api_call') && <KeyFrames ep={ep} kf={kf} camera={cam} endMissing={node.end === null} />}
          {node && cam && <FlipbookSection model={model} ep={ep} selection={selection!} camera={cam} follow={follow && node.open} />}
          {node && !cam && (
            <div className="text-xs text-text-tertiary">
              記録フレーム: {LABELS.notCaptured}（録画バッファが無い設定か、フレームが記録されていません）
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model-input images
// ---------------------------------------------------------------------------

/**
 * Link to the stored render the sent image was matched to (6.2 `source.asset`, the
 * observation_snapshot's harness_render PNG), so the two can be compared by eye.
 */
export function SourceSnapshotLink({ ep, im }: { ep: EpisodeRef; im: ImageInfo }) {
  const asset = im.source && typeof im.source.asset === 'string' ? im.source.asset : null;
  if (!asset) return null;
  return (
    <a
      href={episodeFileUrl(ep, asset)}
      target="_blank"
      rel="noreferrer"
      className="text-accent-light hover:text-accent underline underline-offset-2"
      title="画素が一致した、ハーネスが描画した画像（保存ファイル）"
    >
      一致したスナップショットの画像を開く
    </a>
  );
}

export function imageMeta(im: ImageInfo, sourceText: string) {
  return (
    <div className="space-y-0.5">
      <div>
        order {im.order ?? '?'} · {im.mime ?? '?'} · {im.width && im.height ? `${im.width}×${im.height}` : '?'} · {fmtBytes(im.bytes)}
      </div>
      <div>sha256 {shortSha(im.sha256, 16)}</div>
      <div>path {im.path ?? '?'}</div>
      <div>由来: {sourceText}</div>
    </div>
  );
}

function ModelImages({ model, ep, selection }: { model: EpisodeModel; ep: EpisodeRef; selection: Selection }) {
  const call = selection.node.call!;
  const attempt = selection.attempt ?? mainAttempt(call);
  const d = attempt ? dataOf(attempt.request, 'model_request') : null;
  const images = [...(d?.images ?? [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  // 「この呼び出しは画像入力なし」 only when zero images and zero videos were measured.
  const counts = requestImageCounts(d);
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">送信した画像・動画</span>
        <SentToModelChip />
        <span className="text-[11px] text-text-tertiary">payload 内の順序</span>
      </div>
      {!d && <div className="text-xs text-text-tertiary">{LABELS.notRecorded}</div>}
      {d && counts.noImage === true && images.length === 0 && <Badge tone="neutral">{LABELS.noImageInput}</Badge>}
      {d && counts.noImage === null && images.length === 0 && (
        <div className="text-xs text-text-tertiary">
          画像・動画: {LABELS.notRecorded}（送信本文を解析していないため、画像の有無は分かりません
          {d.reconstruction?.reason ? `: ${d.reconstruction.reason}` : ''}）
        </div>
      )}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-2">
        {images.map((im, i) => {
          const isVideo = !!im.mime && im.mime.startsWith('video/');
          const url = im.asset && im.written !== false ? episodeFileUrl(ep, im.asset) : null;
          const sourceText = imageSourceText(model, im, attempt?.request.seq ?? null, false);
          const amb = ambiguousSource(model, im);
          return (
            <div key={`${im.path}-${i}`} className="space-y-1">
              {im.url && !im.asset ? (
                <div className="p-2 rounded border border-surface-border bg-surface-sunken text-[11px] font-mono break-all">
                  外部URL（{im.fetched ? '取得済み' : 'ビューワーもレコーダーも取得しない'}）: {im.url}
                </div>
              ) : (
                <TraceImage
                  src={url}
                  alt={`model input #${im.order ?? i}`}
                  provenance="model_input"
                  video={isVideo}
                  meta={imageMeta(im, amb ? `${sourceText}（${amb}）` : sourceText)}
                  className="h-36 w-full"
                />
              )}
              <div className="text-[10px] text-text-tertiary font-mono leading-4">
                #{im.order ?? i} {isVideo ? 'video' : 'image'} · {im.width && im.height ? `${im.width}×${im.height}` : ''} · {shortSha(im.sha256)}
                <br />
                由来: {sourceText}
              </div>
              {amb && <div className="text-[10px] text-amber-300 leading-4">{amb}</div>}
              <div className="text-[10px] leading-4">
                <SourceSnapshotLink ep={ep} im={im} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Key frames
// ---------------------------------------------------------------------------

function KeyFrameCell({ ep, k, title, missingNote = 'キーフレームの記録なし' }: { ep: EpisodeRef; k: KeyFrame | undefined; title: string; missingNote?: string }) {
  const url = k?.path ? episodeFileUrl(ep, k.path) : null;
  return (
    <div className="flex-1 min-w-0 space-y-1">
      <div className="text-[11px] text-text-secondary font-medium">{title}</div>
      <TraceImage
        src={url}
        alt={title}
        provenance="debug_frame"
        className="h-40 w-full"
        meta={
          k ? (
            <span>
              camera {k.camera ?? '?'} · epoch {k.epoch ?? '?'} · buffer_index {k.bufferIndex ?? '?'} · frame sim_step {k.frameSimStep ?? '不明'}
            </span>
          ) : undefined
        }
      />
      <div className="text-[10px] text-text-tertiary font-mono leading-4">
        {k && k.path ? (
          <>
            event sim_step {k.eventSimStep ?? '不明'} · frame sim_step {k.frameSimStep ?? '不明'}
            <br />
            {k.offset !== null ? frameOffsetLabel(k.offset) : 'step 差: 不明'} · e{k.epoch ?? '?'} #{k.bufferIndex ?? '?'}
          </>
        ) : (
          <>
            {LABELS.notCaptured}
            {k?.reason ? `（${k.reason}）` : k ? '' : `（${missingNote}）`}
          </>
        )}
      </div>
    </div>
  );
}

function KeyFrames({ ep, kf, camera, endMissing }: { ep: EpisodeRef; kf: { before: KeyFrame[]; after: KeyFrame[] }; camera: string | null; endMissing: boolean }) {
  // The ref of the selected camera, else a camera-less ref that says why there is no frame.
  const before = kf.before.find((k) => k.camera === camera) ?? kf.before.find((k) => k.camera === null);
  const after = kf.after.find((k) => k.camera === camera) ?? kf.after.find((k) => k.camera === null);
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">前後のキーフレーム</span>
        <DebugFrameChip />
      </div>
      <div className="flex gap-2">
        <KeyFrameCell ep={ep} k={before} title="開始時" />
        <KeyFrameCell ep={ep} k={after} title="終了時" missingNote={endMissing ? '終了イベントなし' : undefined} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Flipbook
// ---------------------------------------------------------------------------

function FlipbookSection({ model, ep, selection, camera, follow }: { model: EpisodeModel; ep: EpisodeRef; selection: Selection; camera: string; follow: boolean }) {
  const { frames, start } = useMemo(() => framesAround(model, selection.node, camera), [model, selection.node, camera]);
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[11px] font-bold font-display uppercase tracking-widest text-text-tertiary">記録フレーム（イベント前後）</span>
        <DebugFrameChip />
      </div>
      {frames.length === 0 ? (
        <div className="text-xs text-text-tertiary">
          {LABELS.notCaptured}: このカメラ（{camera}）の記録フレームはこのイベントの前後にありません。
        </div>
      ) : (
        <Flipbook key={`${selection.node.key}|${camera}`} frames={frames} startIndex={follow ? frames.length - 1 : start} ep={ep} follow={follow} />
      )}
    </div>
  );
}

const SPEEDS = [0.25, 0.5, 1, 2, 4];
const BASE_FPS = 4;

export function Flipbook({ frames, startIndex, ep, follow = false }: { frames: FrameRec[]; startIndex: number; ep: EpisodeRef; follow?: boolean }) {
  const [idx, setIdx] = useState(Math.min(startIndex, frames.length - 1));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  // Live: show the newest frame as frames arrive (unless the viewer is playing).
  useEffect(() => {
    if (follow && !playing) setIdx(frames.length - 1);
  }, [follow, playing, frames.length]);
  const i = Math.min(idx, frames.length - 1);
  const f = frames[i];
  const url = (fr: FrameRec | undefined) => (fr && fr.written && fr.path ? episodeFileUrl(ep, fr.path) : null);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setIdx((cur) => {
        if (cur + 1 >= frames.length) {
          setPlaying(false);
          return cur;
        }
        return cur + 1;
      });
    }, 1000 / (BASE_FPS * speed));
    return () => window.clearInterval(id);
  }, [playing, speed, frames.length]);

  // Preload the next frame so playback does not flicker.
  useEffect(() => {
    const next = url(frames[i + 1]);
    if (next) {
      const img = new Image();
      img.src = next;
    }
    // eslint-style deps: url depends only on ep, which is stable per episode.
  }, [i, frames]);

  return (
    <div className="space-y-1.5">
      <TraceImage
        src={url(f)}
        alt={`frame ${f.camera} e${f.epoch} #${f.bufferIndex}`}
        provenance="debug_frame"
        className="h-56 w-full"
        meta={
          <span>
            camera {f.camera} · epoch {f.epoch} · buffer_index {f.bufferIndex} · sim_step {f.simStep ?? '不明'} · {f.reason} · seq {f.event.seq}
          </span>
        }
      />
      <div className="flex flex-wrap items-center gap-1">
        <SmallButton onClick={() => setIdx(0)} title="最初">⏮</SmallButton>
        <SmallButton onClick={() => setIdx(Math.max(0, i - 1))} title="1 フレーム戻る">◀</SmallButton>
        <SmallButton onClick={() => setPlaying(!playing)} active={playing} title={playing ? '一時停止' : '再生'}>
          {playing ? '一時停止' : '再生'}
        </SmallButton>
        <SmallButton onClick={() => setIdx(Math.min(frames.length - 1, i + 1))} title="1 フレーム進む">▶</SmallButton>
        <SmallButton onClick={() => setIdx(frames.length - 1)} title="最後">⏭</SmallButton>
        <label className="flex items-center gap-1 text-[11px] text-text-tertiary ml-1">
          速度
          <select
            name="trace-flipbook-speed"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="bg-surface-sunken border border-surface-border rounded px-1 py-0.5 text-[11px] text-text-primary"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <span className="ml-auto text-[11px] text-text-tertiary font-mono">
          {i + 1}/{frames.length}
        </span>
      </div>
      <input
        type="range"
        name="trace-flipbook-position"
        min={0}
        max={frames.length - 1}
        value={i}
        onChange={(e) => {
          setPlaying(false);
          setIdx(Number(e.target.value));
        }}
        className="w-full accent-[#D4A017]"
        aria-label="フレーム位置"
      />
      <div className="text-[10px] text-text-tertiary font-mono">
        sim_step {f.simStep ?? '不明'} · e{f.epoch} #{f.bufferIndex} · {f.reason}
        {f.forEventId ? ` · for ${f.forEventId}` : ''}
        {follow ? ' · Live: 最新のフレームを表示中' : ''} · 再生はページ内の表示だけで、実験は止めません（{BASE_FPS} fps 基準）
      </div>
    </div>
  );
}
