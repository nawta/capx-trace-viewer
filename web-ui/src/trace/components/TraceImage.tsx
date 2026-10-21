/**
 * Image (or video) with click-to-enlarge, labelled by where it came from
 * (docs/trace-format.md 9.3 media pane): 「モデルに送信」 for model inputs,
 * 「デバッグ用フレーム（モデル入力ではない）」 for recorded frames, and the
 * other provenance labels of 5.1/9.4. The label stays visible in the enlarged
 * view so a debug frame is never mistaken for a model input.
 *
 * Styled after the CaP-X web UI image viewer, which cannot be
 * reused directly: it prefixes every src that is not data:/http with a JPEG
 * base64 header, so relative /api/traces URLs break there.
 *
 * Related: MediaPane.tsx, tabs/ModelInputTab.tsx.
 */

import { useEffect, useRef, useState, type ReactNode } from 'react';
import { LABELS } from '../labels';
import { Badge, type Tone } from './common';

export type ImageProvenance = 'model_input' | 'debug_frame' | 'harness_render' | 'perception' | 'legacy_harness' | 'state';

const PROVENANCE: Record<ImageProvenance, { label: string; tone: Tone }> = {
  model_input: { label: LABELS.sentToModel, tone: 'gold' },
  debug_frame: { label: LABELS.debugFrame, tone: 'sky' },
  harness_render: { label: 'ハーネス描画（モデル入力とは限らない）', tone: 'violet' },
  perception: { label: '知覚サービスの入出力（デバッグ用）', tone: 'sky' },
  legacy_harness: { label: LABELS.legacyHarnessImage, tone: 'amber' },
  state: { label: LABELS.debugBanner, tone: 'sky' },
};

export function ProvenanceChip({ provenance }: { provenance: ImageProvenance }) {
  const p = PROVENANCE[provenance];
  return <Badge tone={p.tone}>{p.label}</Badge>;
}

interface TraceImageProps {
  src: string | null;
  alt: string;
  provenance: ImageProvenance;
  /** Metadata shown under the enlarged image. */
  meta?: ReactNode;
  className?: string;
  video?: boolean;
  /** Hide the provenance chip on the thumbnail (for tiny card thumbnails). */
  compact?: boolean;
  /** Render nothing (instead of 「未取得」) when the image cannot be loaded. */
  hideOnError?: boolean;
}

export function TraceImage({ src, alt, provenance, meta, className = '', video = false, compact = false, hideOnError = false }: TraceImageProps) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  if (hideOnError && (!src || failed)) return null;
  if (!src || failed) {
    return (
      <div
        className={`flex items-center justify-center rounded border border-dashed border-surface-border-light bg-surface-sunken text-[11px] text-text-tertiary ${className}`}
        title={failed ? '画像を読み込めません' : undefined}
      >
        {LABELS.notCaptured}
      </div>
    );
  }

  return (
    <>
      <div className={`relative group ${className}`}>
        {video ? (
          <video src={src} controls className="w-full h-full object-contain rounded border border-surface-border bg-black" onError={() => setFailed(true)} />
        ) : (
          <img
            src={src}
            alt={alt}
            loading="lazy"
            onError={() => setFailed(true)}
            onClick={(e) => {
              e.stopPropagation();
              setOpen(true);
            }}
            className="w-full h-full object-contain rounded border border-surface-border bg-black cursor-zoom-in hover:border-surface-border-light"
          />
        )}
        {!compact && (
          <div className="absolute left-1 top-1 pointer-events-none">
            <ProvenanceChip provenance={provenance} />
          </div>
        )}
      </div>
      {open && <ImageModal src={src} alt={alt} provenance={provenance} meta={meta} onClose={() => setOpen(false)} />}
    </>
  );
}

export function ImageModal({
  src,
  alt,
  provenance,
  meta,
  onClose,
}: {
  src: string;
  alt: string;
  provenance: ImageProvenance;
  meta?: ReactNode;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    ref.current?.focus();
  }, []);
  // Recorded frames and model inputs are often small (128-256 px): the dialog scales them up
  // to the available space, without smoothing, so single pixels stay visible.
  const small = natural !== null && natural.w <= 512 && natural.h <= 512;
  return (
    <div
      ref={ref}
      role="dialog"
      aria-modal="true"
      aria-label={alt}
      tabIndex={-1}
      onClick={(e) => {
        e.stopPropagation();
        onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation();
          onClose();
        }
      }}
      className="fixed inset-0 z-[60] bg-black/85 flex flex-col items-center justify-center p-6 focus:outline-none animate-fade-in"
    >
      <div className="flex items-center gap-3 mb-3" onClick={(e) => e.stopPropagation()}>
        <ProvenanceChip provenance={provenance} />
        <span className="text-xs text-text-secondary font-mono">{alt}</span>
        {natural && (
          <span className="text-xs text-text-tertiary font-mono">
            {natural.w}×{natural.h} px{small ? '（拡大表示・補間なし）' : ''}
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          className="ml-4 w-8 h-8 bg-white/15 hover:bg-white/25 rounded-full flex items-center justify-center text-white"
          aria-label="閉じる"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <img
        src={src}
        alt={alt}
        onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        style={small ? { imageRendering: 'pixelated' } : undefined}
        className="w-[92vw] h-[74vh] object-contain"
        onClick={(e) => e.stopPropagation()}
      />
      {meta && (
        <div className="mt-3 max-w-[92vw] text-xs text-text-secondary font-mono" onClick={(e) => e.stopPropagation()}>
          {meta}
        </div>
      )}
    </div>
  );
}
