/**
 * Fixed UI strings of the trace viewer.
 *
 * The Japanese labels quoted in docs/trace-format.md sections 8.2, 9.2, 9.3
 * and 9.4 are copied here verbatim and used only through this module, so a
 * reviewer can check them in one place. Strings that the spec gives only in
 * English (6.4 later-input report, 6.5 proxy note) are translated here.
 *
 * Related: components/* render these; model.ts and laterInput.ts decide when
 * each one applies.
 */

import type { ModelErrorData, SegmentCategory } from './types';

export const LABELS = {
  // 9.3 details tab names, in order.
  tabs: ['実際のモデル入力', 'モデル出力/コード', 'API/実行ログ', 'デバッグ状態', '入力差分'] as const,

  // 9.3 fixed labels (verbatim).
  notRecorded: '未記録',
  noImageInput: 'この呼び出しは画像入力なし',
  noNewLlmCall: 'このAPI実行時に新しいLLM呼び出しはありません',
  debugBanner: 'モデル入力とは別（デバッグ用）',
  normalizedPayload: '保存用正規化済みpayload',
  unitsUnknown: '単位・座標系: 不明',
  sentToModel: 'モデルに送信',
  debugFrame: 'デバッグ用フレーム（モデル入力ではない）',
  live3dTab: 'ライブ3D',
  live3dCaption: '現在のシミュレータ状態（選択中の過去イベントではありません）',
  past3dUnsupported: '過去3D未対応：記録済みフレームを表示しています',
  openWhileRunning: '終了イベントなし（実行中）',
  openProcessEnded: '終了イベントなし（プロセス終了）',
  lateEvent: 'エピソード終了後に届いたイベント',
  unusedRole: '未使用',
  modelClaimsFinish: 'モデルは完了と主張',
  // 6.4 (verbatim).
  unattributed: '由来不明',
  noLaterModelCall: 'このエピソードでは以後のモデル呼び出しなし',
  // 8.1 (verbatim).
  notCaptured: '未取得',
  // 9.2 status "incomplete" (verbatim).
  incompleteStatus: '記録が途中で止まりました（episode_end なし）',
  // 9.4 (verbatim).
  legacySection: 'トレースなし（既存ログ）',
  legacyHarnessImage: 'ハーネス保存画像（送信を検証していない）',
  legacyHarnessPrompts: 'ハーネスが保存したプロンプト文字列（送信payloadではない）',

  // 5.2 model_output_parsed: the harness reading is shown in these words.
  harnessNoRegenerateFinish: 'harness: no REGENERATE -> finish',
  // 5.2 episode_end.harness_summary.success: the batch trial harness and
  // the web trial harness set it differently.
  harnessSuccessMeaning: '最終ブロックの sandbox_rc == 0（ハーネスの補正後。途中のブロックの失敗は含まない）',
  webUiSuccessMeaning: 'task_completed または num_finishes > 0（Web UI の判定。sandbox_rc は見ない）',
  successIsNotTaskVerdict: 'タスク成功の判定とは別の値です',
  // 9.3: older traces describe success as code running without an exception.
  oldSuccessMeaningNote: '以前のレコーダーが保存した誤った説明。正しい意味は success の行を参照',
} as const;

/**
 * 5.2 model_output_parsed: the harness's reading of the reply, never the model's claim.
 * The harness parser returns "finish" whenever the reply has no REGENERATE keyword, so a
 * "finish" decision is always shown as "harness: no REGENERATE -> finish"; the recorder's
 * own `harness_reading` wins when it is present.
 */
export function harnessReadingLabel(decision: string | null | undefined, reading?: unknown): string | null {
  if (typeof reading === 'string' && reading) return `harness: ${reading}`;
  if (decision === 'finish') return LABELS.harnessNoRegenerateFinish;
  if (decision === 'regenerate') return 'harness: REGENERATE found -> regenerate';
  return decision ? `harness: ${decision}` : null;
}

/**
 * Name of a model_error for cards and tabs. A stream the caller stopped reading
 * (GeneratorExit, for example the web UI's stop button) is named as such; a logical
 * error is the call's final result.
 */
export function modelErrorLabel(d: Pick<ModelErrorData, 'error_type' | 'scope'>): string {
  if (d.error_type === 'GeneratorExit') return 'ストリーム中断（GeneratorExit: 呼び出し側が受信を打ち切り）';
  const base = d.error_type ?? 'error';
  return d.scope === 'logical' ? `論理呼び出しの失敗 ${base}` : base;
}

/** Stream states of the 6.4 later-input report (laterInput.ts StreamCheckState). */
export const STREAM_STATE_LABELS: Record<string, string> = {
  included: '含まれています',
  not_included: '含まれていません（照合済み）',
  empty: '空（出力なし）',
  short: '未確認（8 文字未満のため照合の対象外）',
  unsearched: '未確認（この要求では照合の対象に登録されていません）',
  not_recorded: '未記録（このブロックの出力イベントがありません）',
};

/** Why a check of the block's stream in a later request was not possible (unchecked states only). */
export const STREAM_UNCHECKED_REASONS: Record<string, string> = {
  short: '8 文字未満のため照合の対象外',
  unsearched: 'この要求では照合の対象に登録されていません',
  not_recorded: 'このブロックの出力イベントがありません',
  empty: '出力は空',
};

/** 8.2: 「このフレームは N step 前」. */
export function frameOffsetLabel(steps: number): string {
  return `このフレームは ${steps} step 前`;
}

/** Six-category input summary (requirements: 画像枚数 … その他; spec 6.4 categories). */
export const CATEGORY_LABELS: Record<SegmentCategory | 'images', string> = {
  images: '画像枚数',
  task_api_description: 'タスク/API説明',
  code_history: '実行コード履歴',
  stdout_stderr: 'stdout/stderr',
  vdm_description: 'VDM説明',
  other: 'その他',
};

/** Segment kind -> viewer category label, per the table in spec 6.4. */
export const SEGMENT_KIND_LABELS: Record<string, string> = {
  task_prompt: 'タスク/API説明',
  api_docs: 'タスク/API説明',
  harness_template: 'その他 (harness template)',
  code_history: '実行コード履歴',
  stdout: 'stdout/stderr',
  stderr: 'stdout/stderr',
  vdm_description: 'VDM説明',
  user_feedback: 'その他 (user)',
};

/** Status of a run's unscoped log (events with no episode, served as the episode unscoped.<pid>). */
export const UNSCOPED_STATUS_LABEL = 'エピソードに属さないイベント（unscoped）';

/** Episode status (9.2) -> Japanese label. */
export function episodeStatusLabel(status: string | null | undefined): string {
  switch (status) {
    case 'running':
      return '実行中';
    case 'ended':
      return '終了（記録完了）';
    case 'ended_with_gaps':
      return '終了（記録に欠落あり）';
    case 'incomplete':
      return LABELS.incompleteStatus;
    case 'unscoped':
      return UNSCOPED_STATUS_LABEL;
    default:
      return status ? `状態: ${status}` : '状態不明';
  }
}

/** Short status label for the run list's per-status counts. */
export function episodeStatusShortLabel(status: string | null | undefined): string {
  switch (status) {
    case 'running':
      return '実行中';
    case 'ended':
      return '終了';
    case 'ended_with_gaps':
      return '欠落あり';
    case 'incomplete':
      return '途中で停止';
    default:
      return status ?? '?';
  }
}

/** Plain-language meaning of trace_warning codes (5.2), for the gap summary. */
export const WARNING_CODE_LABELS: Record<string, string> = {
  write_failed: '保存失敗',
  unsafe_path: '安全でない保存先パス',
  disk_low: 'ディスク残量不足',
  queue_full: '書き込み待ちの上限超過',
  dropped_events: 'イベントの欠落',
  hook_install_failed: 'フック導入失敗',
  payload_unparsed: 'payload を解析できず',
  frame_unavailable: 'フレーム取得不可',
  array_budget: '配列の保存上限超過',
  tracer_frame_visible: 'レコーダーの行が出力に混入',
  ambiguous_episode: '所属エピソードを決められないイベント',
  late_event: 'エピソード終了後のイベント',
  internal_error: 'レコーダー内部エラー',
  stale_context: '前の呼び出しの記録用コンテキストが残っていた',
  state_unavailable: '正解状態を取得できず',
};

/**
 * Stored exception texts with `message_redacted: true`: the recorder removed URL
 * userinfo/query, Authorization credentials, keys of known formats (sk-, AIza, ...) and the
 * values of secret key names (the trace recorder redact), so the text differs from what
 * the program printed.
 */
export const REDACTED_MESSAGE_NOTE =
  'レコーダーが秘密情報（URL の userinfo と query、Authorization の資格情報、既知の形式のキー、秘密キー名の値）を伏せて保存した文です。プログラムが出力した文とは異なります。';

/** A short form of REDACTED_MESSAGE_NOTE for table cells and one-line lists (`*_redacted: true`). */
export const REDACTED_SHORT = '秘密情報を伏せて保存';

/**
 * Stored texts with `*_cut: true` (the trace recorder redact): the text was longer than
 * the recorder reads (64 KB), so only its start is stored, followed by a line
 * "[recorder: only the first N of M characters were read]".
 */
export const CUT_TEXT_NOTE =
  'レコーダーが読んだのは先頭の 64 KB だけです（末尾に注記の行を付けて保存）。プログラムが出力した文とは異なります。';

/**
 * array_budget_exceeded (5.3): arrays over CAPX_TRACE_ARRAY_CALL_MB and _EPISODE_MB whose
 * values were not saved (only shape, dtype and a bounded hash are kept: sha256 of the whole
 * array up to 1 MiB, of its first 1 MiB above that, none past the per-summary hash budget).
 */
export function arrayBudgetLabel(n: number): string {
  return `配列 ${n} 件は保存上限を超えたため値を保存せず（shape・dtype とハッシュのみ）`;
}

/** 6.5 note for requests sent to the local proxy (translated). */
export const PROXY_NOTE =
  'この記録はローカルの送信先へ送った本文です。送信先の判定はホストと設定されたポートによる推定です。' +
  'CaP-X 付属の capx/serving/openrouter_server.py を使う場合、プロキシはモデル名の接頭辞を外し、' +
  'temperature=0.2 と max_tokens=256 の既定値を適用します。未知のリクエスト項目は除外され、' +
  '本文は text と image_url 以外を拒否します。OpenAI SDK は 429/5xx を既定回数まで内部で再試行します。' +
  'ストリーミング以外の応答は usage と reasoning を含めずに組み直されます。';

/** 6.3: what 「保存用正規化済みpayload」 means (translated). */
export const NORMALIZED_PAYLOAD_NOTE =
  '送信本文を解析して保存用に正規化したものです。data: URL の画像・動画は assets に移し、' +
  '秘密情報のキーはマスクしています。生の本文そのものは保存していません。';

/** Statement shown for API and code-block events (9.3 tab 1). */
export const NO_CODEGEN_OR_VDM_DURING_EVENT =
  'このイベントの実行中に、コード生成モデルと VDM への呼び出しはありません。' +
  '前のモデル入力は、この時点で再送されたものではありません。';

/** Label of every ground-truth / evaluator value (1: never model input). */
export const NOT_MODEL_INPUT = 'モデル入力ではない';
