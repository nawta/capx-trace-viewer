/** Task verdict for this attempt, never a later retry's result. */
import { episodeKeyOf, runnerRecordFor } from './runner';
import type { EpisodeSummary, RunnerRecord } from './types';
export type TaskVerdict = 'success' | 'failure' | 'unknown' | 'running';
export const VERDICT_LABELS: Record<TaskVerdict, string> = {success:'成功', failure:'失敗', unknown:'未判定', running:'実行中'};
const verdict = (value: unknown): TaskVerdict => value === true ? 'success' : value === false ? 'failure' : 'unknown';
export function episodeVerdict(episode: EpisodeSummary, records: RunnerRecord[] = []): TaskVerdict {
  if (episode.status === 'running') return 'running';
  const match = runnerRecordFor(records, episodeKeyOf(null, episode, episode.start_wall_time));
  if (match.attempt && Object.hasOwn(match.attempt, 'task_success')) return verdict(match.attempt.task_success);
  if (match.isFinal && match.record?.outcome) return verdict(match.record.outcome.task_success);
  const summary = episode.episode?.harness_summary;
  if (summary && typeof summary === 'object' && !Array.isArray(summary)) {
    const data = summary as Record<string, unknown>;
    // Generic `success` is code execution, and must not decide task success.
    if (episode.end_reason === 'completed' || (typeof data.sandbox_rc === 'number' && data.sandbox_rc >= 0)) return verdict(data.task_completed);
  }
  return 'unknown';
}
