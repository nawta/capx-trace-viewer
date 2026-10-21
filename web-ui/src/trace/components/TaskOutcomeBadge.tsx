import { VERDICT_LABELS, type TaskVerdict } from '../outcome';
import { Badge } from './common';
export function TaskOutcomeBadge({ verdict }: { verdict: TaskVerdict }) {
  return <Badge tone={verdict === 'success' ? 'green' : verdict === 'failure' ? 'red' : verdict === 'running' ? 'gold' : 'neutral'} title="シミュレーターのタスク達成判定">{VERDICT_LABELS[verdict]}</Badge>;
}
