/** Group by code execution identity, never by a shared video timestamp. */
import type { EpisodeModel, Selection, TNode } from './model';
import { dataOf, type TraceEvent } from './types';

export function executionSource(model: EpisodeModel, block: TNode): {request: TraceEvent | null; response: TraceEvent | null} {
  const id = dataOf(block.primary, 'code_block_start').source_model_event_id;
  let response = id ? model.byId.get(id) : undefined;
  if (response?.event_type === 'model_output_parsed') {
    const parent = dataOf(response, 'model_output_parsed').model_event_id;
    response = parent ? model.byId.get(parent) : undefined;
  }
  if (response?.event_type !== 'model_response') return {request:null,response:null};
  const call = model.callOfEvent.get(response.event_id);
  const attempt = call?.attempts.find(a => a.responses.some(r => r.event_id === response!.event_id));
  return {response, request:attempt?.request ?? null};
}
export function selectedExecution(model: EpisodeModel, selection: Selection | null): TNode | null {
  if (!selection) return model.blocks[0] ?? null;
  for (let node: TNode | null = selection.node; node; node = node.parent) {
    if (node.kind === 'code_block') return node;
  }
  const generated = selection.node.call?.producedBlocks;
  if (generated?.length) return [...generated].sort((a,b)=>a.seq-b.seq)[0];
  // A final FINISH/evaluation belongs to the preceding execution, not a new run.
  return [...model.blocks].reverse().find(b=>b.seq <= selection.node.seq) ?? model.blocks[0] ?? null;
}
export function executionRange(block: TNode): {start:number;end:number} | null {
  if (!block.end) return null;
  const range = dataOf(block.end,'code_block_end').sim_step_range;
  const start = range?.[0], end = range?.[1];
  if (typeof start !== 'number' || typeof end !== 'number' || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start) return null;
  return {start,end};
}
export function executionResult(block: TNode): string {
  if (!block.end) return '結果未記録';
  const end = dataOf(block.end,'code_block_end');
  if (end.exception || (typeof end.sandbox_rc === 'number' && end.sandbox_rc !== 0)) return '実行エラー';
  if (end.task_completed === true) return '成功';
  if (end.task_completed === false) return '未達成';
  return '結果未記録';
}
export function executionFrame(model: EpisodeModel, block: TNode): string | null {
  const ref=dataOf(block.primary,'code_block_start').frame_ref;
  const first=Array.isArray(ref)?ref[0]:ref;
  if(first?.written !== false && first?.path) return first.path;
  const frame=[...model.frames].reverse().find(f=>f.event.seq <= block.seq && f.epoch === block.primary.sim_epoch && f.written);
  return frame?.path ?? null;
}
