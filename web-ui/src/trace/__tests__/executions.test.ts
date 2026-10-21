import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEpisodeModel, resolveSelection } from '../model';
import { executionRange, executionResult, executionSource, selectedExecution } from '../executions';
import { Rec } from './fixtures';

function fixture() {
  const r=new Rec();r.episodeStart();
  const q1=r.request('code_generator','first');const a1=r.response(q1,200,'exit()');
  const b1=r.blockStart(a1,'exit()', {block_index:0});
  r.blockEnd(b1,0,10,{sandbox_rc:1,task_completed:false,sim_step_range:[0,0]});
  const q2=r.request('code_generator','second');const a2=r.response(q2,200,'move()');
  const b2=r.blockStart(a2,'move()', {block_index:1});
  r.blockEnd(b2,10,0,{sandbox_rc:0,task_completed:true,sim_step_range:[0,60]});
  const q3=r.request('code_generator','finish');r.response(q3,200,'FINISH');
  const model=buildEpisodeModel(r.events);
  return {r,model,q1,q2,q3,a1,a2,b1,b2};
}
test('failure and success at the same video start keep distinct request, response and result',()=>{
  const {model,q1,q2,a1,a2}=fixture();const [first,last]=model.blocks;
  assert.equal(executionSource(model,first).request?.event_id,q1.event_id);
  assert.equal(executionSource(model,first).response?.event_id,a1.event_id);
  assert.equal(executionSource(model,last).request?.event_id,q2.event_id);
  assert.equal(executionSource(model,last).response?.event_id,a2.event_id);
  assert.equal(executionResult(first),'実行エラー');assert.equal(executionResult(last),'成功');
  assert.deepEqual(executionRange(first),{start:0,end:0});assert.deepEqual(executionRange(last),{start:0,end:60});
});
test('selection defaults to first execution and an exact model request opens only the code it generated',()=>{
  const {model,q1,q2,q3}=fixture();
  assert.equal(selectedExecution(model,null),model.blocks[0]);
  assert.equal(selectedExecution(model,resolveSelection(model,q1.event_id)),model.blocks[0]);
  assert.equal(selectedExecution(model,resolveSelection(model,q2.event_id)),model.blocks[1]);
  assert.equal(selectedExecution(model,resolveSelection(model,q3.event_id)),model.blocks[1]);
});
test('missing generation links never borrow the newest or preceding model input',()=>{
  const {r}=fixture();r.blockStart(null,'reference_code()');const m=buildEpisodeModel(r.events);
  assert.deepEqual(executionSource(m,m.blocks.at(-1)!),{request:null,response:null});
});
test('an earlier HTTP response retains its own request after a later retry',()=>{
  const r=new Rec();r.episodeStart();const q1=r.request('code_generator','retry',1);const a1=r.response(q1,200,'one');const block=r.blockStart(a1);
  const q2=r.request('code_generator','retry',2);r.response(q2,200,'two');
  const m=buildEpisodeModel(r.events);assert.equal(executionSource(m,m.blockById.get(block.event_id)!).request?.event_id,q1.event_id);
});
test('missing, invalid and unfinished execution ranges are not interpreted as no motion or success',()=>{
  const r=new Rec();r.episodeStart();const b=r.blockStart(null);let m=buildEpisodeModel(r.events);
  assert.equal(executionRange(m.blocks[0]),null);assert.equal(executionResult(m.blocks[0]),'結果未記録');
  r.blockEnd(b,null,null,{sim_step_range:[9,2]});m=buildEpisodeModel(r.events);assert.equal(executionRange(m.blocks[0]),null);
});
