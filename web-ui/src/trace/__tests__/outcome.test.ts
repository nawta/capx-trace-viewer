import test from 'node:test';
import assert from 'node:assert/strict';
import { episodeVerdict, headerVerdict } from '../outcome';
import { runnerRecordFor, episodeKeyOf } from '../runner';
import type { EpisodeSummary, RunnerRecord } from '../types';
const ep: EpisodeSummary = {episode_id:'e',task:'cube',condition:'S',seed:1,attempt_index:1,start_wall_time:1001,status:'ended'};
const record: RunnerRecord = {task:'cube',condition:'S',seed:1,attempts:[
  {attempt_id:1,started_at:new Date(1000000).toISOString(),ended_at:new Date(1100000).toISOString(),task_success:false},
  {attempt_id:2,started_at:new Date(1101000).toISOString(),ended_at:new Date(1200000).toISOString(),task_success:true}],outcome:{task_success:true}};
test('failed first attempt does not inherit successful retry', () => {
  assert.equal(episodeVerdict(ep,[record]),'failure');
  assert.equal(episodeVerdict({...ep,attempt_index:2,start_wall_time:1150},[record]),'success');
});
test('old recordings use task_completed, never execution success', () => {
  assert.equal(episodeVerdict({...ep,end_reason:'completed',episode:{harness_summary:{task_completed:false,success:true}}}),'failure');
  assert.equal(episodeVerdict({...ep,end_reason:'completed',episode:{harness_summary:{task_completed:true,success:false}}}),'success');
  assert.equal(episodeVerdict({...ep,end_reason:'completed',episode:{harness_summary:{success:true}}}),'unknown');
});
test('missing earlier attempt outcome remains unknown', () => {
  const older:RunnerRecord={...record,attempts:record.attempts?.map(({task_success,...attempt})=>attempt)};
  assert.equal(episodeVerdict(ep,[older]),'unknown');
});
test('unknown is not false; running is live', () => {
  assert.equal(episodeVerdict(ep,[{...record,attempts:[{...record.attempts![0],task_success:null}]}]),'unknown');
  assert.equal(episodeVerdict({...ep,status:'running'},[record]),'running');
});
test('header uses episode task result only after a zero-record upstream run loads',()=>{
 const summary={...ep,end_reason:'completed' as const,episode:{harness_summary:{task_completed:true}}};
 const none=runnerRecordFor([],episodeKeyOf(null,summary,summary.start_wall_time));
 assert.equal(headerVerdict(false,none,summary,'envs.runner',true,0),'成功');
 assert.equal(headerVerdict(false,none,summary,'envs.runner',false,0),'結果未確認');
  assert.equal(headerVerdict(false,none,summary,null,true,0),'結果未確認');
 assert.equal(headerVerdict(false,none,summary,'experiments.run',true,0),'結果未確認');
 assert.equal(headerVerdict(false,none,summary,'web_ui',true,0),'結果未確認');
 assert.equal(headerVerdict(false,none,summary,'envs.runner',true,1),'結果未確認');
});
