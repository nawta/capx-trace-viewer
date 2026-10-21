import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEpisodeModel } from '../model';
import { cueAt, playbackMap, trialVideos } from '../playback';
import type { RunnerMatch } from '../runner';
import { Rec } from './fixtures';
import { dataOf } from '../types';

function recording() {
 const r=new Rec();r.episodeStart();
 r.add('frame',{camera:'main',epoch:0,buffer_index:0,sim_step:0});
 const first=r.request('code_generator','first');first.sim_step=0;first.sim_epoch=0;
 const second=r.request('code_generator','second');second.sim_step=30;second.sim_epoch=0;
 const third=r.request('vdm','third');third.sim_step=30;third.sim_epoch=0;
 r.add('frame',{camera:'main',epoch:0,buffer_index:59,sim_step:59});
 return {r,first,second,third};
}
test('maps simulator frames, not wall-clock inference waits; preserves prompts sharing a video position',()=>{
 const {r}=recording();const m=playbackMap(buildEpisodeModel(r.events),'video_combined.mp4',2);
 assert.equal(m.reason,null);assert.deepEqual(m.prompts.map(c=>c.time),[0,1,1]);
 assert.equal(cueAt(m.prompts,.5)?.node.call?.id,'first');
 assert.equal(cueAt(m.prompts,1)?.node.call?.id,'third');
 assert.equal(cueAt(m.prompts,.1)?.node.call?.id,'first','backward seek must restore earlier prompt');
});
test('refuses synchronization for resets, mismatched length and unrecorded cameras',()=>{
 const {r}=recording();let m=buildEpisodeModel(r.events);
 assert.ok(playbackMap(m,'video_combined.mp4',10).reason);
 assert.ok(playbackMap(m,'video_combined_wrist.mp4',2).reason);
 r.add('env_reset',{});m=buildEpisodeModel(r.events);
 assert.ok(playbackMap(m,'video_combined.mp4',2).reason);
});
test('refuses partial recording and nonzero epochs rather than stretching it over a full video',()=>{
 const {r}=recording();dataOf(r.events.find(e=>e.event_type==='frame'), 'frame').buffer_index=12;
 assert.ok(playbackMap(buildEpisodeModel(r.events),'video_combined.mp4',2).reason);
 dataOf(r.events.find(e=>e.event_type==='frame'), 'frame').buffer_index=0;
 dataOf(r.events.find(e=>e.event_type==='frame'), 'frame').epoch=1;
 assert.ok(playbackMap(buildEpisodeModel(r.events),'video_combined.mp4',2).reason);
});
function match():RunnerMatch {
 return {record:{task:'task',condition:'C',artifacts:{trial_dir:'/tmp/runs/run/task/C/trial_1',videos:['/tmp/runs/run/task/C/trial_1/video_combined.mp4']}},isFinal:true,unverified:false,attempt:null,finalAttempt:null,attemptId:1,candidates:1,problem:null};
}
test('video lookup preserves registered root identity, never guesses from seed',()=>{
 const m=match();assert.match(trialVideos('run',m,['/other','/tmp/runs'])[0].url,/root=1$/);
 assert.deepEqual(trialVideos('run',m,['/different']),[]);
 assert.deepEqual(trialVideos('other-run',m,['/tmp/runs']),[]);
});
test('earlier and unverified attempts never receive the final attempt video',()=>{
 assert.deepEqual(trialVideos('run',{...match(),isFinal:false},['/tmp/runs']),[]);
 assert.deepEqual(trialVideos('run',{...match(),unverified:true},['/tmp/runs']),[]);
});
test('rejects outside and traversal video paths',()=>{
 const m=match();m.record!.artifacts={trial_dir:'/tmp/runs/run/task/C/trial_1',videos:['/outside/video_combined.mp4','/tmp/runs/run/task/C/trial_1/../video_combined.mp4']};
 assert.deepEqual(trialVideos('run',m,['/tmp/runs']),[]);
});
