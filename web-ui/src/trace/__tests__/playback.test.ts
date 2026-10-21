import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEpisodeModel } from '../model';
import { cameraVideos, canPlayExecutionVideo, chooseExecutionVideo, cueAt, episodeVideos, playbackMap, trialVideos, validClip, videosForEpisode } from '../playback';
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
test('runner artifact lookup uses only the original combined MP4',()=>{
 const m=match();m.record!.artifacts={trial_dir:'/tmp/runs/run/task/C/trial_1',videos:[
  '/tmp/runs/run/task/C/trial_1/video_combined_main.mp4',
  '/tmp/runs/run/task/C/trial_1/video_combined.mp4']};
 assert.deepEqual(trialVideos('run',m,['/tmp/runs']).map(v=>v.name),['video_combined.mp4']);
});
test('earlier and unverified attempts never receive the final attempt video',()=>{
 assert.deepEqual(trialVideos('run',{...match(),isFinal:false},['/tmp/runs']),[]);
 assert.deepEqual(trialVideos('run',{...match(),unverified:true},['/tmp/runs']),[]);
});
test('rejects outside and traversal video paths',()=>{
 const m=match();m.record!.artifacts={trial_dir:'/tmp/runs/run/task/C/trial_1',videos:['/outside/video_combined.mp4','/tmp/runs/run/task/C/trial_1/../video_combined.mp4']};
 assert.deepEqual(trialVideos('run',m,['/tmp/runs']),[]);
});
test('upstream episode videos use only written, declared files and retain turn indexes',()=>{
 const r=new Rec();r.episodeStart();r.episodeEnd();
 r.add('trial_videos',{}, {artifacts:[
  {path:'videos/video_turn_00.mp4',mime:'video/mp4',written:true,turn_index:0},
  {path:'videos/video_combined.mp4',mime:'video/mp4',written:true,turn_index:null},
  {path:'videos/video_turn_01.mp4',mime:'video/mp4',written:false,turn_index:1},
  {path:'videos/../secret.mp4',mime:'video/mp4',written:true},
 ]});
 const videos=episodeVideos({runId:'r',episodeId:'e',generation:'gen'},buildEpisodeModel(r.events));
 assert.deepEqual(videos.map(v=>[v.name,v.turnIndex??null]),[['video_turn_00.mp4',0],['video_combined.mp4',null]]);
 assert.match(videos[0].url,/\/episodes\/e\/file\/videos\/video_turn_00\.mp4/);
 assert.match(videos[0].url,/g=gen/);
});
test('turn lookup uses recorded block indexes across missing events and regeneration',()=>{
 const r=new Rec();r.episodeStart();r.blockStart(null,'a',{block_index:0});
 r.request('code_generator','regeneration');r.blockStart(null,'b',{block_index:2});
 const model=buildEpisodeModel(r.events);
 const videos=[0,1,2].map(i=>({name:`video_turn_0${i}.mp4`,url:`/turn/${i}`,turnIndex:i}));
 assert.equal(chooseExecutionVideo(model.blocks[0],videos).video?.turnIndex,0);
 assert.equal(chooseExecutionVideo(model.blocks[1],videos).video?.turnIndex,2);
 assert.equal(chooseExecutionVideo(model.blocks[1],videos.filter(v=>v.turnIndex!==2)).video,undefined);
 assert.equal(chooseExecutionVideo(model.blocks[1],[...videos.filter(v=>v.turnIndex!==2),{name:'video_combined_main.mp4',url:'/episodes/e/file/combined'}]).video?.name,'video_combined_main.mp4');
 assert.equal(chooseExecutionVideo(model.blocks[1],[...videos.filter(v=>v.turnIndex!==2),
  {name:'video_combined_wrist.mp4',url:'/wrist'},{name:'video_combined_main.mp4',url:'/main'}]).video?.name,'video_combined_main.mp4');
 const unknown=new Rec();unknown.episodeStart();unknown.blockStart(null,'c',{block_index:null});
 assert.equal(chooseExecutionVideo(buildEpisodeModel(unknown.events).blocks[0],videos).video,undefined);
 assert.deepEqual(cameraVideos([...videos,{name:'video_combined_main.mp4',url:'/combined'}]).map(v=>v.name),['video_combined_main.mp4']);
 assert.ok(playbackMap(model,'video_combined_main.mp4',2).reason);
 assert.equal(canPlayExecutionVideo({name:'video_combined_main.mp4',url:'/episodes/e',source:'episode'},false,null),true);
 assert.equal(canPlayExecutionVideo({name:'video_combined.mp4',url:'/legacy',source:'runner'},false,null),false);
 assert.equal(validClip({start:1,end:1}),null);
 assert.equal(validClip({start:2,end:1}),null);
 assert.equal(canPlayExecutionVideo({name:'video_combined.mp4',url:'/episodes/e',source:'episode'},false,validClip({start:1,end:1})),true);
});
test('a runner record with no usable MP4 falls back to copied episode video',()=>{
 const r=new Rec();r.episodeStart();r.episodeEnd();r.add('trial_videos',{}, {artifacts:[{path:'videos/video_combined_wrist.mp4',mime:'video/mp4',written:true,turn_index:null}]});
 const model=buildEpisodeModel(r.events);
 const ep={runId:'run',episodeId:'ep',generation:'new'};
 assert.deepEqual(videosForEpisode(ep,model,{...match(),isFinal:false},['/tmp/runs']).map(v=>v.name),['video_combined_wrist.mp4']);
 assert.match(videosForEpisode(ep,model,{...match(),isFinal:false},['/tmp/runs'])[0].url,/g=new/);
 assert.equal(videosForEpisode(ep,model,match(),['/tmp/runs'])[0].name,'video_combined.mp4');
});
test('a stale turn falls back to the copied combined video',()=>{
 const r=new Rec();r.episodeStart();r.blockStart(null,'turn1',{block_index:1});r.episodeEnd();
 r.add('trial_videos',{}, {artifacts:[
  {path:'videos/video_turn_01.mp4',mime:'video/mp4',written:false,reason:'stale',turn_index:1},
  {path:'videos/video_combined.mp4',mime:'video/mp4',written:true,turn_index:null}]});
 const model=buildEpisodeModel(r.events);
 assert.equal(chooseExecutionVideo(model.blocks[0],episodeVideos({runId:'r',episodeId:'e'},model)).video?.name,'video_combined.mp4');
});
