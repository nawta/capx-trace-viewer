import test from 'node:test';
import assert from 'node:assert/strict';
import TraceViewerPage from '../TraceViewerPage';
import { SimplePicker, SimpleViewer } from '../components/SimpleViewer';
import { TopBar } from '../components/TopBar';
import { TaskOutcomeBadge } from '../components/TaskOutcomeBadge';
import { SmallButton } from '../components/common';
import { advance, findAll, net, resetBrowser, setLocation } from './harness/browser';
import { mount } from './harness/fakeReact';
import { addEpisode, ev, makeRun, serve } from './pageServer';
function setup() {
  resetBrowser();const run=makeRun();
  addEpisode(run,'e1',{status:'ended',task:'cube_lift',condition:'C',seed:1,trace_complete:true},[
    ev('e1',1,'episode_start',{task:'cube_lift',condition:'C',seed:1}),
    ev('e1',2,'episode_end',{trace_complete:true}),
  ]);serve(run);
}
test('simple view hides diagnostics; details switch preserves episode and is reversible',async()=>{
  setup();setLocation('/traces?run=r1&episode=e1&event=e1_1');
  const page=mount(TraceViewerPage,{});await advance(1000);
  assert.equal(findAll(page.output,SimpleViewer).length,1);
  assert.equal(findAll(page.output,TopBar).length,0);
  const click=(label:string)=>(findAll(page.output,SmallButton).find(e=>e.props.children===label)!.props.onClick as ()=>void)();
  click('詳細を見る');await advance(0);
  assert.equal(findAll(page.output,SimpleViewer).length,0);
  assert.equal(findAll(page.output,TopBar)[0].props.episodeId,'e1');
  click('シンプル表示に戻る');await advance(0);
  assert.equal(findAll(page.output,SimpleViewer).length,1);page.unmount();
});
test('no run opens simple picker; explicit details links retain diagnostic view',async()=>{
  setup();setLocation('/traces');const first=mount(TraceViewerPage,{});await advance(1000);
  assert.equal(findAll(first.output,SimplePicker).length,1);
  assert.equal(findAll(first.output,TopBar).length,0);first.unmount();
  setLocation('/traces?view=details&run=r1&episode=e1');
  const second=mount(TraceViewerPage,{});await advance(1000);
  assert.equal(findAll(second.output,TopBar).length,1);
  assert.equal(findAll(second.output,SimpleViewer).length,0);second.unmount();
});

test('the selectable trial cards show their own task result and the outcome filter keeps the matching card', async()=>{
  resetBrowser();setLocation('/traces');const run=makeRun();
  for (const [id, success] of [['succeeded',true],['failed',false],['unknown',null]] as const) {
    addEpisode(run,id,{task:'cube_restack',condition:'S1',seed:1,status:'ended',end_reason:'completed',
      episode:{harness_summary:{task_completed:success,sandbox_rc:0}}},[]);
  }
  serve(run);const handler=net.handler;
  net.handler=request=>request.path==='/api/traces/runs'
    ? {json:{runs:[{run_id:run.runId,n_episodes:3}]}}
    : handler!(request);
  const picked:string[]=[];
  const page=mount(SimplePicker,{onPick:(run:string,episode:string)=>picked.push(`${run}/${episode}`),onDetails:()=>{}});
  await advance(1000);
  const select=findAll(page.output,'select').find(el=>el.props['aria-label']==='実験を選ぶ')!;
  (select.props.onChange as Function)({target:{value:'r1'}});await advance(1000);
  const cards=()=>findAll(page.output,'button').filter(el=>el.props['data-task-outcome']!==undefined);
  assert.deepEqual(cards().map(card=>findAll(card,TaskOutcomeBadge)[0]?.props.verdict),['success','failure','unknown']);
  (cards()[1].props.onClick as Function)();assert.deepEqual(picked,['r1/failed']);
  const filter=findAll(page.output,'select').find(el=>el.props['aria-label']==='結果で絞り込み')!;
  (filter.props.onChange as Function)({target:{value:'failure'}});await advance(0);
  assert.equal(cards().length,1);assert.equal(findAll(cards()[0],TaskOutcomeBadge)[0].props.verdict,'failure');
  page.unmount();
});
