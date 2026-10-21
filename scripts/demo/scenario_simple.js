// Simple view: choose the experiment, open a trial, play its video, switch the tabs, jump
// to another code execution, play its video and return to the prompt. `p` comes from
// record_demo.py.
async (p) => {
  const d = window.__demo;
  await d.sleep(600);
  await d.choose('select', '実験を選ぶ', p.run, 700);
  await d.click('button', p.trialLabel, 1000);
  const v = document.querySelector('video');
  await d.moveTo(v); v.currentTime = 0; v.play();
  await d.sleep(1500);
  await d.click('button', 'モデル出力', 1000);
  await d.click('button', '実行ログ', 1000);
  await d.choose('select', '実行回', p.blockLabel, 700);
  const v2 = document.querySelector('video');
  await d.moveTo(v2); v2.currentTime = 0; v2.play();
  await d.sleep(1600);
  await d.click('button', '入力プロンプト', 900);
  return 'ok';
}
