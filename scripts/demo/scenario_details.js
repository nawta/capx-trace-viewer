// Detailed view: open the timeline of the current trial, select an API call and a service
// call, pick a model request in the video panel and search the timeline. `p` comes from
// record_demo.py.
async (p) => {
  const d = window.__demo;
  const B = 'button,[role="button"],[role="tab"]';
  await d.sleep(500);
  await d.click(B, '詳細を見る', 1200);
  await d.click(B, p.apiCall, 900);
  await d.click(B, 'API/実行ログ', 1000);
  await d.click(B, p.serviceCall, 1000);
  await d.click(B, p.modelEvent, 1100);
  await d.click(B, '実際のモデル入力', 700);
  await d.type('input', 'タイムラインを検索', p.search, 1100);
  return 'ok';
}
