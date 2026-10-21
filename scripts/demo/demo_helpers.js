// Helpers for the scripted demo recordings (scripts/demo/record_demo.py): a visible cursor dot
// that moves to each element before it is clicked, and paced clicks, selections and typing.
window.__demo = (() => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let dot = document.getElementById('__demo_dot');
  if (!dot) {
    dot = document.createElement('div');
    dot.id = '__demo_dot';
    Object.assign(dot.style, {
      position: 'fixed', left: '720px', top: '450px', width: '22px', height: '22px',
      marginLeft: '-11px', marginTop: '-11px', borderRadius: '50%',
      background: 'rgba(230,60,60,0.55)', border: '2px solid rgba(255,255,255,0.9)',
      boxShadow: '0 0 6px rgba(0,0,0,0.5)', zIndex: 2147483647, pointerEvents: 'none',
      transition: 'left 330ms ease, top 330ms ease, transform 150ms ease',
    });
    document.body.appendChild(dot);
  }
  const find = (sel, text) => {
    // The shortest matching element is the most specific one (a timeline row, not the code block that mentions it).
    const els = [...document.querySelectorAll(sel)].filter(
      (e) => (e.getAttribute('aria-label') || '').includes(text) || e.textContent.includes(text));
    return els.reduce((best, e) => (!best || e.textContent.length < best.textContent.length ? e : best), null);
  };
  async function waitFor(sel, text, ms = 5000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const e = find(sel, text);
      if (e) return e;
      await sleep(100);
    }
    throw new Error('not found: ' + sel + ' ' + text);
  }
  async function moveTo(el) {
    el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const r = el.getBoundingClientRect();
    dot.style.left = r.left + Math.min(r.width / 2, 120) + 'px';
    dot.style.top = r.top + r.height / 2 + 'px';
    await sleep(360);
  }
  async function press() {
    dot.style.transform = 'scale(0.7)';
    await sleep(150);
    dot.style.transform = 'scale(1)';
  }
  async function click(sel, text, after = 1200) {
    const el = await waitFor(sel, text);
    await moveTo(el);
    await press();
    el.click();
    await sleep(after);
    return el;
  }
  async function choose(sel, text, optionText, after = 1200) {
    const el = await waitFor(sel, text);
    await moveTo(el);
    await press();
    const opt = [...el.options].find((o) => o.text.includes(optionText));
    const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
    setter.call(el, opt.value);
    el.dispatchEvent(new Event('change', { bubbles: true }));
    await sleep(after);
  }
  async function type(sel, text, value, after = 1200) {
    const el = await waitFor(sel, text);
    await moveTo(el);
    await press();
    el.focus();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    for (let i = 1; i <= value.length; i++) {
      setter.call(el, value.slice(0, i));
      el.dispatchEvent(new Event('input', { bubbles: true }));
      await sleep(70);
    }
    await sleep(after);
  }
  return { sleep, click, choose, type, waitFor, moveTo };
})();
