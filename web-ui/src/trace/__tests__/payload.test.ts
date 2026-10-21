/**
 * Reading the stored request payload (payload.ts): JSON paths of text and
 * media parts match the recorder's (`messages[1].content[0].text`,
 * `….image_url.url`), for chat bodies and the Responses-API form.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { imagesByPath, messageText, parsePayload } from '../payload';

test('chat body: text and image parts get the recorder paths; placeholders become asset refs', () => {
  const body = {
    model: 'm',
    temperature: 0.2,
    messages: [
      { role: 'system', content: 'you are a robot' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'look' },
          { type: 'image_url', image_url: { url: { $asset: 'assets/aa/a.png', sha256: 'aa', mime: 'image/png', bytes: 10 } } },
          { type: 'video_url', video_url: { url: { $asset: 'assets/bb/b.mp4', sha256: 'bb', mime: 'video/mp4', bytes: 20 } } },
          { type: 'image_url', image_url: { url: 'https://example.com/x.png' } },
          { type: 'image_url', image_url: { url: { $asset: null, written: false, reason: 'disk_low', sha256: 'cc' } } },
        ],
      },
    ],
  };
  const p = parsePayload(body);
  assert.equal(p.container, 'messages');
  assert.deepEqual(Object.keys(p.params).sort(), ['model', 'temperature']);
  assert.equal(p.messages[0].parts[0].path, 'messages[0].content');
  const parts = p.messages[1].parts;
  assert.deepEqual(parts.map((x) => x.path), [
    'messages[1].content[0].text',
    'messages[1].content[1].image_url.url',
    'messages[1].content[2].video_url.url',
    'messages[1].content[3].image_url.url',
    'messages[1].content[4].image_url.url',
  ]);
  assert.equal(parts[1].media?.asset, 'assets/aa/a.png');
  assert.equal(parts[2].media?.isVideo, true);
  assert.equal(parts[3].media?.externalUrl, 'https://example.com/x.png');
  assert.equal(parts[4].media?.asset, null);
  assert.equal(parts[4].media?.reason, 'disk_low');
  assert.match(messageText(p.messages[1]), /look\n\n\[image aa\]/);
});

test('Responses-API body (input / input_text / input_image)', () => {
  const p = parsePayload({ input: [{ role: 'user', content: [{ type: 'input_text', text: 'hi' }, { type: 'input_image', image_url: { $asset: 'assets/cc/c.png', mime: 'image/png' } }] }] });
  assert.equal(p.container, 'input');
  assert.deepEqual(p.messages[0].parts.map((x) => x.path), ['input[0].content[0].text', 'input[0].content[1].image_url']);
});

test('a body without messages or input is shown raw; images are indexed by path', () => {
  assert.equal(parsePayload({ prompt: 'x' }).container, 'none');
  assert.equal(parsePayload(null).container, 'none');
  const m = imagesByPath([{ path: 'messages[1].content[1].image_url.url', sha256: 'aa' }, { sha256: 'no path' }]);
  assert.equal(m.size, 1);
});
