import test from 'node:test';
import assert from 'node:assert/strict';
import {buildQwenMessages} from '../web/qwen-prompt.mjs';
import {BrowserQwen} from '../web/qwen-client.mjs';

const input = {message: 'I got the job!', mode: 'phase', before: [0.2, 0.4, -0.3], observed: [0.9, 0.8, 0.1], after: [0.5, 0.6, -0.1], phase: 1.2, alignment: 0.75};
test('Qwen prompt actually includes both controllable mood vectors and interaction mode', () => {
  const messages = buildQwenMessages({...input, history: [{user:'Hello', assistant:'Hi there'}]});
  assert.match(messages[0].content, /oscillatory phase/);
  assert.match(messages[0].content, /\[0.20, 0.40, -0.30\]/);
  assert.match(messages[0].content, /\[0.90, 0.80, 0.10\]/);
  assert.match(messages[0].content, /\[0.50, 0.60, -0.10\]/);
  assert.deepEqual(messages.slice(1).map(x=>x.role), ['user','assistant','user']);
  assert.match(messages.at(-1).content, /\/no_think$/);
});
test('bad VAD, mode or nonfinite phase fail before inference', () => {
  assert.throws(() => buildQwenMessages({...input, phase: Number.NaN}), TypeError);
  assert.throws(() => buildQwenMessages({...input, observed: [0, 0, 5]}), RangeError);
  assert.throws(() => buildQwenMessages({...input, mode: 'unknown'}), TypeError);
});
test('browser model adapter reuses one worker and separates loading from generation', async () => {
  const handlers = new Map();
  let posts = 0;
  const fake = {
    addEventListener(type, callback) {handlers.set(type, callback);},
    postMessage({id, type}) {
      posts++;
      queueMicrotask(() => handlers.get('message')({data:{id,type:'result',value:type==='generate'?'Hello, world!':true}}));
    }
  };
  const client = new BrowserQwen({workerFactory:()=>fake});
  await Promise.all([client.load(), client.load()]);
  assert.equal(posts, 1);
  const response = await client.generate([{role:'user',content:'Hi'}]);
  assert.equal(response, 'Hello, world!');
  assert.equal(posts, 2);
});
test('worker failures surface instead of silently displaying templates', async () => {
  const handlers = new Map();
  const fake = {
    addEventListener(type, callback) {handlers.set(type, callback);},
    postMessage({id}) {queueMicrotask(()=>handlers.get('message')({data:{id,type:'error',message:'WebGPU unavailable'}}));}
  };
  const client = new BrowserQwen({workerFactory:()=>fake});
  await assert.rejects(client.load(), /WebGPU unavailable/);
  assert.equal(client.ready, false);
});
