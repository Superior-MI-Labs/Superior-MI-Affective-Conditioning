import test from 'node:test';
import assert from 'node:assert/strict';
import {buildQwenMessages} from '../web/qwen-prompt.mjs';
import {BrowserQwen} from '../web/qwen-client.mjs';
import {probeGeneration} from '../web/qwen-probe.mjs';

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

test('inference backend changes require an explicit reload and preserve returned plan', async () => {
  const handlers = new Map();
  const requests = [];
  const fake = {
    addEventListener(type, callback) {handlers.set(type, callback);},
    postMessage(data) {
      requests.push(data);
      const value = data.type === 'load'
        ? {device:data.preference === 'cpu'?'wasm':'webgpu',dtype:data.preference === 'cpu'?'q4':'q4f16'}
        : {status:'ready',shaderF16:true};
      queueMicrotask(() => handlers.get('message')({data:{id:data.id,type:'result',value}}));
    },
  };
  const client = new BrowserQwen({workerFactory:()=>fake});
  assert.equal((await client.diagnose()).shaderF16,true);
  await client.load('auto');
  assert.equal(client.plan.dtype,'q4f16');
  await client.load('auto');
  assert.equal(requests.filter(x=>x.type==='load').length,1);
  await client.load('cpu');
  assert.equal(client.plan.device,'wasm');
  assert.equal(client.preference,'cpu');
  assert.equal(requests.filter(x=>x.type==='load').length,2);
});

test('streaming progress is delivered to the active generation without resolving it early', async () => {
  const handlers = new Map();
  const fake = {
    addEventListener(type, handler) {handlers.set(type, handler);},
    postMessage({id, type}) {
      if (type === 'load') {
        queueMicrotask(() => handlers.get('message')({data:{id, type:'result', value:{device:'webgpu', dtype:'q4'}}}));
      } else if (type === 'generate') {
        queueMicrotask(() => {
          for (const data of [
            {id, type:'progress', tokens:1},
            {id, type:'progress', delta:'Hello ', tokens:1},
            {id, type:'progress', delta:'world', tokens:2},
            {id, type:'result', value:'Hello world'},
          ]) handlers.get('message')({data});
        });
      }
    },
  };
  const client = new BrowserQwen({workerFactory: () => fake});
  await client.load();
  const updates = [];
  const result = await client.generate([{role:'user',content:'Hi'}], (update) => updates.push(update));
  assert.equal(result, 'Hello world');
  assert.deepEqual(updates.map(x => x.delta).filter(Boolean), ['Hello ', 'world']);
  assert.equal(updates.at(-1).tokens, 2);
  assert.equal(client.pending.size, 0);
});

test('canceling generation terminates the worker and rejects outstanding requests', async () => {
  const handlers = new Map();
  let terminated = false;
  const fake = {
    addEventListener(type, handler) {handlers.set(type, handler);},
    postMessage({id, type}) {
      if (type === 'load') queueMicrotask(() => handlers.get('message')({data:{id, type:'result',value:{device:'webgpu',dtype:'q4'}}}));
    },
    terminate() {terminated = true;},
  };
  const client = new BrowserQwen({workerFactory: () => fake});
  await client.load();
  const inflight = client.generate([{role:'user',content:'Hi'}]);
  client.dispose();
  await assert.rejects(inflight, /Generation canceled/);
  assert.equal(terminated,true);
  assert.equal(client.ready,false);
  assert.equal(client.pending.size,0);
});

test('Qwen2.5 prompt must not inherit the Qwen3 thinking control', () => {
  const prompt = buildQwenMessages({...input, supportsThinking:false});
  assert.equal(prompt.at(-1).content, input.message);
});

test('model is not verified until real token computation finishes', async () => {
  const events = [];
  const generator = async (messages, options) => {
    events.push({messages, options});
    return [{generated_text: [{role:'assistant',content:'Hi'}]}];
  };
  await probeGeneration(generator);
  assert.equal(events.length, 1);
  assert.equal(events[0].options.max_new_tokens, 3);
  assert.equal(events[0].options.min_new_tokens, 3);
  await assert.rejects(probeGeneration(async () => {throw Error('Gather requires f16');}), /Gather requires f16/);
});

test('readiness probe reaches token decoding rather than only prompt prefill', async () => {
  await assert.rejects(
    probeGeneration(async (_messages, options) => {
      if (options.max_new_tokens > 1) throw new Error('decode kernel failed');
    }),
    /decode kernel failed/,
  );
});
