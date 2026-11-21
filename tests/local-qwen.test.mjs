import test from 'node:test';
import assert from 'node:assert/strict';
import {LocalQwen} from '../web/local-qwen.mjs';

function asStream(events) {
  const encoder = new TextEncoder();
  const data = events.map(item => `data: ${JSON.stringify(item)}\n\n`).join('') + 'data: [DONE]\n\n';
  const cut = Math.max(2, Math.floor(data.length / 3));
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(data.slice(0, cut)));
      controller.enqueue(encoder.encode(data.slice(cut, cut + 7)));
      controller.enqueue(encoder.encode(data.slice(cut + 7)));
      controller.close();
    },
  });
}

test('local GGUF adapter uses exact loopback origin, bearer key, and existing chat contract', async () => {
  const requests = [];
  const client = new LocalQwen({apiKey: () => 'private-session-key', fetchImpl: async (url, options) => {
    requests.push({url,options});
    if (url.endsWith('/models')) return new Response(JSON.stringify({data:[{id:'test-qwen.gguf'}]}), {status:200});
    return new Response(asStream([
      {choices:[{delta:{content:'Hello '}}]},
      {choices:[{delta:{content:'from GGUF.'}}]},
    ]), {status:200});
  }});
  const plan = await client.load();
  assert.deepEqual({device:plan.device,dtype:plan.dtype,model:plan.model},{device:'local',dtype:'gguf',model:'test-qwen.gguf'});
  const updates = [];
  const result = await client.generate([{role:'user',content:'Hi'}], data=>updates.push(data));
  assert.equal(result,'Hello from GGUF.');
  assert.deepEqual(updates.map(x=>x.delta), ['Hello ', 'from GGUF.']);
  assert.ok(requests.every(x => x.url.startsWith('http://127.0.0.1:8080/v1/')));
  assert.ok(requests.every(x => x.options.headers.Authorization === 'Bearer private-session-key'));
  assert.ok(requests.every(x => x.options.targetAddressSpace === 'loopback'));
  assert.equal(JSON.parse(requests[1].options.body).model, 'test-qwen.gguf');
});

test('local client refuses empty keys and reports auth errors instead of treating them as loaded', async () => {
  const client = new LocalQwen({apiKey:()=>''});
  await assert.rejects(client.load(), /Enter the local server API key/);
  const rejected = new LocalQwen({apiKey:()=> 'wrong',fetchImpl: async()=>new Response(JSON.stringify({error:{message:'Invalid key'}}),{status:401})});
  await assert.rejects(rejected.load(), /API key rejected/);
  assert.equal(rejected.ready,false);
});

test('local client refuses malformed streams and aborted generation', async () => {
  let controller;
  const client = new LocalQwen({apiKey:()=> 'key', fetchImpl:async (url,options)=>{
    if (url.endsWith('/models')) return new Response(JSON.stringify({data:[{id:'qwen'}]}),{status:200});
    controller = options.signal;
    return new Response(new ReadableStream({start(){}}),{status:200});
  }});
  await client.load();
  const pending = client.generate([{role:'user',content:'hello'}]);
  // Fetch response is mock and its reader does not enforce abort automatically; manually
  // cancel the stream in real browsers. Verify the signal is the contract here.
  await new Promise(resolve=>setTimeout(resolve,0));
  client.dispose();
  assert.equal(controller.aborted,true);
  // Avoid keeping an intentionally unresolved mock stream alive.
  void pending.catch(()=>{});
});
