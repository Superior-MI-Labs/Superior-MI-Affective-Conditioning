import test from 'node:test';
import assert from 'node:assert/strict';
import {browserFamily, browserGuidance, inspectWebGPU, inferencePlan, modelFailureMessage} from '../web/browser-support.mjs';

test('browser classification and manual Vivaldi override are honest', () => {
  assert.equal(browserFamily('Mozilla/5.0 Vivaldi/7.2 Chrome/130.0'), 'vivaldi');
  assert.equal(browserFamily('Mozilla/5.0 Chrome/130.0', true), 'vivaldi');
  assert.equal(browserFamily('Mozilla/5.0 Chrome/130.0'), 'chromium');
  assert.equal(browserFamily('Mozilla/5.0 Edg/130.0 Chrome/130.0'), 'edge');
  assert.equal(browserFamily('Mozilla/5.0 Firefox/130.0'), 'firefox');
  const guide = browserGuidance('vivaldi').steps.join(' ');
  assert.match(guide, /vivaldi:\/\/settings/);
  assert.match(guide, /enable-unsafe-webgpu/);
  assert.match(guide, /at your own risk/);
});

test('WebGPU probe distinguishes missing API, missing adapter, unsupported and supported f16', async () => {
  assert.deepEqual(await inspectWebGPU(undefined), {status:'api-unavailable',shaderF16:false});
  assert.deepEqual(await inspectWebGPU({requestAdapter:async()=>null}), {status:'adapter-unavailable',shaderF16:false});
  const noF16=await inspectWebGPU({requestAdapter:async()=>({features:new Set()})});
  assert.deepEqual(noF16, {status:'ready',shaderF16:false});
  const yesF16=await inspectWebGPU({requestAdapter:async()=>({features:new Set(['shader-f16'])})});
  assert.deepEqual(yesF16, {status:'ready',shaderF16:true});
  assert.equal((await inspectWebGPU({requestAdapter:async()=>{throw Error('blocked')}})).status,'probe-failed');
});

test('the model never selects an f16 dtype when the adapter does not expose shader-f16', () => {
  assert.deepEqual(inferencePlan({status:'ready',shaderF16:false}),
    {device:'webgpu',dtype:'q4',reason:'16-bit shaders unavailable: use float32-compatible q4'});
  assert.equal(inferencePlan({status:'ready',shaderF16:true}).dtype,'q4f16');
  assert.equal(inferencePlan({status:'api-unavailable'},'cpu').device,'wasm');
  assert.throws(()=>inferencePlan({status:'adapter-unavailable'}),/CPU/);
  assert.throws(()=>inferencePlan({status:'ready'},'unknown'),TypeError);
});

test('runtime f16 errors and memory failures produce actionable messages', () => {
  const error=modelFailureMessage(new Error('GenerateSourceCode Program Gather requires f16 but the device does not support it'));
  assert.match(error,/CPU \(slow\)/);
  assert.match(error,/Details: GenerateSourceCode/);
  assert.match(modelFailureMessage('GPU out of memory'),/Insufficient/);
});
