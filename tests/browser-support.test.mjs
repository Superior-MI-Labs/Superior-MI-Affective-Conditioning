import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
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

test('automatic browser mode rejects unsupported WebGPU instead of silently starting huge WASM', () => {
  for (const capability of [
    {status:'ready', shaderF16:false}, {status:'adapter-unavailable', shaderF16:false},
    {status:'api-unavailable', shaderF16:false}, {status:'probe-failed', shaderF16:false},
  ]) {
    assert.throws(() => inferencePlan(capability), /Local GGUF server/);
  }
  const f16 = inferencePlan({status:'ready', shaderF16:true});
  assert.equal(f16.device,'webgpu');
  assert.equal(f16.dtype,'q4f16');
  assert.equal(f16.model, 'onnx-community/Qwen3-0.6B-ONNX');
  assert.equal(inferencePlan({status:'ready',shaderF16:false},'cpu').device, 'wasm');
  assert.throws(() => inferencePlan({status:'ready'},'webgpu'), TypeError);
});

test('runtime f16 errors and memory failures produce actionable messages', () => {
  const error=modelFailureMessage(new Error('GenerateSourceCode Program Gather requires f16 but the device does not support it'));
  assert.match(error,/CPU \/ WebAssembly mode/);
  assert.match(error,/Details: GenerateSourceCode/);
  assert.match(modelFailureMessage('GPU out of memory'),/could not allocate/);
  assert.match(modelFailureMessage('Can\'t create a session. ERROR_CODE: 6, ERROR_MESSAGE: std::bad_alloc'),/Local GGUF server/);
});

test('browser runtime selector matches the supported backend contract', () => {
  const page = readFileSync(new URL('../web/index.html', import.meta.url), 'utf8');
  assert.match(page, /<option value="auto">/);
  assert.match(page, /<option value="cpu">/);
  assert.match(page, /<option value="local">/);
  assert.doesNotMatch(page, /<option value="webgpu">/);
});
