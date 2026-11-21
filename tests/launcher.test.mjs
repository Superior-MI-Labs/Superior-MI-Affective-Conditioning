import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, writeFileSync, chmodSync, readFileSync, existsSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const publicOrigin = 'https://superior-mi-labs.github.io';

function launch(origin) {
  const temp = mkdtempSync(join(tmpdir(), 'affective-launcher-'));
  const model = join(temp, 'sample.gguf');
  const capture = join(temp, 'arguments.txt');
  const mock = join(temp, 'mock-llama-server');
  writeFileSync(model, 'placeholder: not a real model\n');
  writeFileSync(mock, '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n');
  chmodSync(mock, 0o700);
  const env = {...process.env, XDG_CONFIG_HOME: join(temp, 'config'), LLAMA_SERVER:mock, CAPTURE_ARGS:capture};
  delete env.SMI_CORS_ORIGIN;
  if (origin !== undefined) env.SMI_CORS_ORIGIN = origin;
  const result = spawnSync('bash', [join(root, 'scripts/run-local-qwen.sh'), model], {cwd: root, env, encoding:'utf8'});
  const args = existsSync(capture) ? readFileSync(capture, 'utf8').trimEnd().split('\n') : [];
  const key = join(temp, 'config/smi-affective/llama-api-key');
  const keyExists = existsSync(key);
  rmSync(temp, {recursive:true, force:true});
  return {result, args, keyExists};
}
function flag(args, name) {
  const index = args.indexOf(name);
  assert.notEqual(index, -1, `${name} must be present`);
  return args[index + 1];
}
test('default launcher passes one public origin and retains loopback and API key', () => {
  const {result,args,keyExists} = launch();
  assert.equal(result.status, 0, result.stderr);
  assert.equal(flag(args, '--cors-origins'), publicOrigin);
  assert.equal(flag(args, '--host'), '127.0.0.1');
  assert.equal(flag(args, '--port'), '8080');
  assert.equal(args.filter(x=>x === '--cors-origins').length, 1);
  assert.equal(keyExists, true);
  assert.match(flag(args, '--api-key-file'), /llama-api-key$/);
});
test('one selected local development origin is accepted', () => {
  for (const origin of ['http://127.0.0.1:8000', 'http://localhost:8000']) {
    const {result,args} = launch(origin);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(flag(args, '--cors-origins'), origin);
  }
});
test('multiple origins and untrusted origins are rejected', () => {
  for (const origin of [publicOrigin + ',http://localhost:8000', '*', 'https://example.com', ' ']) {
    const {result,args} = launch(origin);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Unsupported SMI_CORS_ORIGIN/);
    assert.equal(args.length, 0);
  }
});
