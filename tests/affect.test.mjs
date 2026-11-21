import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULTS, clamp, initialState, stepAffect, validateVAD} from '../web/affect.mjs';

test('rejects nonfinite and out-of-range VAD', () => {
  assert.throws(() => validateVAD([0, 0, Number.NaN]), RangeError);
  assert.throws(() => validateVAD([0, 0, Infinity]), RangeError);
  assert.throws(() => validateVAD([0, 0]), TypeError);
  assert.throws(() => validateVAD([0, 0, 2]), RangeError);
  assert.throws(() => clamp(Infinity), TypeError);
});
test('neutral vector never causes divide-by-zero', () => {
  const result = stepAffect({ai:[0,0,0],phase:0}, [0,0,0]);
  assert.equal(result.alignment, 0);
  assert.ok(result.ai.every(Number.isFinite));
});
test('state transitions are bounded, deterministic and input-immutable', () => {
  for (const mode of ['vector', 'phase']) {
    const state = initialState(DEFAULTS.ai);
    const frozenBefore = JSON.stringify(state);
    const first = stepAffect(state, [1, 1, -1], {mode});
    assert.equal(JSON.stringify(state), frozenBefore);
    assert.deepEqual(first, stepAffect(state, [1, 1, -1], {mode}));
    assert.ok(first.ai.every(v => v >= -1 && v <= 1));
  }
});
test('both modes produce finite values under extreme turns', () => {
  for (const mode of ['vector', 'phase']) {
    let state = {ai:[-1,1,-1],phase:0};
    for (let i = 0; i < 1200; i++) {
      const out = stepAffect(state, i % 2 ? [1,1,1] : [-1,-1,-1], {mode});
      state = {ai:out.ai,phase:out.phase};
      assert.ok(state.ai.every(v => Number.isFinite(v) && Math.abs(v) <= 1));
      assert.ok(Number.isFinite(state.phase));
    }
  }
});
test('vector and phase preserve different temporal dynamics', () => {
  let vector = initialState([0.6,0.25,0]);
  let phase = initialState([0.6,0.25,0]);
  for (let t = 0; t < 8; t++) {
    const v = stepAffect(vector, [0.9,0.8,0.5], {mode:'vector'});
    const p = stepAffect(phase, [0.9,0.8,0.5], {mode:'phase'});
    vector = {ai:v.ai,phase:v.phase};
    phase = {ai:p.ai,phase:p.phase};
  }
  assert.notDeepEqual(vector.ai, phase.ai);
  assert.notEqual(vector.phase, phase.phase);
});
test('unexpected mode and bad phase are rejected', () => {
  assert.throws(() => stepAffect(initialState(), [0,0,0], {mode:'made-up'}));
  assert.throws(() => stepAffect({ai:[0,0,0],phase:NaN}, [0,0,0]));
});
