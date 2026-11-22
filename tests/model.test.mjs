import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {infer, checkVAD, checkModel, stateDelta} from '../web/model.mjs';

const demo = JSON.parse(readFileSync(new URL('../web/model.json', import.meta.url), 'utf8'));

test('exported checkpoint matches original PyTorch output', () => {
  checkModel(demo);
  const actual=infer(demo,demo.probe.input.slice(0,3),demo.probe.input.slice(3));
  for(let i=0;i<3;i++)assert.ok(Math.abs(actual[i]-demo.probe.output[i])<1e-6,
    `axis ${i}: browser value differs from PyTorch`);
});

test('network returns bounded, finite next VAD for extreme states',()=>{
  for(const current of [[-1,-1,-1],[0,0,0],[1,1,1]]){
    for(const observed of [[-1,1,-1],[0,0,0],[1,-1,1]]){
      const p=infer(demo,current,observed);
      assert.equal(p.length,3);
      assert.ok(p.every(v=>Number.isFinite(v)&&Math.abs(v)<=1));
    }
  }
});

test('changes in all three observed axes affect the actual neural output',()=>{
  const original=[0.2,0.4,-0.3];
  const reference=infer(demo,original,[0,0,0]);
  for(let i=0;i<3;i++){
    const observed=[0,0,0];observed[i]=1;
    const next=infer(demo,original,observed);
    assert.ok(Math.max(...next.map((v,k)=>Math.abs(v-reference[k])))>.05,
      `observed axis ${i} has no meaningful influence`);
  }
});

test('invalid inputs and malformed checkpoint fail rather than making up a state',()=>{
  assert.throws(()=>checkVAD([0,0,2]),RangeError);
  assert.throws(()=>checkVAD([0,0,NaN]),RangeError);
  assert.throws(()=>checkVAD([0,0]),RangeError);
  assert.throws(()=>infer(demo,[0,0,0],[0,NaN,0]),RangeError);
  assert.throws(()=>checkModel({...demo,layers:[]}),/Unsupported/);
  const corrupted=structuredClone(demo);
  corrupted.layers[0].weights[0].pop();
  assert.throws(()=>checkModel(corrupted),/Invalid exported neural weights/);
});

test('delta is predicted minus previous internal state',()=>{
  const delta=stateDelta([.4,.3,-.2],[.2,.4,-.3]);
  for (const [i, expected] of [.2,-.1,.1].entries()) assert.ok(Math.abs(delta[i]-expected)<1e-10);
});
