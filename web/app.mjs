import {AXES, infer, checkModel, stateDelta} from './model.mjs';

const $ = id => document.getElementById(id);
const DEFAULT = Object.freeze({current:[0.2,0.4,-0.3], observed:[0.8,0.7,0.5]});
const state = {
  current:[...DEFAULT.current], observed:[...DEFAULT.observed], target:[0,0,0],
  touched: new Set(), examples:[], transitions:0, model:null, predicted:null,
};
const format = n => `${n >= 0 ? '+' : ''}${n.toFixed(2)}`;

function makeSliders(kind) {
  const host = $(`${kind}-controls`);
  const values = kind === 'target' ? state.target : state[kind];
  for (let i=0; i<3; i++) {
    const item = document.createElement('div');
    item.className = 'slider-row';
    const label = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = AXES[i];
    const output = document.createElement('output');
    output.id = `${kind}-${i}-output`;
    output.textContent = format(values[i]);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = `${kind}-${i}`;
    slider.min = '-1';slider.max = '1';slider.step='any';
    slider.value = String(values[i]);
    slider.setAttribute('aria-label', `${kind} ${AXES[i]}`);
    label.htmlFor = slider.id;
    label.append(name, output);
    const scale = document.createElement('div');
    scale.className = 'scale';
    for (const text of ['−1.0','0','+1.0']) {
      const element = document.createElement('span');
      element.textContent = text;
      scale.append(element);
    }
    item.append(label, slider, scale);
    host.append(item);
    slider.addEventListener('input', () => {
      values[i] = Number(slider.value);
      output.textContent = format(values[i]);
      if (kind === 'target') state.touched.add(i);
      else state.touched.clear(); // User must review the target again for changed inputs.
      update();
    });
  }
}

function setSliders(kind, values) {
  const vector = kind === 'target' ? state.target : state[kind];
  for (let i=0;i<3;i++) {
    vector[i] = values[i];
    $(`${kind}-${i}`).value = String(values[i]);
    $(`${kind}-${i}-output`).textContent = format(values[i]);
  }
}

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgNode(parent, tag, attributes={}, text='') {
  const element=document.createElementNS(SVG_NS,tag);
  for (const [key,value] of Object.entries(attributes)) element.setAttribute(key,String(value));
  if (text) element.textContent=text;
  parent.append(element);
  return element;
}
function project([v,a,d]) {return [320+128*v-64*a-42*d,205+10*v-70*a+78*d];}
function renderSpace() {
  const svg=$('vad-space');
  svg.replaceChildren();
  const corners=[];
  for(const v of [-1,1])for(const a of [-1,1])for(const d of [-1,1])corners.push([v,a,d]);
  for (const from of corners) for (const axis of [0,1,2]) {
    if (from[axis]===1) continue;
    const to=[...from];to[axis]=1;
    const [x1,y1]=project(from),[x2,y2]=project(to);
    svgNode(svg,'line',{x1,y1,x2,y2,stroke:'#496178','stroke-width':1,'stroke-opacity':.55});
  }
  for(const [index,label,color] of [[0,'VALENCE','#63d8cf'],[1,'AROUSAL','#b0cada'],[2,'DOMINANCE','#d2adef']]) {
    const start=[0,0,0],end=[0,0,0];start[index]=-1;end[index]=1;
    const [x1,y1]=project(start),[x2,y2]=project(end);
    svgNode(svg,'line',{x1,y1,x2,y2,stroke:color,'stroke-width':1.5,'stroke-opacity':.72});
    svgNode(svg,'text',{x:x2+(index===0?14:index===1?-18:10),y:y2+(index===0?6:index===1?-12:17),fill:color,'font-size':12,'font-weight':700,'text-anchor':index===1?'end':'start'},label);
    svgNode(svg,'text',{x:x1-7,y:y1+7,fill:'#698096','font-size':11},'−');
  }
  const current=project(state.current),observed=project(state.observed);
  const predicted=state.predicted?project(state.predicted):null;
  const drawLine=(a,b,stroke,dash='')=>svgNode(svg,'line',{'x1':a[0],'y1':a[1],'x2':b[0],'y2':b[1],stroke,'stroke-width':2,'stroke-dasharray':dash,'stroke-opacity':.75});
  drawLine(current,observed,'#a8b0bd','5 7');
  if(predicted) drawLine(current,predicted,'#b8a0fe');
  const addPoint=(point,color)=>{
    svgNode(svg,'circle',{cx:point[0],cy:point[1],r:16,fill:color,'fill-opacity':.08});
    svgNode(svg,'circle',{cx:point[0],cy:point[1],r:6.5,fill:color,stroke:'#0d1a25','stroke-width':2});
  };
  addPoint(observed,'#ebb16f');
  addPoint(current,'#55d3ca');
  if(predicted) addPoint(predicted,'#b8a0fe');
}

function renderDeltas() {
  const host=$('delta-bars');host.replaceChildren();
  for(let i=0;i<3;i++) {
    const value=state.predicted[i]-state.current[i];
    const row=document.createElement('div');row.className='delta-row';
    const axis=document.createElement('span');axis.textContent=AXES[i];
    const track=document.createElement('div');track.className='delta-track';
    const zero=document.createElement('div');zero.className='delta-zero';
    const fill=document.createElement('div');fill.className='delta-fill';
    const pixels=Math.min(50,Math.abs(value)*25); // delta is bounded by [-2,+2]
    fill.style.left = `${value>=0?50:50-pixels}%`;
    fill.style.width=`${pixels}%`;
    fill.style.background=value>=0?'#55d3ca':'#ebb16f';
    track.append(zero,fill);
    const number=document.createElement('span');number.className='delta-value';number.textContent=format(value);
    row.append(axis,track,number);host.append(row);
  }
}
function update() {
  if (state.model) {
    state.predicted=infer(state.model,state.current,state.observed);
    $('next-vector').textContent=`[${state.predicted.map(format).join(', ')}]`;
    $('delta-vector').textContent=`[${stateDelta(state.predicted,state.current).map(format).join(', ')}]`;
    renderDeltas();
  }
  renderSpace();
  $('advance').disabled = !state.predicted;
  $('add-example').disabled = state.touched.size !== 3;
  $('export-data').disabled = state.examples.length === 0;
  $('example-count').textContent = `${state.examples.length} reviewed example${state.examples.length===1?'':'s'} in this session`;
  $('transition-count').textContent = `${state.transitions} transition${state.transitions===1?'':'s'} applied`;
}

function download(name,text,type) {
  const url=URL.createObjectURL(new Blob([text],{type}));
  const a=document.createElement('a');a.href=url;a.download=name;
  document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

for(const kind of ['current','observed','target'])makeSliders(kind);
$('reset').addEventListener('click',()=>{
  setSliders('current',DEFAULT.current);
  setSliders('observed',DEFAULT.observed);
  state.transitions=0;state.touched.clear();update();
});
$('advance').addEventListener('click',()=>{
  if(!state.predicted)return;
  setSliders('current',state.predicted);
  state.transitions++;
  state.touched.clear();update();
});
$('add-example').addEventListener('click',()=>{
  if(state.touched.size!==3)return;
  state.examples.push({source:'human_reviewed',current:[...state.current],observed:[...state.observed],next:[...state.target]});
  state.touched.clear();
  $('dataset-status').textContent='Saved in memory. Export the JSONL before closing this page.';
  update();
});
$('export-data').addEventListener('click',()=>{
  if(!state.examples.length)return;
  download('vad-reviewed.jsonl',state.examples.map(x=>JSON.stringify(x)).join('\n')+'\n','application/x-ndjson');
});

try {
  const response=await fetch('./model.json',{cache:'no-store'});
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const model=checkModel(await response.json());
  const expected=model.probe?.output;
  const actual=infer(model,model.probe?.input?.slice(0,3),model.probe?.input?.slice(3));
  if(!Array.isArray(expected)||expected.length!==3||actual.some((n,i)=>Math.abs(n-expected[i])>1e-5)){
    throw new Error('Exported weights disagree with the PyTorch reference output.');
  }
  state.model=model;
  $('model-status').textContent=`6 → 16 → 3 · ${model.source==='synthetic_toy'?'synthetic weights':'reviewed data'}`;
  $('source-note').textContent=`Training source: ${model.source==='synthetic_toy'?'artificial rule':'human reviewed'}`;
  $('metric-mse').textContent=model.metrics.test_mse.toFixed(5);
  $('metric-baseline').textContent=model.metrics.copy_state_baseline_mse.toFixed(5);
  $('metric-count').textContent=String(model.metrics.train_examples);
} catch(error) {
  $('model-status').textContent=`Model unavailable: ${error.message}`;
}
update();
