import {DEFAULTS, clamp, initialState, stepAffect, affectName, templateResponse} from './affect.mjs';
import {buildQwenMessages} from './qwen-prompt.mjs';
import {BrowserQwen} from './qwen-client.mjs';
import {BROWSER_NAMES, browserFamily, browserGuidance} from './browser-support.mjs';

const $ = (id) => document.getElementById(id);
const copy = (a) => [...a];
const fmt = (n) => `${n >= 0 ? '+' : ''}${n.toFixed(2)}`;

const state = {
  current: initialState(DEFAULTS.ai),
  observed: copy(DEFAULTS.message),
  mode: 'vector',
  evolve: true,
  turns: [],
  recent: null,
  engine: 'simulation',
  busy: false,
};

let qwen;
const inferredBrowser = browserFamily(navigator.userAgent, Boolean(window.vivaldi));
function ensureQwen() {
  qwen ??= new BrowserQwen({onStatus: setModelStatus});
  return qwen;
}
function renderBrowserGuide() {
  const selected = $('browser-choice').value;
  const family = selected === 'auto' ? inferredBrowser : selected;
  const {steps, common} = browserGuidance(family);
  const detected = BROWSER_NAMES[inferredBrowser];
  $('browser-detected').textContent = `Detected: ${detected}. Some browsers (including Vivaldi) identify themselves as Chrome; select Vivaldi manually if needed.`;
  const list = $('browser-steps');
  list.replaceChildren();
  for (const step of steps) {
    const item = document.createElement('li');
    item.textContent = step;
    list.append(item);
  }
  $('browser-warning').textContent = common;
}
async function checkWebGPU() {
  $('gpu-health').textContent = 'Checking GPU adapter…';
  $('gpu-check').disabled = true;
  try {
    const capability = await ensureQwen().diagnose();
    if (capability.status === 'ready') {
      $('gpu-health').textContent = `WebGPU adapter available · shader-f16 ${capability.shaderF16 ? 'supported' : 'unavailable'}`;
      $('gpu-recommendation').textContent = capability.shaderF16
        ? 'WebGPU will load the q4f16 model, which uses half-precision shader operations.'
        : 'WebGPU is available but shader-f16 is not. WebGPU will select the q4 model (float32-compatible activations); CPU mode remains available if inference still fails.';
    } else {
      $('gpu-health').textContent = `WebGPU unavailable (${capability.status})`;
      $('gpu-recommendation').textContent = 'Use the browser setup guide below, or choose CPU / WebAssembly explicitly. CPU inference can be very slow and needs substantial RAM.';
      $('browser-help').open = true;
    }
  } catch (error) {
    $('gpu-health').textContent = 'Could not inspect WebGPU';
    $('gpu-recommendation').textContent = `Check failed: ${error.message}. Use the browser guide or CPU mode.`;
    $('browser-help').open = true;
  } finally {
    $('gpu-check').disabled = state.busy;
  }
}
function setModelStatus(message) { $('model-status').textContent = message; }
function setBusy(value) {
  state.busy = value;
  for (const id of ['submit-turn', 'reset', 'mode-vector', 'mode-phase', 'engine-simulation', 'engine-qwen', 'evolve', 'load-qwen', 'model-backend', 'ai-d', 'msg-d']) $(id).disabled = value;
  $('message').disabled = value;
  $('gpu-check').disabled = value;
  for (const prefix of ['ai', 'msg']) $(prefix + '-plot').setAttribute('aria-disabled', String(value));
}


function render() {
  for (const [prefix, value] of [['ai', state.current.ai], ['msg', state.observed]]) {
    $(prefix + '-point').style.left = `${50 * (value[0] + 1)}%`;
    $(prefix + '-point').style.top = `${50 * (1 - value[1])}%`;
    $(prefix + '-v').textContent = fmt(value[0]);
    $(prefix + '-a').textContent = fmt(value[1]);
    $(prefix + '-d').value = String(value[2]);
    $(prefix + '-d-value').textContent = fmt(value[2]);
    $(prefix + '-name').textContent = affectName(value);
  }
  $('phase').textContent = fmt(state.current.phase);
  $('turns').textContent = String(state.turns.length);
  $('alignment').textContent = state.recent ? fmt(state.recent.alignment) : '—';
  $('gain').textContent = state.recent ? state.recent.gain.toFixed(3) : '—';
  for (const mode of ['vector', 'phase']) {
    const selected = state.mode === mode;
    $('mode-' + mode).classList.toggle('selected', selected);
    $('mode-' + mode).setAttribute('aria-pressed', String(selected));
  }
  $('evolve').checked = state.evolve;
  for (const engine of ['simulation', 'qwen']) {
    const selected = state.engine === engine;
    $('engine-' + engine).classList.toggle('selected', selected);
    $('engine-' + engine).setAttribute('aria-pressed', String(selected));
  }
  $('model-hint').textContent = state.engine === 'qwen'
    ? 'Real Qwen3-0.6B output, prompt-conditioned with numerical affect. No affective LoRA training yet.'
    : 'Deterministic example replies; no neural model runs in this mode.';
}

function setVectorPoint(prefix, v, a) {
  if (state.busy) return;
  const target = prefix === 'ai' ? state.current.ai : state.observed;
  target[0] = clamp(v);
  target[1] = clamp(a);
  if (prefix === 'ai') state.current.phase = Math.atan2(target[1], target[0]);
  state.recent = null;
  render();
}

function makePlotInteractive(prefix) {
  const plot = $(prefix + '-plot');
  const updateFromPointer = (event) => {
    if (state.busy) return;
    const rect = plot.getBoundingClientRect();
    setVectorPoint(prefix,
      2 * (event.clientX - rect.left) / rect.width - 1,
      1 - 2 * (event.clientY - rect.top) / rect.height);
  };
  plot.addEventListener('pointerdown', (event) => {
    if (state.busy || event.button !== 0) return;
    plot.setPointerCapture(event.pointerId);
    updateFromPointer(event);
  });
  plot.addEventListener('pointermove', (event) => {
    if (plot.hasPointerCapture(event.pointerId)) updateFromPointer(event);
  });
  plot.addEventListener('keydown', (event) => {
    if (state.busy) return;
    const vector = prefix === 'ai' ? state.current.ai : state.observed;
    const increments = {ArrowLeft:[-0.05,0],ArrowRight:[0.05,0],ArrowUp:[0,0.05],ArrowDown:[0,-0.05]};
    const increment = increments[event.key];
    if (!increment) return;
    event.preventDefault();
    setVectorPoint(prefix, vector[0] + increment[0], vector[1] + increment[1]);
  });
}

function appendBubble(role, content, label) {
  const list = $('conversation');
  const empty = list.querySelector('.empty');
  if (empty) empty.remove();
  const node = document.createElement('div');
  node.className = `bubble ${role}`;
  const caption = document.createElement('small');
  caption.textContent = label;
  const text = document.createElement('div');
  text.textContent = content;
  node.append(caption, text);
  list.append(node);
  list.scrollTop = list.scrollHeight;
}

function downloadJSON(name, data) {
  const blob = new Blob([JSON.stringify(data, null, 2) + '\n'], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

for (const prefix of ['ai', 'msg']) {
  makePlotInteractive(prefix);
  $(prefix + '-d').addEventListener('input', (event) => {
    if (state.busy) return;
    const target = prefix === 'ai' ? state.current.ai : state.observed;
    target[2] = Number(event.target.value);
    state.recent = null;
    render();
  });
}
for (const mode of ['vector', 'phase']) {
  $('mode-' + mode).addEventListener('click', () => {
    if (state.busy) return;
    state.mode = mode;
    state.recent = null;
    render();
  });
}
$('evolve').addEventListener('change', (event) => {state.evolve = event.target.checked;render();});
for (const engine of ['simulation', 'qwen']) {
  $('engine-' + engine).addEventListener('click', () => {
    if (state.busy) return;
    state.engine = engine;
    render();
  });
}
$('browser-choice').addEventListener('change', renderBrowserGuide);
$('gpu-check').addEventListener('click', checkWebGPU);
$('load-qwen').addEventListener('click', async () => {
  if (state.busy) return;
  setBusy(true);
  try {
    const plan = await ensureQwen().load($('model-backend').value);
    state.engine = 'qwen';
    setModelStatus(`Qwen3-0.6B ready (${plan.device}, ${plan.dtype}).`);
  } catch (error) {
    setModelStatus(`Qwen load failed: ${error.message}`);
    $('browser-help').open = true;
  } finally {
    setBusy(false);
    render();
  }
});
$('reset').addEventListener('click', () => {
  state.current = initialState(DEFAULTS.ai);
  state.observed = copy(DEFAULTS.message);
  state.mode = 'vector';
  state.evolve = true;
  state.turns = [];
  state.recent = null;
  $('conversation').replaceChildren();
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.textContent = 'Enter a message and submit a turn to observe the selected dynamics model.';
  $('conversation').append(empty);
  $('target-response').value = '';
  $('export-hint').textContent = 'Run an interaction before exporting an example.';
  render();
});
$('chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const message = $('message').value.trim();
  if (!message) return;
  if (state.engine === 'qwen' && (!qwen?.ready || qwen.preference !== $('model-backend').value)) {
    setModelStatus('Load Qwen with the selected backend before running real inference.');
    return;
  }
  const before = {ai: copy(state.current.ai), phase: state.current.phase};
  const observed = copy(state.observed);
  const mode = state.mode;
  const evolve = state.evolve;
  const result = stepAffect(before, observed, {mode});
  const effective = evolve ? result.ai : before.ai;
  const effectivePhase = evolve ? result.phase : before.phase;
  setBusy(true);
  try {
    let response;
    if (state.engine === 'qwen') {
      setModelStatus('Qwen is generating locally in your browser…');
      const messages = buildQwenMessages({
        message, mode, before: before.ai, observed, after: effective,
        phase: effectivePhase, alignment: result.alignment,
        history: state.turns.filter((turn) => turn.engine === 'qwen'),
      });
      response = await qwen.generate(messages);
      setModelStatus(`Qwen3-0.6B ready (${qwen.plan.device}, ${qwen.plan.dtype}).`);
    } else {
      response = templateResponse(message, effective);
    }
    const turn = {
      schema_version: 1,
      mode,
      engine: state.engine,
      user: message,
      initial_ai_vad: before.ai,
      observed_vad: observed,
      state_after_vad: copy(effective),
      phase_before: before.phase,
      phase_after: effectivePhase,
      evolve,
      alignment: result.alignment,
      assistant: response,
    };
    state.turns.push(turn);
    state.recent = result;
    if (evolve) state.current = {ai: result.ai, phase: result.phase};
    appendBubble('user', message, `MESSAGE · ${mode}`);
    appendBubble('sim', response, state.engine === 'qwen'
      ? 'QWEN3-0.6B · REAL MODEL / TEXT PROMPT CONDITIONING'
      : 'DETERMINISTIC TEMPLATE · NOT MODEL OUTPUT');
    $('message').value = '';
    $('target-response').value = '';
    $('export-hint').textContent = 'Enter a human-approved target response to export a training example.';
  } catch (error) {
    // Generation failure must not advance internal state or commit a training example.
    setModelStatus(`Generation failed: ${error.message}`);
    $('browser-help').open = true;
  } finally {
    setBusy(false);
    render();
  }
});
$('export-trace').addEventListener('click', () => downloadJSON('affective-trace.json', {
  schema_version: 1, kind: 'simulation_trace', data: state.turns,
}));
$('export-example').addEventListener('click', () => {
  const turn = state.turns.at(-1);
  const target = $('target-response').value.trim();
  if (!turn || !target) {
    $('export-hint').textContent = 'An interaction and an authored target response are required.';
    return;
  }
  const {assistant, engine, alignment, ...record} = turn;
  downloadJSON('affective-approved-example.json', {...record,
    assistant: target, provenance: 'human-authored', approved: true,
  });
  $('export-hint').textContent = 'Example exported. Review the values before adding it to a dataset.';
});
renderBrowserGuide();
render();
void checkWebGPU();
