import {DEFAULTS, clamp, initialState, stepAffect, affectName, templateResponse} from './affect.mjs';

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
};

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
}

function setVectorPoint(prefix, v, a) {
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
    const rect = plot.getBoundingClientRect();
    setVectorPoint(prefix,
      2 * (event.clientX - rect.left) / rect.width - 1,
      1 - 2 * (event.clientY - rect.top) / rect.height);
  };
  plot.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    plot.setPointerCapture(event.pointerId);
    updateFromPointer(event);
  });
  plot.addEventListener('pointermove', (event) => {
    if (plot.hasPointerCapture(event.pointerId)) updateFromPointer(event);
  });
  plot.addEventListener('keydown', (event) => {
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
    const target = prefix === 'ai' ? state.current.ai : state.observed;
    target[2] = Number(event.target.value);
    state.recent = null;
    render();
  });
}
for (const mode of ['vector', 'phase']) {
  $('mode-' + mode).addEventListener('click', () => {
    state.mode = mode;
    state.recent = null;
    render();
  });
}
$('evolve').addEventListener('change', (event) => {state.evolve = event.target.checked;render();});
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
$('chat-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const message = $('message').value.trim();
  if (!message) return;
  const before = {ai: copy(state.current.ai), phase: state.current.phase};
  const result = stepAffect(before, state.observed, {mode: state.mode});
  const effective = state.evolve ? result.ai : before.ai;
  const response = templateResponse(message, effective);
  const turn = {
    schema_version: 1,
    mode: state.mode,
    user: message,
    initial_ai_vad: before.ai,
    observed_vad: copy(state.observed),
    state_after_vad: copy(effective),
    phase_before: before.phase,
    phase_after: state.evolve ? result.phase : before.phase,
    evolve: state.evolve,
    alignment: result.alignment,
    demo_template: response,
  };
  state.turns.push(turn);
  state.recent = result;
  if (state.evolve) state.current = {ai: result.ai, phase: result.phase};
  appendBubble('user', message, `MESSAGE · ${state.mode}`);
  appendBubble('sim', response, 'DETERMINISTIC TEMPLATE · NOT MODEL OUTPUT');
  $('message').value = '';
  $('target-response').value = '';
  $('export-hint').textContent = 'Enter a human-approved target response to export a training example.';
  render();
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
  const {demo_template, alignment, ...record} = turn;
  downloadJSON('affective-approved-example.json', {...record,
    assistant: target, provenance: 'human-authored', approved: true,
  });
  $('export-hint').textContent = 'Example exported. Review the values before adding it to a dataset.';
});
render();
